use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::StreamExt;
use ort::{session::Session, value::Tensor};
use tokenizers::{
    PaddingParams, PaddingStrategy, Tokenizer, TruncationDirection, TruncationParams,
    TruncationStrategy,
};
use tokio::io::AsyncWriteExt;
use tokio::{fs, task};

use crate::embedding::Provider;
use crate::error::{ClaudixError, RecoveryHint, Result};
use crate::types::Dimension;

pub const BUNDLED_MODEL_ID: &str = "bge-small-en-v1.5";
pub const BUNDLED_MODEL_FILENAME: &str = "bge-small-en-v1.5.onnx";
pub const BUNDLED_TOKENIZER_FILENAME: &str = "tokenizer.json";
pub const BUNDLED_OUTPUT_NAME: &str = "last_hidden_state";
pub const BUNDLED_MAX_SEQUENCE_LENGTH: usize = 512;
pub const BUNDLED_DIMENSIONS: Dimension = Dimension(384);
const BUNDLED_MODEL_URL: &str =
    "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/onnx/model.onnx";
const BUNDLED_TOKENIZER_URL: &str =
    "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/tokenizer.json";
const DOWNLOAD_TIMEOUT_SECS: u64 = 600;

#[derive(Debug, Clone)]
pub struct BundledProvider {
    inner: Arc<Inner>,
}

#[derive(Debug)]
struct Inner {
    model_id: String,
    dimensions: Dimension,
    backend: Option<LoadedBackend>,
}

#[derive(Debug)]
struct LoadedBackend {
    tokenizer: Tokenizer,
    session: Mutex<Session>,
    requires_token_type_ids: bool,
}

#[derive(Debug)]
struct AssetPaths {
    model: PathBuf,
    tokenizer: PathBuf,
}

impl BundledProvider {
    pub async fn new(model_id: impl Into<String>, dimensions: Dimension) -> Result<Self> {
        Self::from_cache_dir(default_cache_dir()?, model_id, dimensions).await
    }

    pub async fn from_cache_dir(
        cache_dir: impl AsRef<Path>,
        model_id: impl Into<String>,
        dimensions: Dimension,
    ) -> Result<Self> {
        let model_id = model_id.into();
        validate_model_contract(&model_id, dimensions)?;

        let paths = AssetPaths::new(cache_dir.as_ref());
        ensure_assets_exist(&paths, &model_id).await?;

        let tokenizer = load_tokenizer(&paths.tokenizer)?;
        let session = Session::builder()
            .map_err(ort_error)?
            .commit_from_file(&paths.model)
            .map_err(ort_error)?;
        let requires_token_type_ids = session
            .inputs()
            .iter()
            .any(|input| input.name() == "token_type_ids");

        Ok(Self {
            inner: Arc::new(Inner {
                model_id,
                dimensions,
                backend: Some(LoadedBackend {
                    tokenizer,
                    session: Mutex::new(session),
                    requires_token_type_ids,
                }),
            }),
        })
    }

    #[cfg(test)]
    fn unloaded_for_tests(model_id: impl Into<String>, dimensions: Dimension) -> Self {
        Self {
            inner: Arc::new(Inner {
                model_id: model_id.into(),
                dimensions,
                backend: None,
            }),
        }
    }

    fn embed_blocking(&self, batch: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let Some(loaded) = &self.inner.backend else {
            return Err(ClaudixError::Embedding(
                "bundled provider test instance is not loaded".into(),
            ));
        };

        let encodings = loaded
            .tokenizer
            .encode_batch(batch, true)
            .map_err(tokenizer_error)?;

        if encodings.is_empty() {
            return Ok(Vec::new());
        }

        let batch_size = encodings.len();
        let sequence_length = encodings[0].get_ids().len();

        let input_ids = flatten_u32_fields(&encodings, |encoding| encoding.get_ids())?;
        let attention_mask =
            flatten_u32_fields(&encodings, |encoding| encoding.get_attention_mask())?;
        let token_type_ids = loaded
            .requires_token_type_ids
            .then(|| flatten_u32_fields(&encodings, |encoding| encoding.get_type_ids()))
            .transpose()?;

        let mut session = loaded
            .session
            .lock()
            .map_err(|_| ClaudixError::Embedding("bundled session lock poisoned".into()))?;

        let outputs = match token_type_ids {
            Some(token_type_ids) => session
                .run(ort::inputs! {
                    "input_ids" => Tensor::from_array(([batch_size, sequence_length], input_ids)).map_err(ort_error)?,
                    "attention_mask" => Tensor::from_array(([batch_size, sequence_length], attention_mask.clone())).map_err(ort_error)?,
                    "token_type_ids" => Tensor::from_array(([batch_size, sequence_length], token_type_ids)).map_err(ort_error)?,
                })
                .map_err(ort_error)?,
            None => session
                .run(ort::inputs! {
                    "input_ids" => Tensor::from_array(([batch_size, sequence_length], input_ids)).map_err(ort_error)?,
                    "attention_mask" => Tensor::from_array(([batch_size, sequence_length], attention_mask.clone())).map_err(ort_error)?,
                })
                .map_err(ort_error)?,
        };

        let (shape, values) = outputs[BUNDLED_OUTPUT_NAME]
            .try_extract_tensor::<f32>()
            .map_err(ort_error)?;
        let shape = shape
            .iter()
            .map(|dimension| {
                usize::try_from(*dimension)
                    .map_err(|_| ClaudixError::Embedding("negative output shape".into()))
            })
            .collect::<Result<Vec<_>>>()?;
        let output_dimensions = validate_output_shape(&shape, batch_size, self.inner.dimensions)?;

        Ok(mean_pool_and_normalize(
            values,
            &attention_mask,
            batch_size,
            sequence_length,
            output_dimensions,
        ))
    }
}

#[async_trait]
impl Provider for BundledProvider {
    fn name(&self) -> &str {
        "bundled"
    }

    fn dimensions(&self) -> Dimension {
        self.inner.dimensions
    }

    fn model_id(&self) -> &str {
        &self.inner.model_id
    }

    async fn embed(&self, batch: &[&str]) -> Result<Vec<Vec<f32>>> {
        if batch.is_empty() {
            return Ok(Vec::new());
        }

        let batch = batch
            .iter()
            .map(|text| (*text).to_owned())
            .collect::<Vec<_>>();
        let provider = self.clone();

        task::spawn_blocking(move || provider.embed_blocking(batch))
            .await
            .map_err(|error| ClaudixError::Embedding(error.to_string()))?
    }

    async fn health_check(&self) -> Result<()> {
        if self.inner.backend.is_some() {
            Ok(())
        } else {
            Err(ClaudixError::Embedding(
                "bundled provider test instance is not loaded".into(),
            ))
        }
    }
}

impl AssetPaths {
    fn new(cache_dir: &Path) -> Self {
        Self {
            model: cache_dir.join(BUNDLED_MODEL_FILENAME),
            tokenizer: cache_dir.join(BUNDLED_TOKENIZER_FILENAME),
        }
    }
}

fn validate_model_contract(model_id: &str, dimensions: Dimension) -> Result<()> {
    if model_id != BUNDLED_MODEL_ID {
        return Err(ClaudixError::Embedding(format!(
            "bundled provider only supports model {BUNDLED_MODEL_ID}, got {model_id}"
        )));
    }

    if dimensions != BUNDLED_DIMENSIONS {
        return Err(ClaudixError::DimensionMismatch {
            store_dim: BUNDLED_DIMENSIONS.0,
            model_dim: dimensions.0,
            recovery: RecoveryHint("Set [embedding].dimensions = 384 for the bundled provider"),
        });
    }

    Ok(())
}

async fn ensure_assets_exist(paths: &AssetPaths, model_id: &str) -> Result<()> {
    if paths.model.exists() && paths.tokenizer.exists() {
        return Ok(());
    }

    if let Some(parent) = paths.model.parent() {
        fs::create_dir_all(parent).await?;
    }

    eprintln!("downloading default embeddings model (~120MB)...");
    download_asset(BUNDLED_MODEL_URL, &paths.model).await?;
    download_asset(BUNDLED_TOKENIZER_URL, &paths.tokenizer).await?;

    if paths.model.exists() && paths.tokenizer.exists() {
        return Ok(());
    }

    Err(ClaudixError::BundledAssetsMissing {
        model_id: model_id.to_owned(),
        recovery: RecoveryHint(
            "Run claudix again after restoring network access, or switch to [embedding] provider = \"http\"",
        ),
    })
}

async fn download_asset(url: &str, destination: &Path) -> Result<()> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(DOWNLOAD_TIMEOUT_SECS))
        .build()?;
    let response = client.get(url).send().await?.error_for_status()?;
    let mut stream = response.bytes_stream();
    let temp_path = destination.with_extension("download");
    let mut file = fs::File::create(&temp_path).await?;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        file.write_all(&chunk).await?;
    }
    file.flush().await?;
    drop(file);
    fs::rename(temp_path, destination).await?;
    Ok(())
}

fn default_cache_dir() -> Result<PathBuf> {
    dirs::home_dir()
        .map(|home| home.join(".claude").join("claudix").join("models"))
        .ok_or_else(|| {
            ClaudixError::Embedding("failed to resolve bundled model cache directory".into())
        })
}

fn load_tokenizer(path: &Path) -> Result<Tokenizer> {
    let mut tokenizer = Tokenizer::from_file(path).map_err(tokenizer_error)?;
    let pad_id = tokenizer.token_to_id("[PAD]").unwrap_or(0);

    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length: BUNDLED_MAX_SEQUENCE_LENGTH,
            strategy: TruncationStrategy::LongestFirst,
            stride: 0,
            direction: TruncationDirection::Right,
        }))
        .map_err(tokenizer_error)?;
    tokenizer.with_padding(Some(PaddingParams {
        strategy: PaddingStrategy::BatchLongest,
        pad_id,
        ..PaddingParams::default()
    }));

    Ok(tokenizer)
}

fn flatten_u32_fields<F>(encodings: &[tokenizers::Encoding], field: F) -> Result<Vec<i64>>
where
    F: Fn(&tokenizers::Encoding) -> &[u32],
{
    let sequence_length = encodings[0].get_ids().len();
    let mut values = Vec::with_capacity(encodings.len() * sequence_length);

    for encoding in encodings {
        let slice = field(encoding);
        if slice.len() != sequence_length {
            return Err(ClaudixError::Embedding(
                "tokenizer returned inconsistent batch sequence lengths".into(),
            ));
        }

        for value in slice {
            values.push(i64::from(*value));
        }
    }

    Ok(values)
}

fn validate_output_shape(
    shape: &[usize],
    batch_size: usize,
    dimensions: Dimension,
) -> Result<usize> {
    if shape.len() != 3 {
        return Err(ClaudixError::Embedding(format!(
            "bundled model output must have rank 3, got shape {shape:?}"
        )));
    }

    if shape[0] != batch_size {
        return Err(ClaudixError::Embedding(format!(
            "bundled model returned batch size {}, expected {batch_size}",
            shape[0]
        )));
    }

    let actual_dimensions = shape[2];
    let expected_dimensions = usize::from(dimensions.0);
    if actual_dimensions != expected_dimensions {
        return Err(ClaudixError::DimensionMismatch {
            store_dim: dimensions.0,
            model_dim: u16::try_from(actual_dimensions).unwrap_or(u16::MAX),
            recovery: RecoveryHint(
                "Use the bundled bge-small-en-v1.5 export with 384-dimensional hidden states",
            ),
        });
    }

    Ok(actual_dimensions)
}

fn mean_pool_and_normalize(
    values: &[f32],
    attention_mask: &[i64],
    batch_size: usize,
    sequence_length: usize,
    dimensions: usize,
) -> Vec<Vec<f32>> {
    let mut vectors = Vec::with_capacity(batch_size);

    for batch_index in 0..batch_size {
        let mut vector = vec![0.0; dimensions];
        let mut token_count = 0.0f32;

        for token_index in 0..sequence_length {
            if attention_mask[batch_index * sequence_length + token_index] == 0 {
                continue;
            }

            token_count += 1.0;
            let base = (batch_index * sequence_length + token_index) * dimensions;
            for dimension_index in 0..dimensions {
                vector[dimension_index] += values[base + dimension_index];
            }
        }

        if token_count > 0.0 {
            for value in &mut vector {
                *value /= token_count;
            }
            normalize_l2(&mut vector);
        }

        vectors.push(vector);
    }

    vectors
}

fn normalize_l2(vector: &mut [f32]) {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm == 0.0 {
        return;
    }

    for value in vector {
        *value /= norm;
    }
}

fn ort_error(error: impl std::fmt::Display) -> ClaudixError {
    ClaudixError::Embedding(error.to_string())
}

fn tokenizer_error(error: impl std::fmt::Display) -> ClaudixError {
    ClaudixError::Embedding(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn missing_model_triggers_download_or_network_error() {
        let tempdir = tempdir().ok().unwrap_or_else(|| unreachable!());

        let result =
            BundledProvider::from_cache_dir(tempdir.path(), BUNDLED_MODEL_ID, BUNDLED_DIMENSIONS)
                .await;
        assert!(matches!(result, Ok(_) | Err(ClaudixError::Http(_))));
    }

    #[tokio::test]
    async fn missing_tokenizer_triggers_download_or_network_error() {
        let tempdir = tempdir().ok().unwrap_or_else(|| unreachable!());
        let model_path = tempdir.path().join(BUNDLED_MODEL_FILENAME);
        std::fs::write(model_path, b"placeholder")
            .ok()
            .unwrap_or_else(|| unreachable!());

        let result =
            BundledProvider::from_cache_dir(tempdir.path(), BUNDLED_MODEL_ID, BUNDLED_DIMENSIONS)
                .await;
        assert!(matches!(result, Ok(_) | Err(ClaudixError::Http(_))));
    }

    #[test]
    fn metadata_methods_return_configured_values() {
        let provider = BundledProvider::unloaded_for_tests(BUNDLED_MODEL_ID, BUNDLED_DIMENSIONS);

        assert_eq!(provider.name(), "bundled");
        assert_eq!(provider.model_id(), BUNDLED_MODEL_ID);
        assert_eq!(provider.dimensions(), BUNDLED_DIMENSIONS);
    }

    #[test]
    fn output_dimension_mismatch_returns_typed_error() {
        let error = validate_output_shape(&[1, 3, 2], 1, BUNDLED_DIMENSIONS);

        assert!(matches!(error, Err(ClaudixError::DimensionMismatch { .. })));
    }

    #[test]
    fn mean_pooling_respects_attention_mask_and_normalizes() {
        let values = vec![
            1.0, 0.0, 0.0, 1.0, // token 1
            0.0, 1.0, 1.0, 0.0, // token 2
            3.0, 3.0, 3.0, 3.0, // padding token, ignored
        ];
        let attention_mask = vec![1, 1, 0];

        let vectors = mean_pool_and_normalize(&values, &attention_mask, 1, 3, 4);

        assert_eq!(vectors.len(), 1);
        assert_eq!(vectors[0].len(), 4);
        let norm = vectors[0]
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }
}
