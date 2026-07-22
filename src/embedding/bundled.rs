use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::StreamExt;
use ort::{session::Session, value::Tensor};
use sha2::{Digest, Sha256};
use tokenizers::{Tokenizer, TruncationDirection, TruncationParams, TruncationStrategy};
use tokio::io::AsyncWriteExt;
use tokio::{fs, task};

use crate::embedding::Provider;
use crate::error::{ClaudixError, RecoveryHint, Result};
use crate::prompts::hints;
use crate::types::Dimension;

/// How a model turns `last_hidden_state` into one vector per input. Every model
/// publishes exactly one correct answer for itself and reading the wrong one
/// degrades silently, so it is pinned beside the model id instead of assumed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pooling {
    /// First token, `[CLS]`. What the BGE and GTE families specify.
    Cls,
    /// Attention-masked mean across tokens.
    Mean,
}

/// One selectable bundled ONNX model, with every property that has to agree
/// about it stored in a single record.
///
/// Id, dimensions, and pooling head are only ever read together through a table
/// lookup, so no config path can pair one model's id with another's width or
/// reduction. Splitting them across free-standing constants is what let the
/// provider mean-pool a CLS model for several releases while everything still
/// typechecked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BundledModel {
    /// Config value for `[embedding].model`.
    pub id: &'static str,
    pub dimensions: Dimension,
    pub pooling: Pooling,
    /// Token cap handed to the tokenizer's truncation. Not the model's
    /// architectural ceiling; see the per-entry note for why each value.
    pub max_sequence_length: usize,
    /// Cache filenames carry the pinned revision, so bumping `*_url` to a new
    /// revision without renaming is impossible to do silently: a stale file
    /// under the old name is no longer a name any entry claims.
    pub model_filename: &'static str,
    pub model_url: &'static str,
    pub model_sha256: &'static str,
    pub model_size_bytes: u64,
    pub tokenizer_filename: &'static str,
    pub tokenizer_url: &'static str,
    pub tokenizer_sha256: &'static str,
    pub tokenizer_size_bytes: u64,
}

/// `Alibaba-NLP/gte-modernbert-base`, Apache-2.0, pinned at revision
/// `e7f32e3c00f91d699e8c43b53106206bcc72bb22`. ModernBERT graph: `input_ids` +
/// `attention_mask`, no `token_type_ids`, output `last_hidden_state`. Its card
/// and `1_Pooling/config.json` specify `[CLS]`.
const GTE_MODERNBERT_BASE: BundledModel = BundledModel {
    id: "gte-modernbert-base",
    dimensions: Dimension(768),
    pooling: Pooling::Cls,
    // The model accepts 8192 (`max_position_embeddings`); this is a truncation
    // cap, not that ceiling. Peak memory is bounded by EMBED_TOKEN_BUDGET, not
    // by this value, so the choice here is purely about retrieval: indexed
    // chunks average ~290 tokens, 2048 embeds every realistic whole function
    // intact, and the rare longer one truncates to its first 2048 rather than
    // costing 8192^2 attention as a single-item sub-batch. Raise it if long
    // functions start losing their tails in search.
    max_sequence_length: 2048,
    model_filename: "gte-modernbert-base-e7f32e3c.onnx",
    model_url: "https://huggingface.co/Alibaba-NLP/gte-modernbert-base/resolve/e7f32e3c00f91d699e8c43b53106206bcc72bb22/onnx/model_int8.onnx",
    model_sha256: "bae96b276d342bf86eeee07c1bdbc0c75bb82bf4033941aab7fabc1e33ee3b44",
    model_size_bytes: 150_218_016,
    tokenizer_filename: "gte-modernbert-base-e7f32e3c.tokenizer.json",
    tokenizer_url: "https://huggingface.co/Alibaba-NLP/gte-modernbert-base/resolve/e7f32e3c00f91d699e8c43b53106206bcc72bb22/tokenizer.json",
    tokenizer_sha256: "6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30",
    tokenizer_size_bytes: 3_583_228,
};

/// `BAAI/bge-small-en-v1.5`, pinned at revision
/// `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`. Publishes
/// `pooling_mode_cls_token: true` / `pooling_mode_mean_tokens: false` in
/// `1_Pooling/config.json`, and its card spells out "you select the last hidden
/// state of the first token (i.e. [CLS]) as the sentence embedding". Read that
/// file from any model added here rather than copying this line: at least one
/// published model (`codefuse-ai/F2LLM-v2-80M`) ships a stale one, so
/// cross-check against `config.json` and `modules.json` too.
const BGE_SMALL_EN_V1_5: BundledModel = BundledModel {
    id: "bge-small-en-v1.5",
    dimensions: Dimension(384),
    pooling: Pooling::Cls,
    // BERT-family 512-token position limit; the model cannot read past it.
    max_sequence_length: 512,
    model_filename: "bge-small-en-v1.5-5c38ec7c.onnx",
    model_url: "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/onnx/model.onnx",
    model_sha256: "828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35",
    model_size_bytes: 133_093_490,
    tokenizer_filename: "bge-small-en-v1.5-5c38ec7c.tokenizer.json",
    tokenizer_url: "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/tokenizer.json",
    tokenizer_sha256: "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
    tokenizer_size_bytes: 711_396,
};

/// Every model `[embedding].provider = "bundled"` accepts.
pub const BUNDLED_MODELS: [BundledModel; 2] = [GTE_MODERNBERT_BASE, BGE_SMALL_EN_V1_5];

/// What a config that does not name a model gets.
pub const DEFAULT_BUNDLED_MODEL: &BundledModel = &GTE_MODERNBERT_BASE;

/// Asset filenames written by releases that cached one hardcoded model under
/// revision-less names. Purged by [`remove_legacy_assets`] because their
/// provenance is unknown — they predate digest verification — and no entry can
/// ever reclaim the name. Listed literally so cleanup can only unlink names
/// this crate itself chose.
const LEGACY_ASSET_FILENAMES: [&str; 2] = ["bge-small-en-v1.5.onnx", "tokenizer.json"];

pub const BUNDLED_OUTPUT_NAME: &str = "last_hidden_state";

/// Padded tokens — items x longest sequence — allowed through one ONNX run.
///
/// `[embedding].batch_size` counts items, which bounds nothing: the tokenizer
/// pads a sub-batch to its longest member, so one 2048-token chunk drags all 32
/// items in its batch to 2048 and the model then materialises attention over
/// every padded position. That product, not the item count, is what allocates,
/// and at the default 32 items it reached 19.4GB RSS on a 244-file repo.
///
/// Budgeting the product instead caps peak memory for any model at any sequence
/// length, and leaves `batch_size` meaning "at most this many", which is what
/// callers already assume. 2048 equals one full-length gte chunk, so a
/// max-length chunk is a sub-batch of one and pads nothing, while the ~290-token
/// average packs ~7 per run. Measured peak RSS indexing paru under a 6GB cap:
/// gte 0.97GB, bge 0.43GB — both far under the OOM the item-count batching hit
/// (gte was killed at the same 6GB cap before this). A wider budget (4096)
/// measured 1.75GB for gte with no throughput win worth the lost headroom.
const EMBED_TOKEN_BUDGET: usize = 2048;
const DOWNLOAD_TIMEOUT_SECS: u64 = 600;

/// The table entry for `id`, or `None` when the bundled provider does not
/// ship it.
pub fn bundled_model(id: &str) -> Option<&'static BundledModel> {
    BUNDLED_MODELS.iter().find(|model| model.id == id)
}

/// Comma-separated model ids, for error messages that must stay in step with
/// the table.
pub fn bundled_model_ids() -> String {
    BUNDLED_MODELS
        .iter()
        .map(|model| model.id)
        .collect::<Vec<_>>()
        .join(", ")
}

#[derive(Debug, Clone)]
pub struct BundledProvider {
    inner: Arc<Inner>,
}

#[derive(Debug)]
struct Inner {
    model: &'static BundledModel,
    backend: Option<LoadedBackend>,
}

#[derive(Debug)]
struct LoadedBackend {
    tokenizer: Tokenizer,
    session: Mutex<Session>,
    requires_token_type_ids: bool,
    /// Padding is applied per sub-batch here rather than by the tokenizer, so
    /// the id it would have used has to be carried along.
    pad_id: u32,
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

    async fn from_cache_dir(
        cache_dir: impl AsRef<Path>,
        model_id: impl Into<String>,
        dimensions: Dimension,
    ) -> Result<Self> {
        let model = validate_model_contract(&model_id.into(), dimensions)?;

        let paths = AssetPaths::new(cache_dir.as_ref(), model);
        ensure_assets_exist(&paths, model).await?;

        let (tokenizer, pad_id) = load_tokenizer(&paths.tokenizer, model.max_sequence_length)?;
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
                model,
                backend: Some(LoadedBackend {
                    tokenizer,
                    session: Mutex::new(session),
                    requires_token_type_ids,
                    pad_id,
                }),
            }),
        })
    }

    #[cfg(test)]
    fn unloaded_for_tests(model: &'static BundledModel) -> Self {
        Self {
            inner: Arc::new(Inner {
                model,
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

        let lengths: Vec<usize> = encodings
            .iter()
            .map(|encoding| encoding.get_ids().len())
            .collect();
        let mut vectors: Vec<Option<Vec<f32>>> = vec![None; encodings.len()];

        let mut session = loaded
            .session
            .lock()
            .map_err(|_| ClaudixError::Embedding("bundled session lock poisoned".into()))?;

        for group in plan_sub_batches(&lengths, EMBED_TOKEN_BUDGET) {
            let rows: Vec<&tokenizers::Encoding> =
                group.iter().map(|index| &encodings[*index]).collect();

            for (index, vector) in group.iter().zip(self.run_sub_batch(
                &mut session,
                loaded.requires_token_type_ids,
                loaded.pad_id,
                &rows,
            )?) {
                vectors[*index] = Some(vector);
            }
        }

        vectors
            .into_iter()
            .enumerate()
            .map(|(index, vector)| {
                vector.ok_or_else(|| {
                    ClaudixError::Embedding(format!("sub-batch planning skipped input {index}"))
                })
            })
            .collect()
    }

    /// One ONNX run over a sub-batch the planner already sized.
    fn run_sub_batch(
        &self,
        session: &mut Session,
        requires_token_type_ids: bool,
        pad_id: u32,
        rows: &[&tokenizers::Encoding],
    ) -> Result<Vec<Vec<f32>>> {
        let batch_size = rows.len();
        let dimensions = usize::from(self.inner.model.dimensions.0);
        let Some(sequence_length) = rows.iter().map(|row| row.get_ids().len()).max() else {
            return Ok(Vec::new());
        };
        if sequence_length == 0 {
            return Ok(vec![vec![0.0; dimensions]; batch_size]);
        }

        let input_ids = padded_field(rows, sequence_length, pad_id, |row| row.get_ids());
        let attention_mask = padded_field(rows, sequence_length, 0, |row| row.get_attention_mask());
        let token_type_ids = requires_token_type_ids
            .then(|| padded_field(rows, sequence_length, 0, |row| row.get_type_ids()));

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
        let output_dimensions =
            validate_output_shape(&shape, batch_size, self.inner.model.dimensions)?;

        Ok(pool_and_normalize(
            self.inner.model.pooling,
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
        self.inner.model.dimensions
    }

    fn model_id(&self) -> &str {
        self.inner.model.id
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
    fn new(cache_dir: &Path, model: &BundledModel) -> Self {
        Self {
            model: cache_dir.join(model.model_filename),
            tokenizer: cache_dir.join(model.tokenizer_filename),
        }
    }
}

/// Resolve `model_id` against the table and confirm the caller asked for the
/// width that entry actually emits.
fn validate_model_contract(model_id: &str, dimensions: Dimension) -> Result<&'static BundledModel> {
    let Some(model) = bundled_model(model_id) else {
        return Err(ClaudixError::Embedding(format!(
            "bundled provider supports {}, got {model_id}",
            bundled_model_ids()
        )));
    };

    if dimensions != model.dimensions {
        return Err(ClaudixError::DimensionMismatch {
            store_dim: model.dimensions.0,
            model_dim: dimensions.0,
            recovery: RecoveryHint(hints::BUNDLED_DIMENSIONS),
        });
    }

    Ok(model)
}

async fn ensure_assets_exist(paths: &AssetPaths, model: &BundledModel) -> Result<()> {
    if let Some(parent) = paths.model.parent() {
        fs::create_dir_all(parent).await?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let dir_perms = std::fs::Permissions::from_mode(0o700);
            if let Err(e) = fs::set_permissions(parent, dir_perms).await {
                tracing::warn!("could not set perms on model cache dir: {e}");
            }
        }
    }

    if !is_cached(&paths.model, model.model_size_bytes).await {
        eprintln!(
            "downloading bundled embeddings model {} (~{}MB)...",
            model.id,
            model.model_size_bytes / 1_000_000
        );
        download_verified(
            model.model_url,
            &paths.model,
            model.model_sha256,
            model.model_size_bytes,
        )
        .await?;
    }

    if !is_cached(&paths.tokenizer, model.tokenizer_size_bytes).await {
        download_verified(
            model.tokenizer_url,
            &paths.tokenizer,
            model.tokenizer_sha256,
            model.tokenizer_size_bytes,
        )
        .await?;
    }

    if !paths.model.exists() || !paths.tokenizer.exists() {
        return Err(ClaudixError::BundledAssetsMissing {
            model_id: model.id.to_owned(),
            recovery: RecoveryHint(hints::DOWNLOAD_BUNDLED_ASSETS),
        });
    }

    // Only once this model is confirmed usable. Purging first would strand a
    // user who switches models offline: the working assets go, the replacement
    // download fails, and nothing loadable is left on disk.
    if let Some(parent) = paths.model.parent() {
        remove_legacy_assets(parent).await;
    }

    Ok(())
}

/// Whether a cached asset can be reused without re-downloading.
///
/// Only the length is checked. That is honest because the bytes under this
/// name were hashed *off the disk they now sit on* before being published, and
/// the name encodes the pinned revision, so the remaining question is whether
/// they survived since — which a truncated or partially-written file fails and
/// a stat answers for free. Re-hashing 150MB instead would cost ~0.7s on the
/// first search of every session and on every hook cold-load, to defend a 0700
/// user-owned directory against an attacker who could equally rewrite whatever
/// we hashed it against. If that trade ever changes, hash here rather than
/// adding a sidecar: a sidecar is forgeable by exactly the same attacker.
async fn is_cached(path: &Path, expected_size: u64) -> bool {
    matches!(fs::metadata(path).await, Ok(metadata) if metadata.len() == expected_size)
}

/// Unlink the revision-less asset names older releases wrote.
///
/// Only [`LEGACY_ASSET_FILENAMES`] is touched, never another table entry's
/// assets. Entries are revision-stamped and can coexist: total cache size is
/// bounded by the table (~285MB for two models), which is far cheaper than the
/// alternative, since deleting the inactive entry makes two repos pinned to
/// different models re-download 150MB apiece every session. Legacy names are
/// different — they came from a mutable `resolve/main` with no digest on
/// record, so their provenance is unverifiable and they can never be reclaimed
/// by any entry.
///
/// Best-effort throughout: a stale asset costs disk, never correctness, and
/// this runs on the hook path where a permissions error must not surface.
async fn remove_legacy_assets(cache_dir: &Path) {
    for filename in LEGACY_ASSET_FILENAMES {
        let path = cache_dir.join(filename);
        match fs::remove_file(&path).await {
            Ok(()) => tracing::info!("removed legacy bundled asset {}", path.display()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => tracing::warn!("could not remove {}: {error}", path.display()),
        }
    }
}

/// Stream `url` to a private temp file, verify the bytes that actually landed
/// on disk, and publish under `destination` only if they match.
///
/// Every failure path unlinks the temp file, so a failed fetch can neither be
/// observed under the real name nor leak up to 150MB that nothing later knows
/// how to reclaim — cleanup only knows final names.
async fn download_verified(
    url: &str,
    destination: &Path,
    expected_sha256: &str,
    expected_size: u64,
) -> Result<()> {
    let temp_path = download_temp_path(destination);
    let outcome =
        fetch_and_publish(url, &temp_path, destination, expected_sha256, expected_size).await;

    if outcome.is_err()
        && let Err(error) = fs::remove_file(&temp_path).await
        && error.kind() != std::io::ErrorKind::NotFound
    {
        tracing::warn!(
            "could not remove failed download {}: {error}",
            temp_path.display()
        );
    }

    outcome
}

async fn fetch_and_publish(
    url: &str,
    temp_path: &Path,
    destination: &Path,
    expected_sha256: &str,
    expected_size: u64,
) -> Result<()> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(DOWNLOAD_TIMEOUT_SECS))
        .build()?;
    let response = client.get(url).send().await?.error_for_status()?;
    let mut stream = response.bytes_stream();
    let mut file = fs::File::create(temp_path).await?;

    while let Some(chunk) = stream.next().await {
        file.write_all(&chunk?).await?;
    }
    file.flush().await?;
    drop(file);

    // Hash what is on disk, not what came off the socket. A stream digest only
    // proves the network delivered the right bytes; it says nothing about what
    // the filesystem ended up holding, and it is precisely the on-disk bytes
    // that `is_cached` later reuses on a length check alone.
    let (actual_sha256, actual_size) = file_sha256(temp_path).await?;
    if actual_sha256 != expected_sha256 || actual_size != expected_size {
        return Err(ClaudixError::BundledAssetCorrupt {
            url: url.to_owned(),
            expected: expected_sha256.to_owned(),
            actual: actual_sha256,
            recovery: RecoveryHint(hints::BUNDLED_ASSET_CORRUPT),
        });
    }

    fs::rename(temp_path, destination).await?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let file_perms = std::fs::Permissions::from_mode(0o600);
        if let Err(e) = fs::set_permissions(destination, file_perms).await {
            tracing::warn!("could not set perms on downloaded asset: {e}");
        }
    }
    Ok(())
}

/// sha256 and length of `path`, read back from disk on a blocking thread.
async fn file_sha256(path: &Path) -> Result<(String, u64)> {
    let path = path.to_path_buf();

    task::spawn_blocking(move || {
        use std::io::Read;

        let mut file = std::fs::File::open(&path)?;
        let mut hasher = Sha256::new();
        let mut buffer = vec![0u8; 1 << 20];
        let mut length = 0u64;

        loop {
            let read = file.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
            length += read as u64;
        }

        Ok((hex_encode(&hasher.finalize()), length))
    })
    .await
    .map_err(|error| ClaudixError::Embedding(error.to_string()))?
}

/// A temp path no concurrent download can collide with.
///
/// Two claudix processes fetching one model — two repos on a first run, or the
/// MCP server racing a `claudix index` — both open this path with `O_TRUNC`.
/// On a shared name the second truncates the file while the first keeps its
/// offset, so the first writes past a sparse hole and publishes a file of the
/// right length full of zeroes. The pid separates processes and the counter
/// separates concurrent downloads within one.
///
/// The suffix is appended rather than replacing an extension: `x.tokenizer.json`
/// must not share a temp path with `x.onnx`, which `with_extension` would give.
fn download_temp_path(destination: &Path) -> PathBuf {
    static NONCE: AtomicU64 = AtomicU64::new(0);

    let mut name = destination.as_os_str().to_owned();
    name.push(format!(
        ".{}.{}.download",
        std::process::id(),
        NONCE.fetch_add(1, Ordering::Relaxed)
    ));
    PathBuf::from(name)
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        // Writing to a String cannot fail.
        let _ = write!(encoded, "{byte:02x}");
    }
    encoded
}

fn default_cache_dir() -> Result<PathBuf> {
    dirs::home_dir()
        .map(|home| home.join(".claude").join("claudix").join("models"))
        .ok_or_else(|| {
            ClaudixError::Embedding("failed to resolve bundled model cache directory".into())
        })
}

/// Truncation only. Padding is deliberately left off: the tokenizer's
/// `BatchLongest` pads across everything handed to `encode_batch`, which is the
/// whole point of what has to be split up, so encodings are kept at their true
/// lengths and padded per sub-batch instead.
fn load_tokenizer(path: &Path, max_sequence_length: usize) -> Result<(Tokenizer, u32)> {
    let mut tokenizer = Tokenizer::from_file(path).map_err(tokenizer_error)?;
    let pad_id = tokenizer.token_to_id("[PAD]").unwrap_or(0);

    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length: max_sequence_length,
            strategy: TruncationStrategy::LongestFirst,
            stride: 0,
            direction: TruncationDirection::Right,
        }))
        .map_err(tokenizer_error)?;
    tokenizer.with_padding(None);

    Ok((tokenizer, pad_id))
}

/// Group `lengths` into sub-batches whose (items x longest member) stays within
/// `token_budget`, shortest first so one long input cannot pad short ones up to
/// its own length.
///
/// An input longer than the budget on its own still gets embedded, as a
/// sub-batch of one — dropping it would silently lose a chunk, and splitting it
/// further is not this layer's call (truncation already happened at the
/// tokenizer, bounded by the model's `max_sequence_length`).
///
/// Returns index groups into `lengths`; the caller reassembles in input order.
fn plan_sub_batches(lengths: &[usize], token_budget: usize) -> Vec<Vec<usize>> {
    let mut order: Vec<usize> = (0..lengths.len()).collect();
    order.sort_by_key(|index| lengths[*index]);

    let mut batches: Vec<Vec<usize>> = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    let mut current_max = 0usize;

    for index in order {
        let widened = current_max.max(lengths[index]);
        if !current.is_empty() && widened * (current.len() + 1) > token_budget {
            batches.push(std::mem::take(&mut current));
            current_max = 0;
        }

        current_max = current_max.max(lengths[index]);
        current.push(index);
    }

    if !current.is_empty() {
        batches.push(current);
    }

    batches
}

/// One row per encoding, right-padded to `sequence_length` with `pad`.
fn padded_field<F>(
    encodings: &[&tokenizers::Encoding],
    sequence_length: usize,
    pad: u32,
    field: F,
) -> Vec<i64>
where
    F: Fn(&tokenizers::Encoding) -> &[u32],
{
    let mut values = Vec::with_capacity(encodings.len() * sequence_length);

    for encoding in encodings {
        let row = field(encoding);
        for position in 0..sequence_length {
            values.push(i64::from(row.get(position).copied().unwrap_or(pad)));
        }
    }

    values
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
            recovery: RecoveryHint(hints::BUNDLED_HIDDEN_STATES),
        });
    }

    Ok(actual_dimensions)
}

/// Reduce `last_hidden_state` to one vector per input.
///
/// Which reduction is not a free choice: a model is trained so that one
/// specific position or statistic carries the sentence embedding, and every
/// other reduction returns a vector that is still plausible, still normalized,
/// and quietly worse. Dispatching on a value pinned beside the model id is what
/// keeps that choice visible.
fn pool_and_normalize(
    strategy: Pooling,
    values: &[f32],
    attention_mask: &[i64],
    batch_size: usize,
    sequence_length: usize,
    dimensions: usize,
) -> Vec<Vec<f32>> {
    match strategy {
        Pooling::Cls => cls_pool_and_normalize(values, batch_size, sequence_length, dimensions),
        Pooling::Mean => mean_pool_and_normalize(
            values,
            attention_mask,
            batch_size,
            sequence_length,
            dimensions,
        ),
    }
}

/// Take the first token's hidden state. `[CLS]` is prepended by the tokenizer
/// and never masked, so the attention mask cannot move which row this reads.
fn cls_pool_and_normalize(
    values: &[f32],
    batch_size: usize,
    sequence_length: usize,
    dimensions: usize,
) -> Vec<Vec<f32>> {
    let mut vectors = Vec::with_capacity(batch_size);

    for batch_index in 0..batch_size {
        let base = batch_index * sequence_length * dimensions;
        let mut vector = match values.get(base..base + dimensions) {
            Some(row) => row.to_vec(),
            None => vec![0.0; dimensions],
        };
        normalize_l2(&mut vector);
        vectors.push(vector);
    }

    vectors
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
            // Bounds-checked like the CLS path: a truncated output tensor or a
            // short mask must degrade to a zero vector, not panic. The hook path
            // fails open, and this runs for whichever table entry declares
            // `Pooling::Mean` — adding one is a data edit that touches no code.
            let masked = attention_mask.get(batch_index * sequence_length + token_index);
            if matches!(masked, None | Some(0)) {
                continue;
            }

            let base = (batch_index * sequence_length + token_index) * dimensions;
            let Some(row) = values.get(base..base + dimensions) else {
                continue;
            };

            token_count += 1.0;
            for (slot, value) in vector.iter_mut().zip(row) {
                *slot += value;
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
    use std::time::Duration;
    use tempfile::tempdir;

    #[tokio::test]
    #[ignore = "hits huggingface network"]
    async fn missing_model_triggers_download_or_network_error() {
        let tempdir = tempdir().ok().unwrap_or_else(|| unreachable!());

        let result = BundledProvider::from_cache_dir(
            tempdir.path(),
            DEFAULT_BUNDLED_MODEL.id,
            DEFAULT_BUNDLED_MODEL.dimensions,
        )
        .await;
        assert!(matches!(result, Ok(_) | Err(ClaudixError::Http(_))));
    }

    #[tokio::test]
    #[ignore = "hits huggingface network"]
    async fn missing_tokenizer_triggers_download_or_network_error() {
        let tempdir = tempdir().ok().unwrap_or_else(|| unreachable!());
        let model_path = tempdir.path().join(DEFAULT_BUNDLED_MODEL.model_filename);
        std::fs::write(model_path, b"placeholder")
            .ok()
            .unwrap_or_else(|| unreachable!());

        let result = BundledProvider::from_cache_dir(
            tempdir.path(),
            DEFAULT_BUNDLED_MODEL.id,
            DEFAULT_BUNDLED_MODEL.dimensions,
        )
        .await;
        assert!(matches!(result, Ok(_) | Err(ClaudixError::Http(_))));
    }

    #[test]
    fn metadata_methods_return_configured_values() {
        let provider = BundledProvider::unloaded_for_tests(DEFAULT_BUNDLED_MODEL);

        assert_eq!(provider.name(), "bundled");
        assert_eq!(provider.model_id(), DEFAULT_BUNDLED_MODEL.id);
        assert_eq!(provider.dimensions(), DEFAULT_BUNDLED_MODEL.dimensions);
    }

    #[test]
    fn output_dimension_mismatch_returns_typed_error() {
        let error = validate_output_shape(&[1, 3, 2], 1, DEFAULT_BUNDLED_MODEL.dimensions);

        assert!(matches!(error, Err(ClaudixError::DimensionMismatch { .. })));
    }

    /// The default is what an unconfigured install embeds with, and flipping it
    /// silently re-embeds every user's corpus, so it is asserted as a literal
    /// rather than read back off the table.
    #[test]
    fn default_bundled_model_is_gte_modernbert_base() {
        assert_eq!(DEFAULT_BUNDLED_MODEL.id, "gte-modernbert-base");
        assert_eq!(DEFAULT_BUNDLED_MODEL.dimensions, Dimension(768));
    }

    /// The published pooling head per model, pinned as literals. The provider
    /// read the mean of every token for its first several releases while both
    /// models specify `[CLS]`, which produced vectors that were normalized,
    /// plausible, and wrong. A round-trip test cannot catch that: encoding and
    /// searching through the same wrong head stays self-consistent. Only an
    /// assertion against what the model publishes does.
    #[test]
    fn every_bundled_model_pools_the_head_its_model_card_specifies() {
        for model in BUNDLED_MODELS {
            assert_eq!(model.pooling, Pooling::Cls, "{}", model.id);
        }
    }

    /// Pins the digests and revisions the assets were verified against by hand.
    /// A bumped URL that keeps an old digest would otherwise fail only at
    /// download time, on a user's machine, after 150MB of transfer.
    #[test]
    fn bundled_model_assets_pin_their_verified_digests() {
        let gte = bundled_model("gte-modernbert-base").unwrap_or_else(|| unreachable!());
        assert!(
            gte.model_url
                .contains("e7f32e3c00f91d699e8c43b53106206bcc72bb22")
        );
        assert_eq!(
            gte.model_sha256,
            "bae96b276d342bf86eeee07c1bdbc0c75bb82bf4033941aab7fabc1e33ee3b44"
        );
        assert_eq!(gte.model_size_bytes, 150_218_016);
        assert_eq!(
            gte.tokenizer_sha256,
            "6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30"
        );
        assert_eq!(gte.tokenizer_size_bytes, 3_583_228);

        let bge = bundled_model("bge-small-en-v1.5").unwrap_or_else(|| unreachable!());
        assert!(
            bge.model_url
                .contains("5c38ec7c405ec4b44b94cc5a9bb96e735b38267a")
        );
        assert_eq!(
            bge.model_sha256,
            "828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35"
        );
        assert_eq!(bge.model_size_bytes, 133_093_490);
        assert_eq!(
            bge.tokenizer_sha256,
            "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66"
        );
        assert_eq!(bge.tokenizer_size_bytes, 711_396);
    }

    /// Two entries sharing an on-disk name would let a model switch load the
    /// other model's weights against this model's declared dimensions.
    #[test]
    fn every_bundled_model_owns_distinct_asset_filenames() {
        let mut names: Vec<&str> = BUNDLED_MODELS
            .iter()
            .flat_map(|model| [model.model_filename, model.tokenizer_filename])
            .collect();
        let total = names.len();
        names.sort_unstable();
        names.dedup();

        assert_eq!(names.len(), total, "bundled asset filenames collide");
    }

    /// Cleanup unlinks every legacy name unconditionally, so a live entry
    /// reusing one would delete the active model's own assets on every build.
    #[test]
    fn no_live_asset_filename_collides_with_a_legacy_name() {
        for model in BUNDLED_MODELS {
            for name in [model.model_filename, model.tokenizer_filename] {
                assert!(
                    !LEGACY_ASSET_FILENAMES.contains(&name),
                    "{name} is both a live and a legacy asset name"
                );
            }
        }
    }

    #[test]
    fn unknown_model_id_is_rejected() {
        let error = validate_model_contract("nomic-embed-text", Dimension(768));

        assert!(matches!(error, Err(ClaudixError::Embedding(_))));
    }

    /// The id/dimension pair is checked against one entry, so borrowing the
    /// other model's width cannot pass.
    #[test]
    fn known_model_with_another_models_dimensions_is_rejected() {
        let error = validate_model_contract("bge-small-en-v1.5", Dimension(768));

        assert!(matches!(error, Err(ClaudixError::DimensionMismatch { .. })));
    }

    #[test]
    fn each_model_resolves_to_its_own_entry() {
        for model in BUNDLED_MODELS {
            let resolved = validate_model_contract(model.id, model.dimensions);

            assert_eq!(resolved.ok(), Some(&model), "{}", model.id);
        }
    }

    #[test]
    fn model_and_tokenizer_get_distinct_temp_paths() {
        let model = download_temp_path(Path::new("/cache/gte-modernbert-base-e7f32e3c.onnx"));
        let tokenizer = download_temp_path(Path::new(
            "/cache/gte-modernbert-base-e7f32e3c.tokenizer.json",
        ));

        assert_ne!(model, tokenizer);
        assert!(model.to_string_lossy().contains(".onnx."));
        assert!(model.to_string_lossy().ends_with(".download"));
    }

    /// Two processes fetching one model both open the temp path with `O_TRUNC`.
    /// On a shared name the second's truncate resets the length while the first
    /// keeps its offset, so the first writes past a sparse hole and publishes a
    /// zero-filled file of exactly the right length — which `is_cached` then
    /// trusts forever. The pid is what separates processes; the counter
    /// separates concurrent downloads inside one.
    #[test]
    fn every_download_claims_its_own_temp_path() {
        let destination = Path::new("/cache/model.onnx");
        let paths: std::collections::HashSet<PathBuf> =
            (0..64).map(|_| download_temp_path(destination)).collect();

        assert_eq!(paths.len(), 64, "temp paths collide within one process");
        for path in &paths {
            assert!(
                path.to_string_lossy()
                    .contains(&format!(".{}.", std::process::id())),
                "temp path does not separate this process from another"
            );
        }
    }

    #[test]
    fn hex_encode_pads_every_byte_to_two_digits() {
        assert_eq!(hex_encode(&[0x00, 0x0f, 0xff]), "000fff");
    }

    #[tokio::test]
    async fn cached_asset_is_reused_only_at_the_published_size() {
        let tempdir = tempdir().ok().unwrap_or_else(|| unreachable!());
        let path = tempdir.path().join("asset.onnx");
        std::fs::write(&path, b"1234567890")
            .ok()
            .unwrap_or_else(|| unreachable!());

        assert!(is_cached(&path, 10).await);
        assert!(!is_cached(&path, 11).await, "truncated file was reused");
        assert!(!is_cached(&tempdir.path().join("absent.onnx"), 10).await);
    }

    /// Purging reaches the pre-table names and nothing else. Deleting the
    /// inactive entry's assets was tried and reverted: revision-stamped names
    /// let entries coexist, and clearing them makes two repos on different
    /// models re-download 150MB apiece every session.
    #[tokio::test]
    async fn purging_takes_legacy_names_and_leaves_every_table_entry() {
        let tempdir = tempdir().ok().unwrap_or_else(|| unreachable!());
        let cache = tempdir.path();
        let bge = bundled_model("bge-small-en-v1.5").unwrap_or_else(|| unreachable!());

        let bystander = cache.join("unrelated.onnx");
        let legacy_model = cache.join("bge-small-en-v1.5.onnx");
        let legacy_tokenizer = cache.join("tokenizer.json");
        for path in [
            &cache.join(bge.model_filename),
            &cache.join(bge.tokenizer_filename),
            &cache.join(DEFAULT_BUNDLED_MODEL.model_filename),
            &legacy_model,
            &legacy_tokenizer,
            &bystander,
        ] {
            std::fs::write(path, b"x")
                .ok()
                .unwrap_or_else(|| unreachable!());
        }

        remove_legacy_assets(cache).await;

        assert!(
            !legacy_model.exists(),
            "revision-less legacy asset was kept"
        );
        assert!(!legacy_tokenizer.exists());
        assert!(bystander.exists(), "an unlisted file was deleted");
        for model in BUNDLED_MODELS {
            for name in [model.model_filename, model.tokenizer_filename] {
                let path = cache.join(name);
                assert!(
                    !path.exists() || std::fs::metadata(&path).is_ok(),
                    "{name} vanished"
                );
            }
        }
        assert!(
            cache.join(bge.model_filename).exists(),
            "an inactive table entry's assets were deleted, forcing a re-download"
        );
        assert!(cache.join(DEFAULT_BUNDLED_MODEL.model_filename).exists());
    }

    /// Points at a closed loopback port, so its fetch fails immediately the way
    /// an offline model switch does — no network, no timeout wait.
    const UNREACHABLE_MODEL: BundledModel = BundledModel {
        id: "test-unreachable",
        dimensions: Dimension(8),
        pooling: Pooling::Cls,
        max_sequence_length: 8,
        model_filename: "test-unreachable.onnx",
        model_url: "http://127.0.0.1:1/model.onnx",
        model_sha256: "0000000000000000000000000000000000000000000000000000000000000000",
        model_size_bytes: 1,
        tokenizer_filename: "test-unreachable.tokenizer.json",
        tokenizer_url: "http://127.0.0.1:1/tokenizer.json",
        tokenizer_sha256: "0000000000000000000000000000000000000000000000000000000000000000",
        tokenizer_size_bytes: 1,
    };

    /// Purging runs only after the active model is confirmed on disk, so a
    /// model switch that cannot reach the network leaves the user with
    /// something loadable rather than nothing.
    #[tokio::test]
    async fn a_failed_download_leaves_the_existing_assets_alone() {
        let tempdir = tempdir().ok().unwrap_or_else(|| unreachable!());
        let cache = tempdir.path();
        let legacy = cache.join("bge-small-en-v1.5.onnx");
        std::fs::write(&legacy, b"working weights")
            .ok()
            .unwrap_or_else(|| unreachable!());

        let paths = AssetPaths::new(cache, &UNREACHABLE_MODEL);
        let result = ensure_assets_exist(&paths, &UNREACHABLE_MODEL).await;

        assert!(result.is_err(), "the download was expected to fail");
        assert!(
            legacy.exists(),
            "cleanup ran before the replacement was in place and stranded the user"
        );
    }

    /// Serve `body` once over loopback and return its URL, so the download path
    /// is exercised end to end without reaching the network.
    async fn serve_once(body: &'static [u8]) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .ok()
            .unwrap_or_else(|| unreachable!());
        let port = listener
            .local_addr()
            .map(|addr| addr.port())
            .unwrap_or_else(|_| unreachable!());

        tokio::spawn(async move {
            if let Ok((mut socket, _)) = listener.accept().await {
                let header = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = socket.write_all(header.as_bytes()).await;
                let _ = socket.write_all(body).await;
                let _ = socket.shutdown().await;
            }
        });

        format!("http://127.0.0.1:{port}/asset")
    }

    /// Trickle `body` out in many small writes so the download performs many
    /// write syscalls, giving a competing writer a real window to land in.
    async fn serve_trickle(body: Vec<u8>) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .ok()
            .unwrap_or_else(|| unreachable!());
        let port = listener
            .local_addr()
            .map(|addr| addr.port())
            .unwrap_or_else(|_| unreachable!());

        tokio::spawn(async move {
            if let Ok((mut socket, _)) = listener.accept().await {
                let header = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = socket.write_all(header.as_bytes()).await;
                for piece in body.chunks(32 * 1024) {
                    let _ = socket.write_all(piece).await;
                    let _ = socket.flush().await;
                    tokio::time::sleep(Duration::from_micros(100)).await;
                }
                let _ = socket.shutdown().await;
            }
        });

        format!("http://127.0.0.1:{port}/asset")
    }

    fn lingering_downloads(dir: &Path) -> Vec<PathBuf> {
        std::fs::read_dir(dir)
            .into_iter()
            .flatten()
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| path.to_string_lossy().ends_with(".download"))
            .collect()
    }

    /// The concurrent-download corruption, reproduced against the real
    /// mechanism.
    ///
    /// Two claudix processes fetching one model — two repos on a first run, or
    /// the MCP server racing a `claudix index` — open the same temp path with
    /// `O_TRUNC`. The second's truncate resets the length while the first keeps
    /// its file offset, so the first's remaining bytes land past a sparse hole
    /// and the finished file carries the right *length* and the wrong
    /// *content*. `is_cached` checks length only, so that file is then trusted
    /// on every later run and ORT fails to parse it forever.
    ///
    /// This calls [`fetch_and_publish`] directly with a fixed temp path so the
    /// competing `O_TRUNC` writer targets a known name — reproducing the
    /// collision rather than simulating it. A digest taken over the network
    /// stream cannot catch this: it hashes bytes that never reached the disk.
    /// Hashing the file back off disk is what makes it detectable, and reverting
    /// to a stream digest reds this test.
    #[tokio::test]
    async fn a_temp_file_a_competing_writer_truncated_is_never_published() {
        let tempdir = tempdir().ok().unwrap_or_else(|| unreachable!());
        let destination = tempdir.path().join("asset.onnx");
        let temp_path = tempdir.path().join("asset.onnx.competing.download");
        let body = vec![7u8; 8 * 1024 * 1024];
        let digest = hex_encode(&Sha256::digest(&body));
        let size = body.len() as u64;
        let url = serve_trickle(body).await;

        let victim = temp_path.clone();
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let truncates = Arc::new(AtomicU64::new(0));
        let (flag, counter) = (Arc::clone(&stop), Arc::clone(&truncates));
        let competitor = tokio::spawn(async move {
            while !flag.load(Ordering::Relaxed) {
                // Only truncate a file that already has bytes: resetting an
                // empty one leaves the writer's offset at 0 and does no damage,
                // which is the interleaving that proves nothing.
                if std::fs::metadata(&victim)
                    .map(|meta| meta.len())
                    .unwrap_or(0)
                    > 0
                    && let Ok(file) = std::fs::OpenOptions::new().write(true).open(&victim)
                    && file.set_len(0).is_ok()
                {
                    counter.fetch_add(1, Ordering::Relaxed);
                }
                tokio::time::sleep(Duration::from_micros(100)).await;
            }
        });

        let result = fetch_and_publish(&url, &temp_path, &destination, &digest, size).await;
        stop.store(true, Ordering::Relaxed);
        let _ = competitor.await;

        assert!(
            truncates.load(Ordering::Relaxed) > 0,
            "the competing writer never opened the temp file — the test did not \
             exercise the collision it exists to cover"
        );

        // Either outcome is safe. A published file carrying the right length
        // and the wrong bytes is the one that must not happen.
        match std::fs::read(&destination) {
            Ok(bytes) => {
                assert!(result.is_ok(), "{result:?}");
                assert!(
                    bytes.len() as u64 == size && bytes.iter().all(|byte| *byte == 7),
                    "a corrupt asset was published: len {} with {} zero bytes",
                    bytes.len(),
                    bytes.iter().filter(|byte| **byte == 0).count()
                );
            }
            Err(_) => assert!(
                result.is_err(),
                "download reported success but published nothing"
            ),
        }
    }

    /// A sparse file of exactly the expected length still hashes wrong, which
    /// is the property that makes [`is_cached`]'s length-only reuse safe.
    #[tokio::test]
    async fn file_sha256_reads_holes_rather_than_trusting_length() {
        let tempdir = tempdir().ok().unwrap_or_else(|| unreachable!());
        let path = tempdir.path().join("sparse.bin");
        let body = vec![7u8; 4096];
        let mut holed = vec![0u8; 2048];
        holed.extend_from_slice(&body[2048..]);
        std::fs::write(&path, &holed)
            .ok()
            .unwrap_or_else(|| unreachable!());

        let (digest, length) = file_sha256(&path)
            .await
            .ok()
            .unwrap_or_else(|| unreachable!());

        assert_eq!(length, 4096, "the hole-filled file has the expected length");
        assert_ne!(
            digest,
            hex_encode(&Sha256::digest(&body)),
            "a hole-filled file of the right length hashed as if it were intact"
        );
        assert_eq!(digest, hex_encode(&Sha256::digest(&holed)));
    }

    /// Peak allocation scales with (items x longest sequence) per ONNX run, so
    /// that product is what has to be bounded. Without this the item count is
    /// the only limit: `batch_size` 32 alongside a 2048-token chunk pads every
    /// item in the batch to 2048 and the model allocates attention over all of
    /// it, which measured 19.4GB RSS on a 244-file repo and was OOM-killed at a
    /// 6GB cap on a 34-file one.
    #[test]
    fn no_sub_batch_exceeds_the_token_budget() {
        let lengths = vec![2048, 12, 300, 7, 1900, 64, 512, 288, 33, 1024, 290, 41];

        for group in plan_sub_batches(&lengths, EMBED_TOKEN_BUDGET) {
            let longest = group
                .iter()
                .map(|index| lengths[*index])
                .max()
                .unwrap_or_else(|| unreachable!());
            let padded = longest * group.len();

            assert!(
                group.len() == 1 || padded <= EMBED_TOKEN_BUDGET,
                "sub-batch of {} items padded to {longest} costs {padded} tokens, \
                 over the {EMBED_TOKEN_BUDGET} budget",
                group.len()
            );
        }
    }

    /// Every input reaches exactly one sub-batch. Reassembly is by index, so a
    /// dropped one would surface as a missing embedding and a duplicated one
    /// would overwrite a neighbour's vector.
    #[test]
    fn sub_batch_planning_partitions_every_input_exactly_once() {
        let lengths = vec![2048, 12, 300, 7, 1900, 64, 512, 288, 33, 1024, 290, 41];

        let mut seen: Vec<usize> = plan_sub_batches(&lengths, EMBED_TOKEN_BUDGET)
            .into_iter()
            .flatten()
            .collect();
        seen.sort_unstable();

        assert_eq!(seen, (0..lengths.len()).collect::<Vec<_>>());
    }

    /// An input past the budget on its own is still embedded, alone. Dropping
    /// it would silently lose a chunk from the index and splitting it further
    /// is the tokenizer's job, already done via `max_sequence_length`.
    #[test]
    fn an_input_over_budget_gets_its_own_sub_batch() {
        let groups = plan_sub_batches(&[9000, 10, 10], 4096);

        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0], vec![1, 2]);
        assert_eq!(
            groups[1],
            vec![0],
            "the over-budget input was dropped or split"
        );
    }

    /// Short inputs must not be dragged up to a long one's padded width, which
    /// is the whole reason the planner sorts before grouping.
    #[test]
    fn a_long_input_does_not_pad_short_ones_up_to_its_length() {
        let lengths = vec![2048, 8, 8, 8, 8];

        let groups = plan_sub_batches(&lengths, 4096);
        let shorts = groups
            .iter()
            .find(|group| group.contains(&1))
            .unwrap_or_else(|| unreachable!());

        assert!(
            !shorts.contains(&0),
            "the 2048-token input was batched with 8-token ones, padding them 256x"
        );
    }

    #[test]
    fn planning_an_empty_batch_yields_no_sub_batches() {
        assert!(plan_sub_batches(&[], EMBED_TOKEN_BUDGET).is_empty());
    }

    const KNOWN_BODY: &[u8] = b"bundled asset";
    /// Any digest that is not `KNOWN_BODY`'s: stands in for a mirror serving
    /// different bytes than the table pinned.
    const WRONG_SHA256: &str = "0000000000000000000000000000000000000000000000000000000000000000";

    #[tokio::test]
    async fn a_digest_mismatch_leaves_nothing_cached() {
        let tempdir = tempdir().ok().unwrap_or_else(|| unreachable!());
        let destination = tempdir.path().join("asset.onnx");
        let url = serve_once(KNOWN_BODY).await;

        let error =
            download_verified(&url, &destination, WRONG_SHA256, KNOWN_BODY.len() as u64).await;

        assert!(matches!(
            error,
            Err(ClaudixError::BundledAssetCorrupt { .. })
        ));
        assert!(
            !destination.exists(),
            "a file that failed verification was published under its real name"
        );
        assert!(
            lingering_downloads(tempdir.path()).is_empty(),
            "the unverified download was left on disk for a later run to reuse"
        );
    }

    #[tokio::test]
    async fn a_matching_digest_publishes_the_asset() {
        let tempdir = tempdir().ok().unwrap_or_else(|| unreachable!());
        let destination = tempdir.path().join("asset.onnx");
        let url = serve_once(KNOWN_BODY).await;
        let expected = hex_encode(&Sha256::digest(KNOWN_BODY));

        let result =
            download_verified(&url, &destination, &expected, KNOWN_BODY.len() as u64).await;

        assert!(result.is_ok(), "{result:?}");
        assert_eq!(
            std::fs::read(&destination).ok().as_deref(),
            Some(KNOWN_BODY)
        );
        assert!(lingering_downloads(tempdir.path()).is_empty());
    }

    /// A body that hashes correctly but arrives short still fails: the length is
    /// what a truncated transfer breaks first.
    #[tokio::test]
    async fn a_size_mismatch_fails_verification() {
        let tempdir = tempdir().ok().unwrap_or_else(|| unreachable!());
        let destination = tempdir.path().join("asset.onnx");
        let url = serve_once(KNOWN_BODY).await;
        let expected = hex_encode(&Sha256::digest(KNOWN_BODY));

        let error = download_verified(&url, &destination, &expected, 999).await;

        assert!(matches!(
            error,
            Err(ClaudixError::BundledAssetCorrupt { .. })
        ));
        assert!(!destination.exists());
    }

    /// Distinguishes the two heads by construction: token 0 is orthogonal to
    /// token 1, so a mean-pooled result cannot equal a CLS-pooled one.
    #[test]
    fn cls_pooling_reads_the_first_token_not_the_mean() {
        let values = vec![
            1.0, 0.0, 0.0, 0.0, // token 0, the [CLS] row
            0.0, 1.0, 0.0, 0.0, // token 1
            9.0, 9.0, 9.0, 9.0, // padding, ignored by both heads
        ];
        let attention_mask = vec![1, 1, 0];

        let cls = pool_and_normalize(Pooling::Cls, &values, &attention_mask, 1, 3, 4);
        assert_eq!(cls[0], vec![1.0, 0.0, 0.0, 0.0]);

        let mean = pool_and_normalize(Pooling::Mean, &values, &attention_mask, 1, 3, 4);
        let half = std::f32::consts::FRAC_1_SQRT_2;
        assert!((mean[0][0] - half).abs() < 1e-6);
        assert!((mean[0][1] - half).abs() < 1e-6);
        assert_ne!(cls[0], mean[0]);
    }

    /// A short output tensor yields zeros rather than panicking: the hook path
    /// must fail open, and a truncated model output is exactly the kind of
    /// corruption that would otherwise take the session down.
    #[test]
    fn cls_pooling_survives_a_truncated_output_tensor() {
        let vectors = cls_pool_and_normalize(&[1.0, 0.0], 1, 3, 4);

        assert_eq!(vectors, vec![vec![0.0; 4]]);
    }

    /// Same fail-open guarantee for the mean head. It is unreachable today (both
    /// table entries pool `[CLS]`), so this pins the defense before a future
    /// `Pooling::Mean` entry — a data edit that touches no code — makes a
    /// truncated tensor or short mask reachable and able to panic the hook path.
    #[test]
    fn mean_pooling_survives_a_truncated_output_tensor_and_short_mask() {
        let short_values = mean_pool_and_normalize(&[1.0, 0.0], &[1, 1, 0], 1, 3, 4);
        assert_eq!(short_values, vec![vec![0.0; 4]]);

        let short_mask = mean_pool_and_normalize(&[0.0; 12], &[1], 1, 3, 4);
        assert_eq!(short_mask, vec![vec![0.0; 4]]);
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
