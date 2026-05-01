use std::path::Path;

use claudix::chunking::{Chunker, MultiLanguageChunker};
use claudix::config::Config;
use claudix::embedding::Provider;
use claudix::enumeration::FileEnumerator;
use claudix::store::Store;
use claudix::types::EmbeddedChunk;
use claudix::ClaudixError;

pub fn stub_config() -> Config {
    stub_config_with_model("stub-v1")
}

pub fn stub_config_with_model(model: impl Into<String>) -> Config {
    let mut config = Config::default();
    config.embedding.model = model.into();
    config.embedding.dimensions = 8;
    config.hooks.session_start_warmup = false;
    config
}

pub async fn index_fixture(
    store: &Store,
    embedder: &dyn Provider,
    project_root: &Path,
    config: &Config,
) -> claudix::Result<()> {
    let enumerator = FileEnumerator::new(project_root.to_path_buf(), config.clone())?;
    let files = enumerator.enumerate()?;
    let mut chunks = Vec::new();

    for file in files {
        let content = tokio::fs::read_to_string(&file.absolute_path).await?;
        let path = file.relative_path.clone();
        let language = file.language;
        let file_hash = file.file_hash;

        let file_chunks = tokio::task::spawn_blocking(move || {
            MultiLanguageChunker::new().chunk(&path, language, file_hash, &content)
        })
        .await
        .map_err(|error| ClaudixError::TreeSitter(error.to_string()))??;
        chunks.extend(file_chunks);
    }

    let inputs = chunks
        .iter()
        .map(|chunk| chunk.content.as_str())
        .collect::<Vec<_>>();
    let vectors = embedder.embed(&inputs).await?;
    let embedded_chunks = chunks
        .into_iter()
        .zip(vectors)
        .map(|(chunk, vector)| EmbeddedChunk { chunk, vector })
        .collect::<Vec<_>>();

    store.replace_chunks(&embedded_chunks, config).await?;
    Ok(())
}
