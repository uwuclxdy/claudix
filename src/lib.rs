pub mod chunking;
pub mod cli;
pub mod config;
pub mod embedding;
pub mod enumeration;
pub mod error;
pub mod hooks;
pub mod mcp;
pub mod search;
pub mod store;
pub mod types;
pub mod util;

pub use error::{ClaudixError, Result};
pub use types::{
    ByteRange, Chunk, ChunkId, ChunkKind, Dimension, EmbeddedChunk, FileHash, Language, LineRange,
    RelativePath,
};

use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use chunking::{Chunker, MultiLanguageChunker};
use config::{Config, EmbeddingProvider};
#[cfg(any(test, feature = "test-stub"))]
use embedding::StubProvider;
use embedding::{BundledProvider, HttpProvider, Provider};
use enumeration::{EnumeratedFile, FileEnumerator};
use error::RecoveryHint;
use search::{SearchQuery, SearchResult, Searcher};
use store::Store;
use tokio::{fs, task};

pub struct Claudix {
    config: Arc<Config>,
    project_root: PathBuf,
    embedder: Arc<dyn Provider>,
    store: Store,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexStats {
    pub file_count: usize,
    pub chunk_count: usize,
}

impl Claudix {
    pub async fn new(project_root: PathBuf, config: Arc<Config>) -> Result<Self> {
        let embedder = build_provider(config.as_ref()).await?;
        let store = Store::new(&project_root, config.as_ref())?;
        store.validate_manifest_compatibility(embedder.model_id(), embedder.dimensions().0)?;

        Ok(Self {
            config,
            project_root,
            embedder,
            store,
        })
    }

    #[cfg(test)]
    pub(crate) fn from_parts(
        project_root: PathBuf,
        config: Arc<Config>,
        embedder: Arc<dyn Provider>,
        store: Store,
    ) -> Self {
        Self {
            config,
            project_root,
            embedder,
            store,
        }
    }

    pub fn config(&self) -> &Config {
        self.config.as_ref()
    }

    pub fn project_root(&self) -> &Path {
        &self.project_root
    }

    pub async fn index_full(&self) -> Result<IndexStats> {
        let files = FileEnumerator::new(self.project_root.clone(), self.config.as_ref().clone())?
            .enumerate()?;
        let chunks = self.collect_chunks(&files).await?;
        let embedded_chunks = self.embed_chunks(chunks).await?;
        let stats = self
            .store
            .replace_chunks(&embedded_chunks, self.config.as_ref())
            .await?;

        Ok(IndexStats {
            file_count: stats.file_count,
            chunk_count: stats.chunk_count,
        })
    }

    pub async fn reindex_file(&self, path: &Path) -> Result<IndexStats> {
        let relative_path = self.relative_path_from_input(path)?;
        let files = FileEnumerator::new(self.project_root.clone(), self.config.as_ref().clone())?
            .enumerate()?;

        let Some(file) = files
            .into_iter()
            .find(|file| file.relative_path == relative_path)
        else {
            let stats = self
                .store
                .delete_file_chunks(&relative_path, self.config.as_ref())
                .await?;
            return Ok(IndexStats {
                file_count: stats.file_count,
                chunk_count: stats.chunk_count,
            });
        };

        if let Ok(Some(stored_hash)) = self.store.stored_file_hash(&relative_path).await {
            if stored_hash == file.file_hash.0 {
                let stats = self.store.chunk_stats().await?;
                return Ok(IndexStats {
                    file_count: stats.file_count,
                    chunk_count: stats.chunk_count,
                });
            }
        }

        let chunks = self.collect_file_chunks(&file).await?;
        let embedded_chunks = self.embed_chunks(chunks).await?;
        let stats = if embedded_chunks.is_empty() {
            self.store
                .delete_file_chunks(&relative_path, self.config.as_ref())
                .await?
        } else {
            self.store
                .replace_file_chunks(&embedded_chunks, self.config.as_ref())
                .await?
        };

        Ok(IndexStats {
            file_count: stats.file_count,
            chunk_count: stats.chunk_count,
        })
    }

    pub async fn search(&self, query: SearchQuery) -> Result<Vec<SearchResult>> {
        let searcher = Searcher::new(
            self.store.clone(),
            Arc::clone(&self.embedder),
            self.config.search.clone(),
        );
        searcher.search(query).await
    }

    pub async fn embedder_health_check(&self) -> Result<()> {
        self.embedder.health_check().await
    }

    async fn collect_chunks(&self, files: &[EnumeratedFile]) -> Result<Vec<Chunk>> {
        let mut chunks = Vec::new();

        for file in files {
            chunks.extend(self.collect_file_chunks(file).await?);
        }

        Ok(chunks)
    }

    async fn collect_file_chunks(&self, file: &EnumeratedFile) -> Result<Vec<Chunk>> {
        let content = match fs::read_to_string(&file.absolute_path).await {
            Ok(content) => content,
            Err(error) if error.kind() == std::io::ErrorKind::InvalidData => return Ok(Vec::new()),
            Err(error) => return Err(error.into()),
        };
        let path = file.relative_path.clone();
        let language = file.language;
        let file_hash = file.file_hash;
        let force_indexed = file.force_indexed;
        let overlap_lines = self.config.indexing.chunk_overlap_lines;

        task::spawn_blocking(move || {
            let chunker = MultiLanguageChunker::with_fallback_params(60, overlap_lines);
            if force_indexed && language == Language::Unknown {
                chunker.chunk_as_text(&path, language, file_hash, &content)
            } else {
                chunker.chunk(&path, language, file_hash, &content)
            }
        })
        .await
        .map_err(|error| ClaudixError::TreeSitter(error.to_string()))?
    }

    async fn embed_chunks(&self, chunks: Vec<Chunk>) -> Result<Vec<EmbeddedChunk>> {
        let mut embedded_chunks = Vec::with_capacity(chunks.len());
        let batch_size = self.config.embedding.batch_size;
        let expected_dimensions = self.embedder.dimensions();

        for batch in chunks.chunks(batch_size) {
            let inputs = batch
                .iter()
                .map(|chunk| chunk.content.as_str())
                .collect::<Vec<_>>();
            let vectors = self.embedder.embed(&inputs).await?;

            if vectors.len() != batch.len() {
                return Err(ClaudixError::Embedding(format!(
                    "provider returned {} vectors for {} chunks",
                    vectors.len(),
                    batch.len()
                )));
            }

            for (chunk, vector) in batch.iter().cloned().zip(vectors) {
                let actual_dimensions = u16::try_from(vector.len()).unwrap_or(u16::MAX);
                if actual_dimensions != expected_dimensions.0 {
                    return Err(ClaudixError::DimensionMismatch {
                        store_dim: expected_dimensions.0,
                        model_dim: actual_dimensions,
                        recovery: RecoveryHint(
                            "Reindex the project after aligning embedding dimensions with the active model",
                        ),
                    });
                }

                embedded_chunks.push(EmbeddedChunk { chunk, vector });
            }
        }

        Ok(embedded_chunks)
    }

    fn relative_path_from_input(&self, path: &Path) -> Result<RelativePath> {
        if path.is_absolute() {
            let relative =
                path.strip_prefix(&self.project_root)
                    .map_err(|_| ClaudixError::PathTraversal {
                        path: path.to_path_buf(),
                        recovery: RecoveryHint("Only reindex files inside $CLAUDE_PROJECT_DIR"),
                    })?;
            reject_relative_escape(relative)?;
            return Ok(RelativePath::from_path(relative));
        }

        reject_relative_escape(path)?;
        Ok(RelativePath::from_path(path))
    }
}

async fn build_provider(config: &Config) -> Result<Arc<dyn Provider>> {
    let dimensions = Dimension(config.embedding.dimensions);

    #[cfg(any(test, feature = "test-stub"))]
    if config.embedding.model.starts_with("stub") {
        return Ok(Arc::new(StubProvider::with_model_id(
            config.embedding.model.clone(),
            dimensions,
        )));
    }

    match config.embedding.provider {
        EmbeddingProvider::Bundled => Ok(Arc::new(
            BundledProvider::new(config.embedding.model.clone(), dimensions).await?,
        )),
        EmbeddingProvider::Http => Ok(Arc::new(HttpProvider::new(
            config.embedding.endpoint.clone(),
            config.embedding.model.clone(),
            dimensions,
            Duration::from_millis(config.embedding.timeout_ms),
            None,
        )?)),
    }
}

fn reject_relative_escape(path: &Path) -> Result<()> {
    for component in path.components() {
        if matches!(
            component,
            Component::ParentDir | Component::RootDir | Component::Prefix(_)
        ) {
            return Err(ClaudixError::PathTraversal {
                path: path.to_path_buf(),
                recovery: RecoveryHint("Only reindex files inside $CLAUDE_PROJECT_DIR"),
            });
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::StubProvider;
    use std::collections::BTreeSet;

    mod fixture {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/common/fixture.rs"
        ));
    }

    mod config_support {
        use crate as claudix;

        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/common/config_support.rs"
        ));
    }

    use config_support::stub_config;
    use fixture::TestFixture;

    fn test_claudix(project_root: PathBuf, config: Config) -> Result<Claudix> {
        let store = Store::new(&project_root, &config)?;
        let embedder: Arc<dyn Provider> = Arc::new(StubProvider::with_model_id(
            config.embedding.model.clone(),
            Dimension(config.embedding.dimensions),
        ));

        Ok(Claudix {
            config: Arc::new(config),
            project_root,
            embedder,
            store,
        })
    }

    #[tokio::test]
    async fn index_full_persists_fixture_chunks() {
        let fixture = TestFixture::new("small_rust");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());
        let config = stub_config();

        let claudix = test_claudix(fixture.root().to_path_buf(), config.clone());
        assert!(claudix.is_ok());
        let claudix = claudix.ok().unwrap_or_else(|| unreachable!());

        let stats = claudix.index_full().await;
        assert!(stats.is_ok());
        assert_eq!(
            stats.ok().unwrap_or_else(|| unreachable!()),
            IndexStats {
                file_count: 2,
                chunk_count: 3,
            }
        );

        let rows = claudix.store.read_chunks().await;
        assert!(rows.is_ok());
        let rows = rows.ok().unwrap_or_else(|| unreachable!());

        let names = rows
            .iter()
            .filter_map(|row| row.name.clone())
            .collect::<BTreeSet<_>>();
        assert!(names.contains("greet"));
        assert!(names.contains("add"));
        assert!(rows.iter().all(|row| row.vector.len() == 8));

        let manifest = claudix.store.read_manifest();
        assert!(manifest.is_ok());
        let manifest = manifest.ok().unwrap_or_else(|| unreachable!());
        assert!(manifest.is_some());
        let manifest = manifest.unwrap_or_else(|| unreachable!());
        assert_eq!(manifest.embedding_model, "stub-v1");
        assert_eq!(manifest.dimensions, 8);
        assert_eq!(manifest.file_count, 2);
        assert_eq!(manifest.chunk_count, 3);
    }

    #[tokio::test]
    async fn index_full_replaces_stale_chunks() {
        let fixture = TestFixture::new("small_rust");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());
        let config = stub_config();

        let claudix = test_claudix(fixture.root().to_path_buf(), config);
        assert!(claudix.is_ok());
        let claudix = claudix.ok().unwrap_or_else(|| unreachable!());

        assert!(claudix.index_full().await.is_ok());
        assert!(
            fs::write(
                fixture.root().join("src/lib.rs"),
                "pub mod math;\n\npub fn salute(name: &str) -> String {\n    format!(\"hi {name}\")\n}\n",
            )
            .await
            .is_ok()
        );

        let stats = claudix.index_full().await;
        assert!(stats.is_ok());
        assert_eq!(
            stats.ok().unwrap_or_else(|| unreachable!()),
            IndexStats {
                file_count: 2,
                chunk_count: 3,
            }
        );

        let rows = claudix.store.read_chunks().await;
        assert!(rows.is_ok());
        let rows = rows.ok().unwrap_or_else(|| unreachable!());

        let names = rows
            .iter()
            .filter_map(|row| row.name.clone())
            .collect::<BTreeSet<_>>();
        assert!(names.contains("salute"));
        assert!(!names.contains("greet"));
    }

    #[tokio::test]
    async fn reindex_file_updates_only_target_file() {
        let fixture = TestFixture::new("small_rust");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());
        let config = stub_config();

        let claudix = test_claudix(fixture.root().to_path_buf(), config);
        assert!(claudix.is_ok());
        let claudix = claudix.ok().unwrap_or_else(|| unreachable!());

        assert!(claudix.index_full().await.is_ok());
        assert!(
            fs::write(
                fixture.root().join("src/math.rs"),
                "pub fn multiply(left: i32, right: i32) -> i32 {\n    left * right\n}\n",
            )
            .await
            .is_ok()
        );

        let stats = claudix.reindex_file(Path::new("src/math.rs")).await;
        assert!(stats.is_ok());
        assert_eq!(
            stats.ok().unwrap_or_else(|| unreachable!()),
            IndexStats {
                file_count: 2,
                chunk_count: 3,
            }
        );

        let rows = claudix.store.read_chunks().await;
        assert!(rows.is_ok());
        let rows = rows.ok().unwrap_or_else(|| unreachable!());

        let names = rows
            .iter()
            .filter_map(|row| row.name.clone())
            .collect::<BTreeSet<_>>();
        assert!(names.contains("greet"));
        assert!(names.contains("multiply"));
        assert!(!names.contains("add"));
    }

    #[tokio::test]
    async fn reindex_file_skips_embedding_when_hash_unchanged() {
        let fixture = TestFixture::new("small_rust");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());
        let config = stub_config();

        let claudix = test_claudix(fixture.root().to_path_buf(), config);
        assert!(claudix.is_ok());
        let claudix = claudix.ok().unwrap_or_else(|| unreachable!());

        assert!(claudix.index_full().await.is_ok());

        // Reindex the same file without modifying it — hash matches stored hash, must skip.
        let stats = claudix.reindex_file(Path::new("src/math.rs")).await;
        assert!(stats.is_ok());
        let stats = stats.ok().unwrap_or_else(|| unreachable!());
        // Chunk count unchanged — no re-embedding happened.
        assert_eq!(stats.chunk_count, 3);
    }

    #[tokio::test]
    async fn index_full_skips_invalid_utf8_files() {
        let fixture = TestFixture::new("small_rust");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());
        assert!(
            fs::write(fixture.root().join("binary.rs"), [0xff, 0xfe, 0xfd])
                .await
                .is_ok()
        );

        let config = stub_config();
        let claudix = test_claudix(fixture.root().to_path_buf(), config);
        assert!(claudix.is_ok());
        let claudix = claudix.ok().unwrap_or_else(|| unreachable!());

        let stats = claudix.index_full().await;
        assert!(stats.is_ok());
        assert_eq!(
            stats.ok().unwrap_or_else(|| unreachable!()),
            IndexStats {
                file_count: 2,
                chunk_count: 3,
            }
        );
    }

    #[tokio::test]
    async fn reindex_file_deletes_missing_file_chunks() {
        let fixture = TestFixture::new("small_rust");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());
        let config = stub_config();

        let claudix = test_claudix(fixture.root().to_path_buf(), config);
        assert!(claudix.is_ok());
        let claudix = claudix.ok().unwrap_or_else(|| unreachable!());

        assert!(claudix.index_full().await.is_ok());
        assert!(
            fs::remove_file(fixture.root().join("src/math.rs"))
                .await
                .is_ok()
        );

        let stats = claudix.reindex_file(Path::new("src/math.rs")).await;
        assert!(stats.is_ok());
        assert_eq!(
            stats.ok().unwrap_or_else(|| unreachable!()),
            IndexStats {
                file_count: 1,
                chunk_count: 2,
            }
        );

        let rows = claudix.store.read_chunks().await;
        assert!(rows.is_ok());
        let rows = rows.ok().unwrap_or_else(|| unreachable!());
        assert!(rows.iter().all(|row| row.file_path == "src/lib.rs"));
    }

    #[tokio::test]
    async fn reindex_file_removes_stale_chunks_when_file_becomes_empty() {
        let fixture = TestFixture::new("small_rust");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());
        let config = stub_config();

        let claudix = test_claudix(fixture.root().to_path_buf(), config);
        assert!(claudix.is_ok());
        let claudix = claudix.ok().unwrap_or_else(|| unreachable!());

        assert!(claudix.index_full().await.is_ok());
        assert!(fs::write(fixture.root().join("src/math.rs"), b"").await.is_ok());

        let stats = claudix.reindex_file(Path::new("src/math.rs")).await;
        assert!(stats.is_ok());

        let rows = claudix.store.read_chunks().await;
        assert!(rows.is_ok());
        let rows = rows.ok().unwrap_or_else(|| unreachable!());
        assert!(
            rows.iter().all(|row| row.file_path != "src/math.rs"),
            "stale chunks from emptied file must be removed"
        );
    }
}
