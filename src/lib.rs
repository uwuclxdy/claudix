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
use embedding::bundled::{BUNDLED_DIMENSIONS, BUNDLED_MODEL_ID};
use embedding::{BundledProvider, FallbackProvider, HttpProvider, Provider};
use enumeration::{EnumeratedFile, FileEnumerator, WatchFilter};
use error::RecoveryHint;
use search::{SearchQuery, SearchResult, Searcher};
use store::{Store, stored_chunks_from_embedded};
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

pub enum IndexFileStatus {
    Indexed,
    Verified,
    Skipped(&'static str),
}

pub trait IndexProgress {
    fn file(&mut self, path: &RelativePath, status: IndexFileStatus) -> Result<()>;
}

pub struct NoopIndexProgress;

impl IndexProgress for NoopIndexProgress {
    fn file(&mut self, _path: &RelativePath, _status: IndexFileStatus) -> Result<()> {
        Ok(())
    }
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
        let mut progress = NoopIndexProgress;
        self.index_full_with_progress(&mut progress).await
    }

    pub async fn index_full_with_progress(
        &self,
        progress: &mut dyn IndexProgress,
    ) -> Result<IndexStats> {
        let files = FileEnumerator::new(self.project_root.clone(), self.config.as_ref().clone())?
            .enumerate_with_progress(Some(progress))?;

        let current_files: Vec<(String, [u8; 16])> = files
            .iter()
            .map(|f| (f.relative_path.as_str().to_owned(), f.file_hash.0))
            .collect();

        let (changed_paths, unchanged_rows) = self
            .store
            .incremental_file_state_with_progress(&current_files, Some(progress))
            .await?;

        if changed_paths.is_empty()
            && let Some(stats) = self
                .store
                .touch_manifest_if_in_sync(&current_files, self.config.as_ref())?
        {
            return Ok(IndexStats {
                file_count: stats.file_count,
                chunk_count: stats.chunk_count,
            });
        }

        let mut rows = unchanged_rows;

        for file in files
            .iter()
            .filter(|file| changed_paths.contains(file.relative_path.as_str()))
        {
            let chunks = self.collect_file_chunks(file).await?;
            if chunks.is_empty() {
                progress.file(
                    &file.relative_path,
                    IndexFileStatus::Skipped("no indexable chunks"),
                )?;
                continue;
            }

            let embedded_chunks = self.embed_chunks(chunks).await?;
            rows.retain(|row| row.file_path != file.relative_path.as_str());
            rows.extend(stored_chunks_from_embedded(
                &embedded_chunks,
                Dimension(self.config.embedding.dimensions),
            )?);
            progress.file(&file.relative_path, IndexFileStatus::Indexed)?;
        }

        let stats = self
            .store
            .persist_incremental(&[], rows, self.config.as_ref(), &current_files)
            .await?;

        Ok(IndexStats {
            file_count: stats.file_count,
            chunk_count: stats.chunk_count,
        })
    }

    pub async fn reindex_file(&self, path: &Path) -> Result<IndexStats> {
        let relative_path = self.relative_path_from_input(path)?;

        // Honour the same ignore set the watcher uses so a direct CLI/MCP call
        // on `.claudix/manifest.json` or a gitignored build artifact does not
        // embed index metadata back into the store.
        let filter = WatchFilter::load(&self.project_root)?;
        if !filter.is_watchable(&relative_path.to_path_buf()) {
            let manifest = self.store.read_manifest()?;
            return Ok(IndexStats {
                file_count: manifest
                    .as_ref()
                    .map(|m| m.file_count as usize)
                    .unwrap_or(0),
                chunk_count: manifest
                    .as_ref()
                    .map(|m| m.chunk_count as usize)
                    .unwrap_or(0),
            });
        }

        if let Some(stats) = self.skip_unchanged_target(&relative_path).await? {
            return Ok(stats);
        }

        let enumerator =
            FileEnumerator::new(self.project_root.clone(), self.config.as_ref().clone())?;
        let Some(file) = enumerator.enumerate_one(relative_path.clone(), false)? else {
            let stats = self
                .store
                .delete_file_chunks(&relative_path, self.config.as_ref())
                .await?;
            return Ok(IndexStats {
                file_count: stats.file_count,
                chunk_count: stats.chunk_count,
            });
        };

        if let Ok((Some(stored_hash), stats)) =
            self.store.stored_file_hash_and_stats(&relative_path).await
            && stored_hash == file.file_hash.0
        {
            return Ok(IndexStats {
                file_count: stats.file_count,
                chunk_count: stats.chunk_count,
            });
        }

        let chunks = self.collect_file_chunks(&file).await?;
        let embedded_chunks = self.embed_chunks(chunks).await?;
        let stats = if embedded_chunks.is_empty() {
            let stats = self
                .store
                .delete_file_chunks(&relative_path, self.config.as_ref())
                .await?;
            // File still exists but produces no chunks; record its hash so that
            // subsequent calls don't re-process it until the content changes.
            self.store
                .note_file_hash(&relative_path, file.file_hash.0, self.config.as_ref())?;
            stats
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
            self.project_root.clone(),
            self.store.clone(),
            Arc::clone(&self.embedder),
            self.config.search.clone(),
        );
        searcher.search(query).await
    }

    pub async fn embedder_health_check(&self) -> Result<()> {
        self.embedder.health_check().await
    }

    async fn skip_unchanged_target(
        &self,
        relative_path: &RelativePath,
    ) -> Result<Option<IndexStats>> {
        let Some(manifest) = self.store.read_manifest()? else {
            return Ok(None);
        };
        let Some(stored_hash) = manifest.file_hashes.get(relative_path.as_str()).copied() else {
            return Ok(None);
        };
        if manifest.embedding_model != self.config.embedding.model
            || manifest.dimensions != self.config.embedding.dimensions
        {
            return Ok(None);
        }

        let absolute_path = self.project_root.join(relative_path.to_path_buf());
        let bytes = match fs::read(&absolute_path).await {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        if bytes.len() as u64 > self.config.indexing.max_file_size_kb.saturating_mul(1024) {
            return Ok(None);
        }
        if enumeration::hash_bytes(&bytes).0 != stored_hash {
            return Ok(None);
        }

        Ok(Some(IndexStats {
            file_count: usize::try_from(manifest.file_count).unwrap_or(usize::MAX),
            chunk_count: usize::try_from(manifest.chunk_count).unwrap_or(usize::MAX),
        }))
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
            let chunker = MultiLanguageChunker::with_fallback_params(
                chunking::DEFAULT_CHUNK_LINES,
                overlap_lines,
            );
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
        )) as Arc<dyn Provider>);
    }

    match config.embedding.provider {
        EmbeddingProvider::Bundled => Ok(Arc::new(
            BundledProvider::new(config.embedding.model.clone(), dimensions).await?,
        ) as Arc<dyn Provider>),
        EmbeddingProvider::Http => {
            let primary = Arc::new(HttpProvider::new(
                config.embedding.endpoint.clone(),
                config.embedding.model.clone(),
                dimensions,
                Duration::from_millis(config.embedding.timeout_ms),
                None,
            )?) as Arc<dyn Provider>;
            if config.embedding.model == BUNDLED_MODEL_ID && dimensions == BUNDLED_DIMENSIONS {
                let fallback = Arc::new(
                    BundledProvider::new(config.embedding.model.clone(), dimensions).await?,
                ) as Arc<dyn Provider>;
                Ok(Arc::new(FallbackProvider::new(primary, fallback)) as Arc<dyn Provider>)
            } else {
                Ok(primary)
            }
        }
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
    use async_trait::async_trait;
    use std::collections::BTreeSet;
    use std::sync::atomic::{AtomicUsize, Ordering};

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

    struct CountingProvider {
        inner: StubProvider,
        calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl Provider for CountingProvider {
        fn name(&self) -> &str {
            self.inner.name()
        }

        fn dimensions(&self) -> Dimension {
            self.inner.dimensions()
        }

        fn model_id(&self) -> &str {
            self.inner.model_id()
        }

        async fn embed(&self, batch: &[&str]) -> Result<Vec<Vec<f32>>> {
            self.calls.fetch_add(batch.len(), Ordering::Relaxed);
            self.inner.embed(batch).await
        }

        async fn health_check(&self) -> Result<()> {
            self.inner.health_check().await
        }
    }

    fn test_claudix_with_embedder(
        project_root: PathBuf,
        config: Config,
        embedder: Arc<dyn Provider>,
    ) -> Result<Claudix> {
        let store = Store::new(&project_root, &config)?;

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
                "pub mod math;\n\npub fn salute(name: &str) -> String {\n    format!(\"hi {name}\")\n}\n\npub fn wave(name: &str) -> String {\n    format!(\"bye {name}\")\n}\n",
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
    async fn reindex_file_records_hash_for_no_chunk_file() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        // Write a binary file that will produce no chunks.
        fs::write(fixture.root().join("binary.bin"), [0xff, 0xfe, 0xfd]).await?;
        let config = stub_config();
        let claudix = test_claudix(fixture.root().to_path_buf(), config)?;

        claudix.index_full().await?;

        // After index_full the binary file's hash is recorded.
        let (hash_before, _) = claudix
            .store
            .stored_file_hash_and_stats(&RelativePath::new("binary.bin"))
            .await?;
        assert!(
            hash_before.is_some(),
            "hash must be stored for no-chunk file after index_full"
        );

        // Calling reindex_file on the same (unchanged) file must return immediately
        // and not clear the stored hash.
        let calls_before = {
            let calls = Arc::new(AtomicUsize::new(0));
            let embedder: Arc<dyn Provider> = Arc::new(CountingProvider {
                inner: StubProvider::with_model_id(
                    claudix.config().embedding.model.clone(),
                    Dimension(claudix.config().embedding.dimensions),
                ),
                calls: calls.clone(),
            });
            let c2 = test_claudix_with_embedder(
                claudix.project_root().to_path_buf(),
                claudix.config().clone(),
                embedder,
            )?;
            c2.reindex_file(std::path::Path::new("binary.bin")).await?;
            calls.load(Ordering::Relaxed)
        };
        assert_eq!(
            calls_before, 0,
            "unchanged no-chunk file must not trigger embedding"
        );
        Ok(())
    }

    #[tokio::test]
    async fn index_full_skips_unchanged_files_without_chunks() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        fs::write(fixture.root().join("src/empty.rs"), "pub mod child;\n").await?;
        let config = stub_config();
        let calls = Arc::new(AtomicUsize::new(0));
        let embedder: Arc<dyn Provider> = Arc::new(CountingProvider {
            inner: StubProvider::with_model_id(
                config.embedding.model.clone(),
                Dimension(config.embedding.dimensions),
            ),
            calls: calls.clone(),
        });
        let claudix = test_claudix_with_embedder(fixture.root().to_path_buf(), config, embedder)?;

        claudix.index_full().await?;
        let first_call_count = calls.load(Ordering::Relaxed);

        claudix.index_full().await?;

        assert_eq!(calls.load(Ordering::Relaxed), first_call_count);
        Ok(())
    }

    #[tokio::test]
    async fn index_full_preserves_unchanged_file_chunks_on_second_run() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let config = stub_config();
        let claudix = test_claudix(fixture.root().to_path_buf(), config)?;

        claudix.index_full().await?;

        // Modify only src/lib.rs; src/math.rs is untouched.
        fs::write(
            fixture.root().join("src/lib.rs"),
            "pub mod math;\n\npub fn salute(name: &str) -> String { format!(\"hi {name}\") }\n",
        )
        .await?;

        claudix.index_full().await?;

        let rows = claudix.store.read_chunks().await?;
        let names: BTreeSet<_> = rows.iter().filter_map(|r| r.name.clone()).collect();

        assert!(names.contains("salute"), "changed file must be re-embedded");
        assert!(!names.contains("greet"), "stale chunk must be gone");
        assert!(
            names.contains("add"),
            "unchanged file chunks must be preserved"
        );
        Ok(())
    }

    #[tokio::test]
    async fn index_full_skips_lancedb_rewrite_when_nothing_changed() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let config = stub_config();
        let claudix = test_claudix(fixture.root().to_path_buf(), config)?;

        claudix.index_full().await?;

        let chunks_dir = claudix
            .store
            .state_dir_path()
            .join("index")
            .join("chunks.lance");
        let before = snapshot_dir(&chunks_dir);
        assert!(
            !before.is_empty(),
            "first index_full should have written chunks.lance"
        );

        claudix.index_full().await?;

        let after = snapshot_dir(&chunks_dir);
        assert_eq!(
            before, after,
            "chunks.lance must not be rewritten when every file is verified as unchanged"
        );

        let manifest = claudix.store.read_manifest()?;
        let manifest = manifest.unwrap_or_else(|| unreachable!());
        assert!(
            manifest.last_full_index_at.is_some(),
            "verification run must still bump last_full_index_at"
        );
        Ok(())
    }

    fn snapshot_dir(dir: &Path) -> BTreeSet<(PathBuf, u64)> {
        fn walk(dir: &Path, into: &mut BTreeSet<(PathBuf, u64)>) {
            let Ok(entries) = std::fs::read_dir(dir) else {
                return;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                let Ok(metadata) = entry.metadata() else {
                    continue;
                };
                if metadata.is_dir() {
                    walk(&path, into);
                } else {
                    into.insert((path, metadata.len()));
                }
            }
        }
        let mut set = BTreeSet::new();
        walk(dir, &mut set);
        set
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
        assert!(
            fs::write(fixture.root().join("src/math.rs"), b"")
                .await
                .is_ok()
        );

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

    #[tokio::test]
    async fn reindex_file_skips_index_internal_paths() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let config = stub_config();
        let claudix = test_claudix(fixture.root().to_path_buf(), config)?;
        claudix.index_full().await?;
        let baseline = claudix.store.read_chunks().await?;

        // `.claudix/` is the index's own state dir — embedding files under it
        // would round-trip manifest data through the embedder.
        let internal = fixture.root().join(".claudix").join("stray.rs");
        if let Some(parent) = internal.parent() {
            fs::create_dir_all(parent).await?;
        }
        fs::write(&internal, b"pub fn stray() -> u32 { 1 }\n").await?;

        let stats = claudix.reindex_file(Path::new(".claudix/stray.rs")).await?;
        assert_eq!(stats.chunk_count, baseline.len());

        let after = claudix.store.read_chunks().await?;
        assert!(
            after
                .iter()
                .all(|row| !row.file_path.starts_with(".claudix")),
            "no chunk under .claudix/ should be embedded"
        );
        Ok(())
    }
}
