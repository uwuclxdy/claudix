pub mod chunking;
pub mod cli;
pub mod config;
pub mod embedding;
pub mod enumeration;
pub mod error;
pub mod hooks;
pub mod mcp;
pub mod prompts;
pub mod search;
pub mod store;
pub mod types;
pub mod util;

pub use error::{ClaudixError, Result};
pub use types::{
    ByteRange, Chunk, ChunkId, ChunkKind, Dimension, EmbeddedChunk, FileHash, Language, LineRange,
    RelativePath,
};

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use chunking::MultiLanguageChunker;
use config::{Config, EmbeddingProvider};
#[cfg(any(test, feature = "test-stub"))]
use embedding::StubProvider;
use embedding::bundled::bundled_model;
use embedding::{BundledProvider, FallbackProvider, HttpProvider, Provider};
use enumeration::{EnumeratedFile, FileEnumerator, PathFilters, WatchFilter};
use error::RecoveryHint;
use prompts::hints;
use search::neighbors::neighbors;
use search::{SearchQuery, SearchResults, Searcher};
use store::marker::change_neighbors::{
    ChangeNeighborsMarker, NeighborEntry, write as write_neighbors_marker,
};
use store::{Store, stored_chunks_from_embedded};
use tokio::{fs, task};
use types::reject_path_escape;

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

/// Unit implements `IndexProgress` as a no-op, so callers that don't care
/// about per-file events can pass `&mut ()` instead of a wrapper struct.
impl IndexProgress for () {
    fn file(&mut self, _path: &RelativePath, _status: IndexFileStatus) -> Result<()> {
        Ok(())
    }
}

impl Claudix {
    pub async fn new(project_root: PathBuf, config: Arc<Config>) -> Result<Self> {
        let embedder = build_provider(config.as_ref()).await?;
        Self::with_embedder(project_root, config, embedder)
    }

    /// Construct around an already-built provider (e.g. from a
    /// [`crate::embedding::ProviderCache`]) so a long-lived process skips the
    /// per-call provider build. Same manifest validation as [`Claudix::new`].
    pub fn with_embedder(
        project_root: PathBuf,
        config: Arc<Config>,
        embedder: Arc<dyn Provider>,
    ) -> Result<Self> {
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

    pub async fn index_full(&self, progress: &mut dyn IndexProgress) -> Result<IndexStats> {
        let enumerator =
            FileEnumerator::new(self.project_root.clone(), self.config.as_ref().clone())?;
        let files = enumerator.enumerate(&mut *progress)?;

        let current_files: Vec<(String, [u8; 16])> = files
            .iter()
            .map(|f| (f.relative_path.as_str().to_owned(), f.file_hash.0))
            .collect();

        // Files the active `.indexinclude` rules force into the index. One that
        // the store holds at zero chunks was last indexed before its rule
        // existed (e.g. the per-file hook recorded it via `note_file_hash` while
        // it routed through the no-op `Unknown` chunker). Its content hash is
        // unchanged, so the hash fast paths below would skip it forever; collect
        // such paths so they re-chunk on this incremental pass instead of only
        // after a full `force` rebuild.
        let force_included: HashSet<&str> = files
            .iter()
            .filter(|file| file.force_indexed)
            .map(|file| file.relative_path.as_str())
            .collect();
        let force_recheck = self
            .store
            .force_included_without_chunks(&force_included)
            .await?;

        // Check once whether the on-disk chunks table actually holds the rows
        // the manifest claims.  A crash between `drop_table` and `add` in
        // `persist_rows` leaves the table missing/empty while the manifest
        // still records the previous run's full file_hashes + chunk_count.
        // Both fast paths below guard on this result so that neither can
        // early-exit when the table is corrupt, permanently blinding search
        // until a manual `force` / `clear`.
        let table_matches = self.store.table_matches_manifest_chunk_count().await?;

        // Fast path: if the manifest already lists exactly these files with the
        // same hashes under the same embedding model — and no force-included
        // file is stuck at zero chunks — skip the LanceDB row read entirely.
        // touch_manifest_if_in_sync handles the timestamp bump.
        if force_recheck.is_empty()
            && table_matches
            && self
                .store
                .manifest_hashes_match(&current_files, self.config.as_ref())?
            && let Some(stats) = self
                .store
                .touch_manifest_if_in_sync(&current_files, self.config.as_ref())?
        {
            return Ok(IndexStats {
                file_count: stats.file_count,
                chunk_count: stats.chunk_count,
            });
        }

        let (changed_paths, unchanged_rows) = self
            .store
            .incremental_file_state(&current_files, &force_recheck, &mut *progress)
            .await?;

        if table_matches
            && changed_paths.is_empty()
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

        // Collect chunks for all changed files before embedding so the provider
        // can batch across file boundaries — 10 files × 8 chunks → 1 round-trip
        // at batch_size=32 instead of 10 serial round-trips.  Vectors for all
        // changed files are held in memory simultaneously; acceptable because the
        // total size is bounded by the number of changed chunks × dimension size.
        let mut file_chunk_counts: Vec<(&EnumeratedFile, usize)> = Vec::new();
        let mut all_chunks: Vec<Chunk> = Vec::new();

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
            file_chunk_counts.push((file, chunks.len()));
            all_chunks.extend(chunks);
        }

        // Single cross-file embed call; provider's batch_size governs request sizes.
        let all_embedded = self.embed_chunks(all_chunks).await?;

        // Partition embedded results back per file in original order and write rows.
        let mut offset = 0;
        for (file, count) in file_chunk_counts {
            let embedded_chunks = &all_embedded[offset..offset + count];
            offset += count;

            rows.retain(|row| row.file_path != file.relative_path.as_str());
            rows.extend(stored_chunks_from_embedded(
                embedded_chunks,
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

        let (skip_stats, preread_bytes) = self.skip_unchanged_target(&relative_path).await?;
        if let Some(stats) = skip_stats {
            // Prune chunks for files deleted out-of-band between no-op watch events;
            // a metadata scan per event is acceptable until watcher throughput matters.
            let stats = match self.store.prune_missing_files(self.config.as_ref()).await {
                Ok(Some(pruned)) => IndexStats {
                    file_count: pruned.file_count,
                    chunk_count: pruned.chunk_count,
                },
                _ => stats,
            };
            return Ok(stats);
        }

        let enumerator =
            FileEnumerator::new(self.project_root.clone(), self.config.as_ref().clone())?;
        // Mirror the full-reindex path: an `.indexinclude`d file of an unknown
        // language (e.g. a `.md` doc) only chunks when force-indexed, so a watch
        // or hook reindex must compute the same flag instead of hard-coding it.
        // `for_path` consults only the rule files on this file's ancestor chain,
        // keeping the per-edit path cheap while honoring nested rules.
        let force_indexed = PathFilters::for_path(&self.project_root, &relative_path)?
            .is_force_included(&relative_path);
        let Some(file) = enumerator.enumerate_one_with_bytes(
            relative_path.clone(),
            force_indexed,
            preread_bytes,
        )?
        else {
            let stats = self
                .store
                .delete_file_chunks(&relative_path, self.config.as_ref())
                .await?;
            // Prune other files deleted out-of-band alongside this explicit delete;
            // a metadata scan per watch event is acceptable until watcher throughput matters.
            let stats = match self.store.prune_missing_files(self.config.as_ref()).await {
                Ok(Some(pruned)) => pruned,
                _ => stats,
            };
            return Ok(IndexStats {
                file_count: stats.file_count,
                chunk_count: stats.chunk_count,
            });
        };

        let chunks = self.collect_file_chunks(&file).await?;
        let embedded_chunks = self.embed_chunks(chunks).await?;

        let surface_related =
            !embedded_chunks.is_empty() && self.config.hooks.surface_related_on_edit;
        // Snapshot the pre-edit chunk contents before the replace below drops
        // them; the neighbor pass then queries with what this edit introduced
        // instead of the whole file. An empty set means "everything is new",
        // which is both the brand-new-file case and the fail-open one: losing
        // the narrowing is better than losing the hint.
        let previous_contents: HashSet<FileHash> = if surface_related {
            self.store
                .read_file_chunks(&relative_path)
                .await
                .map(|rows| {
                    rows.iter()
                        .map(|row| enumeration::hash_bytes(row.content.as_bytes()))
                        .collect()
                })
                .unwrap_or_default()
        } else {
            HashSet::new()
        };

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

        // A per-file reindex only touches the edited file, so chunks for files
        // deleted out-of-band linger until the next full index. Prune them now,
        // before neighbor surfacing, so a deleted file is never offered as
        // related code. Fail-open: a prune error keeps the replace/delete stats.
        let stats = match self.store.prune_missing_files(self.config.as_ref()).await {
            Ok(Some(pruned)) => pruned,
            _ => stats,
        };

        // Compute change-neighbors using the fresh vectors — no extra embed call.
        // Fail-open: neighbor computation errors are discarded; the index is already updated.
        if surface_related {
            self.write_change_neighbors_marker(
                &relative_path,
                &embedded_chunks,
                &previous_contents,
            )
            .await;
        }

        Ok(IndexStats {
            file_count: stats.file_count,
            chunk_count: stats.chunk_count,
        })
    }

    /// Compute semantic neighbors of the chunks this edit introduced (those
    /// whose content is absent from `previous_contents`) and write the marker.
    /// Runs inside the detached reindex-file child — ONNX is already
    /// warm, `read_chunks` is a fast LanceDB scan. Fail-open: any error is
    /// silently discarded so the hook session continues normally.
    ///
    /// Ceiling: the narrowing rides on chunk content being stable across an
    /// unrelated edit, which holds for the tree-sitter chunkers but not for
    /// `chunk_fallback` — it windows on line index, so inserting or deleting a
    /// line rewrites every later window and the whole file reads as changed.
    /// Files without a grammar (markdown, toml, yaml, shell) therefore keep the
    /// old file-wide query. Upgrade path is content-defined chunk boundaries.
    async fn write_change_neighbors_marker(
        &self,
        relative_path: &RelativePath,
        embedded_chunks: &[EmbeddedChunk],
        previous_contents: &HashSet<FileHash>,
    ) {
        // Query with the chunks this edit actually introduced. Every chunk of
        // the file makes the neighbor set a property of the file rather than of
        // the edit, so every save re-injects the same "related code" list.
        let changed: Vec<&EmbeddedChunk> = embedded_chunks
            .iter()
            .filter(|ec| {
                !previous_contents.contains(&enumeration::hash_bytes(ec.chunk.content.as_bytes()))
            })
            .collect();

        // Chunkers emit a container (impl, class, inline mod) alongside the
        // items inside it, and the container's bytes cover theirs — so editing
        // one method marks the enclosing block changed too. Its vector is a
        // whole-block blur that reproduces the file-wide neighbor set this
        // filter exists to kill. Keep only the innermost changed chunks.
        let query_vectors: Vec<Vec<f32>> = changed
            .iter()
            .filter(|outer| {
                !changed.iter().any(|inner| {
                    strictly_contains(&outer.chunk.byte_range, &inner.chunk.byte_range)
                })
            })
            .map(|ec| ec.vector.clone())
            .collect();
        if query_vectors.is_empty() {
            return;
        }

        let Ok(all_rows) = self.store.read_chunks().await else {
            return;
        };

        let exclude = relative_path.clone();
        // Over-fetch the candidate pool. The per-session seen-filter runs later, at
        // ack time in hooks::post_tool_use, and can only subtract — so a marker cut
        // to top_k here goes silent for an edit whose every entry was already
        // surfaced, even while unshown candidates ranked below the cut qualify.
        // Depth gives that filter a tail to fall through to; it does not widen how
        // many hints an edit shows, which stays capped at top_k where the filter runs.
        //
        // Measured on this repo (88 files, 1344 chunks) the pool tops out near 23
        // neighbor files at the 0.80 floor, so 5x covers it outright. That ceiling is
        // a small-repo figure — re-measure before assuming it holds on a much larger
        // codebase, where this multiplier could truncate again.
        let top_k = self
            .config
            .hooks
            .related_top_k
            .saturating_mul(NEIGHBOR_CANDIDATE_DEPTH);
        let min_similarity = self.config.hooks.related_min_similarity;
        let Ok(hits) = task::spawn_blocking(move || {
            neighbors(&all_rows, &query_vectors, &exclude, top_k, min_similarity)
        })
        .await
        else {
            return;
        };

        if hits.is_empty() {
            return;
        }

        let marker_path = self.store.change_neighbors_marker_path();
        let entries: Vec<NeighborEntry> = hits
            .iter()
            .map(|n| NeighborEntry {
                file_path: n.file_path.clone(),
                line_start: n.line_start,
                line_end: n.line_end,
                name: n.name.clone(),
                score: n.score,
            })
            .collect();
        write_neighbors_marker(
            &marker_path,
            &ChangeNeighborsMarker {
                edited_path: relative_path.as_str().to_owned(),
                neighbors: entries,
            },
        );
    }

    pub async fn search(&self, query: SearchQuery) -> Result<SearchResults> {
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

    /// Check whether `relative_path` is unchanged according to the manifest.
    ///
    /// Returns `(Some(stats), None)` when the file is unchanged and processing
    /// can be skipped entirely. Returns `(None, Some(bytes))` when the manifest
    /// was checked but the hash didn't match — the bytes are returned so the
    /// caller can pass them to `enumerate_one_with_bytes` and avoid a second
    /// disk read. Returns `(None, None)` when the manifest can't be used as a
    /// guard (no manifest, model mismatch, file not found, oversized, etc.).
    async fn skip_unchanged_target(
        &self,
        relative_path: &RelativePath,
    ) -> Result<(Option<IndexStats>, Option<Vec<u8>>)> {
        let Some(manifest) = self.store.read_manifest()? else {
            return Ok((None, None));
        };
        let Some(stored_hash) = manifest.file_hashes.get(relative_path.as_str()).copied() else {
            return Ok((None, None));
        };
        if manifest.embedding_model != self.config.embedding.model
            || manifest.dimensions != self.config.embedding.dimensions
        {
            return Ok((None, None));
        }

        let absolute_path = self.project_root.join(relative_path.to_path_buf());
        let bytes = match fs::read(&absolute_path).await {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok((None, None)),
            Err(error) => return Err(error.into()),
        };
        if bytes.len() as u64 > self.config.indexing.max_file_size_kb.saturating_mul(1024) {
            return Ok((None, None));
        }
        if enumeration::hash_bytes(&bytes).0 != stored_hash {
            // Hash mismatch — return the bytes so the caller avoids re-reading.
            return Ok((None, Some(bytes)));
        }

        Ok((
            Some(IndexStats {
                file_count: usize::try_from(manifest.file_count).unwrap_or(usize::MAX),
                chunk_count: usize::try_from(manifest.chunk_count).unwrap_or(usize::MAX),
            }),
            None,
        ))
    }

    async fn collect_file_chunks(&self, file: &EnumeratedFile) -> Result<Vec<Chunk>> {
        let content = if let Some(bytes) = &file.content {
            // Bytes were pre-read by the caller; convert without a disk round-trip.
            match String::from_utf8(bytes.clone()) {
                Ok(s) => s,
                Err(_) => return Ok(Vec::new()),
            }
        } else {
            match fs::read_to_string(&file.absolute_path).await {
                Ok(content) => content,
                Err(error) if error.kind() == std::io::ErrorKind::InvalidData => {
                    return Ok(Vec::new());
                }
                Err(error) => return Err(error.into()),
            }
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
                        recovery: RecoveryHint(hints::REINDEX_ALIGN_DIMENSIONS),
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
                        recovery: RecoveryHint(hints::REINDEX_INSIDE_PROJECT_DIR),
                    })?;
            reject_path_escape(relative, hints::REINDEX_INSIDE_PROJECT_DIR)?;
            return Ok(RelativePath::from_path(relative));
        }

        reject_path_escape(path, hints::REINDEX_INSIDE_PROJECT_DIR)?;
        Ok(RelativePath::from_path(path))
    }
}

/// How far past the hint budget the neighbor candidate pool is fetched. The
/// per-session seen-filter runs at ack time and can only subtract, so the
/// marker needs a ranked tail to fall through to. Re-derive it with the
/// hint-distribution harness rather than editing it against intuition.
pub(crate) const NEIGHBOR_CANDIDATE_DEPTH: usize = 5;

/// True when `outer` covers `inner` and is strictly larger.
///
/// Strictness carries the whole edge case: two chunks over the identical range
/// would each read the other as contained, eliminate each other, and leave the
/// edit with nothing to query. Equal ranges keep both — neither is the
/// narrower signal.
fn strictly_contains(outer: &ByteRange, inner: &ByteRange) -> bool {
    outer.start <= inner.start
        && inner.end <= outer.end
        && (outer.start < inner.start || inner.end < outer.end)
}

pub(crate) async fn build_provider(config: &Config) -> Result<Arc<dyn Provider>> {
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
            if bundled_model(&config.embedding.model)
                .is_some_and(|model| model.dimensions == dimensions)
            {
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

/// Measurement harness, not a correctness test — kept out of `src/` but linked
/// in so it can reach crate internals. Every case in it is `#[ignore]`d.
#[cfg(test)]
#[path = "../tests/measure/hint_distribution.rs"]
mod hint_distribution;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::StubProvider;
    use crate::store::marker::change_neighbors as cn_marker;
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

    /// Provider that counts how many times `embed` is invoked (not how many
    /// items are passed).  Used to assert cross-file batching in `index_full`
    /// reduces call count to `ceil(total_chunks / batch_size)`.
    struct InvocationCountingProvider {
        inner: StubProvider,
        invocations: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl Provider for InvocationCountingProvider {
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
            self.invocations.fetch_add(1, Ordering::Relaxed);
            self.inner.embed(batch).await
        }

        async fn health_check(&self) -> Result<()> {
            self.inner.health_check().await
        }
    }

    /// Provider that returns a fixed per-call rotation of vectors, enabling
    /// deterministic control over cosine similarities in tests.
    struct RotatingProvider {
        dimension: Dimension,
        /// Vectors returned in round-robin per item in a batch.
        vectors: Vec<Vec<f32>>,
        calls: std::sync::Mutex<usize>,
    }

    impl RotatingProvider {
        fn new(dimension: Dimension, vectors: Vec<Vec<f32>>) -> Self {
            Self {
                dimension,
                vectors,
                calls: std::sync::Mutex::new(0),
            }
        }
    }

    #[async_trait]
    impl Provider for RotatingProvider {
        fn name(&self) -> &str {
            "rotating"
        }

        fn dimensions(&self) -> Dimension {
            self.dimension
        }

        fn model_id(&self) -> &str {
            "stub-v1"
        }

        async fn embed(&self, batch: &[&str]) -> Result<Vec<Vec<f32>>> {
            let mut idx = self.calls.lock().unwrap_or_else(|e| e.into_inner());
            let result = batch
                .iter()
                .map(|_| {
                    let v = self.vectors[*idx % self.vectors.len()].clone();
                    *idx += 1;
                    v
                })
                .collect();
            Ok(result)
        }

        async fn health_check(&self) -> Result<()> {
            Ok(())
        }
    }

    /// Three mutually orthogonal axes, picked by a marker word in the text.
    /// Identical text always maps to the same axis, so which chunks seeded a
    /// neighbor query is directly readable off the resulting neighbor list.
    fn content_keyed_vector(text: &str) -> Vec<f32> {
        let axis = if text.contains("alpha") {
            0
        } else if text.contains("beta") {
            1
        } else {
            2
        };
        let mut vector = vec![0.0_f32; 8];
        vector[axis] = 1.0;
        vector
    }

    /// Provider whose output depends only on the chunk text, so a chunk that
    /// survives an edit unchanged re-embeds to the vector already in the store.
    struct ContentKeyedProvider;

    #[async_trait]
    impl Provider for ContentKeyedProvider {
        fn name(&self) -> &str {
            "content-keyed"
        }

        fn dimensions(&self) -> Dimension {
            Dimension(8)
        }

        fn model_id(&self) -> &str {
            "stub-v1"
        }

        async fn embed(&self, batch: &[&str]) -> Result<Vec<Vec<f32>>> {
            Ok(batch
                .iter()
                .map(|text| content_keyed_vector(text))
                .collect())
        }

        async fn health_check(&self) -> Result<()> {
            Ok(())
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

        let stats = claudix.index_full(&mut ()).await;
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

        assert!(claudix.index_full(&mut ()).await.is_ok());
        assert!(
            fs::write(
                fixture.root().join("src/lib.rs"),
                "pub mod math;\n\npub fn salute(name: &str) -> String {\n    format!(\"hi {name}\")\n}\n\npub fn wave(name: &str) -> String {\n    format!(\"bye {name}\")\n}\n",
            )
            .await
            .is_ok()
        );

        let stats = claudix.index_full(&mut ()).await;
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

        assert!(claudix.index_full(&mut ()).await.is_ok());
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

        assert!(claudix.index_full(&mut ()).await.is_ok());

        // Reindex the same file without modifying it — hash matches stored hash, must skip.
        let stats = claudix.reindex_file(Path::new("src/math.rs")).await;
        assert!(stats.is_ok());
        let stats = stats.ok().unwrap_or_else(|| unreachable!());
        // Chunk count unchanged — no re-embedding happened.
        assert_eq!(stats.chunk_count, 3);
    }

    #[tokio::test]
    async fn reindex_file_unchanged_skip_triggers_zero_embed_calls() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let config = stub_config();
        let claudix = test_claudix(fixture.root().to_path_buf(), config)?;
        claudix.index_full(&mut ()).await?;

        // Wrap the same store+config in a CountingProvider and call reindex_file
        // on an unchanged file — the manifest hash guard must fire before embedding.
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
        c2.reindex_file(Path::new("src/math.rs")).await?;

        assert_eq!(
            calls.load(Ordering::Relaxed),
            0,
            "unchanged file must skip embed entirely"
        );
        Ok(())
    }

    #[tokio::test]
    async fn reindex_file_records_hash_for_no_chunk_file() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        // Write a binary file that will produce no chunks.
        fs::write(fixture.root().join("binary.bin"), [0xff, 0xfe, 0xfd]).await?;
        let config = stub_config();
        let claudix = test_claudix(fixture.root().to_path_buf(), config)?;

        claudix.index_full(&mut ()).await?;

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

        claudix.index_full(&mut ()).await?;
        let first_call_count = calls.load(Ordering::Relaxed);

        claudix.index_full(&mut ()).await?;

        assert_eq!(calls.load(Ordering::Relaxed), first_call_count);
        Ok(())
    }

    #[tokio::test]
    async fn index_full_preserves_unchanged_file_chunks_on_second_run() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let config = stub_config();
        let claudix = test_claudix(fixture.root().to_path_buf(), config)?;

        claudix.index_full(&mut ()).await?;

        // Modify only src/lib.rs; src/math.rs is untouched.
        fs::write(
            fixture.root().join("src/lib.rs"),
            "pub mod math;\n\npub fn salute(name: &str) -> String { format!(\"hi {name}\") }\n",
        )
        .await?;

        claudix.index_full(&mut ()).await?;

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

        claudix.index_full(&mut ()).await?;

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

        claudix.index_full(&mut ()).await?;

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

    /// Verifies that a second `index_full` on an unchanged fixture takes the
    /// manifest-first early exit: `manifest_hashes_match` returns `true` so
    /// `incremental_file_state` (and therefore any LanceDB read) is never
    /// reached. Behaviorally this mirrors
    /// `index_full_skips_lancedb_rewrite_when_nothing_changed` but explicitly
    /// asserts the manifest guard condition, not just the side effect.
    #[tokio::test]
    async fn index_full_takes_manifest_first_early_exit_when_hashes_match() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let config = stub_config();
        let claudix = test_claudix(fixture.root().to_path_buf(), config.clone())?;

        // First index populates the manifest with file_hashes.
        let first = claudix.index_full(&mut ()).await?;
        assert!(first.file_count > 0);

        // Before the second call the manifest must report all hashes in sync.
        let manifest = claudix
            .store
            .read_manifest()?
            .unwrap_or_else(|| unreachable!());
        assert!(
            !manifest.file_hashes.is_empty(),
            "first index_full must populate file_hashes"
        );
        assert!(
            claudix.store.manifest_hashes_match(
                &manifest
                    .file_hashes
                    .iter()
                    .map(|(p, h)| (p.clone(), *h))
                    .collect::<Vec<_>>(),
                &config
            )?,
            "manifest_hashes_match must return true before second index_full"
        );

        // Second call must return identical stats via the early exit.
        let second = claudix.index_full(&mut ()).await?;
        assert_eq!(
            first, second,
            "early-exit must return the same stats as the first index"
        );

        // Timestamp must still be bumped.
        let manifest2 = claudix
            .store
            .read_manifest()?
            .unwrap_or_else(|| unreachable!());
        assert!(
            manifest2.last_full_index_at.is_some(),
            "early-exit must still bump last_full_index_at"
        );
        Ok(())
    }

    /// Empty `file_hashes` in the manifest (pre-migration index) must fall
    /// through to the normal incremental path — not take the early exit.
    #[tokio::test]
    async fn index_full_falls_through_when_manifest_file_hashes_empty() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let config = stub_config();
        let claudix = test_claudix(fixture.root().to_path_buf(), config.clone())?;

        // Seed a manifest with empty file_hashes to simulate a pre-migration index.
        claudix.store.ensure_layout()?;
        let mut manifest =
            crate::store::Manifest::new(&config.embedding.model, config.embedding.dimensions);
        manifest.file_count = 0;
        manifest.chunk_count = 0;
        // file_hashes intentionally left empty.
        claudix.store.write_manifest(&manifest)?;

        // manifest_hashes_match must return false for this shape.
        let current: Vec<(String, [u8; 16])> = vec![("src/lib.rs".to_owned(), [1u8; 16])];
        assert!(
            !claudix.store.manifest_hashes_match(&current, &config)?,
            "empty file_hashes must not trigger the manifest-first guard"
        );

        // index_full must still succeed via the incremental path.
        let stats = claudix.index_full(&mut ()).await?;
        assert!(stats.file_count > 0);
        Ok(())
    }

    /// A doc the per-file hook recorded at zero chunks before its
    /// `.indexinclude` rule existed must re-chunk on the next incremental
    /// `index_full`, not stay invisible until a full `force` rebuild. The
    /// content hash is unchanged across the rule addition, so the manifest-cache
    /// fast paths would otherwise skip it forever.
    #[tokio::test]
    async fn index_full_rechunks_force_included_zero_chunk_doc() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let config = stub_config();

        // `docs/` is gitignored, so without a rule the bulk walk never sees it.
        fs::write(fixture.root().join(".gitignore"), "docs/\n").await?;
        let doc = "# Internals\n\nLoad-bearing design notes for the project.\n";
        fs::create_dir_all(fixture.root().join("docs")).await?;
        fs::write(fixture.root().join("docs/internals.md"), doc).await?;

        let claudix = test_claudix(fixture.root().to_path_buf(), config)?;

        // 1. Index with no rule: docs/ is pruned by gitignore, absent from the store.
        claudix.index_full(&mut ()).await?;
        let baseline = claudix.store.read_chunks().await?;
        assert!(
            baseline.iter().all(|r| r.file_path != "docs/internals.md"),
            "doc must not be indexed before its rule exists"
        );

        // 2. Simulate the PostToolUse hook having touched the doc earlier (a
        //    global gitignore makes it watchable) → Unknown chunker → 0 chunks →
        //    hash recorded in the manifest.
        claudix.store.note_file_hash(
            &RelativePath::new("docs/internals.md"),
            crate::enumeration::hash_bytes(doc.as_bytes()).0,
            claudix.config.as_ref(),
        )?;

        // 3. Add the include rule. The doc's content is unchanged, so its
        //    manifest hash still matches — this is where the bug skipped it.
        fs::write(fixture.root().join(".indexinclude"), "docs/**\n").await?;

        claudix.index_full(&mut ()).await?;
        let rows = claudix.store.read_chunks().await?;
        assert!(
            rows.iter().any(|r| r.file_path == "docs/internals.md"),
            "force-included doc must re-chunk on incremental reindex after rule add"
        );

        // 4. Self-heal is stable: once it has chunks, force_recheck is empty so a
        //    further reindex takes the fast path and keeps the doc — no flapping.
        claudix.index_full(&mut ()).await?;
        let rows = claudix.store.read_chunks().await?;
        assert!(
            rows.iter().any(|r| r.file_path == "docs/internals.md"),
            "doc must stay indexed on subsequent reindexes"
        );
        Ok(())
    }

    /// `index_full` must batch chunks across file boundaries so the number of
    /// `embed` invocations equals `ceil(total_chunks / batch_size)`, not one
    /// call per changed file.
    #[tokio::test]
    async fn index_full_batches_embed_calls_across_files() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let mut config = stub_config();
        // Force batch_size=1 so each embed() call takes exactly one chunk; with
        // two files producing 3 chunks total we expect 3 invocations regardless
        // of how many files there are (proving cross-file batching is active).
        config.embedding.batch_size = 1;

        let invocations = Arc::new(AtomicUsize::new(0));
        let embedder: Arc<dyn Provider> = Arc::new(InvocationCountingProvider {
            inner: StubProvider::with_model_id(
                config.embedding.model.clone(),
                Dimension(config.embedding.dimensions),
            ),
            invocations: invocations.clone(),
        });
        let claudix = test_claudix_with_embedder(fixture.root().to_path_buf(), config, embedder)?;

        let stats = claudix.index_full(&mut ()).await?;
        // small_rust has 2 files with 3 chunks total (greet + add + module stub).
        let total_chunks = stats.chunk_count;
        let observed = invocations.load(Ordering::Relaxed);
        // With batch_size=1: expected = total_chunks; proves each chunk went
        // through a single flat embed pass, not one per-file pass.
        assert_eq!(
            observed, total_chunks,
            "expected {total_chunks} embed invocations (batch_size=1, cross-file), got {observed}"
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

        let stats = claudix.index_full(&mut ()).await;
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

        assert!(claudix.index_full(&mut ()).await.is_ok());
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
    async fn reindex_file_prunes_chunks_for_out_of_band_deletion() {
        let fixture = TestFixture::new("small_rust");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());
        let config = stub_config();

        let claudix = test_claudix(fixture.root().to_path_buf(), config);
        assert!(claudix.is_ok());
        let claudix = claudix.ok().unwrap_or_else(|| unreachable!());

        assert!(claudix.index_full(&mut ()).await.is_ok());

        // Delete math.rs out-of-band (rm / git / branch switch) — nothing
        // reindexes it directly. Then edit a DIFFERENT file so its reindex runs
        // for real; the per-file pass must still prune the deleted file's chunks.
        assert!(
            fs::remove_file(fixture.root().join("src/math.rs"))
                .await
                .is_ok()
        );
        assert!(
            fs::write(
                fixture.root().join("src/lib.rs"),
                b"pub fn greet() -> &'static str {\n    \"hello\"\n}\n",
            )
            .await
            .is_ok()
        );

        let stats = claudix.reindex_file(Path::new("src/lib.rs")).await;
        assert!(stats.is_ok());

        let rows = claudix.store.read_chunks().await;
        assert!(rows.is_ok());
        let rows = rows.ok().unwrap_or_else(|| unreachable!());
        assert!(
            rows.iter().all(|row| row.file_path != "src/math.rs"),
            "chunks for a file deleted out-of-band must be pruned on the next per-file reindex"
        );
        assert!(
            rows.iter().any(|row| row.file_path == "src/lib.rs"),
            "the reindexed file's chunks must remain"
        );
    }

    #[tokio::test]
    async fn reindex_file_unchanged_prunes_out_of_band_deleted_chunks() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let config = stub_config();
        let claudix = test_claudix(fixture.root().to_path_buf(), config)?;

        claudix.index_full(&mut ()).await?;

        // Delete math.rs out-of-band without touching lib.rs so the next
        // reindex_file call hits the hash-unchanged skip rather than the normal
        // replace path. The skip path must still prune the deleted file.
        fs::remove_file(fixture.root().join("src/math.rs")).await?;

        claudix.reindex_file(Path::new("src/lib.rs")).await?;

        let rows = claudix.store.read_chunks().await?;
        assert!(
            rows.iter().all(|row| row.file_path != "src/math.rs"),
            "the hash-unchanged skip must prune chunks for a file deleted out-of-band"
        );
        assert!(
            rows.iter().any(|row| row.file_path == "src/lib.rs"),
            "the unchanged reindexed file's chunks must remain"
        );
        Ok(())
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

        assert!(claudix.index_full(&mut ()).await.is_ok());
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
        claudix.index_full(&mut ()).await?;
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

    // ── change-neighbors ───────────────────────────────────────────────────

    /// Build a `Claudix` backed by a `RotatingProvider` that cycles through
    /// the given `vectors` across all embedding calls. Use this to control
    /// cosine similarities deterministically in neighbor tests.
    fn claudix_with_rotating(
        project_root: std::path::PathBuf,
        mut config: Config,
        vectors: Vec<Vec<f32>>,
    ) -> Result<Claudix> {
        config.embedding.dimensions = vectors.first().map(|v| v.len() as u16).unwrap_or(8);
        let store = Store::new(&project_root, &config)?;
        let embedder: Arc<dyn Provider> = Arc::new(RotatingProvider::new(
            Dimension(config.embedding.dimensions),
            vectors,
        ));
        Ok(Claudix {
            config: Arc::new(config),
            project_root,
            embedder,
            store,
        })
    }

    /// Seed the store with a pre-computed chunk so the neighbor scan can find it.
    async fn seed_chunk(
        store: &Store,
        config: &Config,
        file_path: &str,
        name: &str,
        vector: Vec<f32>,
    ) -> Result<()> {
        seed_chunks(store, config, &[(file_path, name, vector)]).await
    }

    /// Multi-file form of [`seed_chunk`]. `replace_chunks` rewrites the whole
    /// table, so seeding several files needs one call, not one call per file.
    async fn seed_chunks(
        store: &Store,
        config: &Config,
        entries: &[(&str, &str, Vec<f32>)],
    ) -> Result<()> {
        use crate::types::{ByteRange, ChunkId, ChunkKind, EmbeddedChunk, FileHash, LineRange};
        let embedded: Vec<EmbeddedChunk> = entries
            .iter()
            .enumerate()
            .map(|(index, (file_path, name, vector))| EmbeddedChunk {
                chunk: Chunk {
                    id: ChunkId(index as u64 + 1),
                    file_path: RelativePath::new(*file_path),
                    language: crate::types::Language::Rust,
                    kind: ChunkKind::Function,
                    name: Some((*name).to_owned()),
                    line_range: LineRange { start: 1, end: 5 },
                    byte_range: ByteRange { start: 0, end: 50 },
                    file_hash: FileHash([0u8; 16]),
                    content: format!("pub fn {name}() {{}}"),
                },
                vector: vector.clone(),
            })
            .collect();
        store.replace_chunks(&embedded, config).await?;
        Ok(())
    }

    #[tokio::test]
    async fn reindex_file_writes_change_neighbors_marker_for_near_duplicate() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let mut config = stub_config();
        // Set a zero floor so any similarity causes a hit.
        config.hooks.surface_related_on_edit = true;
        config.hooks.related_top_k = 5;
        config.hooks.related_min_similarity = 0.0;

        // Both the query vector (used for the edited file's chunks) and the
        // seed vector (stored for src/other.rs) are [1,0,...,0] → cosine = 1.0.
        let shared_vector = vec![1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let claudix = claudix_with_rotating(
            fixture.root().to_path_buf(),
            config.clone(),
            vec![shared_vector.clone()],
        )?;
        claudix.store.ensure_layout()?;

        // Seed "src/other.rs" with the same vector as the to-be-edited file.
        // It must also exist on disk or the reindex-time prune (which drops
        // chunks for deleted files) would remove it before neighbor surfacing.
        seed_chunk(
            &claudix.store,
            claudix.config.as_ref(),
            "src/other.rs",
            "other_fn",
            shared_vector,
        )
        .await?;
        tokio::fs::write(
            fixture.root().join("src/other.rs"),
            b"pub fn other_fn() {}\n",
        )
        .await?;

        // Write a real file for reindex_file to pick up (it must exist on disk).
        tokio::fs::write(
            fixture.root().join("src/lib.rs"),
            b"pub fn greet(name: &str) -> String { format!(\"Hello, {name}!\") }\n",
        )
        .await?;

        claudix.reindex_file(Path::new("src/lib.rs")).await?;

        let marker_path = claudix.store.change_neighbors_marker_path();
        assert!(
            marker_path.exists(),
            "change-neighbors marker must be written after editing a file with a near-duplicate"
        );

        let marker = cn_marker::read(&marker_path);
        assert!(marker.is_some(), "marker must parse correctly");
        let marker = marker.unwrap_or_else(|| unreachable!());

        assert_eq!(marker.edited_path, "src/lib.rs");
        assert!(
            marker
                .neighbors
                .iter()
                .any(|n| n.file_path == "src/other.rs"),
            "near-duplicate src/other.rs must appear in marker neighbors"
        );
        assert!(
            marker.neighbors.iter().all(|n| n.file_path != "src/lib.rs"),
            "edited file must not appear in its own neighbor list"
        );
        Ok(())
    }

    #[tokio::test]
    async fn reindex_file_no_marker_when_no_similar_code() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let mut config = stub_config();
        config.hooks.surface_related_on_edit = true;
        config.hooks.related_top_k = 5;
        // Use a very high floor — nothing will pass.
        config.hooks.related_min_similarity = 1.1;

        let claudix = claudix_with_rotating(
            fixture.root().to_path_buf(),
            config.clone(),
            vec![vec![1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        )?;
        claudix.store.ensure_layout()?;

        // Seed a dissimilar chunk.
        seed_chunk(
            &claudix.store,
            claudix.config.as_ref(),
            "src/other.rs",
            "other_fn",
            vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        .await?;

        tokio::fs::write(fixture.root().join("src/lib.rs"), b"pub fn greet() {}\n").await?;

        claudix.reindex_file(Path::new("src/lib.rs")).await?;

        assert!(
            !claudix.store.change_neighbors_marker_path().exists(),
            "no marker must be written when nothing passes the similarity floor"
        );
        Ok(())
    }

    #[tokio::test]
    async fn reindex_file_no_marker_when_feature_disabled() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let mut config = stub_config();
        config.hooks.surface_related_on_edit = false;
        config.hooks.related_min_similarity = 0.0;

        let shared_vector = vec![1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let claudix = claudix_with_rotating(
            fixture.root().to_path_buf(),
            config.clone(),
            vec![shared_vector.clone()],
        )?;
        claudix.store.ensure_layout()?;

        seed_chunk(
            &claudix.store,
            claudix.config.as_ref(),
            "src/other.rs",
            "other_fn",
            shared_vector,
        )
        .await?;

        tokio::fs::write(fixture.root().join("src/lib.rs"), b"pub fn greet() {}\n").await?;

        claudix.reindex_file(Path::new("src/lib.rs")).await?;

        assert!(
            !claudix.store.change_neighbors_marker_path().exists(),
            "no marker must be written when surface_related_on_edit = false"
        );
        Ok(())
    }

    const ALPHA_FN: &str = "pub fn alpha() {\n    1\n}\n";
    const BETA_FN: &str = "pub fn beta() {\n    2\n}\n";

    /// Store + provider wired so each chunk of the edited file maps to exactly
    /// one seeded neighbor file. Which chunks seeded the neighbor query is then
    /// readable straight off the marker's file list. `top_k` is a parameter so a
    /// caller can set a budget below the two seeded neighbors and observe the
    /// candidate pool independently of it.
    async fn changed_chunk_fixture(top_k: usize) -> Result<(TestFixture, Claudix)> {
        let fixture = TestFixture::new("small_rust")?;
        let mut config = stub_config();
        config.hooks.surface_related_on_edit = true;
        config.hooks.related_top_k = top_k;
        config.hooks.related_min_similarity = 0.5;

        let claudix = test_claudix_with_embedder(
            fixture.root().to_path_buf(),
            config,
            Arc::new(ContentKeyedProvider),
        )?;
        claudix.store.ensure_layout()?;

        seed_chunks(
            &claudix.store,
            claudix.config.as_ref(),
            &[
                (
                    "src/near_alpha.rs",
                    "near_alpha",
                    content_keyed_vector("alpha"),
                ),
                (
                    "src/near_beta.rs",
                    "near_beta",
                    content_keyed_vector("beta"),
                ),
            ],
        )
        .await?;
        // Both must exist on disk: the reindex-time prune drops chunks for
        // missing files before neighbor surfacing runs.
        for path in ["src/near_alpha.rs", "src/near_beta.rs"] {
            fs::write(fixture.root().join(path), b"pub fn seeded() {}\n").await?;
        }
        Ok((fixture, claudix))
    }

    fn marker_neighbor_paths(marker_path: &Path) -> Vec<String> {
        cn_marker::read(marker_path)
            .map(|marker| {
                marker
                    .neighbors
                    .iter()
                    .map(|n| n.file_path.clone())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// The marker is a candidate pool, not the hint list. The per-session
    /// seen-filter runs later (at ack, in `hooks::post_tool_use`) and can only
    /// subtract, so a pool cut to `related_top_k` here leaves an edit whose every
    /// entry was already surfaced with nothing to fall through to.
    #[tokio::test]
    async fn reindex_file_over_fetches_neighbor_pool_beyond_top_k() -> Result<()> {
        // A budget below the two seeded neighbors: without the over-fetch the
        // marker holds one path, with it both.
        let (fixture, claudix) = changed_chunk_fixture(1).await?;
        let edited = fixture.root().join("src/edited.rs");

        fs::write(&edited, format!("{ALPHA_FN}\n{BETA_FN}")).await?;
        claudix.reindex_file(Path::new("src/edited.rs")).await?;

        // Count, not identity: on an exact score tie `neighbor_rank` still orders
        // deterministically by path, but which entry survives a budget of 1 is not
        // the property under test.
        let pooled = marker_neighbor_paths(&claudix.store.change_neighbors_marker_path()).len();
        assert_eq!(
            pooled, 2,
            "marker must pool both qualifying neighbors despite a top_k of 1, so the \
             ack-time seen-filter has a tail to fall through to; got {pooled}"
        );
        Ok(())
    }

    /// The neighbor query must be seeded by the chunks the edit introduced, not
    /// by every chunk in the file — otherwise the surfaced set is a property of
    /// the file and every save re-injects the same list.
    #[tokio::test]
    async fn reindex_file_queries_only_chunks_whose_content_changed() -> Result<()> {
        let (fixture, claudix) = changed_chunk_fixture(5).await?;
        let edited = fixture.root().join("src/edited.rs");
        let marker_path = claudix.store.change_neighbors_marker_path();

        fs::write(&edited, format!("{ALPHA_FN}\n{BETA_FN}")).await?;
        claudix.reindex_file(Path::new("src/edited.rs")).await?;

        // Control: with nothing stored for the file, both chunks are new and
        // both neighbors are reachable. Without it the assertion below could
        // pass on a fixture where `near_alpha` never surfaces at all.
        assert_eq!(
            marker_neighbor_paths(&marker_path),
            vec!["src/near_alpha.rs", "src/near_beta.rs"],
            "control: every chunk of an unindexed file seeds the query"
        );
        fs::remove_file(&marker_path).await?;

        // Rewrite only the beta function; the alpha function's bytes are
        // byte-identical across the edit.
        fs::write(
            &edited,
            format!("{ALPHA_FN}\npub fn beta() {{\n    22\n}}\n"),
        )
        .await?;
        claudix.reindex_file(Path::new("src/edited.rs")).await?;

        assert_eq!(
            marker_neighbor_paths(&marker_path),
            vec!["src/near_beta.rs"],
            "only the chunk whose content changed may seed the neighbor query"
        );
        Ok(())
    }

    /// An edit that leaves every chunk's content intact (here: reordering two
    /// functions) introduces nothing to surface, so no marker is written.
    #[tokio::test]
    async fn reindex_file_writes_no_marker_when_no_chunk_content_changed() -> Result<()> {
        let (fixture, claudix) = changed_chunk_fixture(5).await?;
        let edited = fixture.root().join("src/edited.rs");
        let marker_path = claudix.store.change_neighbors_marker_path();

        fs::write(&edited, format!("{ALPHA_FN}\n{BETA_FN}")).await?;
        claudix.reindex_file(Path::new("src/edited.rs")).await?;
        assert!(marker_path.exists(), "control: the first index surfaces");
        fs::remove_file(&marker_path).await?;

        fs::write(&edited, format!("{BETA_FN}\n{ALPHA_FN}")).await?;
        claudix.reindex_file(Path::new("src/edited.rs")).await?;

        // Liveness control: the file hash changed, so the reindex re-chunked
        // rather than short-circuiting on an unchanged hash. Rows sort by byte
        // offset, so beta leading proves the swap landed in the store.
        let rows = claudix
            .store
            .read_file_chunks(&RelativePath::new("src/edited.rs"))
            .await?;
        assert_eq!(
            rows.first().and_then(|row| row.name.clone()),
            Some("beta".to_owned()),
            "the reindex must have re-chunked the reordered file"
        );
        assert!(
            !marker_path.exists(),
            "an edit that changes no chunk content must surface nothing"
        );
        Ok(())
    }

    /// The shape that dominates real code: an `impl` is stored as its own chunk
    /// covering every method inside it, so editing one method marks the whole
    /// block changed too. That block's vector is a file-level blur — if it
    /// reaches the query, the narrowing is undone on almost every real edit.
    #[tokio::test]
    async fn reindex_file_drops_container_chunks_wrapping_a_changed_chunk() -> Result<()> {
        let (fixture, claudix) = changed_chunk_fixture(5).await?;
        let edited = fixture.root().join("src/edited.rs");
        let marker_path = claudix.store.change_neighbors_marker_path();

        // The `impl` chunk's content spans both methods, so it contains the
        // word "alpha" and embeds onto the alpha axis. Leaking it into the
        // query is therefore directly visible as a near_alpha hit.
        let with_body = |beta_body: &str| {
            format!(
                "pub struct Thing;\n\nimpl Thing {{\n    pub fn alpha(&self) -> u32 {{\n        1\n    }}\n\n    pub fn beta(&self) -> u32 {{\n        {beta_body}\n    }}\n}}\n"
            )
        };

        fs::write(&edited, with_body("2")).await?;
        claudix.reindex_file(Path::new("src/edited.rs")).await?;

        // Control: the impl block really is stored as its own chunk covering
        // both methods, so the test is exercising the container shape and not
        // a flat file the filter would handle trivially.
        let rows = claudix
            .store
            .read_file_chunks(&RelativePath::new("src/edited.rs"))
            .await?;
        let impl_row = rows
            .iter()
            .find(|row| row.kind == "impl")
            .ok_or_else(|| ClaudixError::Store("fixture lost its impl chunk".to_owned()))?;
        assert!(
            impl_row.content.contains("fn alpha") && impl_row.content.contains("fn beta"),
            "the impl chunk must span both methods for this test to mean anything"
        );
        fs::remove_file(&marker_path).await?;

        // Change only beta's body. The method chunk and the enclosing impl
        // chunk both change; alpha's own chunk does not.
        fs::write(&edited, with_body("22")).await?;
        claudix.reindex_file(Path::new("src/edited.rs")).await?;

        assert_eq!(
            marker_neighbor_paths(&marker_path),
            vec!["src/near_beta.rs"],
            "the enclosing impl chunk must not seed the query alongside the edited method"
        );
        Ok(())
    }

    #[test]
    fn strictly_contains_keeps_both_chunks_on_an_identical_range() {
        let outer = ByteRange { start: 0, end: 100 };
        let inner = ByteRange { start: 10, end: 50 };
        assert!(strictly_contains(&outer, &inner));
        assert!(!strictly_contains(&inner, &outer));

        // Equal ranges must not eliminate each other: mutual containment would
        // drop both and leave the edit with no query vectors at all.
        let same = ByteRange { start: 0, end: 100 };
        assert!(!strictly_contains(&outer, &same));
        assert!(!strictly_contains(&same, &outer));

        // Sharing one edge is still strict containment on the other.
        let flush_start = ByteRange { start: 0, end: 40 };
        assert!(strictly_contains(&outer, &flush_start));
        let flush_end = ByteRange {
            start: 60,
            end: 100,
        };
        assert!(strictly_contains(&outer, &flush_end));

        // Overlap without containment eliminates nothing.
        let overlapping = ByteRange {
            start: 50,
            end: 150,
        };
        assert!(!strictly_contains(&outer, &overlapping));
        assert!(!strictly_contains(&overlapping, &outer));
    }

    #[tokio::test]
    async fn reindex_file_new_file_queries_every_chunk() -> Result<()> {
        let (fixture, claudix) = changed_chunk_fixture(5).await?;
        let fresh = fixture.root().join("src/fresh.rs");

        fs::write(&fresh, format!("{ALPHA_FN}\n{BETA_FN}")).await?;
        claudix.reindex_file(Path::new("src/fresh.rs")).await?;

        assert_eq!(
            marker_neighbor_paths(&claudix.store.change_neighbors_marker_path()),
            vec!["src/near_alpha.rs", "src/near_beta.rs"],
            "a file with no stored chunks has every chunk seed the query"
        );
        Ok(())
    }

    /// Regression test for the manifest-vs-table corruption scenario: if the
    /// process crashes between `drop_table` and `add` in `persist_rows`, the
    /// chunks table is left missing/empty while `manifest.json` still records
    /// the previous run's full `file_hashes` + `chunk_count`.  The
    /// manifest-first fast path must detect this and fall through to a real
    /// rebuild, not early-exit with an empty search index.
    #[tokio::test]
    async fn index_full_rebuilds_after_chunks_table_corruption() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let config = stub_config();
        let claudix = test_claudix(fixture.root().to_path_buf(), config.clone())?;

        // First index: populates the chunks table and manifest normally.
        let first = claudix.index_full(&mut ()).await?;
        assert!(
            first.chunk_count > 0,
            "fixture must produce at least one chunk"
        );

        // Simulate the mid-rewrite crash: use the LanceDB API to drop the
        // chunks table while leaving manifest.json intact.  This mirrors the
        // state after `drop_table` completes but before `add` in `persist_rows`
        // — the exact window where a kill/crash leaves the store corrupted.
        // Using the API (vs. remove_dir_all) keeps commit-handler state clean
        // so subsequent connections open without internal inconsistency.
        claudix.store.drop_chunks_table_for_test().await?;

        // The row-count gate must detect the corruption (table gone, manifest
        // still claims chunk_count > 0) and return false so the fast path is
        // bypassed on the next index_full call.
        let matches = claudix.store.table_matches_manifest_chunk_count().await?;
        assert!(
            !matches,
            "table_matches_manifest_chunk_count must return false when table is dropped"
        );

        // index_full must detect the mismatch and fall through to a real
        // rebuild, not early-exit with an empty search index.
        let second = claudix.index_full(&mut ()).await?;
        assert_eq!(
            second.chunk_count, first.chunk_count,
            "index_full must rebuild to the original chunk count after corruption"
        );

        // Verify the rows are actually present — not just counted from the manifest.
        let rows = claudix.store.read_chunks().await?;
        assert_eq!(
            rows.len(),
            second.chunk_count,
            "stored chunk rows must match the reported chunk_count after rebuild"
        );
        Ok(())
    }
}
