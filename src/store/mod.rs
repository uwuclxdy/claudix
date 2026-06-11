//! Persistence layer: LanceDB table + JSON manifest + cross-process lock.
//!
//! Submodules carry the heavy details — Arrow plumbing in `arrow`, lock
//! state machines in `lock`, pid markers in `marker`, the on-disk
//! manifest in `manifest`, and the in-memory row shape in `chunk_row`.
//! `mod.rs` keeps the `Store` aggregate and the high-level read/write
//! API the rest of the crate depends on.

mod arrow;
mod chunk_row;
pub mod lock;
pub mod manifest;
pub mod marker;

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use lancedb::{Connection, Table};
use tokio::sync::OnceCell;

use crate::config::Config;
use crate::error::{ClaudixError, RecoveryHint, Result};
use crate::types::{Dimension, EmbeddedChunk, RelativePath, reject_path_escape};
use crate::util::now_rfc3339;
use crate::{IndexFileStatus, IndexProgress};

use arrow::{chunk_schema, read_all_rows, read_metadata_rows, record_batch_from_rows};

pub(crate) use chunk_row::stored_chunks_from_embedded;
pub use chunk_row::{ChunkMetadata, StoredChunk};
pub use lock::IndexLockGuard;
pub use manifest::{Manifest, SCHEMA_VERSION};

const GITIGNORE_FILE_NAME: &str = ".gitignore";
const GITIGNORE_CONTENTS: &str = "*\n";
const CHUNKS_TABLE_NAME: &str = "chunks";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StorePaths {
    state_dir: PathBuf,
    index_dir: PathBuf,
    manifest_path: PathBuf,
    gitignore_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoreStats {
    pub chunk_count: usize,
    pub file_count: usize,
}

pub struct Store {
    project_root: PathBuf,
    paths: StorePaths,
    /// Cached LanceDB connection — opened once per `Store` instance, cloned on each use.
    connection: OnceCell<Connection>,
}

impl std::fmt::Debug for Store {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Store")
            .field("project_root", &self.project_root)
            .field("paths", &self.paths)
            .field("connection", &self.connection.get().map(|_| "<connected>"))
            .finish()
    }
}

impl Clone for Store {
    fn clone(&self) -> Self {
        // Clones share the same paths but get a fresh connection cell; the
        // connection will be re-established on first use in the clone.
        Self {
            project_root: self.project_root.clone(),
            paths: self.paths.clone(),
            connection: OnceCell::new(),
        }
    }
}

impl Store {
    pub fn new(project_root: impl AsRef<Path>, config: &Config) -> Result<Self> {
        let project_root = project_root.as_ref().canonicalize()?;
        let index_dir = resolve_project_path(&project_root, &config.paths.index_dir)?;
        let state_dir = index_dir
            .parent()
            .ok_or_else(|| ClaudixError::Store("index path has no parent directory".to_owned()))?
            .to_path_buf();

        let paths = StorePaths {
            manifest_path: state_dir.join(manifest::MANIFEST_FILE_NAME),
            gitignore_path: state_dir.join(GITIGNORE_FILE_NAME),
            state_dir,
            index_dir,
        };

        Ok(Self {
            project_root,
            paths,
            connection: OnceCell::new(),
        })
    }

    pub fn project_root(&self) -> &Path {
        &self.project_root
    }

    pub fn pending_index_marker_path(&self) -> PathBuf {
        self.paths.state_dir.join("indexing-pending")
    }

    pub fn watch_marker_path(&self) -> PathBuf {
        self.paths.state_dir.join("watch.pid")
    }

    pub fn change_neighbors_marker_path(&self) -> PathBuf {
        self.paths.state_dir.join("change-neighbors")
    }

    pub fn state_dir_path(&self) -> &Path {
        &self.paths.state_dir
    }

    pub fn ensure_layout(&self) -> Result<()> {
        fs::create_dir_all(&self.paths.index_dir)?;
        fs::write(&self.paths.gitignore_path, GITIGNORE_CONTENTS)?;
        Ok(())
    }

    pub fn read_manifest(&self) -> Result<Option<Manifest>> {
        if !self.paths.manifest_path.exists() {
            return Ok(None);
        }

        let text = fs::read_to_string(&self.paths.manifest_path)?;
        let manifest = serde_json::from_str(&text)?;
        Ok(Some(manifest))
    }

    pub fn write_manifest(&self, manifest: &Manifest) -> Result<()> {
        self.ensure_layout()?;

        let temp_path = self.paths.manifest_path.with_extension("json.tmp");
        let bytes = serde_json::to_vec_pretty(manifest)?;
        fs::write(&temp_path, bytes)?;
        fs::rename(temp_path, &self.paths.manifest_path)?;
        Ok(())
    }

    pub fn validate_manifest_compatibility(
        &self,
        expected_model: &str,
        expected_dimensions: u16,
    ) -> Result<Option<Manifest>> {
        self.read_manifest()?
            .map(|manifest| {
                manifest::validate_manifest_compatibility(
                    manifest,
                    expected_model,
                    expected_dimensions,
                )
            })
            .transpose()
    }

    pub async fn read_chunks(&self) -> Result<Vec<StoredChunk>> {
        let Some(table) = self.open_chunks_table().await? else {
            return Ok(Vec::new());
        };

        let mut rows = read_all_rows(&table).await?;
        sort_rows(&mut rows);
        Ok(rows)
    }

    /// Projected read that returns only lightweight scalar metadata, omitting
    /// embedding vectors. Use this when callers only need `file_path`,
    /// `file_hash`, `language`, or `name` — avoids loading large float arrays.
    pub async fn read_chunk_metadata(&self) -> Result<Vec<ChunkMetadata>> {
        let Some(table) = self.open_chunks_table().await? else {
            return Ok(Vec::new());
        };
        read_metadata_rows(&table).await
    }

    pub async fn stored_file_hash_and_stats(
        &self,
        relative_path: &RelativePath,
    ) -> Result<(Option<[u8; 16]>, StoreStats)> {
        let rows = self.read_chunks().await?;
        let file_hashes = self.stored_file_hashes(&rows)?;
        let hash = file_hashes.get(relative_path.as_str()).copied();
        let stats = stats_from_rows(&rows);
        Ok((hash, stats))
    }

    pub async fn incremental_file_state(
        &self,
        current_files: &[(String, [u8; 16])],
        progress: &mut dyn IndexProgress,
    ) -> Result<(HashSet<String>, Vec<StoredChunk>)> {
        // Use the projected metadata read to build the hash map — no vectors needed here.
        let metadata = self.read_chunk_metadata().await?;
        let stored_file_hashes = self.stored_file_hashes_from_metadata(&metadata)?;
        let stored_path_set: HashSet<&str> =
            stored_file_hashes.keys().map(String::as_str).collect();

        // Full rows are still required for the unchanged_rows output (callers
        // pass them straight into persist_incremental), so we load them after
        // the cheap hash-comparison pass.
        let stored_rows = self.read_chunks().await?;

        let mut changed_paths: HashSet<String> = HashSet::new();
        for (path, hash) in current_files {
            match stored_file_hashes.get(path.as_str()) {
                Some(stored_hash) if stored_hash == hash => {
                    progress.file(&RelativePath::new(path.as_str()), IndexFileStatus::Verified)?;
                }
                _ => {
                    changed_paths.insert(path.clone());
                    if !stored_path_set.contains(path.as_str()) {
                        progress.file(
                            &RelativePath::new(path.as_str()),
                            IndexFileStatus::Skipped("not present in index"),
                        )?;
                    }
                }
            }
        }

        let current_path_set: HashSet<&str> =
            current_files.iter().map(|(p, _)| p.as_str()).collect();

        let unchanged_rows = stored_rows
            .into_iter()
            .filter(|row| {
                current_path_set.contains(row.file_path.as_str())
                    && !changed_paths.contains(&row.file_path)
            })
            .collect();

        Ok((changed_paths, unchanged_rows))
    }

    /// Returns `true` when the manifest's `file_hashes` already covers exactly
    /// `current_files` (same set of paths, same hashes) under the active
    /// embedding model and dimensions.
    ///
    /// A `true` result means the caller can skip every LanceDB row read and
    /// jump straight to [`Self::touch_manifest_if_in_sync`].
    ///
    /// Returns `false` when the manifest is missing, when `file_hashes` is
    /// empty (pre-migration index), or when any path or hash differs.
    pub fn manifest_hashes_match(
        &self,
        current_files: &[(String, [u8; 16])],
        config: &Config,
    ) -> Result<bool> {
        let Some(manifest) = self.read_manifest()? else {
            return Ok(false);
        };
        if manifest.embedding_model != config.embedding.model
            || manifest.dimensions != config.embedding.dimensions
        {
            return Ok(false);
        }
        if manifest.file_hashes.is_empty() {
            return Ok(false);
        }
        if manifest.file_hashes.len() != current_files.len() {
            return Ok(false);
        }
        for (path, hash) in current_files {
            match manifest.file_hashes.get(path) {
                Some(stored) if stored == hash => {}
                _ => return Ok(false),
            }
        }
        Ok(true)
    }

    /// Bump `last_full_index_at` without rewriting the LanceDB table when the
    /// stored manifest already lists exactly `current_files` (same paths, same
    /// hashes) under the active embedding model.
    ///
    /// Returns `Some(stats)` if the fast path applied; `None` if the caller
    /// must fall through to [`Self::persist_incremental`].
    pub fn touch_manifest_if_in_sync(
        &self,
        current_files: &[(String, [u8; 16])],
        config: &Config,
    ) -> Result<Option<StoreStats>> {
        let Some(mut manifest) = self.read_manifest()? else {
            return Ok(None);
        };
        if manifest.embedding_model != config.embedding.model
            || manifest.dimensions != config.embedding.dimensions
        {
            return Ok(None);
        }
        if manifest.file_hashes.len() != current_files.len() {
            return Ok(None);
        }
        for (path, hash) in current_files {
            match manifest.file_hashes.get(path) {
                Some(stored) if stored == hash => {}
                _ => return Ok(None),
            }
        }

        let stats = StoreStats {
            chunk_count: usize::try_from(manifest.chunk_count).unwrap_or(usize::MAX),
            file_count: usize::try_from(manifest.file_count).unwrap_or(usize::MAX),
        };
        let timestamp = now_rfc3339();
        manifest.last_incremental_at = Some(timestamp.clone());
        manifest.last_full_index_at = Some(timestamp);
        self.write_manifest(&manifest)?;
        Ok(Some(stats))
    }

    pub async fn persist_incremental(
        &self,
        new_chunks: &[EmbeddedChunk],
        unchanged_rows: Vec<StoredChunk>,
        config: &Config,
        current_files: &[(String, [u8; 16])],
    ) -> Result<StoreStats> {
        let dimension = Dimension(config.embedding.dimensions);
        let new_rows = stored_chunks_from_embedded(new_chunks, dimension)?;

        let mut merged_rows = unchanged_rows;
        merged_rows.extend(new_rows);
        sort_rows(&mut merged_rows);

        let stats = stats_from_rows(&merged_rows);
        let file_hashes = current_files.iter().cloned().collect();
        self.persist_rows(merged_rows, dimension).await?;
        self.sync_manifest_with_timestamp(config, &stats, file_hashes, true)?;
        Ok(stats)
    }

    #[cfg(test)]
    pub(crate) async fn replace_chunks(
        &self,
        chunks: &[EmbeddedChunk],
        config: &Config,
    ) -> Result<StoreStats> {
        let dimension = Dimension(config.embedding.dimensions);
        let rows = stored_chunks_from_embedded(chunks, dimension)?;
        let stats = stats_from_rows(&rows);
        let file_hashes = file_hashes_from_rows(&rows);
        self.persist_rows(rows, dimension).await?;
        self.sync_manifest_with_timestamp(config, &stats, file_hashes, true)?;
        Ok(stats)
    }

    pub async fn replace_file_chunks(
        &self,
        chunks: &[EmbeddedChunk],
        config: &Config,
    ) -> Result<StoreStats> {
        let dimension = Dimension(config.embedding.dimensions);
        let replacement_rows = stored_chunks_from_embedded(chunks, dimension)?;
        let replacement_paths = distinct_file_paths(&replacement_rows);
        let existing_rows = self.read_chunks().await?;

        let mut merged_rows: Vec<_> = existing_rows
            .into_iter()
            .filter(|row| !replacement_paths.contains(&row.file_path))
            .collect();
        merged_rows.extend(replacement_rows);
        sort_rows(&mut merged_rows);

        let stats = stats_from_rows(&merged_rows);
        // Preserve no-chunk file hashes from the existing manifest so that files
        // that produce no chunks but were tracked by a prior index_full are not
        // forgotten by a per-file replace operation.
        let base = self
            .read_manifest()?
            .map(|m| m.file_hashes)
            .unwrap_or_default();
        let mut file_hashes = base;
        for path in &replacement_paths {
            file_hashes.remove(path.as_str());
        }
        file_hashes.extend(file_hashes_from_rows(&merged_rows));
        self.persist_rows(merged_rows, dimension).await?;
        self.sync_manifest_with_timestamp(config, &stats, file_hashes, false)?;
        Ok(stats)
    }

    pub async fn delete_file_chunks(
        &self,
        relative_path: &RelativePath,
        config: &Config,
    ) -> Result<StoreStats> {
        let dimension = Dimension(config.embedding.dimensions);
        let mut remaining_rows: Vec<_> = self
            .read_chunks()
            .await?
            .into_iter()
            .filter(|row| row.file_path != relative_path.as_str())
            .collect();
        sort_rows(&mut remaining_rows);

        let stats = stats_from_rows(&remaining_rows);
        let base = self
            .read_manifest()?
            .map(|m| m.file_hashes)
            .unwrap_or_default();
        let mut file_hashes = base;
        file_hashes.remove(relative_path.as_str());
        file_hashes.extend(file_hashes_from_rows(&remaining_rows));
        self.persist_rows(remaining_rows, dimension).await?;
        self.sync_manifest_with_timestamp(config, &stats, file_hashes, false)?;
        Ok(stats)
    }

    /// Drop indexed chunks for files that no longer exist under the project
    /// root, returning the post-prune stats when anything was removed.
    ///
    /// The per-file reindex path only ever touches the edited file, so a file
    /// deleted out-of-band (`rm`, `git rm`, a branch switch) keeps its chunks —
    /// and would surface as a phantom search hit or read-time neighbor — until
    /// the next full index. This closes that window cheaply: a projected
    /// metadata scan (no vectors) finds the missing paths, and the table is
    /// rewritten only when at least one path is actually gone. Returns `None`
    /// (no write) when every indexed file is still present.
    pub async fn prune_missing_files(&self, config: &Config) -> Result<Option<StoreStats>> {
        let metadata = self.read_chunk_metadata().await?;
        let missing = self.missing_paths_from_metadata(&metadata);
        if missing.is_empty() {
            return Ok(None);
        }

        let dimension = Dimension(config.embedding.dimensions);
        let remaining_rows: Vec<_> = self
            .read_chunks()
            .await?
            .into_iter()
            .filter(|row| !missing.contains(&row.file_path))
            .collect();

        let stats = stats_from_rows(&remaining_rows);
        let mut file_hashes = self
            .read_manifest()?
            .map(|m| m.file_hashes)
            .unwrap_or_default();
        for path in &missing {
            file_hashes.remove(path.as_str());
        }
        file_hashes.extend(file_hashes_from_rows(&remaining_rows));
        self.persist_rows(remaining_rows, dimension).await?;
        self.sync_manifest_with_timestamp(config, &stats, file_hashes, false)?;
        Ok(Some(stats))
    }

    /// Distinct row paths whose file no longer exists under the project root.
    /// Each distinct path is stat'd once.
    fn missing_paths_from_metadata(&self, metadata: &[ChunkMetadata]) -> HashSet<String> {
        let mut distinct: HashSet<&str> = HashSet::new();
        for entry in metadata {
            distinct.insert(entry.file_path.as_str());
        }
        distinct
            .into_iter()
            .filter(|path| !self.project_root.join(path).exists())
            .map(str::to_owned)
            .collect()
    }

    pub fn note_file_hash(
        &self,
        path: &RelativePath,
        hash: [u8; 16],
        config: &Config,
    ) -> Result<()> {
        let mut manifest = self
            .read_manifest()?
            .unwrap_or_else(|| Manifest::for_config(config));
        manifest.file_hashes.insert(path.as_str().to_owned(), hash);
        self.write_manifest(&manifest)
    }

    pub async fn clear_chunks(&self, config: &Config) -> Result<()> {
        self.stop_index_lock_holder();
        self.ensure_layout()?;

        let connection = self.open_connection().await?;
        if self.chunks_table_exists(&connection).await? {
            connection.drop_table(CHUNKS_TABLE_NAME, &[]).await?;
        }

        let manifest = Manifest::for_config(config);
        self.write_manifest(&manifest)
    }

    async fn persist_rows(&self, rows: Vec<StoredChunk>, dimension: Dimension) -> Result<()> {
        use arrow_array::{RecordBatchIterator, RecordBatchReader};

        self.ensure_layout()?;
        let connection = self.open_connection().await?;

        if self.chunks_table_exists(&connection).await? {
            connection.drop_table(CHUNKS_TABLE_NAME, &[]).await?;
        }

        let table = connection
            .create_empty_table(CHUNKS_TABLE_NAME, chunk_schema(dimension))
            .execute()
            .await?;

        if rows.is_empty() {
            return Ok(());
        }

        let batch = record_batch_from_rows(&rows, dimension)?;
        let reader: Box<dyn RecordBatchReader + Send> = Box::new(RecordBatchIterator::new(
            vec![Ok(batch)],
            chunk_schema(dimension),
        ));
        table.add(reader).execute().await?;
        Ok(())
    }

    async fn open_connection(&self) -> Result<Connection> {
        self.connection
            .get_or_try_init(|| async {
                lancedb::connect(&self.paths.index_dir.to_string_lossy())
                    .execute()
                    .await
                    .map_err(ClaudixError::from)
            })
            .await
            .cloned()
    }

    async fn open_chunks_table(&self) -> Result<Option<Table>> {
        let connection = self.open_connection().await?;
        if !self.chunks_table_exists(&connection).await? {
            return Ok(None);
        }

        connection
            .open_table(CHUNKS_TABLE_NAME)
            .execute()
            .await
            .map(Some)
            .map_err(ClaudixError::from)
    }

    async fn chunks_table_exists(&self, connection: &Connection) -> Result<bool> {
        let table_names = connection.table_names().execute().await?;
        Ok(table_names.iter().any(|name| name == CHUNKS_TABLE_NAME))
    }

    fn sync_manifest_with_timestamp(
        &self,
        config: &Config,
        stats: &StoreStats,
        file_hashes: BTreeMap<String, [u8; 16]>,
        full_index: bool,
    ) -> Result<()> {
        let mut manifest = self
            .read_manifest()?
            .unwrap_or_else(|| Manifest::for_config(config));
        let timestamp = now_rfc3339();
        manifest.embedding_model = config.embedding.model.clone();
        manifest.dimensions = config.embedding.dimensions;
        manifest.chunk_count = u64::try_from(stats.chunk_count).unwrap_or(u64::MAX);
        manifest.file_count = u64::try_from(stats.file_count).unwrap_or(u64::MAX);
        manifest.file_hashes = file_hashes;
        manifest.last_incremental_at = Some(timestamp.clone());
        if full_index {
            manifest.last_full_index_at = Some(timestamp);
        }
        self.write_manifest(&manifest)
    }

    fn stored_file_hashes(
        &self,
        stored_rows: &[StoredChunk],
    ) -> Result<BTreeMap<String, [u8; 16]>> {
        let Some(manifest) = self.read_manifest()? else {
            return Ok(file_hashes_from_rows(stored_rows));
        };

        if manifest.file_hashes.is_empty() {
            return Ok(file_hashes_from_rows(stored_rows));
        }

        Ok(manifest.file_hashes)
    }

    /// Same logic as [`stored_file_hashes`], but accepts lightweight metadata
    /// rows instead of full [`StoredChunk`]s so callers can avoid loading vectors.
    fn stored_file_hashes_from_metadata(
        &self,
        metadata: &[ChunkMetadata],
    ) -> Result<BTreeMap<String, [u8; 16]>> {
        let Some(manifest) = self.read_manifest()? else {
            return Ok(file_hashes_from_metadata(metadata));
        };

        if manifest.file_hashes.is_empty() {
            return Ok(file_hashes_from_metadata(metadata));
        }

        Ok(manifest.file_hashes)
    }
}

fn resolve_project_path(project_root: &Path, relative_path: &Path) -> Result<PathBuf> {
    use crate::prompts::hints::STORE_INSIDE_PROJECT_DIR;
    reject_path_escape(relative_path, STORE_INSIDE_PROJECT_DIR)?;

    let resolved = project_root.join(relative_path);
    if resolved.starts_with(project_root) {
        return Ok(resolved);
    }

    Err(ClaudixError::PathTraversal {
        path: resolved,
        recovery: RecoveryHint(STORE_INSIDE_PROJECT_DIR),
    })
}

fn distinct_file_paths(rows: &[StoredChunk]) -> BTreeSet<String> {
    rows.iter().map(|row| row.file_path.clone()).collect()
}

fn file_hashes_from_rows(rows: &[StoredChunk]) -> BTreeMap<String, [u8; 16]> {
    rows.iter()
        .map(|row| (row.file_path.clone(), row.file_hash))
        .collect()
}

fn file_hashes_from_metadata(metadata: &[ChunkMetadata]) -> BTreeMap<String, [u8; 16]> {
    metadata
        .iter()
        .map(|m| (m.file_path.clone(), m.file_hash))
        .collect()
}

fn stats_from_rows(rows: &[StoredChunk]) -> StoreStats {
    StoreStats {
        chunk_count: rows.len(),
        file_count: distinct_file_paths(rows).len(),
    }
}

fn sort_rows(rows: &mut [StoredChunk]) {
    rows.sort_by(|left, right| {
        left.file_path
            .cmp(&right.file_path)
            .then(left.byte_start.cmp(&right.byte_start))
            .then(left.chunk_id.cmp(&right.chunk_id))
    });
}
#[cfg(test)]
mod tests {
    use super::arrow::{batches_to_rows, chunk_schema, record_batch_from_rows_unchecked};
    use super::lock::LOCK_FILE_NAME;
    use super::marker::read_pid;
    use super::*;
    use crate::types::{
        ByteRange, Chunk, ChunkId, ChunkKind, FileHash, Language, LineRange, RelativePath,
    };
    use ::arrow_array::types::Float32Type;
    use ::arrow_array::{
        ArrayRef, FixedSizeBinaryArray, FixedSizeListArray, RecordBatch, StringArray, UInt32Array,
        UInt64Array,
    };
    use std::sync::Arc;
    use tempfile::tempdir;

    fn manifest_with_schema(
        schema_version: u32,
        embedding_model: &str,
        dimensions: u16,
    ) -> Manifest {
        let mut manifest = Manifest::new(embedding_model, dimensions);
        manifest.schema_version = schema_version;
        manifest
    }

    fn sample_chunk(
        chunk_id: u64,
        file_path: &str,
        name: &str,
        content: &str,
        vector: &[f32],
    ) -> EmbeddedChunk {
        EmbeddedChunk {
            chunk: Chunk {
                id: ChunkId(chunk_id),
                file_path: RelativePath::new(file_path),
                language: Language::Rust,
                kind: ChunkKind::Function,
                name: Some(name.to_owned()),
                line_range: LineRange { start: 1, end: 3 },
                byte_range: ByteRange { start: 0, end: 32 },
                file_hash: FileHash([u8::try_from(chunk_id).unwrap_or(0); 16]),
                content: content.to_owned(),
            },
            vector: vector.to_vec(),
        }
    }

    #[test]
    fn manifest_new_uses_schema_defaults() {
        let manifest = Manifest::new("stub-model", 512);

        assert_eq!(manifest.schema_version, SCHEMA_VERSION);
        assert_eq!(manifest.embedding_model, "stub-model");
        assert_eq!(manifest.dimensions, 512);
        assert_eq!(manifest.chunk_count, 0);
        assert_eq!(manifest.file_count, 0);
        assert!(manifest.file_hashes.is_empty());
        assert!(manifest.last_full_index_at.is_none());
        assert!(manifest.last_incremental_at.is_none());
    }

    #[test]
    fn store_resolves_default_layout_inside_project() {
        let project_root = tempdir();
        assert!(project_root.is_ok());
        let project_root = project_root.ok().unwrap_or_else(|| unreachable!());

        let store = Store::new(project_root.path(), &Config::default());
        assert!(store.is_ok());
        let store = store.ok().unwrap_or_else(|| unreachable!());

        assert_eq!(store.project_root(), project_root.path());
        assert_eq!(store.paths.state_dir, project_root.path().join(".claudix"));
        assert_eq!(
            store.paths.index_dir,
            project_root.path().join(".claudix/index")
        );
        assert_eq!(
            store.paths.manifest_path,
            project_root.path().join(".claudix/manifest.json")
        );
    }

    #[test]
    fn ensure_layout_creates_index_state_and_gitignore() {
        let project_root = tempdir();
        assert!(project_root.is_ok());
        let project_root = project_root.ok().unwrap_or_else(|| unreachable!());

        let store = Store::new(project_root.path(), &Config::default());
        assert!(store.is_ok());
        let store = store.ok().unwrap_or_else(|| unreachable!());

        let state_dir = project_root.path().join(".claudix");
        let index_dir = state_dir.join("index");
        let gitignore_path = state_dir.join(".gitignore");

        assert!(store.ensure_layout().is_ok());
        assert!(state_dir.exists());
        assert!(index_dir.exists());

        let gitignore = fs::read_to_string(gitignore_path);
        assert!(gitignore.is_ok());
        assert_eq!(gitignore.ok().unwrap_or_else(|| unreachable!()), "*\n");
    }

    #[test]
    fn manifest_round_trips_through_disk() {
        let project_root = tempdir();
        assert!(project_root.is_ok());
        let project_root = project_root.ok().unwrap_or_else(|| unreachable!());

        let store = Store::new(project_root.path(), &Config::default());
        assert!(store.is_ok());
        let store = store.ok().unwrap_or_else(|| unreachable!());

        let mut manifest = Manifest::new("stub-model", 384);
        manifest.last_full_index_at = Some("2026-04-27T12:00:00Z".to_owned());
        manifest.chunk_count = 42;
        manifest.file_count = 7;

        assert!(store.write_manifest(&manifest).is_ok());

        let loaded = store.read_manifest();
        assert!(loaded.is_ok());
        assert_eq!(
            loaded.ok().unwrap_or_else(|| unreachable!()),
            Some(manifest)
        );
    }

    #[test]
    fn missing_manifest_returns_none() {
        let project_root = tempdir();
        assert!(project_root.is_ok());
        let project_root = project_root.ok().unwrap_or_else(|| unreachable!());

        let store = Store::new(project_root.path(), &Config::default());
        assert!(store.is_ok());
        let store = store.ok().unwrap_or_else(|| unreachable!());

        let manifest = store.read_manifest();
        assert!(manifest.is_ok());
        assert!(manifest.ok().unwrap_or_else(|| unreachable!()).is_none());
    }

    #[test]
    fn store_rejects_escape_paths() {
        let project_root = tempdir();
        assert!(project_root.is_ok());
        let project_root = project_root.ok().unwrap_or_else(|| unreachable!());

        let mut config = Config::default();
        config.paths.index_dir = PathBuf::from("../outside/index");

        let store = Store::new(project_root.path(), &config);
        assert!(matches!(store, Err(ClaudixError::PathTraversal { .. })));
    }

    #[test]
    fn manifest_compatibility_accepts_matching_state() {
        let project_root = tempdir();
        assert!(project_root.is_ok());
        let project_root = project_root.ok().unwrap_or_else(|| unreachable!());

        let store = Store::new(project_root.path(), &Config::default());
        assert!(store.is_ok());
        let store = store.ok().unwrap_or_else(|| unreachable!());

        let manifest = Manifest::new("stub-model", 512);
        assert!(store.write_manifest(&manifest).is_ok());

        let loaded = store.validate_manifest_compatibility("stub-model", 512);
        assert!(loaded.is_ok());
        assert_eq!(
            loaded.ok().unwrap_or_else(|| unreachable!()),
            Some(manifest)
        );
    }

    #[test]
    fn manifest_compatibility_rejects_schema_mismatch() {
        let project_root = tempdir();
        assert!(project_root.is_ok());
        let project_root = project_root.ok().unwrap_or_else(|| unreachable!());

        let store = Store::new(project_root.path(), &Config::default());
        assert!(store.is_ok());
        let store = store.ok().unwrap_or_else(|| unreachable!());

        let manifest = manifest_with_schema(SCHEMA_VERSION + 1, "stub-model", 512);
        assert!(store.write_manifest(&manifest).is_ok());

        let error = store.validate_manifest_compatibility("stub-model", 512);
        assert!(matches!(error, Err(ClaudixError::SchemaMismatch { .. })));
    }

    #[test]
    fn manifest_compatibility_rejects_model_mismatch() {
        let project_root = tempdir();
        assert!(project_root.is_ok());
        let project_root = project_root.ok().unwrap_or_else(|| unreachable!());

        let store = Store::new(project_root.path(), &Config::default());
        assert!(store.is_ok());
        let store = store.ok().unwrap_or_else(|| unreachable!());

        let manifest = Manifest::new("old-model", 512);
        assert!(store.write_manifest(&manifest).is_ok());

        let error = store.validate_manifest_compatibility("new-model", 512);
        assert!(matches!(
            error,
            Err(ClaudixError::EmbeddingModelMismatch { .. })
        ));
    }

    #[test]
    fn manifest_compatibility_rejects_dimension_mismatch() {
        let project_root = tempdir();
        assert!(project_root.is_ok());
        let project_root = project_root.ok().unwrap_or_else(|| unreachable!());

        let store = Store::new(project_root.path(), &Config::default());
        assert!(store.is_ok());
        let store = store.ok().unwrap_or_else(|| unreachable!());

        let manifest = Manifest::new("stub-model", 384);
        assert!(store.write_manifest(&manifest).is_ok());

        let error = store.validate_manifest_compatibility("stub-model", 512);
        assert!(matches!(error, Err(ClaudixError::DimensionMismatch { .. })));
    }

    #[tokio::test]
    async fn replace_chunks_persists_and_reads_rows() {
        let project_root = tempdir();
        assert!(project_root.is_ok());
        let project_root = project_root.ok().unwrap_or_else(|| unreachable!());
        let config = Config::default();

        let store = Store::new(project_root.path(), &config);
        assert!(store.is_ok());
        let store = store.ok().unwrap_or_else(|| unreachable!());

        let chunks = vec![
            sample_chunk(1, "src/lib.rs", "alpha", "pub fn alpha() {}", &[1.0; 384]),
            sample_chunk(2, "src/lib.rs", "beta", "pub fn beta() {}", &[2.0; 384]),
        ];

        let stats = store.replace_chunks(&chunks, &config).await;
        assert!(stats.is_ok());
        assert_eq!(
            stats.ok().unwrap_or_else(|| unreachable!()),
            StoreStats {
                chunk_count: 2,
                file_count: 1,
            }
        );

        let rows = store.read_chunks().await;
        assert!(rows.is_ok());
        let rows = rows.ok().unwrap_or_else(|| unreachable!());
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].name.as_deref(), Some("alpha"));
        assert_eq!(rows[1].name.as_deref(), Some("beta"));
        assert_eq!(rows[0].vector.len(), 384);

        let manifest = store.read_manifest();
        assert!(manifest.is_ok());
        let manifest = manifest.ok().unwrap_or_else(|| unreachable!());
        assert!(manifest.is_some());
        let manifest = manifest.unwrap_or_else(|| unreachable!());
        assert_eq!(manifest.chunk_count, 2);
        assert_eq!(manifest.file_count, 1);
        assert!(manifest.last_full_index_at.is_some());
        assert_eq!(manifest.last_full_index_at, manifest.last_incremental_at);
    }

    #[tokio::test]
    async fn replace_chunks_rejects_non_finite_vectors() {
        let project_root = tempdir();
        assert!(project_root.is_ok());
        let project_root = project_root.ok().unwrap_or_else(|| unreachable!());
        let config = Config::default();

        let store = Store::new(project_root.path(), &config);
        assert!(store.is_ok());
        let store = store.ok().unwrap_or_else(|| unreachable!());

        let chunks = vec![sample_chunk(
            1,
            "src/lib.rs",
            "alpha",
            "pub fn alpha() {}",
            &[f32::INFINITY; 384],
        )];

        let error = store.replace_chunks(&chunks, &config).await;
        assert!(
            matches!(error, Err(ClaudixError::Store(message)) if message.contains("non-finite"))
        );
    }

    #[test]
    fn batches_to_rows_rejects_non_finite_vectors() {
        let row = StoredChunk {
            chunk_id: 1,
            file_path: "src/lib.rs".to_owned(),
            language: "rust".to_owned(),
            kind: "function".to_owned(),
            name: Some("alpha".to_owned()),
            line_start: 1,
            line_end: 3,
            byte_start: 0,
            byte_end: 32,
            file_hash: [1; 16],
            content: "pub fn alpha() {}".to_owned(),
            vector: vec![f32::NAN, 1.0],
        };
        let batch = record_batch_from_rows_unchecked(&[row], Dimension(2));
        assert!(batch.is_ok());
        let batch = batch.ok().unwrap_or_else(|| unreachable!());

        let error = batches_to_rows(vec![batch]);
        assert!(
            matches!(error, Err(ClaudixError::Store(message)) if message.contains("non-finite"))
        );
    }

    #[test]
    fn batches_to_rows_rejects_null_vector_values() {
        let batch = record_batch_with_vector(Some(vec![Some(1.0), None]));
        assert!(batch.is_ok());
        let batch = batch.ok().unwrap_or_else(|| unreachable!());

        let error = batches_to_rows(vec![batch]);
        assert!(matches!(error, Err(ClaudixError::Store(message)) if message.contains("null")));
    }

    #[test]
    fn batches_to_rows_rejects_null_vectors() {
        let batch = record_batch_with_vector(None);
        assert!(batch.is_ok());
        let batch = batch.ok().unwrap_or_else(|| unreachable!());

        let error = batches_to_rows(vec![batch]);
        assert!(matches!(error, Err(ClaudixError::Store(message)) if message.contains("null")));
    }

    fn record_batch_with_vector(vector: Option<Vec<Option<f32>>>) -> Result<RecordBatch> {
        RecordBatch::try_new(
            chunk_schema(Dimension(2)),
            vec![
                Arc::new(UInt64Array::from(vec![1])) as ArrayRef,
                Arc::new(StringArray::from(vec!["src/lib.rs"])) as ArrayRef,
                Arc::new(StringArray::from(vec!["rust"])) as ArrayRef,
                Arc::new(StringArray::from(vec!["function"])) as ArrayRef,
                Arc::new(StringArray::from(vec![Some("alpha")])) as ArrayRef,
                Arc::new(UInt32Array::from(vec![1])) as ArrayRef,
                Arc::new(UInt32Array::from(vec![3])) as ArrayRef,
                Arc::new(UInt32Array::from(vec![0])) as ArrayRef,
                Arc::new(UInt32Array::from(vec![32])) as ArrayRef,
                Arc::new(
                    FixedSizeBinaryArray::try_from_sparse_iter_with_size(
                        vec![Some([1_u8; 16].as_slice())].into_iter(),
                        16,
                    )
                    .map_err(|error| ClaudixError::Store(error.to_string()))?,
                ) as ArrayRef,
                Arc::new(StringArray::from(vec!["pub fn alpha() {}"])) as ArrayRef,
                Arc::new(
                    FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(vec![vector], 2),
                ) as ArrayRef,
            ],
        )
        .map_err(|error| ClaudixError::Store(error.to_string()))
    }

    #[tokio::test]
    async fn incremental_updates_refresh_only_incremental_timestamp() {
        let project_root = tempdir();
        assert!(project_root.is_ok());
        let project_root = project_root.ok().unwrap_or_else(|| unreachable!());
        let config = Config::default();

        let store = Store::new(project_root.path(), &config);
        assert!(store.is_ok());
        let store = store.ok().unwrap_or_else(|| unreachable!());

        let initial = vec![sample_chunk(
            1,
            "src/lib.rs",
            "alpha",
            "pub fn alpha() {}",
            &[1.0; 384],
        )];
        assert!(store.replace_chunks(&initial, &config).await.is_ok());

        let manifest = store.read_manifest();
        assert!(manifest.is_ok());
        let manifest = manifest.ok().unwrap_or_else(|| unreachable!());
        assert!(manifest.is_some());
        let manifest = manifest.unwrap_or_else(|| unreachable!());
        let original_full = manifest.last_full_index_at.clone();
        assert!(original_full.is_some());

        let mut seed_manifest = manifest.clone();
        seed_manifest.last_incremental_at = Some("2026-04-26T00:00:00Z".to_owned());
        assert!(store.write_manifest(&seed_manifest).is_ok());

        let replacement = vec![sample_chunk(
            2,
            "src/lib.rs",
            "beta",
            "pub fn beta() {}",
            &[2.0; 384],
        )];
        assert!(
            store
                .replace_file_chunks(&replacement, &config)
                .await
                .is_ok()
        );

        let manifest = store.read_manifest();
        assert!(manifest.is_ok());
        let manifest = manifest.ok().unwrap_or_else(|| unreachable!());
        assert!(manifest.is_some());
        let manifest = manifest.unwrap_or_else(|| unreachable!());
        assert_eq!(manifest.last_full_index_at, original_full);
        assert_ne!(
            manifest.last_incremental_at,
            Some("2026-04-26T00:00:00Z".to_owned())
        );
        assert!(manifest.last_incremental_at.is_some());
    }

    #[tokio::test]
    async fn replace_file_chunks_replaces_only_target_file() {
        let project_root = tempdir();
        assert!(project_root.is_ok());
        let project_root = project_root.ok().unwrap_or_else(|| unreachable!());
        let config = Config::default();

        let store = Store::new(project_root.path(), &config);
        assert!(store.is_ok());
        let store = store.ok().unwrap_or_else(|| unreachable!());

        let initial = vec![
            sample_chunk(1, "src/lib.rs", "alpha", "pub fn alpha() {}", &[1.0; 384]),
            sample_chunk(2, "src/other.rs", "omega", "pub fn omega() {}", &[2.0; 384]),
        ];
        assert!(store.replace_chunks(&initial, &config).await.is_ok());

        let replacement = vec![sample_chunk(
            3,
            "src/lib.rs",
            "beta",
            "pub fn beta() {}",
            &[3.0; 384],
        )];

        let stats = store.replace_file_chunks(&replacement, &config).await;
        assert!(stats.is_ok());
        assert_eq!(
            stats.ok().unwrap_or_else(|| unreachable!()),
            StoreStats {
                chunk_count: 2,
                file_count: 2,
            }
        );

        let rows = store.read_chunks().await;
        assert!(rows.is_ok());
        let rows = rows.ok().unwrap_or_else(|| unreachable!());
        assert_eq!(rows.len(), 2);
        assert!(rows.iter().any(|row| row.name.as_deref() == Some("beta")));
        assert!(rows.iter().any(|row| row.name.as_deref() == Some("omega")));
        assert!(!rows.iter().any(|row| row.name.as_deref() == Some("alpha")));
    }

    #[tokio::test]
    async fn delete_file_chunks_removes_only_matching_path() {
        let project_root = tempdir();
        assert!(project_root.is_ok());
        let project_root = project_root.ok().unwrap_or_else(|| unreachable!());
        let config = Config::default();

        let store = Store::new(project_root.path(), &config);
        assert!(store.is_ok());
        let store = store.ok().unwrap_or_else(|| unreachable!());

        let initial = vec![
            sample_chunk(1, "src/lib.rs", "alpha", "pub fn alpha() {}", &[1.0; 384]),
            sample_chunk(2, "src/other.rs", "omega", "pub fn omega() {}", &[2.0; 384]),
        ];
        assert!(store.replace_chunks(&initial, &config).await.is_ok());

        let stats = store
            .delete_file_chunks(&RelativePath::new("src/lib.rs"), &config)
            .await;
        assert!(stats.is_ok());
        assert_eq!(
            stats.ok().unwrap_or_else(|| unreachable!()),
            StoreStats {
                chunk_count: 1,
                file_count: 1,
            }
        );

        let rows = store.read_chunks().await;
        assert!(rows.is_ok());
        let rows = rows.ok().unwrap_or_else(|| unreachable!());
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].file_path, "src/other.rs");
    }

    #[tokio::test]
    async fn prune_missing_files_drops_chunks_for_deleted_paths() -> Result<()> {
        let project_root = tempdir()?;
        let config = Config::default();
        let store = Store::new(project_root.path(), &config)?;

        // present.rs exists on disk; gone.rs does not (deleted out-of-band).
        let src = store.project_root().join("src");
        fs::create_dir_all(&src)?;
        fs::write(src.join("present.rs"), "fn present() {}")?;

        let initial = vec![
            sample_chunk(
                1,
                "src/present.rs",
                "present",
                "fn present() {}",
                &[1.0; 384],
            ),
            sample_chunk(2, "src/gone.rs", "gone", "fn gone() {}", &[2.0; 384]),
        ];
        store.replace_chunks(&initial, &config).await?;

        let pruned = store.prune_missing_files(&config).await?;
        assert_eq!(
            pruned,
            Some(StoreStats {
                chunk_count: 1,
                file_count: 1,
            }),
            "prune must drop the deleted file and report post-prune stats"
        );

        let rows = store.read_chunks().await?;
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].file_path, "src/present.rs");

        let manifest = store.read_manifest()?.unwrap_or_else(|| unreachable!());
        assert!(
            !manifest.file_hashes.contains_key("src/gone.rs"),
            "deleted path must be dropped from manifest file_hashes"
        );
        assert!(manifest.file_hashes.contains_key("src/present.rs"));

        // Every indexed file now exists → second prune is a noop (no rewrite).
        assert!(store.prune_missing_files(&config).await?.is_none());
        Ok(())
    }

    #[tokio::test]
    async fn clear_chunks_drops_table_and_resets_manifest_counts() {
        let project_root = tempdir();
        assert!(project_root.is_ok());
        let project_root = project_root.ok().unwrap_or_else(|| unreachable!());
        let config = Config::default();

        let store = Store::new(project_root.path(), &config);
        assert!(store.is_ok());
        let store = store.ok().unwrap_or_else(|| unreachable!());

        let chunks = vec![sample_chunk(
            1,
            "src/lib.rs",
            "alpha",
            "pub fn alpha() {}",
            &[1.0; 384],
        )];
        assert!(store.replace_chunks(&chunks, &config).await.is_ok());

        let cleared = store.clear_chunks(&config).await;
        assert!(cleared.is_ok());

        let rows = store.read_chunks().await;
        assert!(rows.is_ok());
        assert!(rows.ok().unwrap_or_else(|| unreachable!()).is_empty());

        let manifest = store.read_manifest();
        assert!(manifest.is_ok());
        let manifest = manifest.ok().unwrap_or_else(|| unreachable!());
        assert!(manifest.is_some());
        let manifest = manifest.unwrap_or_else(|| unreachable!());
        assert_eq!(manifest.chunk_count, 0);
        assert_eq!(manifest.file_count, 0);
        assert_eq!(manifest.embedding_model, config.embedding.model);
        assert_eq!(manifest.dimensions, config.embedding.dimensions);
    }

    #[tokio::test]
    async fn clear_chunks_removes_stale_index_lock() {
        let project_root = tempdir();
        assert!(project_root.is_ok());
        let project_root = project_root.ok().unwrap_or_else(|| unreachable!());
        let config = Config::default();

        let store = Store::new(project_root.path(), &config);
        assert!(store.is_ok());
        let store = store.ok().unwrap_or_else(|| unreachable!());
        assert!(store.ensure_layout().is_ok());

        let lock_path = store.paths.state_dir.join(LOCK_FILE_NAME);
        assert!(fs::write(&lock_path, "999999999\n").is_ok());

        let cleared = store.clear_chunks(&config).await;
        assert!(cleared.is_ok());
        assert!(!lock_path.exists());
    }

    #[test]
    fn acquire_index_lock_writes_current_pid() {
        let project_root = tempdir();
        assert!(project_root.is_ok());
        let project_root = project_root.ok().unwrap_or_else(|| unreachable!());
        let config = Config::default();

        let store = Store::new(project_root.path(), &config);
        assert!(store.is_ok());
        let store = store.ok().unwrap_or_else(|| unreachable!());

        let lock = store.acquire_index_lock();
        assert!(lock.is_some());

        let lock_path = store.paths.state_dir.join(LOCK_FILE_NAME);
        assert_eq!(read_pid(&lock_path), Some(std::process::id()));
    }

    #[tokio::test]
    async fn clear_chunks_resets_manifest_model_and_dimensions() {
        let project_root = tempdir();
        assert!(project_root.is_ok());
        let project_root = project_root.ok().unwrap_or_else(|| unreachable!());
        let mut config = Config::default();
        config.embedding.model = "active-model".to_owned();
        config.embedding.dimensions = 768;

        let store = Store::new(project_root.path(), &config);
        assert!(store.is_ok());
        let store = store.ok().unwrap_or_else(|| unreachable!());

        let old_manifest = Manifest::new("old-model", 384);
        assert!(store.write_manifest(&old_manifest).is_ok());

        let cleared = store.clear_chunks(&config).await;
        assert!(cleared.is_ok());

        let manifest = store.read_manifest();
        assert!(manifest.is_ok());
        let manifest = manifest.ok().unwrap_or_else(|| unreachable!());
        let manifest = manifest.unwrap_or_else(|| unreachable!());
        assert_eq!(manifest.embedding_model, "active-model");
        assert_eq!(manifest.dimensions, 768);
        assert_eq!(manifest.chunk_count, 0);
        assert_eq!(manifest.file_count, 0);
        assert!(manifest.file_hashes.is_empty());
    }

    #[tokio::test]
    async fn replace_chunks_rejects_dimension_mismatch() {
        let project_root = tempdir();
        assert!(project_root.is_ok());
        let project_root = project_root.ok().unwrap_or_else(|| unreachable!());
        let config = Config::default();

        let store = Store::new(project_root.path(), &config);
        assert!(store.is_ok());
        let store = store.ok().unwrap_or_else(|| unreachable!());

        let chunks = vec![sample_chunk(
            1,
            "src/lib.rs",
            "alpha",
            "pub fn alpha() {}",
            &[1.0; 8],
        )];

        let result = store.replace_chunks(&chunks, &config).await;
        assert!(matches!(
            result,
            Err(ClaudixError::DimensionMismatch { .. })
        ));
    }

    #[tokio::test]
    async fn read_chunks_returns_more_than_default_lancedb_limit() -> Result<()> {
        // Regression test: LanceDB Query::new() sets limit=Some(10) by default.
        // read_all_rows must override it or >10 chunks are silently truncated.
        let project_root = tempdir()?;
        let config = Config::default();
        let store = Store::new(project_root.path(), &config)?;

        let chunks: Vec<_> = (1u64..=15)
            .map(|i| {
                sample_chunk(
                    i,
                    &format!("src/f{i}.rs"),
                    &format!("fn{i}"),
                    "fn body",
                    &[i as f32 / 15.0; 384],
                )
            })
            .collect();

        store.replace_chunks(&chunks, &config).await?;

        let rows = store.read_chunks().await?;
        assert_eq!(
            rows.len(),
            15,
            "read_chunks must return all rows, not just the LanceDB default of 10"
        );
        Ok(())
    }

    #[tokio::test]
    async fn incremental_file_state_splits_changed_and_unchanged() -> Result<()> {
        let project_root = tempdir()?;
        let config = Config::default();
        let store = Store::new(project_root.path(), &config)?;

        // Seed: two files, file_hash = chunk_id byte repeated
        let initial = vec![
            sample_chunk(1, "src/a.rs", "fn_a", "fn a() {}", &[1.0; 384]),
            sample_chunk(2, "src/b.rs", "fn_b", "fn b() {}", &[2.0; 384]),
        ];
        store.replace_chunks(&initial, &config).await?;

        // a.rs unchanged (same hash [1;16]), b.rs changed (new hash [9;16])
        let current_files = vec![
            ("src/a.rs".to_owned(), [1u8; 16]),
            ("src/b.rs".to_owned(), [9u8; 16]),
        ];

        let (changed_paths, unchanged_rows) = store
            .incremental_file_state(&current_files, &mut ())
            .await?;

        assert!(
            !changed_paths.contains("src/a.rs"),
            "unchanged file must not be in changed_paths"
        );
        assert!(
            changed_paths.contains("src/b.rs"),
            "changed file must be in changed_paths"
        );
        assert_eq!(unchanged_rows.len(), 1);
        assert_eq!(unchanged_rows[0].file_path, "src/a.rs");
        Ok(())
    }

    #[tokio::test]
    async fn incremental_file_state_uses_manifest_hashes_without_chunks() -> Result<()> {
        let project_root = tempdir()?;
        let config = Config::default();
        let store = Store::new(project_root.path(), &config)?;

        let mut manifest = Manifest::for_config(&config);
        manifest.file_count = 1;
        manifest
            .file_hashes
            .insert("src/empty.rs".to_owned(), [7u8; 16]);
        store.write_manifest(&manifest)?;

        let current_files = vec![("src/empty.rs".to_owned(), [7u8; 16])];
        let (changed_paths, unchanged_rows) = store
            .incremental_file_state(&current_files, &mut ())
            .await?;

        assert!(changed_paths.is_empty());
        assert!(unchanged_rows.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn incremental_file_state_drops_deleted_files() -> Result<()> {
        let project_root = tempdir()?;
        let config = Config::default();
        let store = Store::new(project_root.path(), &config)?;

        let initial = vec![
            sample_chunk(1, "src/a.rs", "fn_a", "fn a() {}", &[1.0; 384]),
            sample_chunk(2, "src/b.rs", "fn_b", "fn b() {}", &[2.0; 384]),
        ];
        store.replace_chunks(&initial, &config).await?;

        // b.rs is not present in current_files (deleted)
        let current_files = vec![("src/a.rs".to_owned(), [1u8; 16])];

        let (changed_paths, unchanged_rows) = store
            .incremental_file_state(&current_files, &mut ())
            .await?;

        assert!(changed_paths.is_empty());
        assert_eq!(unchanged_rows.len(), 1);
        assert_eq!(unchanged_rows[0].file_path, "src/a.rs");
        Ok(())
    }

    #[tokio::test]
    async fn replace_file_chunks_preserves_no_chunk_hashes() -> Result<()> {
        let project_root = tempdir()?;
        let config = Config::default();
        let store = Store::new(project_root.path(), &config)?;

        // Record a no-chunk file hash directly in the manifest.
        let mut manifest = Manifest::for_config(&config);
        manifest
            .file_hashes
            .insert("src/empty.rs".to_owned(), [7u8; 16]);
        store.write_manifest(&manifest)?;

        // Now replace chunks for a different chunk-having file.
        let chunks = vec![sample_chunk(
            1,
            "src/a.rs",
            "fn_a",
            "fn a() {}",
            &[1.0; 384],
        )];
        store.replace_file_chunks(&chunks, &config).await?;

        let saved = store.read_manifest()?.unwrap_or_else(|| unreachable!());
        assert_eq!(
            saved.file_hashes.get("src/empty.rs").copied(),
            Some([7u8; 16]),
            "replace_file_chunks must not discard no-chunk file hashes"
        );
        assert!(saved.file_hashes.contains_key("src/a.rs"));
        Ok(())
    }

    #[tokio::test]
    async fn delete_file_chunks_preserves_no_chunk_hashes() -> Result<()> {
        let project_root = tempdir()?;
        let config = Config::default();
        let store = Store::new(project_root.path(), &config)?;

        let mut manifest = Manifest::for_config(&config);
        manifest
            .file_hashes
            .insert("src/empty.rs".to_owned(), [7u8; 16]);
        store.write_manifest(&manifest)?;

        let chunks = vec![
            sample_chunk(1, "src/a.rs", "fn_a", "fn a() {}", &[1.0; 384]),
            sample_chunk(2, "src/b.rs", "fn_b", "fn b() {}", &[2.0; 384]),
        ];
        store.replace_chunks(&chunks, &config).await?;
        // Patch the manifest to include the no-chunk entry (replace_chunks doesn't merge).
        let mut manifest = store.read_manifest()?.unwrap_or_else(|| unreachable!());
        manifest
            .file_hashes
            .insert("src/empty.rs".to_owned(), [7u8; 16]);
        store.write_manifest(&manifest)?;

        store
            .delete_file_chunks(&RelativePath::new("src/b.rs"), &config)
            .await?;

        let saved = store.read_manifest()?.unwrap_or_else(|| unreachable!());
        assert_eq!(
            saved.file_hashes.get("src/empty.rs").copied(),
            Some([7u8; 16]),
            "delete_file_chunks must not discard no-chunk file hashes"
        );
        assert!(!saved.file_hashes.contains_key("src/b.rs"));
        Ok(())
    }

    #[tokio::test]
    async fn note_file_hash_records_entry_without_touching_chunks() -> Result<()> {
        let project_root = tempdir()?;
        let config = Config::default();
        let store = Store::new(project_root.path(), &config)?;

        let chunks = vec![sample_chunk(
            1,
            "src/a.rs",
            "fn_a",
            "fn a() {}",
            &[1.0; 384],
        )];
        store.replace_chunks(&chunks, &config).await?;

        store.note_file_hash(&RelativePath::new("src/empty.rs"), [5u8; 16], &config)?;

        let saved = store.read_manifest()?.unwrap_or_else(|| unreachable!());
        assert_eq!(
            saved.file_hashes.get("src/empty.rs").copied(),
            Some([5u8; 16])
        );
        // chunk count for a.rs must be unchanged
        let rows = store.read_chunks().await?;
        assert_eq!(rows.len(), 1);
        Ok(())
    }

    #[tokio::test]
    async fn clear_chunks_resets_last_full_index_at() {
        let project_root = tempdir();
        assert!(project_root.is_ok());
        let project_root = project_root.ok().unwrap_or_else(|| unreachable!());
        let config = Config::default();

        let store = Store::new(project_root.path(), &config);
        assert!(store.is_ok());
        let store = store.ok().unwrap_or_else(|| unreachable!());

        let chunks = vec![sample_chunk(
            1,
            "src/lib.rs",
            "foo",
            "pub fn foo() {}",
            &[1.0; 384],
        )];
        assert!(store.replace_chunks(&chunks, &config).await.is_ok());

        let manifest = store.read_manifest();
        assert!(manifest.is_ok());
        let manifest = manifest.ok().unwrap_or_else(|| unreachable!());
        assert!(
            manifest
                .unwrap_or_else(|| unreachable!())
                .last_full_index_at
                .is_some()
        );

        assert!(store.clear_chunks(&config).await.is_ok());

        let manifest = store.read_manifest();
        assert!(manifest.is_ok());
        let manifest = manifest.ok().unwrap_or_else(|| unreachable!());
        let manifest = manifest.unwrap_or_else(|| unreachable!());
        assert!(
            manifest.last_full_index_at.is_none(),
            "clear must reset last_full_index_at so auto-reindex triggers on next session"
        );
        assert!(
            manifest.last_incremental_at.is_none(),
            "clear must reset last_incremental_at to avoid skipping files on next incremental run"
        );
        assert_eq!(manifest.chunk_count, 0);
        assert_eq!(manifest.file_count, 0);
    }

    #[tokio::test]
    async fn read_chunk_metadata_on_empty_store_returns_empty() {
        let project_root = tempdir();
        assert!(project_root.is_ok());
        let project_root = project_root.ok().unwrap_or_else(|| unreachable!());
        let config = Config::default();

        let store = Store::new(project_root.path(), &config);
        assert!(store.is_ok());
        let store = store.ok().unwrap_or_else(|| unreachable!());

        let metadata = store.read_chunk_metadata().await;
        assert!(metadata.is_ok());
        assert!(
            metadata.ok().unwrap_or_else(|| unreachable!()).is_empty(),
            "metadata read on empty store must return an empty vec"
        );
    }

    #[tokio::test]
    async fn read_chunk_metadata_returns_same_set_as_read_chunks() {
        let project_root = tempdir();
        assert!(project_root.is_ok());
        let project_root = project_root.ok().unwrap_or_else(|| unreachable!());
        let config = Config::default();

        let store = Store::new(project_root.path(), &config);
        assert!(store.is_ok());
        let store = store.ok().unwrap_or_else(|| unreachable!());

        let chunks = vec![
            sample_chunk(1, "src/lib.rs", "alpha", "pub fn alpha() {}", &[1.0; 384]),
            sample_chunk(2, "src/util.rs", "beta", "pub fn beta() {}", &[2.0; 384]),
            sample_chunk(3, "src/util.rs", "gamma", "pub fn gamma() {}", &[3.0; 384]),
        ];
        assert!(store.replace_chunks(&chunks, &config).await.is_ok());

        let full_rows = store.read_chunks().await;
        assert!(full_rows.is_ok());
        let full_rows = full_rows.ok().unwrap_or_else(|| unreachable!());

        let metadata = store.read_chunk_metadata().await;
        assert!(metadata.is_ok());
        let metadata = metadata.ok().unwrap_or_else(|| unreachable!());

        // Metadata count must equal full-row count.
        assert_eq!(
            metadata.len(),
            full_rows.len(),
            "metadata row count must match full read"
        );

        // Build a sorted view of (file_path, name) from both sides for comparison.
        let mut full_keys: Vec<(String, Option<String>)> = full_rows
            .iter()
            .map(|r| (r.file_path.clone(), r.name.clone()))
            .collect();
        full_keys.sort();

        let mut meta_keys: Vec<(String, Option<String>)> = metadata
            .iter()
            .map(|m| (m.file_path.clone(), m.name.clone()))
            .collect();
        meta_keys.sort();

        assert_eq!(
            meta_keys, full_keys,
            "metadata (file_path, name) pairs must match full read"
        );

        // file_hash must also be preserved correctly.
        let mut full_hashes: Vec<(String, [u8; 16])> = full_rows
            .iter()
            .map(|r| (r.file_path.clone(), r.file_hash))
            .collect();
        full_hashes.sort_by_key(|(p, _)| p.clone());

        let mut meta_hashes: Vec<(String, [u8; 16])> = metadata
            .iter()
            .map(|m| (m.file_path.clone(), m.file_hash))
            .collect();
        meta_hashes.sort_by_key(|(p, _)| p.clone());

        assert_eq!(
            meta_hashes, full_hashes,
            "file_hash values must match between metadata and full read"
        );

        // language must be preserved.
        for m in &metadata {
            assert!(
                !m.language.is_empty(),
                "metadata language must not be empty"
            );
        }

        // Vectors must NOT be present in metadata rows (they don't exist on the type).
        // This is enforced by ChunkMetadata not having a vector field — compile-time guarantee.
    }

    mod config_support {
        use crate as claudix;

        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/common/config_support.rs"
        ));
    }

    use config_support::stub_config;

    #[test]
    fn manifest_hashes_match_returns_false_when_no_manifest() -> Result<()> {
        let dir = tempdir()?;
        let config = stub_config();
        let store = Store::new(dir.path(), &config)?;

        let current = vec![("src/lib.rs".to_owned(), [1u8; 16])];
        assert!(!store.manifest_hashes_match(&current, &config)?);
        Ok(())
    }

    #[test]
    fn manifest_hashes_match_returns_false_for_empty_file_hashes() -> Result<()> {
        let dir = tempdir()?;
        let config = stub_config();
        let store = Store::new(dir.path(), &config)?;
        store.ensure_layout()?;

        // Manifest with empty file_hashes (pre-migration shape).
        let manifest = Manifest::new(&config.embedding.model, config.embedding.dimensions);
        store.write_manifest(&manifest)?;

        let current = vec![("src/lib.rs".to_owned(), [1u8; 16])];
        assert!(!store.manifest_hashes_match(&current, &config)?);
        Ok(())
    }

    #[test]
    fn manifest_hashes_match_returns_false_on_model_mismatch() -> Result<()> {
        let dir = tempdir()?;
        let config = stub_config();
        let store = Store::new(dir.path(), &config)?;
        store.ensure_layout()?;

        let mut manifest = Manifest::new("other-model", config.embedding.dimensions);
        manifest
            .file_hashes
            .insert("src/lib.rs".to_owned(), [1u8; 16]);
        manifest.file_count = 1;
        store.write_manifest(&manifest)?;

        let current = vec![("src/lib.rs".to_owned(), [1u8; 16])];
        assert!(!store.manifest_hashes_match(&current, &config)?);
        Ok(())
    }

    #[test]
    fn manifest_hashes_match_returns_false_when_hash_differs() -> Result<()> {
        let dir = tempdir()?;
        let config = stub_config();
        let store = Store::new(dir.path(), &config)?;
        store.ensure_layout()?;

        let mut manifest = Manifest::new(&config.embedding.model, config.embedding.dimensions);
        manifest
            .file_hashes
            .insert("src/lib.rs".to_owned(), [1u8; 16]);
        manifest.file_count = 1;
        store.write_manifest(&manifest)?;

        let current = vec![("src/lib.rs".to_owned(), [2u8; 16])];
        assert!(!store.manifest_hashes_match(&current, &config)?);
        Ok(())
    }

    #[test]
    fn manifest_hashes_match_returns_false_when_file_set_differs() -> Result<()> {
        let dir = tempdir()?;
        let config = stub_config();
        let store = Store::new(dir.path(), &config)?;
        store.ensure_layout()?;

        let mut manifest = Manifest::new(&config.embedding.model, config.embedding.dimensions);
        manifest
            .file_hashes
            .insert("src/lib.rs".to_owned(), [1u8; 16]);
        manifest.file_count = 1;
        store.write_manifest(&manifest)?;

        // Extra file not in manifest.
        let current = vec![
            ("src/lib.rs".to_owned(), [1u8; 16]),
            ("src/main.rs".to_owned(), [2u8; 16]),
        ];
        assert!(!store.manifest_hashes_match(&current, &config)?);
        Ok(())
    }

    #[test]
    fn manifest_hashes_match_returns_true_when_fully_in_sync() -> Result<()> {
        let dir = tempdir()?;
        let config = stub_config();
        let store = Store::new(dir.path(), &config)?;
        store.ensure_layout()?;

        let mut manifest = Manifest::new(&config.embedding.model, config.embedding.dimensions);
        manifest
            .file_hashes
            .insert("src/lib.rs".to_owned(), [1u8; 16]);
        manifest
            .file_hashes
            .insert("src/math.rs".to_owned(), [2u8; 16]);
        manifest.file_count = 2;
        store.write_manifest(&manifest)?;

        let current = vec![
            ("src/lib.rs".to_owned(), [1u8; 16]),
            ("src/math.rs".to_owned(), [2u8; 16]),
        ];
        assert!(store.manifest_hashes_match(&current, &config)?);
        Ok(())
    }
}
