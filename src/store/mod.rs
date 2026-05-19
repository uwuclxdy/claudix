use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs;
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use arrow_array::types::Float32Type;
use arrow_array::{
    Array, ArrayRef, FixedSizeBinaryArray, FixedSizeListArray, Float32Array, RecordBatch,
    RecordBatchIterator, RecordBatchReader, StringArray, UInt32Array, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::{Connection, Table};

use crate::config::Config;
use crate::error::{ClaudixError, RecoveryHint, Result};
use crate::types::{Dimension, EmbeddedChunk, RelativePath};
use crate::util::now_rfc3339;
use crate::{IndexFileStatus, IndexProgress};

pub const SCHEMA_VERSION: u32 = 1;
const MANIFEST_FILE_NAME: &str = "manifest.json";
const LOCK_FILE_NAME: &str = "index.lock";
// Long enough to outlast a full index on a large repo; the per-file reindex
// is a background subprocess, so a generous deadline is preferable to giving
// up and silently losing the user's edit.
const REINDEX_LOCK_WAIT_MS: u64 = 1_800_000;
const REINDEX_LOCK_POLL_MS: u64 = 50;
const LOCK_TERMINATION_GRACE_MS: u64 = 2_000;
const LOCK_TERMINATION_POLL_MS: u64 = 100;
const GITIGNORE_FILE_NAME: &str = ".gitignore";
const GITIGNORE_CONTENTS: &str = "*\n";
const CHUNKS_TABLE_NAME: &str = "chunks";

const FIELD_CHUNK_ID: &str = "chunk_id";
const FIELD_FILE_PATH: &str = "file_path";
const FIELD_LANGUAGE: &str = "language";
const FIELD_KIND: &str = "kind";
const FIELD_NAME: &str = "name";
const FIELD_LINE_START: &str = "line_start";
const FIELD_LINE_END: &str = "line_end";
const FIELD_BYTE_START: &str = "byte_start";
const FIELD_BYTE_END: &str = "byte_end";
const FIELD_FILE_HASH: &str = "file_hash";
const FIELD_CONTENT: &str = "content";
const FIELD_VECTOR: &str = "vector";

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Manifest {
    pub schema_version: u32,
    pub embedding_model: String,
    pub dimensions: u16,
    pub last_full_index_at: Option<String>,
    pub last_incremental_at: Option<String>,
    pub chunk_count: u64,
    pub file_count: u64,
    #[serde(default)]
    pub file_hashes: BTreeMap<String, [u8; 16]>,
}

impl Manifest {
    pub fn new(embedding_model: impl Into<String>, dimensions: u16) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            embedding_model: embedding_model.into(),
            dimensions,
            last_full_index_at: None,
            last_incremental_at: None,
            chunk_count: 0,
            file_count: 0,
            file_hashes: BTreeMap::new(),
        }
    }
}

pub struct IndexLockGuard {
    path: PathBuf,
}

impl Drop for IndexLockGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

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

#[derive(Debug, Clone, PartialEq)]
pub struct StoredChunk {
    pub chunk_id: u64,
    pub file_path: String,
    pub language: String,
    pub kind: String,
    pub name: Option<String>,
    pub line_start: u32,
    pub line_end: u32,
    pub byte_start: u32,
    pub byte_end: u32,
    pub file_hash: [u8; 16],
    pub content: String,
    pub vector: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct Store {
    project_root: PathBuf,
    paths: StorePaths,
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
            manifest_path: state_dir.join(MANIFEST_FILE_NAME),
            gitignore_path: state_dir.join(GITIGNORE_FILE_NAME),
            state_dir,
            index_dir,
        };

        Ok(Self {
            project_root,
            paths,
        })
    }

    pub fn project_root(&self) -> &Path {
        &self.project_root
    }

    pub fn pending_index_marker_path(&self) -> std::path::PathBuf {
        self.paths.state_dir.join("indexing-pending")
    }

    pub fn state_dir_path(&self) -> &Path {
        &self.paths.state_dir
    }

    /// Block until the shared chunk-writer lock is available, then claim it.
    ///
    /// Shares [`LOCK_FILE_NAME`] with [`Self::acquire_index_lock`] so a full
    /// index and a per-file reindex can never rewrite the chunk table at the
    /// same time. The deadline is long enough to outlast a full index on a
    /// large repo; if the holder dies or never wrote its PID, the dead-lock
    /// recovery branch reclaims it on the next poll.
    pub fn acquire_reindex_lock(&self) -> Result<IndexLockGuard> {
        fs::create_dir_all(&self.paths.state_dir)?;
        let lock_path = self.paths.state_dir.join(LOCK_FILE_NAME);
        let deadline = std::time::Instant::now() + Duration::from_millis(REINDEX_LOCK_WAIT_MS);

        loop {
            if let Ok(mut file) = fs::File::create_new(&lock_path) {
                let _ = writeln!(file, "{}", std::process::id());
                return Ok(IndexLockGuard { path: lock_path });
            }
            if let Some(pid) = read_lock_pid(&lock_path)
                && !process_running(pid)
            {
                let _ = fs::remove_file(&lock_path);
                continue;
            }
            if std::time::Instant::now() >= deadline {
                return Err(ClaudixError::Store(
                    "reindex lock contention: another reindex job is in progress".to_owned(),
                ));
            }
            thread::sleep(Duration::from_millis(REINDEX_LOCK_POLL_MS));
        }
    }

    pub fn acquire_index_lock(&self) -> Option<IndexLockGuard> {
        fs::create_dir_all(&self.paths.state_dir).ok()?;
        let lock_path = self.paths.state_dir.join(LOCK_FILE_NAME);

        if let Ok(mut file) = fs::File::create_new(&lock_path) {
            writeln!(file, "{}", std::process::id()).ok()?;
            return Some(IndexLockGuard { path: lock_path });
        }

        if self.full_index_running() {
            return None;
        }
        let _ = fs::remove_file(&lock_path);
        if let Ok(mut file) = fs::File::create_new(&lock_path) {
            writeln!(file, "{}", std::process::id()).ok()?;
            return Some(IndexLockGuard { path: lock_path });
        }
        None
    }

    pub fn full_index_running(&self) -> bool {
        let lock_path = self.paths.state_dir.join(LOCK_FILE_NAME);
        let Ok(content) = fs::read_to_string(&lock_path) else {
            return false;
        };
        match content.trim().parse::<u32>() {
            // Lock has a parseable PID — defer to the OS.
            Ok(pid) => process_running(pid),
            // Lock exists but has no PID yet. This is the small window between
            // `create_new` and `writeln!` in `acquire_index_lock`; the writer
            // is racing to fill it in. Anything older than this window is
            // corrupt and should not block recovery.
            Err(_) if content.trim().is_empty() => fs::metadata(&lock_path)
                .ok()
                .and_then(|m| m.modified().ok())
                .and_then(|t| std::time::SystemTime::now().duration_since(t).ok())
                .is_some_and(|age| age < Duration::from_secs(5)),
            // Lock has non-empty unparseable contents — almost certainly a
            // crash mid-write or unrelated debris. Treat as dead so the next
            // acquire can replace it instead of waiting hours.
            Err(_) => false,
        }
    }

    pub fn stop_index_lock_holder(&self) {
        let lock_path = self.paths.state_dir.join(LOCK_FILE_NAME);
        let Some(pid) = read_lock_pid(&lock_path) else {
            let _ = fs::remove_file(lock_path);
            return;
        };

        if pid == std::process::id() {
            return;
        }

        terminate_process(pid);
        wait_for_process_exit(pid, Duration::from_millis(LOCK_TERMINATION_GRACE_MS));
        if process_running(pid) {
            kill_process(pid);
        }
        let _ = fs::remove_file(lock_path);
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
                validate_manifest_compatibility(manifest, expected_model, expected_dimensions)
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

    pub async fn chunk_stats(&self) -> Result<StoreStats> {
        let rows = self.read_chunks().await?;
        Ok(stats_from_rows(&rows))
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
    ) -> Result<(HashSet<String>, Vec<StoredChunk>)> {
        self.incremental_file_state_with_progress(current_files, None)
            .await
    }

    pub async fn incremental_file_state_with_progress(
        &self,
        current_files: &[(String, [u8; 16])],
        mut progress: Option<&mut dyn IndexProgress>,
    ) -> Result<(HashSet<String>, Vec<StoredChunk>)> {
        let stored_rows = self.read_chunks().await?;
        let stored_file_hashes = self.stored_file_hashes(&stored_rows)?;
        let stored_path_set: HashSet<&str> =
            stored_file_hashes.keys().map(String::as_str).collect();

        let mut changed_paths: HashSet<String> = HashSet::new();
        for (path, hash) in current_files {
            match stored_file_hashes.get(path.as_str()) {
                Some(stored_hash) if stored_hash == hash => {
                    if let Some(progress) = progress.as_deref_mut() {
                        progress
                            .file(&RelativePath::new(path.as_str()), IndexFileStatus::Verified)?;
                    }
                }
                _ => {
                    changed_paths.insert(path.clone());
                    if !stored_path_set.contains(path.as_str())
                        && let Some(progress) = progress.as_deref_mut()
                    {
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
        self.sync_manifest(config, &stats, file_hashes)?;
        Ok(stats)
    }

    pub async fn replace_chunks(
        &self,
        chunks: &[EmbeddedChunk],
        config: &Config,
    ) -> Result<StoreStats> {
        let dimension = Dimension(config.embedding.dimensions);
        let rows = stored_chunks_from_embedded(chunks, dimension)?;
        let stats = stats_from_rows(&rows);
        let file_hashes = file_hashes_from_rows(&rows);
        self.persist_rows(rows, dimension).await?;
        self.sync_manifest(config, &stats, file_hashes)?;
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

    pub fn note_file_hash(
        &self,
        path: &RelativePath,
        hash: [u8; 16],
        config: &Config,
    ) -> Result<()> {
        let mut manifest = self
            .read_manifest()?
            .unwrap_or_else(|| Manifest::new(&config.embedding.model, config.embedding.dimensions));
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

        let manifest = Manifest::new(&config.embedding.model, config.embedding.dimensions);
        self.write_manifest(&manifest)
    }

    async fn persist_rows(&self, rows: Vec<StoredChunk>, dimension: Dimension) -> Result<()> {
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
        lancedb::connect(&self.paths.index_dir.to_string_lossy())
            .execute()
            .await
            .map_err(ClaudixError::from)
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

    fn sync_manifest(
        &self,
        config: &Config,
        stats: &StoreStats,
        file_hashes: BTreeMap<String, [u8; 16]>,
    ) -> Result<()> {
        self.sync_manifest_with_timestamp(config, stats, file_hashes, true)
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
            .unwrap_or_else(|| Manifest::new(&config.embedding.model, config.embedding.dimensions));
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
}

fn read_lock_pid(lock_path: &Path) -> Option<u32> {
    fs::read_to_string(lock_path).ok()?.trim().parse().ok()
}

fn wait_for_process_exit(pid: u32, timeout: Duration) {
    let deadline = std::time::Instant::now() + timeout;
    while std::time::Instant::now() < deadline {
        if !process_running(pid) {
            return;
        }
        thread::sleep(Duration::from_millis(LOCK_TERMINATION_POLL_MS));
    }
}

#[cfg(unix)]
pub(crate) fn process_running(pid: u32) -> bool {
    Command::new("kill")
        .args(["-0", &pid.to_string()])
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

#[cfg(unix)]
fn terminate_process(pid: u32) {
    let _ = Command::new("kill")
        .args(["-TERM", &pid.to_string()])
        .stderr(Stdio::null())
        .status();
}

#[cfg(unix)]
fn kill_process(pid: u32) {
    let _ = Command::new("kill")
        .args(["-KILL", &pid.to_string()])
        .stderr(Stdio::null())
        .status();
}

#[cfg(windows)]
pub(crate) fn process_running(pid: u32) -> bool {
    Command::new("tasklist")
        .args(["/FI", &format!("PID eq {pid}"), "/NH"])
        .output()
        .is_ok_and(|output| {
            output.status.success()
                && String::from_utf8_lossy(&output.stdout).contains(&pid.to_string())
        })
}

#[cfg(windows)]
fn terminate_process(pid: u32) {
    let _ = Command::new("taskkill")
        .args(["/PID", &pid.to_string()])
        .status();
}

#[cfg(windows)]
fn kill_process(pid: u32) {
    let _ = Command::new("taskkill")
        .args(["/PID", &pid.to_string(), "/F"])
        .status();
}

impl StoredChunk {
    fn from_embedded_chunk(chunk: &EmbeddedChunk, dimension: Dimension) -> Result<Self> {
        validate_vector(&chunk.vector, dimension)?;

        Ok(Self {
            chunk_id: chunk.chunk.id.0,
            file_path: chunk.chunk.file_path.as_str().to_owned(),
            language: chunk.chunk.language.to_string(),
            kind: chunk.chunk.kind.to_string(),
            name: chunk.chunk.name.clone(),
            line_start: chunk.chunk.line_range.start,
            line_end: chunk.chunk.line_range.end,
            byte_start: chunk.chunk.byte_range.start,
            byte_end: chunk.chunk.byte_range.end,
            file_hash: chunk.chunk.file_hash.0,
            content: chunk.chunk.content.clone(),
            vector: chunk.vector.clone(),
        })
    }
}

fn validate_manifest_compatibility(
    manifest: Manifest,
    expected_model: &str,
    expected_dimensions: u16,
) -> Result<Manifest> {
    if manifest.schema_version != SCHEMA_VERSION {
        return Err(ClaudixError::SchemaMismatch {
            store: manifest.schema_version,
            binary: SCHEMA_VERSION,
            recovery: RecoveryHint(
                "Reindex the project to rebuild the store with the current schema version",
            ),
        });
    }

    if manifest.embedding_model != expected_model {
        return Err(ClaudixError::EmbeddingModelMismatch {
            store_model: manifest.embedding_model,
            active_model: expected_model.to_owned(),
            recovery: RecoveryHint(
                "Reindex the project after changing the configured embedding model",
            ),
        });
    }

    if manifest.dimensions != expected_dimensions {
        return Err(ClaudixError::DimensionMismatch {
            store_dim: manifest.dimensions,
            model_dim: expected_dimensions,
            recovery: RecoveryHint(
                "Reindex the project after changing the configured embedding dimensions",
            ),
        });
    }

    Ok(manifest)
}

fn resolve_project_path(project_root: &Path, relative_path: &Path) -> Result<PathBuf> {
    reject_path_escape(relative_path)?;

    let resolved = project_root.join(relative_path);
    if resolved.starts_with(project_root) {
        return Ok(resolved);
    }

    Err(ClaudixError::PathTraversal {
        path: resolved,
        recovery: RecoveryHint("Only use store paths inside $CLAUDE_PROJECT_DIR"),
    })
}

fn reject_path_escape(path: &Path) -> Result<()> {
    if path.is_absolute() {
        return Err(ClaudixError::PathTraversal {
            path: path.to_path_buf(),
            recovery: RecoveryHint("Only use store paths inside $CLAUDE_PROJECT_DIR"),
        });
    }

    for component in path.components() {
        if matches!(
            component,
            Component::ParentDir | Component::RootDir | Component::Prefix(_)
        ) {
            return Err(ClaudixError::PathTraversal {
                path: path.to_path_buf(),
                recovery: RecoveryHint("Only use store paths inside $CLAUDE_PROJECT_DIR"),
            });
        }
    }

    Ok(())
}

fn validate_vector(vector: &[f32], dimension: Dimension) -> Result<()> {
    if vector.len() != usize::from(dimension.0) {
        return Err(ClaudixError::DimensionMismatch {
            store_dim: dimension.0,
            model_dim: u16::try_from(vector.len()).unwrap_or(u16::MAX),
            recovery: RecoveryHint(
                "Reindex the project after aligning embedding dimensions with the active model",
            ),
        });
    }

    if vector.iter().any(|value| !value.is_finite()) {
        return Err(ClaudixError::Store(
            "embedding vector contains non-finite values".to_owned(),
        ));
    }

    Ok(())
}

pub(crate) fn stored_chunks_from_embedded(
    chunks: &[EmbeddedChunk],
    dimension: Dimension,
) -> Result<Vec<StoredChunk>> {
    chunks
        .iter()
        .map(|chunk| StoredChunk::from_embedded_chunk(chunk, dimension))
        .collect()
}

fn chunk_schema(dimension: Dimension) -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new(FIELD_CHUNK_ID, DataType::UInt64, false),
        Field::new(FIELD_FILE_PATH, DataType::Utf8, false),
        Field::new(FIELD_LANGUAGE, DataType::Utf8, false),
        Field::new(FIELD_KIND, DataType::Utf8, false),
        Field::new(FIELD_NAME, DataType::Utf8, true),
        Field::new(FIELD_LINE_START, DataType::UInt32, false),
        Field::new(FIELD_LINE_END, DataType::UInt32, false),
        Field::new(FIELD_BYTE_START, DataType::UInt32, false),
        Field::new(FIELD_BYTE_END, DataType::UInt32, false),
        Field::new(FIELD_FILE_HASH, DataType::FixedSizeBinary(16), false),
        Field::new(FIELD_CONTENT, DataType::Utf8, false),
        Field::new(
            FIELD_VECTOR,
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                i32::from(dimension.0),
            ),
            true,
        ),
    ]))
}

fn record_batch_from_rows(rows: &[StoredChunk], dimension: Dimension) -> Result<RecordBatch> {
    for row in rows {
        validate_vector(&row.vector, dimension)?;
    }

    record_batch_from_rows_unchecked(rows, dimension)
}

fn record_batch_from_rows_unchecked(
    rows: &[StoredChunk],
    dimension: Dimension,
) -> Result<RecordBatch> {
    let names: Vec<Option<String>> = rows.iter().map(|row| row.name.clone()).collect();
    let hash_refs: Vec<&[u8; 16]> = rows.iter().map(|row| &row.file_hash).collect();
    let vectors = rows
        .iter()
        .map(|row| Some(row.vector.iter().copied().map(Some).collect::<Vec<_>>()));

    RecordBatch::try_new(
        chunk_schema(dimension),
        vec![
            Arc::new(UInt64Array::from(
                rows.iter().map(|row| row.chunk_id).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(StringArray::from(
                rows.iter()
                    .map(|row| row.file_path.clone())
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(StringArray::from(
                rows.iter()
                    .map(|row| row.language.clone())
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(StringArray::from(
                rows.iter().map(|row| row.kind.clone()).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(StringArray::from(names)) as ArrayRef,
            Arc::new(UInt32Array::from(
                rows.iter().map(|row| row.line_start).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(UInt32Array::from(
                rows.iter().map(|row| row.line_end).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(UInt32Array::from(
                rows.iter().map(|row| row.byte_start).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(UInt32Array::from(
                rows.iter().map(|row| row.byte_end).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(
                FixedSizeBinaryArray::try_from_iter(hash_refs.into_iter())
                    .map_err(|error| ClaudixError::Store(error.to_string()))?,
            ) as ArrayRef,
            Arc::new(StringArray::from(
                rows.iter()
                    .map(|row| row.content.clone())
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(
                FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
                    vectors,
                    i32::from(dimension.0),
                ),
            ) as ArrayRef,
        ],
    )
    .map_err(|error| ClaudixError::Store(error.to_string()))
}

async fn read_all_rows(table: &Table) -> Result<Vec<StoredChunk>> {
    let batches = table
        .query()
        .limit(i64::MAX as usize)
        .execute()
        .await?
        .try_collect::<Vec<_>>()
        .await?;
    batches_to_rows(batches)
}

fn batches_to_rows(batches: Vec<RecordBatch>) -> Result<Vec<StoredChunk>> {
    let mut rows = Vec::new();

    for batch in batches {
        let dimension = vector_dimension(&batch)?;
        for row_index in 0..batch.num_rows() {
            let vector = read_vector(&batch, row_index)?;
            validate_vector(&vector, dimension)?;
            rows.push(StoredChunk {
                chunk_id: read_u64(&batch, FIELD_CHUNK_ID, row_index)?,
                file_path: read_string(&batch, FIELD_FILE_PATH, row_index)?,
                language: read_string(&batch, FIELD_LANGUAGE, row_index)?,
                kind: read_string(&batch, FIELD_KIND, row_index)?,
                name: read_optional_string(&batch, FIELD_NAME, row_index)?,
                line_start: read_u32(&batch, FIELD_LINE_START, row_index)?,
                line_end: read_u32(&batch, FIELD_LINE_END, row_index)?,
                byte_start: read_u32(&batch, FIELD_BYTE_START, row_index)?,
                byte_end: read_u32(&batch, FIELD_BYTE_END, row_index)?,
                file_hash: read_file_hash(&batch, row_index)?,
                content: read_string(&batch, FIELD_CONTENT, row_index)?,
                vector,
            });
        }
    }

    Ok(rows)
}

fn read_string(batch: &RecordBatch, column: &str, row: usize) -> Result<String> {
    let array = batch
        .column_by_name(column)
        .ok_or_else(|| ClaudixError::Store(format!("missing column {column}")))?;
    let array = array
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| ClaudixError::Store(format!("column {column} was not utf8")))?;
    Ok(array.value(row).to_owned())
}

fn read_optional_string(batch: &RecordBatch, column: &str, row: usize) -> Result<Option<String>> {
    let array = batch
        .column_by_name(column)
        .ok_or_else(|| ClaudixError::Store(format!("missing column {column}")))?;
    let array = array
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| ClaudixError::Store(format!("column {column} was not utf8")))?;

    if array.is_null(row) {
        return Ok(None);
    }

    Ok(Some(array.value(row).to_owned()))
}

fn read_u32(batch: &RecordBatch, column: &str, row: usize) -> Result<u32> {
    let array = batch
        .column_by_name(column)
        .ok_or_else(|| ClaudixError::Store(format!("missing column {column}")))?;
    let array = array
        .as_any()
        .downcast_ref::<UInt32Array>()
        .ok_or_else(|| ClaudixError::Store(format!("column {column} was not u32")))?;
    Ok(array.value(row))
}

fn read_u64(batch: &RecordBatch, column: &str, row: usize) -> Result<u64> {
    let array = batch
        .column_by_name(column)
        .ok_or_else(|| ClaudixError::Store(format!("missing column {column}")))?;
    let array = array
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or_else(|| ClaudixError::Store(format!("column {column} was not u64")))?;
    Ok(array.value(row))
}

fn read_file_hash(batch: &RecordBatch, row: usize) -> Result<[u8; 16]> {
    let array = batch
        .column_by_name(FIELD_FILE_HASH)
        .ok_or_else(|| ClaudixError::Store(format!("missing column {FIELD_FILE_HASH}")))?;
    let array = array
        .as_any()
        .downcast_ref::<FixedSizeBinaryArray>()
        .ok_or_else(|| {
            ClaudixError::Store("file_hash column was not fixed-size binary".to_owned())
        })?;

    <[u8; 16]>::try_from(array.value(row))
        .map_err(|_| ClaudixError::Store("file_hash value was not 16 bytes".to_owned()))
}

fn vector_dimension(batch: &RecordBatch) -> Result<Dimension> {
    let array = batch
        .column_by_name(FIELD_VECTOR)
        .ok_or_else(|| ClaudixError::Store(format!("missing column {FIELD_VECTOR}")))?;
    let array = array
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .ok_or_else(|| ClaudixError::Store("vector column was not fixed-size list".to_owned()))?;
    u16::try_from(array.value_length())
        .map(Dimension)
        .map_err(|_| ClaudixError::Store("vector dimension overflowed u16".to_owned()))
}

fn read_vector(batch: &RecordBatch, row: usize) -> Result<Vec<f32>> {
    let array = batch
        .column_by_name(FIELD_VECTOR)
        .ok_or_else(|| ClaudixError::Store(format!("missing column {FIELD_VECTOR}")))?;
    let array = array
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .ok_or_else(|| ClaudixError::Store("vector column was not fixed-size list".to_owned()))?;
    if array.is_null(row) {
        return Err(ClaudixError::Store("vector value was null".to_owned()));
    }

    let values = array.value(row);
    let values = values
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| ClaudixError::Store("vector values were not float32".to_owned()))?;
    if values.null_count() > 0 {
        return Err(ClaudixError::Store(
            "vector values contained nulls".to_owned(),
        ));
    }

    Ok((0..values.len()).map(|index| values.value(index)).collect())
}

fn distinct_file_paths(rows: &[StoredChunk]) -> BTreeSet<String> {
    rows.iter().map(|row| row.file_path.clone()).collect()
}

fn file_hashes_from_rows(rows: &[StoredChunk]) -> BTreeMap<String, [u8; 16]> {
    rows.iter()
        .map(|row| (row.file_path.clone(), row.file_hash))
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
    use super::*;
    use crate::types::{
        ByteRange, Chunk, ChunkId, ChunkKind, FileHash, Language, LineRange, RelativePath,
    };
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
        assert_eq!(read_lock_pid(&lock_path), Some(std::process::id()));
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

        let (changed_paths, unchanged_rows) = store.incremental_file_state(&current_files).await?;

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

        let mut manifest = Manifest::new(&config.embedding.model, config.embedding.dimensions);
        manifest.file_count = 1;
        manifest
            .file_hashes
            .insert("src/empty.rs".to_owned(), [7u8; 16]);
        store.write_manifest(&manifest)?;

        let current_files = vec![("src/empty.rs".to_owned(), [7u8; 16])];
        let (changed_paths, unchanged_rows) = store.incremental_file_state(&current_files).await?;

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

        let (changed_paths, unchanged_rows) = store.incremental_file_state(&current_files).await?;

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
        let mut manifest = Manifest::new(&config.embedding.model, config.embedding.dimensions);
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

        let mut manifest = Manifest::new(&config.embedding.model, config.embedding.dimensions);
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
}
