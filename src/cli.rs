use std::collections::{HashSet, VecDeque};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use notify::{EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use serde::Serialize;
use tokio::fs;
use tokio::sync::mpsc;

use crate::config::{self, validate_project_relative_path};
use crate::enumeration::WatchFilter;
use crate::error::{ClaudixError, RecoveryHint, Result};
use crate::hooks::HookEvent;
use crate::search::SearchQuery;
use crate::store::{IndexLockGuard, Store};
use crate::types::{Language, RelativePath};
use crate::{Claudix, IndexFileStatus, IndexProgress};

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SearchHit {
    pub file_path: String,
    pub language: String,
    pub kind: String,
    pub name: Option<String>,
    pub line_start: u32,
    pub line_end: u32,
    pub score: f32,
    pub stale: bool,
    pub snippet: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SearchOutput {
    pub hits: Vec<SearchHit>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct IndexOutput {
    pub file_count: usize,
    pub chunk_count: usize,
}

pub struct StderrIndexProgress;

impl IndexProgress for StderrIndexProgress {
    fn file(&mut self, path: &RelativePath, status: IndexFileStatus) -> Result<()> {
        let mut stderr = io::stderr().lock();
        match status {
            IndexFileStatus::Indexed => writeln!(stderr, "indexed {}", path.as_str())?,
            IndexFileStatus::Verified => writeln!(stderr, "verified {}", path.as_str())?,
            IndexFileStatus::Skipped(reason) => {
                writeln!(stderr, "skipped {}: {reason}", path.as_str())?
            }
        }
        stderr.flush()?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ClearOutput {
    pub cleared: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct StatusOutput {
    pub chunk_count: usize,
    pub file_count: usize,
    pub model: Option<String>,
    pub dimensions: Option<u16>,
    pub last_full_index_at: Option<String>,
    pub last_incremental_at: Option<String>,
    /// True when the index is missing or older than `reindex_after_hours`.
    pub stale: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DoctorOutput {
    pub project_root: String,
    pub index_present: bool,
    pub chunk_count: usize,
    pub file_count: usize,
    pub model: Option<String>,
    pub dimensions: Option<u16>,
    pub embedding_provider: String,
    pub embedding_healthy: bool,
    /// True when the stored index model differs from the active config model.
    /// Distinct from `embedding_healthy = false` caused by the server being unreachable.
    pub embedding_model_mismatch: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct InstallOutput {
    pub plugin_root: String,
    pub binary_path: String,
    pub config_path: String,
    pub wrote_config: bool,
    pub embedding_healthy: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SetupState {
    Ready,
    Missing(Vec<&'static str>),
}

pub async fn run_search(
    project_root: impl AsRef<Path>,
    query: String,
    top_k: Option<usize>,
    language_filter: Option<Vec<String>>,
    path_prefix: Option<String>,
) -> Result<SearchOutput> {
    validate_search_query(&query)?;
    let project_root = canonical_project_root(project_root.as_ref())?;
    let config = config::load(&project_root)?;
    let top_k = top_k.unwrap_or(config.search.top_k);
    validate_search_top_k(top_k)?;
    let claudix = Claudix::new(project_root, Arc::new(config)).await?;

    run_search_with_claudix(&claudix, query, top_k, language_filter, path_prefix).await
}

pub async fn run_index(project_root: impl AsRef<Path>) -> Result<IndexOutput> {
    let session = IndexSession::new(project_root).await?;
    let stats = session.claudix.index_full().await?;

    Ok(IndexOutput {
        file_count: stats.file_count,
        chunk_count: stats.chunk_count,
    })
}

pub async fn run_index_with_progress(project_root: impl AsRef<Path>) -> Result<IndexOutput> {
    let session = IndexSession::new(project_root).await?;
    let mut progress = StderrIndexProgress;
    let stats = session
        .claudix
        .index_full_with_progress(&mut progress)
        .await?;

    Ok(IndexOutput {
        file_count: stats.file_count,
        chunk_count: stats.chunk_count,
    })
}

struct IndexSession {
    claudix: Claudix,
    _lock: IndexLockGuard,
}

impl IndexSession {
    async fn new(project_root: impl AsRef<Path>) -> Result<Self> {
        let project_root = canonical_project_root(project_root.as_ref())?;
        require_git_repo(&project_root)?;
        let config = config::load(&project_root)?;
        let store = Store::new(&project_root, &config)?;
        let lock = store
            .acquire_index_lock()
            .ok_or_else(|| crate::error::ClaudixError::Store("index already running".to_owned()))?;
        let claudix = match Claudix::new(project_root.clone(), Arc::new(config.clone())).await {
            Ok(claudix) => claudix,
            Err(error) if requires_clean_reindex(&error) => {
                store.clear_chunks(&config).await?;
                Claudix::new(project_root, Arc::new(config)).await?
            }
            Err(error) => return Err(error),
        };

        Ok(Self {
            claudix,
            _lock: lock,
        })
    }
}

pub async fn run_status(project_root: impl AsRef<Path>) -> Result<StatusOutput> {
    let project_root = canonical_project_root(project_root.as_ref())?;
    let config = config::load(&project_root)?;
    let store = Store::new(&project_root, &config)?;
    status_from_store(&store, &config).await
}

const WATCH_MARKER_FILE_NAME: &str = "watch.pid";
const WATCH_HEARTBEAT_SECS: u64 = 30;

struct WatchMarkerGuard {
    path: PathBuf,
}

impl WatchMarkerGuard {
    /// Claim the watcher marker for this process.
    ///
    /// Coordinates with `spawn_background_watch`, which `create_new`s the marker
    /// and pre-writes the spawned child's PID — that hand-off case finds our own
    /// PID already stored and adopts the file. A stale marker (PID dead) is
    /// reclaimed; a live foreign PID returns an error so the duplicate watcher
    /// exits instead of clobbering the original.
    fn install(path: PathBuf) -> Result<Self> {
        use std::fs::OpenOptions;
        use std::io::Write;

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let my_pid = std::process::id();
        for _ in 0..2 {
            if let Ok(mut file) = OpenOptions::new().write(true).create_new(true).open(&path) {
                let _ = writeln!(file, "{my_pid}");
                return Ok(Self { path });
            }

            let existing = std::fs::read_to_string(&path)
                .ok()
                .and_then(|content| content.trim().parse::<u32>().ok());
            match existing {
                Some(pid) if pid == my_pid => {
                    let _ = std::fs::write(&path, format!("{my_pid}\n"));
                    return Ok(Self { path });
                }
                Some(pid) if !crate::store::process_running(pid) => {
                    let _ = std::fs::remove_file(&path);
                }
                Some(_) => {
                    return Err(ClaudixError::Store(
                        "another claudix watch process is already running".to_owned(),
                    ));
                }
                None => {
                    let _ = std::fs::remove_file(&path);
                }
            }
        }
        Err(ClaudixError::Store(
            "watcher marker claim failed".to_owned(),
        ))
    }

    fn heartbeat(&self) {
        let _ = std::fs::write(&self.path, std::process::id().to_string());
    }
}

impl Drop for WatchMarkerGuard {
    fn drop(&mut self) {
        // Only remove if we still hold the marker. Another process may have
        // reclaimed it after our heartbeat task stalled (e.g. SIGSTOP).
        let existing = std::fs::read_to_string(&self.path)
            .ok()
            .and_then(|content| content.trim().parse::<u32>().ok());
        if existing == Some(std::process::id()) {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

pub async fn run_watch(project_root: impl AsRef<Path>) -> Result<()> {
    let project_root = canonical_project_root(project_root.as_ref())?;
    let config = config::load(&project_root)?;
    if !config.watch {
        return Ok(());
    }

    let store = Store::new(&project_root, &config)?;
    store.ensure_layout()?;
    let marker = Arc::new(WatchMarkerGuard::install(
        store.state_dir_path().join(WATCH_MARKER_FILE_NAME),
    )?);

    // Cold ONNX loads can exceed the marker stale window; refresh the marker
    // from a side task while the watcher itself is still booting so concurrent
    // SessionStarts do not misclassify us as dead and spawn a duplicate.
    let early_heartbeat = {
        let marker = Arc::clone(&marker);
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(Duration::from_secs(WATCH_HEARTBEAT_SECS));
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            tick.tick().await; // consume the immediate first tick
            loop {
                tick.tick().await;
                marker.heartbeat();
            }
        })
    };

    let (event_tx, mut event_rx) = mpsc::unbounded_channel();
    let mut watcher = RecommendedWatcher::new(
        move |event| {
            let _ = event_tx.send(event);
        },
        notify::Config::default(),
    )
    .map_err(|error| ClaudixError::Store(format!("file watcher failed: {error}")))?;
    watcher
        .watch(&project_root, RecursiveMode::Recursive)
        .map_err(|error| ClaudixError::Store(format!("file watcher failed: {error}")))?;

    let filter = WatchFilter::load(&project_root)?;
    let claudix = Claudix::new(project_root.clone(), Arc::new(config)).await?;

    early_heartbeat.abort();
    let _ = early_heartbeat.await;

    let mut pending = VecDeque::new();
    let mut debounce_deadline: Option<tokio::time::Instant> = None;
    let mut heartbeat = tokio::time::interval(Duration::from_secs(WATCH_HEARTBEAT_SECS));
    heartbeat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    heartbeat.tick().await; // first tick fires immediately; consume it before the loop
    loop {
        tokio::select! {
            event = event_rx.recv() => {
                let Some(event) = event else {
                    return Ok(());
                };
                queue_reindex_paths(&project_root, &filter, event, &mut pending);
                // Anchor the debounce window when the first event lands; further
                // events do NOT extend it so a continuous file-event stream
                // still gets drained on schedule instead of starving.
                if debounce_deadline.is_none() && !pending.is_empty() {
                    debounce_deadline =
                        Some(tokio::time::Instant::now() + Duration::from_millis(250));
                }
            }
            _ = async {
                match debounce_deadline {
                    Some(deadline) => tokio::time::sleep_until(deadline).await,
                    None => std::future::pending::<()>().await,
                }
            } => {
                debounce_deadline = None;
                let paths = drain_unique_paths(&mut pending);
                for path in paths {
                    // Serialize against concurrent reindex-file CLI/MCP calls and
                    // any duplicate watcher that slipped through the marker claim.
                    let _reindex_lock = match store.acquire_reindex_lock() {
                        Ok(lock) => lock,
                        Err(error) => {
                            tracing::warn!(
                                "claudix watch skipped reindex of {}: {error}",
                                path.display()
                            );
                            continue;
                        }
                    };
                    if let Err(error) = claudix.reindex_file(&path).await {
                        tracing::warn!("claudix watch failed to reindex {}: {error}", path.display());
                    }
                }
            }
            _ = heartbeat.tick() => {
                marker.heartbeat();
            }
        }
    }
}

pub async fn run_reindex_file(
    project_root: impl AsRef<Path>,
    path: impl AsRef<Path>,
) -> Result<IndexOutput> {
    let project_root = canonical_project_root(project_root.as_ref())?;
    let config = config::load(&project_root)?;
    let store = Store::new(&project_root, &config)?;

    if store.full_index_running() {
        let status = status_from_store(&store, &config).await?;
        return Ok(IndexOutput {
            file_count: status.file_count,
            chunk_count: status.chunk_count,
        });
    }

    let _reindex_lock = store.acquire_reindex_lock()?;
    let claudix = Claudix::new(project_root, Arc::new(config)).await?;
    let stats = claudix.reindex_file(path.as_ref()).await?;

    Ok(IndexOutput {
        file_count: stats.file_count,
        chunk_count: stats.chunk_count,
    })
}

pub async fn run_doctor(project_root: impl AsRef<Path>) -> Result<DoctorOutput> {
    let project_root = canonical_project_root(project_root.as_ref())?;
    let config = config::load(&project_root)?;
    let store = Store::new(&project_root, &config)?;
    let status = status_from_store(&store, &config).await?;

    let claudix = Claudix::new(project_root.clone(), Arc::new(config.clone())).await;
    let (embedding_healthy, embedding_model_mismatch) = match claudix {
        Ok(claudix) => (claudix.embedder_health_check().await.is_ok(), false),
        Err(ClaudixError::EmbeddingModelMismatch { .. }) => (false, true),
        Err(_) => (false, false),
    };

    Ok(DoctorOutput {
        project_root: project_root.display().to_string(),
        index_present: status.chunk_count > 0 || status.model.is_some(),
        chunk_count: status.chunk_count,
        file_count: status.file_count,
        model: status.model,
        dimensions: status.dimensions,
        embedding_provider: match config.embedding.provider {
            config::EmbeddingProvider::Bundled => "bundled".to_owned(),
            config::EmbeddingProvider::Http => "http".to_owned(),
        },
        embedding_healthy,
        embedding_model_mismatch,
    })
}

pub async fn run_clear_index(project_root: impl AsRef<Path>) -> Result<ClearOutput> {
    let project_root = canonical_project_root(project_root.as_ref())?;
    let config = config::load(&project_root)?;
    let store = Store::new(&project_root, &config)?;
    store.clear_chunks(&config).await?;

    Ok(ClearOutput { cleared: true })
}

fn requires_clean_reindex(error: &ClaudixError) -> bool {
    matches!(
        error,
        ClaudixError::SchemaMismatch { .. }
            | ClaudixError::EmbeddingModelMismatch { .. }
            | ClaudixError::DimensionMismatch { .. }
    )
}

fn queue_reindex_paths(
    project_root: &Path,
    filter: &WatchFilter,
    event: notify::Result<notify::Event>,
    pending: &mut VecDeque<PathBuf>,
) {
    let Ok(event) = event else {
        return;
    };
    if !is_reindex_event(&event.kind) {
        return;
    }

    pending.extend(event.paths.into_iter().filter_map(|path| {
        let relative = path.strip_prefix(project_root).ok()?;
        if relative.components().next().is_none() {
            return None;
        }
        if !filter.is_watchable(relative) {
            return None;
        }
        Some(relative.to_path_buf())
    }));
}

fn is_reindex_event(kind: &EventKind) -> bool {
    matches!(
        kind,
        EventKind::Create(_) | EventKind::Modify(_) | EventKind::Remove(_)
    )
}

fn drain_unique_paths(pending: &mut VecDeque<PathBuf>) -> Vec<PathBuf> {
    let mut seen = HashSet::new();
    let mut paths = Vec::new();
    while let Some(path) = pending.pop_front() {
        if seen.insert(path.clone()) {
            paths.push(path);
        }
    }
    paths
}

pub async fn run_install(project_root: impl AsRef<Path>) -> Result<InstallOutput> {
    let project_root = canonical_project_root(project_root.as_ref())?;
    let source_root = install_source_root(&project_root)?;
    let plugin_root = plugin_root_from_env(&project_root, std::env::var_os("CLAUDE_PLUGIN_ROOT"))?;
    let binary_path = plugin_root.join("bin").join(binary_name());
    let config_path = global_config_path()?;

    install_plugin_assets(&source_root, &plugin_root).await?;

    let wrote_config = ensure_global_config(&config_path).await?;
    let config = config::load(&project_root)?;
    let claudix = Claudix::new(project_root, Arc::new(config)).await?;
    let embedding_healthy = claudix.embedder_health_check().await.is_ok();

    Ok(InstallOutput {
        plugin_root: plugin_root.display().to_string(),
        binary_path: binary_path.display().to_string(),
        config_path: config_path.display().to_string(),
        wrote_config,
        embedding_healthy,
    })
}

pub async fn setup_state(project_root: impl AsRef<Path>) -> SetupState {
    let project_root = project_root.as_ref();
    let mut missing = Vec::new();

    if plugin_root_from_env(project_root, std::env::var_os("CLAUDE_PLUGIN_ROOT")).is_err() {
        missing.push("plugin files");
    }
    match global_config_path() {
        Ok(config_path) if config_path.try_exists().unwrap_or(false) => {}
        _ => missing.push("global config"),
    }
    if config::load(project_root).is_err() {
        missing.push("valid config");
    }

    if missing.is_empty() {
        SetupState::Ready
    } else {
        SetupState::Missing(missing)
    }
}

pub fn parse_hook_event(value: &str) -> Result<HookEvent> {
    match value {
        "SessionStart" => Ok(HookEvent::SessionStart),
        "PostToolUse" => Ok(HookEvent::PostToolUse),
        "PreToolUse" => Ok(HookEvent::PreToolUse),
        _ => Err(ClaudixError::ConfigInvalid {
            message: format!("unknown hook event: {value}"),
            recovery: RecoveryHint("Use one of: SessionStart, PostToolUse, PreToolUse"),
        }),
    }
}

async fn run_search_with_claudix(
    claudix: &Claudix,
    query: String,
    top_k: usize,
    language_filter: Option<Vec<String>>,
    path_prefix: Option<String>,
) -> Result<SearchOutput> {
    validate_search_top_k(top_k)?;
    let query = SearchQuery {
        query,
        top_k,
        language_filter: parse_language_filter(language_filter)?,
        path_prefix: parse_path_prefix(path_prefix)?,
    };
    let results = claudix.search(query).await?;

    Ok(SearchOutput {
        hits: results
            .into_iter()
            .map(|result| SearchHit {
                file_path: result.chunk.file_path.to_string(),
                language: result.chunk.language.to_string(),
                kind: result.chunk.kind.to_string(),
                name: result.chunk.name,
                line_start: result.chunk.line_range.start,
                line_end: result.chunk.line_range.end,
                score: result.score,
                stale: result.stale,
                snippet: result.chunk.content,
            })
            .collect(),
    })
}

async fn status_from_store(store: &Store, config: &crate::config::Config) -> Result<StatusOutput> {
    let manifest = store.read_manifest()?;
    let chunk_count = manifest
        .as_ref()
        .map(|m| m.chunk_count as usize)
        .unwrap_or(0);
    let file_count = manifest
        .as_ref()
        .map(|m| m.file_count as usize)
        .unwrap_or(0);

    let stale = manifest
        .as_ref()
        .map(|m| crate::hooks::index_is_stale(m, config))
        .unwrap_or(true);

    Ok(StatusOutput {
        chunk_count,
        file_count,
        model: manifest
            .as_ref()
            .map(|manifest| manifest.embedding_model.clone()),
        dimensions: manifest.as_ref().map(|manifest| manifest.dimensions),
        last_full_index_at: manifest
            .as_ref()
            .and_then(|manifest| manifest.last_full_index_at.clone()),
        last_incremental_at: manifest
            .as_ref()
            .and_then(|manifest| manifest.last_incremental_at.clone()),
        stale,
    })
}

async fn install_plugin_assets(project_root: &Path, plugin_root: &Path) -> Result<bool> {
    let mut changed = false;
    changed |= copy_plugin_asset(
        project_root,
        ".claude-plugin/plugin.json",
        plugin_root.join(".claude-plugin").join("plugin.json"),
    )
    .await?;
    changed |= copy_plugin_asset(
        project_root,
        "hooks/hooks.json",
        plugin_root.join("hooks").join("hooks.json"),
    )
    .await?;
    changed |= copy_plugin_asset(
        project_root,
        "bin/claudix",
        plugin_root.join("bin").join(binary_name()),
    )
    .await?;
    make_executable(&plugin_root.join("bin").join(binary_name())).await?;
    changed |=
        copy_plugin_directory(project_root, "commands", plugin_root.join("commands")).await?;
    changed |= copy_plugin_directory(project_root, "scripts", plugin_root.join("scripts")).await?;
    Ok(changed)
}

async fn copy_plugin_asset(
    project_root: &Path,
    source_relative: &str,
    destination: PathBuf,
) -> Result<bool> {
    let source = required_plugin_asset(project_root, source_relative).await?;

    if source == destination || files_match(&source, &destination).await? {
        return Ok(false);
    }

    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent).await?;
    }
    fs::copy(source, &destination).await?;
    Ok(true)
}

async fn copy_plugin_directory(
    project_root: &Path,
    source_relative: &str,
    destination: PathBuf,
) -> Result<bool> {
    let source = required_plugin_asset(project_root, source_relative).await?;
    if source == destination || directories_match(&source, &destination).await? {
        return Ok(false);
    }

    if fs::try_exists(&destination).await? {
        fs::remove_dir_all(&destination).await?;
    }
    fs::create_dir_all(&destination).await?;

    let mut entries = fs::read_dir(source).await?;
    while let Some(entry) = entries.next_entry().await? {
        let file_type = entry.file_type().await?;
        if file_type.is_file() {
            let destination_file = destination.join(entry.file_name());
            fs::copy(entry.path(), &destination_file).await?;
            if destination_file
                .extension()
                .is_some_and(|extension| extension == "sh")
            {
                make_executable(&destination_file).await?;
            }
        }
    }

    Ok(true)
}

async fn files_match(left: &Path, right: &Path) -> Result<bool> {
    if !fs::try_exists(right).await? {
        return Ok(false);
    }

    let left_metadata = fs::metadata(left).await?;
    let right_metadata = fs::metadata(right).await?;
    if left_metadata.len() != right_metadata.len() {
        return Ok(false);
    }

    Ok(fs::read(left).await? == fs::read(right).await?)
}

async fn directories_match(left: &Path, right: &Path) -> Result<bool> {
    if !fs::try_exists(right).await? {
        return Ok(false);
    }

    let mut left_entries = directory_file_names(left).await?;
    let mut right_entries = directory_file_names(right).await?;
    left_entries.sort();
    right_entries.sort();
    if left_entries != right_entries {
        return Ok(false);
    }

    for entry in left_entries {
        if !files_match(&left.join(&entry), &right.join(&entry)).await? {
            return Ok(false);
        }
    }

    Ok(true)
}

async fn directory_file_names(path: &Path) -> Result<Vec<std::ffi::OsString>> {
    let mut file_names = Vec::new();
    let mut entries = fs::read_dir(path).await?;
    while let Some(entry) = entries.next_entry().await? {
        if entry.file_type().await?.is_file() {
            file_names.push(entry.file_name());
        }
    }
    Ok(file_names)
}

async fn required_plugin_asset(project_root: &Path, source_relative: &str) -> Result<PathBuf> {
    let source = project_root.join(source_relative);
    if fs::try_exists(&source).await? {
        return Ok(source);
    }

    Err(ClaudixError::ConfigInvalid {
        message: format!("required plugin asset missing: {}", source.display()),
        recovery: RecoveryHint("Restore the plugin metadata files before running claudix install"),
    })
}

fn local_plugin_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("claudix-plugin")
}

async fn make_executable(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        let mut permissions = fs::metadata(path).await?.permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(path, permissions).await?;
    }

    Ok(())
}

async fn ensure_global_config(config_path: &Path) -> Result<bool> {
    if fs::try_exists(config_path).await? {
        return Ok(false);
    }

    if let Some(parent) = config_path.parent() {
        fs::create_dir_all(parent).await?;
    }

    fs::write(config_path, default_global_config()).await?;
    Ok(true)
}

fn default_global_config() -> &'static str {
    "\
# Global claudix configuration — uncomment and edit as needed.
# Project-level overrides go in .claude/claudix.toml (project wins).
# watch = false                  # opt-in file watcher for saved files

[embedding]
# provider = \"bundled\"           # bundled | http
# model = \"bge-small-en-v1.5\"    # only used by bundled provider
# dimensions = 384               # must match the model
# endpoint = \"http://localhost:11434\"  # for http provider (LM Studio / Ollama)

[indexing]
# reindex_after_hours = 24       # auto-reindex threshold on session start

[hooks]
# auto_index_on_session_start = true  # trigger background reindex when stale
# intercept_grep = true               # redirect conceptual Grep/rg to search_code
# auto_reembed_on_edit = true         # re-embed edited files in background

[search]
# top_k = 10                     # default result count for search_code
"
}

fn install_source_root(project_root: &Path) -> Result<PathBuf> {
    if is_claudix_plugin_root(project_root) {
        return Ok(project_root.to_path_buf());
    }

    match std::env::var_os("CLAUDE_PLUGIN_ROOT") {
        Some(path) => Ok(PathBuf::from(path)),
        None => Ok(PathBuf::from(env!("CARGO_MANIFEST_DIR"))),
    }
}

fn plugin_root_from_env(
    project_root: &Path,
    plugin_root_env: Option<std::ffi::OsString>,
) -> Result<PathBuf> {
    if let Some(path) = plugin_root_env {
        let plugin_root = PathBuf::from(path);
        if is_claudix_plugin_root(&plugin_root) {
            return Ok(plugin_root);
        }
    }

    if is_claudix_plugin_root(project_root) {
        return Ok(local_plugin_root());
    }

    Err(ClaudixError::ConfigInvalid {
        message: "CLAUDE_PLUGIN_ROOT is not set".into(),
        recovery: RecoveryHint(
            "Run claudix install from the plugin directory or plugin environment",
        ),
    })
}

fn is_claudix_plugin_root(path: &Path) -> bool {
    let manifest_path = path.join(".claude-plugin").join("plugin.json");
    let Ok(manifest) = std::fs::read_to_string(manifest_path) else {
        return false;
    };

    manifest.contains("\"name\": \"claudix\"")
}

fn global_config_path() -> Result<PathBuf> {
    dirs::home_dir()
        .map(|home| home.join(".claude").join("claudix.toml"))
        .ok_or_else(|| ClaudixError::ConfigInvalid {
            message: "home directory is not available".into(),
            recovery: RecoveryHint("Set HOME before running claudix install"),
        })
}

fn binary_name() -> &'static str {
    if cfg!(windows) {
        "claudix.exe"
    } else {
        "claudix"
    }
}

fn require_git_repo(project_root: &Path) -> Result<()> {
    if !is_git_repo(project_root) {
        return Err(ClaudixError::NotAGitRepository {
            path: project_root.to_path_buf(),
            recovery: RecoveryHint("Run claudix index from inside a git repository"),
        });
    }
    Ok(())
}

pub fn is_git_repo(path: &Path) -> bool {
    let mut current = path;
    loop {
        if current.join(".git").exists() {
            return true;
        }
        match current.parent() {
            Some(parent) => current = parent,
            None => return false,
        }
    }
}

fn canonical_project_root(project_root: &Path) -> Result<PathBuf> {
    project_root.canonicalize().map_err(ClaudixError::from)
}

fn validate_search_query(query: &str) -> Result<()> {
    if query.trim().is_empty() {
        return Err(ClaudixError::ConfigInvalid {
            message: "search query cannot be empty".into(),
            recovery: RecoveryHint("Pass a non-empty search query"),
        });
    }
    Ok(())
}

fn validate_search_top_k(top_k: usize) -> Result<()> {
    if top_k == 0 {
        return Err(ClaudixError::ConfigInvalid {
            message: "top_k must be > 0".into(),
            recovery: RecoveryHint("Pass a positive top_k value"),
        });
    }

    Ok(())
}

fn parse_language_filter(language_filter: Option<Vec<String>>) -> Result<Option<Vec<Language>>> {
    let Some(language_filter) = language_filter else {
        return Ok(None);
    };
    if language_filter.is_empty() {
        return Ok(None);
    }

    let mut parsed = Vec::with_capacity(language_filter.len());
    for language in language_filter {
        parsed.push(parse_language(&language)?);
    }

    Ok(Some(parsed))
}

fn parse_path_prefix(path_prefix: Option<String>) -> Result<Option<RelativePath>> {
    let Some(prefix) = path_prefix else {
        return Ok(None);
    };
    let trimmed = prefix.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }

    validate_project_relative_path(Path::new(trimmed), "search.path_prefix")?;
    Ok(Some(RelativePath::new(trimmed.to_owned())))
}

fn parse_language(value: &str) -> Result<Language> {
    let value = value.trim();
    match value.to_ascii_lowercase().as_str() {
        "rust" => Ok(Language::Rust),
        "python" => Ok(Language::Python),
        "javascript" | "js" => Ok(Language::JavaScript),
        "typescript" | "ts" => Ok(Language::TypeScript),
        "go" => Ok(Language::Go),
        "java" => Ok(Language::Java),
        "c" => Ok(Language::C),
        "cpp" | "c++" => Ok(Language::Cpp),
        "unknown" => Ok(Language::Unknown),
        _ => Err(ClaudixError::ConfigInvalid {
            message: format!("unsupported language filter: {value}"),
            recovery: RecoveryHint(
                "Use one of: rust, python, javascript, typescript, go, java, c, cpp, unknown",
            ),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::embedding::{Provider, StubProvider};
    use crate::store::{Manifest, Store};
    use crate::types::Dimension;
    use tempfile::tempdir;

    mod fixture {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/common/fixture.rs"
        ));
    }

    mod test_support {
        use crate as claudix;

        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/common/test_support.rs"
        ));
    }

    use fixture::TestFixture;
    use test_support::{index_fixture, stub_config};

    struct CliHarness {
        _fixture: TestFixture,
        claudix: Claudix,
        store: Store,
    }

    fn test_claudix(project_root: PathBuf, config: Config) -> Result<Claudix> {
        let store = Store::new(&project_root, &config)?;
        let config = Arc::new(config);
        let embedder: Arc<dyn Provider> = Arc::new(StubProvider::with_model_id(
            config.embedding.model.clone(),
            Dimension(config.embedding.dimensions),
        ));

        Ok(Claudix::from_parts(project_root, config, embedder, store))
    }

    async fn cli_harness() -> Result<CliHarness> {
        let fixture = TestFixture::new("small_rust")?;
        let config = stub_config();
        let claudix = test_claudix(fixture.root().to_path_buf(), config.clone())?;
        index_fixture(
            &claudix.store,
            claudix.embedder.as_ref(),
            claudix.project_root(),
            &config,
        )
        .await?;
        let store = Store::new(fixture.root(), &config)?;

        Ok(CliHarness {
            _fixture: fixture,
            claudix,
            store,
        })
    }

    #[cfg(unix)]
    #[test]
    fn watch_marker_install_refuses_live_foreign_pid() {
        let dir = tempdir().ok().unwrap_or_else(|| unreachable!());
        let path = dir.path().join("watch.pid");
        let mut child = std::process::Command::new("sleep")
            .arg("60")
            .spawn()
            .ok()
            .unwrap_or_else(|| unreachable!());
        let foreign_pid = child.id();
        assert!(std::fs::write(&path, foreign_pid.to_string()).is_ok());

        let result = WatchMarkerGuard::install(path.clone());
        let stored_after = std::fs::read_to_string(&path).ok();
        let _ = child.kill();
        let _ = child.wait();

        assert!(
            matches!(result, Err(ClaudixError::Store(_))),
            "expected error when a live foreign PID owns the marker"
        );
        assert_eq!(
            stored_after.as_deref().map(str::trim),
            Some(foreign_pid.to_string().as_str()),
            "foreign marker contents must be untouched"
        );
    }

    #[test]
    fn watch_marker_install_clears_malformed_marker() {
        let dir = tempdir().ok().unwrap_or_else(|| unreachable!());
        let path = dir.path().join("watch.pid");
        assert!(std::fs::write(&path, "not-a-pid").is_ok());

        let marker = WatchMarkerGuard::install(path.clone());
        assert!(marker.is_ok(), "malformed marker must be reclaimable");
    }

    #[test]
    fn watch_marker_install_takes_over_dead_pid() {
        let dir = tempdir().ok().unwrap_or_else(|| unreachable!());
        let path = dir.path().join("watch.pid");
        // PID 0 is never running on Linux; use a guaranteed-dead value via crate helper.
        let dead_pid = pick_dead_pid();
        assert!(std::fs::write(&path, dead_pid.to_string()).is_ok());

        let marker = WatchMarkerGuard::install(path.clone());
        assert!(marker.is_ok(), "must reclaim a stale marker");
        let stored = std::fs::read_to_string(&path).ok();
        assert_eq!(
            stored.as_deref().map(str::trim),
            Some(std::process::id().to_string().as_str())
        );
    }

    #[test]
    fn watch_marker_install_adopts_handoff_with_own_pid() {
        let dir = tempdir().ok().unwrap_or_else(|| unreachable!());
        let path = dir.path().join("watch.pid");
        assert!(std::fs::write(&path, std::process::id().to_string()).is_ok());

        let marker = WatchMarkerGuard::install(path.clone());
        assert!(marker.is_ok(), "must adopt parent's hand-off claim");
    }

    #[test]
    fn watch_marker_drop_leaves_foreign_pid_alone() {
        let dir = tempdir().ok().unwrap_or_else(|| unreachable!());
        let path = dir.path().join("watch.pid");
        let marker = WatchMarkerGuard::install(path.clone())
            .ok()
            .unwrap_or_else(|| unreachable!());
        // Simulate another process reclaiming the marker before our drop runs.
        let foreign_pid = if std::process::id() == 1 { 2 } else { 1 };
        assert!(std::fs::write(&path, foreign_pid.to_string()).is_ok());

        drop(marker);
        assert!(
            path.exists(),
            "drop must not remove a marker reclaimed by another process"
        );
    }

    fn pick_dead_pid() -> u32 {
        for candidate in [9_999_999u32, 8_888_888, 7_777_777] {
            if !crate::store::process_running(candidate) {
                return candidate;
            }
        }
        9_999_999
    }

    #[test]
    fn parse_hook_event_accepts_known_values() {
        let event = parse_hook_event("SessionStart");
        assert!(matches!(event, Ok(HookEvent::SessionStart)));

        let event = parse_hook_event("PostToolUse");
        assert!(matches!(event, Ok(HookEvent::PostToolUse)));

        let event = parse_hook_event("PreToolUse");
        assert!(matches!(event, Ok(HookEvent::PreToolUse)));
    }

    #[test]
    fn parse_hook_event_rejects_unknown_values() {
        let result = parse_hook_event("Unknown");
        assert!(matches!(result, Err(ClaudixError::ConfigInvalid { .. })));
    }

    #[test]
    fn validate_search_query_rejects_empty_and_whitespace() {
        for query in ["", "   ", "\t", "\n"] {
            let result = validate_search_query(query);
            assert!(matches!(result, Err(ClaudixError::ConfigInvalid { .. })));
        }
    }

    #[test]
    fn validate_search_top_k_rejects_zero() {
        let result = validate_search_top_k(0);

        assert!(matches!(result, Err(ClaudixError::ConfigInvalid { .. })));
    }

    #[test]
    fn parse_language_filter_accepts_aliases() {
        let parsed = parse_language_filter(Some(vec!["rs".to_owned(), "ts".to_owned()]));
        assert!(parsed.is_err());

        let parsed = parse_language_filter(Some(vec![" rust ".to_owned(), "ts".to_owned()]));
        assert!(matches!(
            parsed,
            Ok(Some(ref languages)) if languages == &vec![Language::Rust, Language::TypeScript]
        ));
    }

    #[test]
    fn parse_language_filter_treats_empty_list_as_no_filter() {
        let parsed = parse_language_filter(Some(Vec::new()));

        assert!(matches!(parsed, Ok(None)));
    }

    #[test]
    fn parse_path_prefix_treats_blank_values_as_no_filter() {
        assert!(matches!(parse_path_prefix(None), Ok(None)));
        assert!(matches!(
            parse_path_prefix(Some("   ".to_owned())),
            Ok(None)
        ));
        assert_eq!(
            parse_path_prefix(Some(" src/math ".to_owned()))
                .ok()
                .flatten()
                .as_ref()
                .map(RelativePath::as_str),
            Some("src/math")
        );
    }

    #[test]
    fn parse_path_prefix_rejects_paths_outside_project() {
        for path_prefix in ["../src", "/tmp/src"] {
            let parsed = parse_path_prefix(Some(path_prefix.to_owned()));
            assert!(matches!(parsed, Err(ClaudixError::ConfigInvalid { .. })));
        }
    }

    #[tokio::test]
    async fn run_search_returns_ranked_hits() {
        let harness = cli_harness().await;
        assert!(harness.is_ok());
        let harness = harness.ok().unwrap_or_else(|| unreachable!());

        let output =
            run_search_with_claudix(&harness.claudix, "add".to_owned(), 5, None, None).await;
        assert!(output.is_ok());
        let output = output.ok().unwrap_or_else(|| unreachable!());

        assert!(!output.hits.is_empty());
        assert_eq!(output.hits[0].name.as_deref(), Some("add"));
        assert_eq!(output.hits[0].file_path, "src/math.rs");
    }

    #[tokio::test]
    async fn run_search_applies_filters() {
        let harness = cli_harness().await;
        assert!(harness.is_ok());
        let harness = harness.ok().unwrap_or_else(|| unreachable!());

        let output = run_search_with_claudix(
            &harness.claudix,
            "add".to_owned(),
            5,
            Some(vec!["rust".to_owned()]),
            Some(RelativePath::new("src/math").to_string()),
        )
        .await;
        assert!(output.is_ok());
        let output = output.ok().unwrap_or_else(|| unreachable!());

        assert_eq!(output.hits.len(), 1);
        assert_eq!(output.hits[0].file_path, "src/math.rs");
    }

    #[test]
    fn queue_reindex_paths_ignores_internal_state_paths() {
        let tmp = tempfile::tempdir();
        assert!(tmp.is_ok());
        let tmp = tmp.ok().unwrap_or_else(|| unreachable!());
        let root = tmp.path();
        let event = notify::Event {
            kind: EventKind::Modify(notify::event::ModifyKind::Data(
                notify::event::DataChange::Content,
            )),
            paths: vec![
                root.join("src/lib.rs"),
                root.join(".claudix/manifest.json"),
                root.join(".git/HEAD"),
            ],
            attrs: notify::event::EventAttributes::new(),
        };
        let filter = WatchFilter::load(root);
        assert!(filter.is_ok());
        let filter = filter.ok().unwrap_or_else(|| unreachable!());
        let mut pending = VecDeque::new();

        queue_reindex_paths(root, &filter, Ok(event), &mut pending);

        assert_eq!(
            pending.into_iter().collect::<Vec<_>>(),
            vec![PathBuf::from("src/lib.rs")]
        );
    }

    #[test]
    fn queue_reindex_paths_respects_project_gitignore() {
        let tmp = tempfile::tempdir();
        assert!(tmp.is_ok());
        let tmp = tmp.ok().unwrap_or_else(|| unreachable!());
        let root = tmp.path();
        assert!(std::fs::write(root.join(".gitignore"), "target/\nnode_modules/\n").is_ok());

        let event = notify::Event {
            kind: EventKind::Modify(notify::event::ModifyKind::Data(
                notify::event::DataChange::Content,
            )),
            paths: vec![
                root.join("src/lib.rs"),
                root.join("target/debug/build/foo"),
                root.join("node_modules/pkg/index.js"),
            ],
            attrs: notify::event::EventAttributes::new(),
        };
        let filter = WatchFilter::load(root);
        assert!(filter.is_ok());
        let filter = filter.ok().unwrap_or_else(|| unreachable!());
        let mut pending = VecDeque::new();

        queue_reindex_paths(root, &filter, Ok(event), &mut pending);

        assert_eq!(
            pending.into_iter().collect::<Vec<_>>(),
            vec![PathBuf::from("src/lib.rs")]
        );
    }

    #[test]
    fn drain_unique_paths_deduplicates_in_order() {
        let mut pending = VecDeque::from(vec![
            PathBuf::from("src/lib.rs"),
            PathBuf::from("src/lib.rs"),
            PathBuf::from("src/main.rs"),
        ]);

        assert_eq!(
            drain_unique_paths(&mut pending),
            vec![PathBuf::from("src/lib.rs"), PathBuf::from("src/main.rs")]
        );
    }

    #[test]
    fn clean_reindex_required_for_manifest_compatibility_errors() {
        assert!(requires_clean_reindex(&ClaudixError::SchemaMismatch {
            store: 0,
            binary: 1,
            recovery: RecoveryHint("reindex"),
        }));
        assert!(requires_clean_reindex(
            &ClaudixError::EmbeddingModelMismatch {
                store_model: "old".to_owned(),
                active_model: "new".to_owned(),
                recovery: RecoveryHint("reindex"),
            }
        ));
        assert!(requires_clean_reindex(&ClaudixError::DimensionMismatch {
            store_dim: 384,
            model_dim: 768,
            recovery: RecoveryHint("reindex"),
        }));
        assert!(!requires_clean_reindex(&ClaudixError::Store(
            "index already running".to_owned()
        )));
    }

    #[tokio::test]
    async fn run_index_clears_model_mismatch_and_reindexes() {
        let fixture = TestFixture::new("small_rust");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());
        let config = stub_config();
        let claude_dir = fixture.root().join(".claude");
        assert!(std::fs::create_dir_all(&claude_dir).is_ok());
        let config_text = toml::to_string(&config);
        assert!(config_text.is_ok());
        assert!(
            std::fs::write(
                claude_dir.join("claudix.toml"),
                config_text.ok().unwrap_or_default(),
            )
            .is_ok()
        );

        let store = Store::new(fixture.root(), &config);
        assert!(store.is_ok());
        let store = store.ok().unwrap_or_else(|| unreachable!());
        let old_manifest = Manifest::new("old-model", config.embedding.dimensions);
        assert!(store.write_manifest(&old_manifest).is_ok());

        let output = run_index(fixture.root()).await;
        assert!(output.is_ok());
        let output = output.ok().unwrap_or_else(|| unreachable!());
        assert!(output.chunk_count > 0);

        let manifest = store.read_manifest();
        assert!(manifest.is_ok());
        let manifest = manifest.ok().unwrap_or_else(|| unreachable!());
        let manifest = manifest.unwrap_or_else(|| unreachable!());
        assert_eq!(manifest.embedding_model, config.embedding.model);
        assert_eq!(manifest.dimensions, config.embedding.dimensions);
        assert_eq!(manifest.chunk_count as usize, output.chunk_count);
    }

    #[tokio::test]
    async fn run_status_reports_manifest_and_counts() {
        let harness = cli_harness().await;
        assert!(harness.is_ok());
        let harness = harness.ok().unwrap_or_else(|| unreachable!());

        let config = stub_config();
        let status = status_from_store(&harness.store, &config).await;
        assert!(status.is_ok());
        let status = status.ok().unwrap_or_else(|| unreachable!());

        assert_eq!(status.chunk_count, 3);
        assert_eq!(status.file_count, 2);
        assert_eq!(status.model.as_deref(), Some("stub-v1"));
        assert_eq!(status.dimensions, Some(8));
        assert!(!status.stale, "freshly indexed should not be stale");
    }

    #[tokio::test]
    async fn run_doctor_reports_index_and_embedding_health() {
        let harness = cli_harness().await;
        assert!(harness.is_ok());
        let harness = harness.ok().unwrap_or_else(|| unreachable!());

        let config = stub_config();
        let claude_dir = harness.claudix.project_root().join(".claude");
        assert!(std::fs::create_dir_all(&claude_dir).is_ok());
        let config_text = toml::to_string(&config);
        assert!(config_text.is_ok());
        assert!(
            std::fs::write(
                claude_dir.join("claudix.toml"),
                config_text.ok().unwrap_or_default(),
            )
            .is_ok()
        );

        let output = run_doctor(harness.claudix.project_root()).await;
        assert!(output.is_ok());
        let output = output.ok().unwrap_or_else(|| unreachable!());

        assert!(output.index_present);
        assert_eq!(output.chunk_count, 3);
        assert_eq!(output.file_count, 2);
        assert_eq!(output.model.as_deref(), Some("stub-v1"));
        assert_eq!(output.embedding_provider, "bundled");
        assert!(output.embedding_healthy);
    }

    #[tokio::test]
    async fn ensure_global_config_writes_default_once() {
        let temp = tempdir();
        assert!(temp.is_ok());
        let temp = temp.ok().unwrap_or_else(|| unreachable!());
        let config_path = temp.path().join(".claude").join("claudix.toml");

        let wrote_config = ensure_global_config(&config_path).await;
        assert!(wrote_config.is_ok());
        assert!(wrote_config.ok().unwrap_or(false));

        let contents = fs::read_to_string(&config_path).await;
        assert!(contents.is_ok());
        assert!(contents.ok().unwrap_or_default().contains("[embedding]"));

        let wrote_config = ensure_global_config(&config_path).await;
        assert!(wrote_config.is_ok());
        assert!(!wrote_config.ok().unwrap_or(true));
    }

    #[tokio::test]
    async fn install_copies_plugin_assets_into_plugin_root() {
        let fixture = TestFixture::new("small_rust");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());
        let plugin_root = fixture.root().join("plugin-root");

        let result =
            install_plugin_assets(Path::new(env!("CARGO_MANIFEST_DIR")), &plugin_root).await;
        assert!(result.is_ok());
        assert!(result.ok().unwrap_or(false));

        let second_result =
            install_plugin_assets(Path::new(env!("CARGO_MANIFEST_DIR")), &plugin_root).await;
        assert!(second_result.is_ok());
        assert!(!second_result.ok().unwrap_or(true));

        let plugin_manifest =
            fs::read_to_string(plugin_root.join(".claude-plugin").join("plugin.json")).await;
        assert!(plugin_manifest.is_ok());
        let plugin_manifest = plugin_manifest.ok().unwrap_or_default();
        assert!(plugin_manifest.contains("\"name\": \"claudix\""));
        assert!(plugin_manifest.contains("\"mcpServers\""));
        assert!(plugin_manifest.contains("\"args\": [\"mcp\"]"));

        let hooks_manifest = fs::read_to_string(plugin_root.join("hooks").join("hooks.json")).await;
        assert!(hooks_manifest.is_ok());
        assert!(
            hooks_manifest
                .ok()
                .unwrap_or_default()
                .contains("scripts/session-start.sh")
        );

        let wrapper = fs::read_to_string(plugin_root.join("bin").join("claudix")).await;
        assert!(wrapper.is_ok());
        assert!(wrapper.ok().unwrap_or_default().contains("CARGO_BIN"));

        let search_command =
            fs::read_to_string(plugin_root.join("commands").join("search.md")).await;
        assert!(search_command.is_ok());
        let search_command = search_command.ok().unwrap_or_default();
        assert!(search_command.contains("!`claudix search"));
        assert!(!search_command.contains("CLAUDE_PLUGIN_ROOT"));

        let updater = fs::read_to_string(plugin_root.join("scripts").join("check-update.sh")).await;
        assert!(updater.is_ok());
        assert!(updater.ok().unwrap_or_default().contains("github.com"));
    }

    #[test]
    fn plugin_root_uses_claudix_environment_value_outside_local_checkout() {
        let fixture = TestFixture::new("small_rust");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());
        let env_root = fixture.root().join("env-plugin-root");
        assert!(std::fs::create_dir_all(env_root.join(".claude-plugin")).is_ok());
        assert!(
            std::fs::write(
                env_root.join(".claude-plugin").join("plugin.json"),
                "{\"name\": \"claudix\"}",
            )
            .is_ok()
        );
        let project_root = fixture.root().join("project");
        assert!(std::fs::create_dir_all(&project_root).is_ok());

        let result = plugin_root_from_env(&project_root, Some(env_root.clone().into_os_string()));
        assert!(result.is_ok());
        assert_eq!(result.ok().unwrap_or_default(), env_root);
    }

    #[test]
    fn plugin_root_ignores_foreign_environment_value() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let fixture = TestFixture::new("small_rust");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());

        let result = plugin_root_from_env(root, Some(fixture.root().as_os_str().to_os_string()));
        assert!(result.is_ok());
        assert_eq!(
            result.ok().unwrap_or_default(),
            root.join("target").join("claudix-plugin")
        );
    }

    #[test]
    fn plugin_root_falls_back_to_local_manifest() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));

        let result = plugin_root_from_env(root, None);
        assert!(result.is_ok());
        assert_eq!(
            result.ok().unwrap_or_default(),
            root.join("target").join("claudix-plugin")
        );
    }

    #[test]
    fn plugin_root_requires_environment_or_local_manifest() {
        let fixture = TestFixture::new("small_rust");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());

        let result = plugin_root_from_env(fixture.root(), None);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn setup_state_reports_missing_plugin_files() {
        let fixture = TestFixture::new("small_rust");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());

        let setup_state = setup_state(fixture.root()).await;
        assert!(
            matches!(setup_state, SetupState::Missing(parts) if parts.contains(&"plugin files"))
        );
    }

    #[test]
    fn default_global_config_includes_commented_defaults() {
        let config = default_global_config();
        assert!(config.contains("[embedding]"));
        assert!(config.contains("provider = \"bundled\""));
        assert!(config.contains("reindex_after_hours = 24"));
        assert!(config.contains("[hooks]"));
        assert!(config.contains("intercept_grep = true"));
        assert!(config.contains("auto_reembed_on_edit = true"));
        assert!(config.contains("[search]"));
        assert!(config.contains("top_k = 10"));
    }

    #[test]
    fn binary_name_matches_platform() {
        if cfg!(windows) {
            assert_eq!(binary_name(), "claudix.exe");
        } else {
            assert_eq!(binary_name(), "claudix");
        }
    }
}
