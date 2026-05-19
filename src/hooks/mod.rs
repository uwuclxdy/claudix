use std::ffi::OsStr;
use std::fs;
use std::path::Path;
use std::time::{Duration, SystemTime};

const WATCH_MARKER_FILE_NAME: &str = "watch.pid";
const WATCH_MARKER_STALE_SECS: u64 = 120;
const HEALTH_CACHE_FILE_NAME: &str = "embedder-health";
const HEALTH_CACHE_HEALTHY_TTL_SECS: u64 = 60;
const HEALTH_CACHE_UNHEALTHY_TTL_SECS: u64 = 30;
const PRE_TOOL_USE_TIMEOUT_MS: u64 = 1_500;
const PENDING_INDEX_READY_GRACE_SECS: u64 = 3;
const PENDING_INDEX_FAILURE_GRACE_SECS: u64 = 60;
const PENDING_INDEX_ACK_FILE_NAME: &str = "indexing-pending-acked";

use serde::Deserialize;
use serde_json::{Value, json};

use std::sync::Arc;

use crate::Claudix;
use crate::cli;
use crate::config::{self, Config};
use crate::error::{ClaudixError, Result};
use crate::search::SearchQuery;
use crate::store::{Manifest, Store};
use crate::util::{now_rfc3339, parse_rfc3339};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HookEvent {
    SessionStart,
    PostToolUse,
    PreToolUse,
}

pub async fn run(project_root: &Path, event: HookEvent, payload: &str) -> Result<Option<Value>> {
    let payload: HookPayload = if payload.trim().is_empty() {
        HookPayload {
            tool_name: None,
            tool_input: None,
        }
    } else {
        serde_json::from_str(payload)?
    };

    match event {
        HookEvent::SessionStart => handle_session_start(project_root, payload).await,
        HookEvent::PostToolUse => handle_post_tool_use(project_root, payload).await,
        HookEvent::PreToolUse => handle_pre_tool_use(project_root, payload).await,
    }
}

fn is_git_repo(path: &Path) -> bool {
    cli::is_git_repo(path)
}

fn spawn_background_index(project_root: &Path, config: &crate::config::Config) -> bool {
    let Ok(store) = Store::new(project_root, config) else {
        return false;
    };
    if store.full_index_running() {
        return false;
    }
    if store.ensure_layout().is_err() {
        return false;
    }

    let marker_path = store.pending_index_marker_path();
    let manifest = store.read_manifest().ok().flatten();
    let prior_ts = manifest
        .as_ref()
        .and_then(|m| m.last_full_index_at.as_deref())
        .unwrap_or("none");

    // The marker doubles as the "spawn in progress" sentinel. `create_new`
    // serializes concurrent SessionStarts so we don't double-spawn before
    // the child has taken the full-index lock. A fresh marker from a prior
    // spawn means an index is already running (or just failed and hasn't
    // aged out yet); either way, leave it alone so its `created_at` keeps
    // anchoring the failure-grace clock.
    let placeholder = format!("{prior_ts}\n{}\n0\n", now_rfc3339());
    if !try_claim_pending_index_marker(&marker_path, &placeholder) {
        return false;
    }
    let ack_path = store.state_dir_path().join(PENDING_INDEX_ACK_FILE_NAME);
    let _ = fs::remove_file(&ack_path);

    let Some(child_pid) = spawn_detached_claudix(project_root, [OsStr::new("index")]) else {
        let _ = fs::remove_file(&marker_path);
        return false;
    };
    // Overwrite the placeholder with the spawned PID so `check_index_ready`
    // can gate the failure decision on the child still being alive — slow
    // ONNX cold loads or large config parses must not be misclassified as
    // a crashed index.
    let payload = format!("{prior_ts}\n{}\n{child_pid}\n", now_rfc3339());
    let _ = fs::write(&marker_path, payload);
    true
}

fn try_claim_pending_index_marker(marker_path: &Path, payload: &str) -> bool {
    use std::fs::OpenOptions;
    use std::io::Write;

    for _ in 0..2 {
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(marker_path)
        {
            Ok(mut file) => return file.write_all(payload.as_bytes()).is_ok(),
            Err(_) => {
                if pending_index_marker_is_fresh(marker_path) {
                    return false;
                }
                if fs::remove_file(marker_path).is_err() {
                    return false;
                }
            }
        }
    }
    false
}

fn pending_index_marker_is_fresh(marker_path: &Path) -> bool {
    let Some(marker) = read_pending_index_marker(marker_path) else {
        return false;
    };
    // `duration_since` errors when `created_at` is in the future — clock skew
    // or restore-from-backup. Treat that as expired so a single bad timestamp
    // can't permanently jam the auto-indexer.
    SystemTime::now()
        .duration_since(marker.created_at)
        .map(|age| age < Duration::from_secs(PENDING_INDEX_FAILURE_GRACE_SECS.saturating_mul(2)))
        .unwrap_or(false)
}

struct PendingIndexMarker {
    prior_ts: String,
    created_at: SystemTime,
    /// PID of the spawned `claudix index` child, or `None` if the marker is
    /// still the placeholder written before the child was forked (or a legacy
    /// marker missing this line entirely).
    child_pid: Option<u32>,
}

fn read_pending_index_marker(marker_path: &Path) -> Option<PendingIndexMarker> {
    let content = fs::read_to_string(marker_path).ok()?;
    let mut lines = content.lines();
    let prior_ts = lines.next()?.to_owned();
    let created_at = parse_rfc3339(lines.next()?).ok()?;
    let child_pid = lines
        .next()
        .and_then(|line| line.trim().parse::<u32>().ok())
        .filter(|pid| *pid != 0);
    Some(PendingIndexMarker {
        prior_ts,
        created_at,
        child_pid,
    })
}

fn spawn_detached_claudix<const N: usize, S>(project_root: &Path, args: [S; N]) -> Option<u32>
where
    S: AsRef<OsStr>,
{
    let binary = std::env::current_exe().ok()?;
    spawn_detached_command(project_root, binary.as_os_str(), args)
}

fn spawn_background_watch(project_root: &Path, config: &Config) -> bool {
    if !config.watch || !config.hooks.auto_reembed_on_edit {
        return false;
    }
    let Ok(store) = Store::new(project_root, config) else {
        return false;
    };
    if store.ensure_layout().is_err() {
        return false;
    }
    let marker_path = store.state_dir_path().join(WATCH_MARKER_FILE_NAME);
    let Some(mut marker_file) = try_claim_watch_marker(&marker_path) else {
        return false;
    };

    let Some(child_pid) = spawn_detached_claudix(project_root, [OsStr::new("watch")]) else {
        drop(marker_file);
        let _ = fs::remove_file(&marker_path);
        return false;
    };
    // Replace the placeholder claim with the spawned child PID so concurrent
    // SessionStarts see a live entry before the child finishes booting.
    use std::io::Write as _;
    let _ = marker_file.write_all(child_pid.to_string().as_bytes());
    true
}

/// Atomically reserve the watch marker for the duration of one watcher.
///
/// Returns the opened file handle on success so the caller can overwrite the
/// placeholder content with the spawned child PID. Returns `None` if a live
/// watcher already holds the marker. A stale marker is removed once and the
/// claim retried; persistent contention after retry returns `None`.
fn try_claim_watch_marker(marker_path: &Path) -> Option<std::fs::File> {
    use std::fs::OpenOptions;
    for _ in 0..2 {
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(marker_path)
        {
            Ok(file) => return Some(file),
            Err(_) => {
                if watch_marker_is_alive(marker_path) {
                    return None;
                }
                if fs::remove_file(marker_path).is_err() {
                    return None;
                }
            }
        }
    }
    None
}

fn watch_marker_is_alive(marker_path: &Path) -> bool {
    let Ok(metadata) = fs::metadata(marker_path) else {
        return false;
    };
    let Ok(modified) = metadata.modified() else {
        return false;
    };
    let fresh = SystemTime::now()
        .duration_since(modified)
        .map(|age| age < Duration::from_secs(WATCH_MARKER_STALE_SECS))
        .unwrap_or(true);
    if !fresh {
        return false;
    }
    let Ok(content) = fs::read_to_string(marker_path) else {
        return false;
    };
    let Ok(pid) = content.trim().parse::<u32>() else {
        return false;
    };
    crate::store::process_running(pid)
}

#[cfg(unix)]
fn spawn_detached_command<const N: usize, S>(
    project_root: &Path,
    binary: &OsStr,
    args: [S; N],
) -> Option<u32>
where
    S: AsRef<OsStr>,
{
    use std::os::unix::process::CommandExt;

    // `nohup` re-execs the binary, so its PID is the nohup process itself;
    // for our liveness checks that's fine because the wrapper stays alive
    // for the whole runtime of the child it execs into.
    std::process::Command::new("nohup")
        .arg(binary)
        .args(args.iter().map(AsRef::as_ref))
        .current_dir(project_root)
        .env("CLAUDE_PROJECT_DIR", project_root)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .process_group(0)
        .spawn()
        .ok()
        .map(|child| child.id())
}

#[cfg(windows)]
fn spawn_detached_command<const N: usize, S>(
    project_root: &Path,
    binary: &OsStr,
    args: [S; N],
) -> Option<u32>
where
    S: AsRef<OsStr>,
{
    use std::os::windows::process::CommandExt;

    const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
    const DETACHED_PROCESS: u32 = 0x0000_0008;

    std::process::Command::new(binary)
        .args(args.iter().map(AsRef::as_ref))
        .current_dir(project_root)
        .env("CLAUDE_PROJECT_DIR", project_root)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .creation_flags(DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP)
        .spawn()
        .ok()
        .map(|child| child.id())
}

async fn handle_session_start(project_root: &Path, _payload: HookPayload) -> Result<Option<Value>> {
    let config = config::load(project_root).ok();

    let indexing = if let Some(ref config) = config
        && config.hooks.auto_index_on_session_start
        && is_git_repo(project_root)
    {
        spawn_background_index(project_root, config)
    } else {
        false
    };
    if let Some(ref config) = config
        && is_git_repo(project_root)
    {
        let _ = spawn_background_watch(project_root, config);
    }

    let manifest = config
        .as_ref()
        .and_then(|config| Store::new(project_root, config).ok())
        .and_then(|store| store.read_manifest().ok().flatten());

    let indexed_file_count = manifest.as_ref().map(|m| m.file_count).unwrap_or(0);
    let indexed_chunk_count = manifest.as_ref().map(|m| m.chunk_count).unwrap_or(0);
    let index_stale = match (&manifest, &config) {
        (Some(m), Some(c)) => index_is_stale(m, c),
        _ => indexed_chunk_count == 0,
    };
    let model_mismatch = match (&manifest, &config) {
        (Some(m), Some(c)) => m.embedding_model != c.embedding.model,
        _ => false,
    };

    let mut response = session_start_response(
        indexed_file_count,
        indexed_chunk_count,
        index_stale,
        model_mismatch,
        indexing,
    );
    let user_message = match consume_pending_restart().await {
        Some(message) => message,
        None => session_start_message(cli::setup_state(project_root).await),
    };
    response["systemMessage"] = Value::String(user_message);
    Ok(Some(response))
}

fn session_start_message(setup_state: cli::SetupState) -> String {
    match setup_state {
        cli::SetupState::Ready => String::new(),
        cli::SetupState::Missing(parts) => format!(
            "claudix setup incomplete (missing {}); run the install script again",
            parts.join(", ")
        ),
    }
}

async fn handle_post_tool_use(project_root: &Path, payload: HookPayload) -> Result<Option<Value>> {
    let Some(tool_name) = payload.tool_name.as_deref() else {
        return Ok(None);
    };

    let config = config::load(project_root).ok();

    // Always spawn the per-file reindex when the watcher is configured to run.
    // The watcher's `WatchFilter` is loaded once at startup, so a `.gitignore`
    // edit mid-session can desync the hook's view from the watcher's; the only
    // correctness-preserving choice is to always spawn and let
    // `acquire_reindex_lock` serialize against the watcher.
    if is_write_tool(tool_name)
        && let Some(cfg) = config.as_ref()
        && cfg.hooks.auto_reembed_on_edit
        && let Some(input) = payload.tool_input
        && let Some(file_path) = input.file_path.or(input.notebook_path)
    {
        spawn_background_reindex_file(project_root, &file_path);
    }

    Ok(config
        .as_ref()
        .and_then(|cfg| check_index_ready(project_root, cfg)))
}

fn check_index_ready(project_root: &Path, config: &Config) -> Option<Value> {
    let store = Store::new(project_root, config).ok()?;
    let marker_path = store.pending_index_marker_path();
    let ack_path = store.state_dir_path().join(PENDING_INDEX_ACK_FILE_NAME);
    let marker = read_pending_index_marker(&marker_path)?;

    // `created_at` in the future (clock skew, restored backup) would make
    // `duration_since` error forever; treat that as "past every grace window"
    // so the marker can be cleaned up instead of jamming the auto-indexer.
    let now = SystemTime::now();
    let age = match now.duration_since(marker.created_at) {
        Ok(age) => age,
        Err(_) => Duration::from_secs(PENDING_INDEX_FAILURE_GRACE_SECS.saturating_mul(2)),
    };
    if age < Duration::from_secs(PENDING_INDEX_READY_GRACE_SECS) {
        return None;
    }

    if store.full_index_running() {
        return None;
    }

    let manifest = store.read_manifest().ok().flatten();
    let current_ts = manifest
        .as_ref()
        .and_then(|m| m.last_full_index_at.as_deref())
        .unwrap_or("none");

    if current_ts != marker.prior_ts {
        let _ = fs::remove_file(&marker_path);
        let _ = fs::remove_file(&ack_path);
        let Some(manifest) = manifest else {
            // ts changed but the manifest is gone — treat as a failed run so
            // the user is informed instead of silently dropping the signal.
            return Some(indexing_failed_response());
        };
        return Some(json!({
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "additionalContext": format!(
                    "claudix indexing complete — {} files, {} chunks. Semantic search is now ready: \
                     use search_code for conceptual queries, identifier lookups, and cross-file discovery.",
                    manifest.file_count, manifest.chunk_count
                ),
            }
        }));
    }

    // Manifest timestamp unchanged AND no index lock holder. If the spawned
    // child is still alive we're inside the slow-boot window (cold ONNX
    // load, large config parse) — never declare failure yet. Only after the
    // PID exits and the failure grace has elapsed do we surface a failure.
    if marker
        .child_pid
        .is_some_and(crate::store::process_running)
    {
        return None;
    }
    if age < Duration::from_secs(PENDING_INDEX_FAILURE_GRACE_SECS) {
        return None;
    }

    // Pre-first-index state has `prior_ts == "none"` indefinitely, so suppressing
    // duplicate-ts acks would silence every consecutive failure on a fresh repo.
    // Always surface failure for that sentinel; gate dedup on real timestamps.
    let is_first_index = marker.prior_ts == "none";
    let already_acked = !is_first_index
        && fs::read_to_string(&ack_path)
            .ok()
            .map(|content| content.trim() == marker.prior_ts)
            .unwrap_or(false);
    let _ = fs::write(&ack_path, &marker.prior_ts);
    let _ = fs::remove_file(&marker_path);

    if already_acked {
        return None;
    }

    Some(indexing_failed_response())
}

fn indexing_failed_response() -> Value {
    json!({
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext":
                "claudix background indexing ended without updating the index — it may have failed. \
                 Run /claudix:doctor to diagnose.",
        }
    })
}

fn is_write_tool(tool_name: &str) -> bool {
    matches!(tool_name, "Edit" | "Write" | "NotebookEdit" | "MultiEdit")
}

fn spawn_background_reindex_file(project_root: &Path, file_path: &str) {
    spawn_detached_claudix(
        project_root,
        [OsStr::new("reindex-file"), OsStr::new(file_path)],
    );
}

async fn handle_pre_tool_use(project_root: &Path, payload: HookPayload) -> Result<Option<Value>> {
    let config = config::load(project_root)?;
    if !config.hooks.intercept_grep {
        return Ok(None);
    }

    let Some(tool_name) = payload.tool_name.as_deref() else {
        return Ok(None);
    };
    let Some(tool_input) = payload.tool_input else {
        return Ok(None);
    };

    let query = match tool_name {
        "Grep" => {
            if tool_input.path.is_some() || tool_input.include.is_some() {
                return Ok(None);
            }
            tool_input.pattern
        }
        "Bash" => extract_search_command(tool_input.command.as_deref()),
        _ => None,
    };

    let Some(query) = query else {
        return Ok(None);
    };
    if should_passthrough(&query) {
        return Ok(None);
    }

    let store = Store::new(project_root, &config)?;
    let manifest = store.read_manifest()?;
    let Some(manifest) = manifest else {
        return Ok(None);
    };
    if index_is_stale(&manifest, &config) {
        return Ok(None);
    }
    if manifest.chunk_count == 0 {
        return Ok(None);
    }

    let health_cache_path = store.state_dir_path().join(HEALTH_CACHE_FILE_NAME);
    if matches!(read_cached_health(&health_cache_path), Some(false)) {
        return Ok(None);
    }

    let search_query = SearchQuery {
        query: query.clone(),
        top_k: config.search.top_k,
        language_filter: None,
        path_prefix: None,
    };
    let project_root = project_root.to_path_buf();
    let config_arc = Arc::new(config.clone());
    let work = async move {
        let claudix = Claudix::new(project_root, config_arc).await?;
        let results = claudix.search(search_query).await?;
        Ok::<_, ClaudixError>(results)
    };

    match tokio::time::timeout(Duration::from_millis(PRE_TOOL_USE_TIMEOUT_MS), work).await {
        Ok(Ok(results)) => {
            write_cached_health(&health_cache_path, true);
            if results.is_empty() {
                Ok(None)
            } else {
                Ok(Some(pre_tool_use_search_response(&query, results)))
            }
        }
        Ok(Err(_)) => {
            write_cached_health(&health_cache_path, false);
            Ok(None)
        }
        // Timeout doesn't necessarily mean the embedder is unhealthy — bundled
        // ONNX cold-loads can exceed the budget — so pass through without
        // poisoning the cache.
        Err(_) => Ok(None),
    }
}

fn read_cached_health(marker_path: &Path) -> Option<bool> {
    let metadata = fs::metadata(marker_path).ok()?;
    let modified = metadata.modified().ok()?;
    let age = SystemTime::now().duration_since(modified).ok()?;
    let content = fs::read_to_string(marker_path).ok()?;
    let healthy = match content.trim() {
        "1" => true,
        "0" => false,
        _ => return None,
    };
    let ttl = if healthy {
        HEALTH_CACHE_HEALTHY_TTL_SECS
    } else {
        HEALTH_CACHE_UNHEALTHY_TTL_SECS
    };
    (age < Duration::from_secs(ttl)).then_some(healthy)
}

fn write_cached_health(marker_path: &Path, healthy: bool) {
    let _ = fs::write(marker_path, if healthy { "1" } else { "0" });
}

async fn consume_pending_restart() -> Option<String> {
    let data_dir = pending_restart_path()?;
    let content = tokio::fs::read_to_string(&data_dir).await.ok()?;
    let version = content.trim();
    if version.is_empty() {
        return None;
    }
    let msg = format!("claudix updated to v{version} — restart Claude Code to activate");
    let _ = tokio::fs::remove_file(&data_dir).await;
    Some(msg)
}

fn pending_restart_path() -> Option<std::path::PathBuf> {
    let base = std::env::var_os("CLAUDIX_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|| {
            std::env::var_os("XDG_DATA_HOME").map(|p| std::path::PathBuf::from(p).join("claudix"))
        })
        .or_else(|| dirs::home_dir().map(|h| h.join(".local").join("share").join("claudix")))?;
    Some(base.join("pending-restart"))
}

fn session_start_response(
    file_count: u64,
    chunk_count: u64,
    stale: bool,
    model_mismatch: bool,
    indexing_spawned: bool,
) -> Value {
    let additional_context = if model_mismatch {
        "claudix semantic search unavailable — embedding model mismatch. Run `claudix clear && claudix index` to rebuild.".to_owned()
    } else if chunk_count == 0 {
        "claudix is installed but the index is empty. Run /claudix:index to build it; until then use Grep or Read for code discovery.".to_owned()
    } else if indexing_spawned {
        format!(
            "claudix semantic search ready — {file_count} files, {chunk_count} chunks (reindexing in background; you'll be notified when complete). \
             Use search_code for fast semantic search: conceptual queries, identifier lookups, cross-file discovery. \
             Use Grep for exact literals, regexes, or path-filtered scans."
        )
    } else if stale {
        format!(
            "claudix semantic search ready — {file_count} files, {chunk_count} chunks (index stale). \
             Use search_code for fast semantic search: conceptual queries, identifier lookups, cross-file discovery. \
             Use Grep for exact literals, regexes, or path-filtered scans."
        )
    } else {
        format!(
            "claudix semantic search ready — {file_count} files, {chunk_count} chunks. \
             Use search_code for fast semantic search: conceptual queries, identifier lookups, cross-file discovery. \
             Use Grep for exact literals, regexes, or path-filtered scans."
        )
    };
    json!({
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": additional_context,
        }
    })
}

fn pre_tool_use_search_response(query: &str, results: Vec<crate::search::SearchResult>) -> Value {
    let mut lines = vec![
        format!("claudix search results for '{query}':"),
        String::new(),
    ];
    for result in &results {
        let chunk = &result.chunk;
        let name_part = chunk
            .name
            .as_deref()
            .map(|n| format!(" {n}"))
            .unwrap_or_default();
        let stale_warning = if result.stale {
            " [STALE - file modified since index]"
        } else {
            ""
        };
        lines.push(format!(
            "{}:{}-{} [{}] {}{name_part} ({:.3}){}",
            chunk.file_path,
            chunk.line_range.start,
            chunk.line_range.end,
            chunk.language,
            chunk.kind,
            result.score,
            stale_warning,
        ));
        if !chunk.content.is_empty() {
            lines.push(truncate_snippet(&chunk.content, 20));
        }
        lines.push(String::new());
    }
    lines.push(
        "Tip: call search_code MCP tool directly next time to skip this interception round-trip."
            .to_owned(),
    );
    let context = lines.join("\n");
    json!({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": format!("claudix found {} semantic matches for '{query}' — see additionalContext.", results.len()),
            "additionalContext": context,
        }
    })
}

fn truncate_snippet(content: &str, max_lines: usize) -> String {
    let mut lines = content.lines();
    let taken: Vec<&str> = lines.by_ref().take(max_lines).collect();
    if lines.next().is_some() {
        format!("{}\n…", taken.join("\n"))
    } else {
        taken.join("\n")
    }
}

fn extract_search_command(command: Option<&str>) -> Option<String> {
    let command = command?.trim();
    let args = command
        .strip_prefix("rg ")
        .or_else(|| command.strip_prefix("grep "))
        .or_else(|| command.strip_prefix("ag "))?;
    let args = args.trim();
    extract_quoted_pattern(args).or_else(|| extract_unquoted_pattern(args))
}

fn extract_quoted_pattern(args: &str) -> Option<String> {
    let bytes = args.as_bytes();
    let dq_pos = bytes.iter().position(|&b| b == b'"');
    let sq_pos = bytes.iter().position(|&b| b == b'\'');

    let (start, quote) = match (dq_pos, sq_pos) {
        (Some(d), Some(s)) => {
            if d < s {
                (d, b'"')
            } else {
                (s, b'\'')
            }
        }
        (Some(d), None) => (d, b'"'),
        (None, Some(s)) => (s, b'\''),
        (None, None) => return None,
    };

    let after = &args[start + 1..];
    let quote = char::from(quote);
    let mut pattern = String::new();
    let mut escaped = false;
    let mut closed = false;

    for character in after.chars() {
        if escaped {
            if character == quote {
                pattern.push(character);
            } else {
                pattern.push('\\');
                pattern.push(character);
            }
            escaped = false;
            continue;
        }

        if character == '\\' {
            escaped = true;
            continue;
        }

        if character == quote {
            closed = true;
            break;
        }

        pattern.push(character);
    }

    if escaped {
        pattern.push('\\');
    }
    if !closed {
        return None;
    }

    let pattern = pattern.trim();
    if pattern.is_empty() {
        None
    } else {
        Some(pattern.to_owned())
    }
}

fn extract_unquoted_pattern(args: &str) -> Option<String> {
    let tokens: Vec<&str> = args.split_whitespace().collect();
    if tokens.iter().any(|t| t.starts_with('-')) {
        return None;
    }
    let pattern_tokens: Vec<&str> = tokens
        .iter()
        .filter(|t| !t.contains('/') && !t.contains('\\'))
        .copied()
        .collect();
    if pattern_tokens.len() == 1 && !pattern_tokens[0].is_empty() {
        Some(pattern_tokens[0].to_owned())
    } else {
        None
    }
}

fn should_passthrough(query: &str) -> bool {
    let trimmed = query.trim();
    trimmed.is_empty()
        || looks_like_regex(trimmed)
        || looks_like_file_target(trimmed)
        || token_count(trimmed) < 3
}

fn looks_like_regex(query: &str) -> bool {
    query.contains('^')
        || query.contains('$')
        || query.contains('\\')
        || query.contains('[')
        || query.contains(']')
        || query.contains('(')
        || query.contains(')')
        || query.contains('+')
        || query.contains('?')
        || query.contains('{')
        || query.contains('}')
        || query.contains(".*")
}

fn looks_like_file_target(query: &str) -> bool {
    const FILE_EXTENSIONS: &[&str] = &[
        ".rs", ".py", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".go", ".java", ".c", ".h", ".cpp",
        ".cc", ".cxx", ".hpp", ".hxx", ".cs", ".sql",
    ];

    query.contains("--glob")
        || query.contains("--include")
        || query.contains("*.")
        || query.contains("src/")
        || FILE_EXTENSIONS.iter().any(|ext| query.contains(ext))
}

fn token_count(query: &str) -> usize {
    query
        .split(|character: char| !character.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .count()
}

pub(crate) fn index_is_stale(manifest: &Manifest, config: &Config) -> bool {
    let Some(last_full_index_at) = manifest.last_full_index_at.as_deref() else {
        return true;
    };
    let Ok(last_full_index_at) = parse_rfc3339(last_full_index_at) else {
        return true;
    };
    let Ok(age) = SystemTime::now().duration_since(last_full_index_at) else {
        return false;
    };

    age > Duration::from_secs(config.indexing.reindex_after_hours.saturating_mul(3_600))
}

#[derive(Debug, Deserialize)]
struct HookPayload {
    tool_name: Option<String>,
    tool_input: Option<ToolInput>,
}

#[derive(Debug, Deserialize)]
struct ToolInput {
    file_path: Option<String>,
    notebook_path: Option<String>,
    pattern: Option<String>,
    command: Option<String>,
    path: Option<String>,
    include: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn looks_like_regex_detects_metacharacters() {
        for pattern in [
            "^pub fn",
            "fn\\s+\\w+",
            "[a-z]+",
            "fn(x)",
            "x+",
            "x?",
            "{3}",
            "x$",
            ".*",
        ] {
            assert!(
                looks_like_regex(pattern),
                "expected regex detection for: {pattern}"
            );
        }
        assert!(!looks_like_regex("error handling"));
        assert!(!looks_like_regex("handle_session_start"));
    }

    #[test]
    fn looks_like_file_target_covers_all_supported_extensions() {
        for ext in &[
            ".rs", ".py", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".go", ".java", ".c", ".h", ".cpp",
            ".cc", ".cxx", ".hpp", ".hxx", ".cs", ".sql",
        ] {
            let query = format!("search routes{ext}");
            assert!(
                looks_like_file_target(&query),
                "expected passthrough for query containing {ext}"
            );
        }
        assert!(!looks_like_file_target("error handling retry logic"));
        assert!(looks_like_file_target("find *.rs files"));
        assert!(looks_like_file_target("search in src/"));
    }

    #[test]
    fn extract_unquoted_pattern_returns_sole_non_path_token() {
        assert_eq!(
            extract_unquoted_pattern("handle_session_start"),
            Some("handle_session_start".to_owned())
        );
        assert_eq!(
            extract_unquoted_pattern("handle_session_start src/"),
            Some("handle_session_start".to_owned())
        );
        // Multiple non-path tokens → ambiguous, return None.
        assert_eq!(extract_unquoted_pattern("foo bar"), None);
        // Any flag → bail out entirely (flag value might be misidentified as pattern).
        assert_eq!(
            extract_unquoted_pattern("--type rust handle_session_start"),
            None
        );
        // Path-only → None.
        assert_eq!(extract_unquoted_pattern("src/lib.rs"), None);
    }

    #[test]
    fn extract_quoted_pattern_uses_first_quote_type_as_delimiter() {
        // Single-quoted pattern containing double quotes — must extract the full inner string.
        assert_eq!(
            extract_quoted_pattern(r#"'say "hello"'"#),
            Some(r#"say "hello""#.to_owned())
        );
        // Double-quoted pattern (common case).
        assert_eq!(
            extract_quoted_pattern(r#""error handling""#),
            Some("error handling".to_owned())
        );
        // Double-quoted pattern with trailing flags.
        assert_eq!(
            extract_quoted_pattern(r#"-rn "pattern" src/"#),
            Some("pattern".to_owned())
        );
        assert_eq!(
            extract_quoted_pattern(r#""error \"quoted\" message" src/"#),
            Some(r#"error "quoted" message"#.to_owned())
        );
        // No quotes → None.
        assert_eq!(extract_quoted_pattern("add src/"), None);
    }
    use std::fs;
    use std::sync::Arc;
    use tempfile::tempdir;

    use crate::Claudix;
    use crate::config::Config;
    use crate::store::Manifest;

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

    fn write_config(project_root: &Path, config: &Config) {
        let claude_dir = project_root.join(".claude");
        assert!(fs::create_dir_all(&claude_dir).is_ok());
        let config_text = toml::to_string(config);
        assert!(config_text.is_ok());
        assert!(
            fs::write(
                claude_dir.join("claudix.toml"),
                config_text.ok().unwrap_or_default()
            )
            .is_ok()
        );
    }

    #[tokio::test]
    async fn session_start_handles_empty_payload() {
        let fixture = TestFixture::new("small_rust");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());
        write_config(fixture.root(), &stub_config());

        let response = run(fixture.root(), HookEvent::SessionStart, "").await;
        assert!(response.is_ok(), "empty payload must not error");
        assert!(response.ok().unwrap_or_else(|| unreachable!()).is_some());
    }

    #[tokio::test]
    async fn session_start_reports_incomplete_setup() {
        let fixture = TestFixture::new("small_rust");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());
        write_config(fixture.root(), &stub_config());

        let response = run(fixture.root(), HookEvent::SessionStart, "{}").await;
        assert!(response.is_ok());
        let response = response.ok().unwrap_or_else(|| unreachable!());
        assert!(response.is_some());
        let response = response.unwrap_or(Value::Null);
        let user_message = response["systemMessage"].as_str().unwrap_or_default();
        assert!(user_message.contains("run the install script again"));

        let model_context = response["hookSpecificOutput"]["additionalContext"]
            .as_str()
            .unwrap_or_default();
        assert_eq!(
            model_context,
            "claudix is installed but the index is empty. Run /claudix:index to build it; until then use Grep or Read for code discovery."
        );
    }

    #[test]
    fn session_start_message_reports_ready_setup() {
        assert_eq!(session_start_message(cli::SetupState::Ready), "");
        assert_eq!(
            session_start_message(cli::SetupState::Missing(vec!["bin"])),
            "claudix setup incomplete (missing bin); run the install script again"
        );
    }

    #[test]
    fn session_start_context_guides_tool_choice() {
        let response = session_start_response(42, 683, false, false, false);
        let context = response["hookSpecificOutput"]["additionalContext"]
            .as_str()
            .unwrap_or_default();

        assert!(context.contains("Use search_code"));
        assert!(context.contains("fast semantic search"));
        assert!(context.contains("conceptual queries"));
        assert!(context.contains("identifier lookups"));
        assert!(context.contains("cross-file discovery"));
        assert!(context.contains("Use Grep for exact literals"));
    }

    #[tokio::test]
    async fn post_tool_use_spawns_background_reindex_and_returns_none() {
        let fixture = TestFixture::new("small_rust");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());
        write_config(fixture.root(), &stub_config());

        let payload = json!({
            "tool_name": "Write",
            "tool_input": {
                "file_path": fixture.root().join("src/math.rs"),
            }
        });
        let response = run(fixture.root(), HookEvent::PostToolUse, &payload.to_string()).await;
        assert!(response.is_ok());
        assert!(response.ok().unwrap_or_else(|| unreachable!()).is_none());
    }

    #[tokio::test]
    async fn post_tool_use_ignores_read_tool() {
        let fixture = TestFixture::new("small_rust");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());
        write_config(fixture.root(), &stub_config());

        let payload = json!({
            "tool_name": "Read",
            "tool_input": {
                "file_path": fixture.root().join("src/math.rs"),
            }
        });
        let response = run(fixture.root(), HookEvent::PostToolUse, &payload.to_string()).await;
        assert!(response.is_ok());
        assert!(
            response.ok().unwrap_or_else(|| unreachable!()).is_none(),
            "Read tool must not trigger reindex"
        );
    }

    #[tokio::test]
    async fn post_tool_use_triggers_reindex_for_notebook_edit() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        write_config(fixture.root(), &stub_config());

        let payload = json!({
            "tool_name": "NotebookEdit",
            "tool_input": {
                "notebook_path": fixture.root().join("analysis.ipynb"),
            }
        });
        let response = run(fixture.root(), HookEvent::PostToolUse, &payload.to_string()).await?;
        assert!(
            response.is_none(),
            "NotebookEdit must trigger reindex and return None"
        );
        Ok(())
    }

    #[tokio::test]
    async fn post_tool_use_passes_through_when_auto_reembed_disabled() {
        let fixture = TestFixture::new("small_rust");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());
        let mut config = stub_config();
        config.hooks.auto_reembed_on_edit = false;
        write_config(fixture.root(), &config);

        let payload = json!({
            "tool_name": "Write",
            "tool_input": {
                "file_path": fixture.root().join("src/math.rs"),
            }
        });
        let response = run(fixture.root(), HookEvent::PostToolUse, &payload.to_string()).await;
        assert!(response.is_ok());
        assert!(response.ok().unwrap_or_else(|| unreachable!()).is_none());
    }

    #[tokio::test]
    async fn pre_tool_use_denies_conceptual_grep_when_index_ready() {
        let fixture = TestFixture::new("small_rust");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());
        write_config(fixture.root(), &stub_config());

        let claudix = Claudix::new(fixture.root().to_path_buf(), Arc::new(stub_config())).await;
        assert!(claudix.is_ok());
        assert!(
            claudix
                .ok()
                .unwrap_or_else(|| unreachable!())
                .index_full()
                .await
                .is_ok()
        );

        let payload = json!({
            "tool_name": "Grep",
            "tool_input": {
                "pattern": "where is config loaded"
            }
        });
        let response = run(fixture.root(), HookEvent::PreToolUse, &payload.to_string()).await;
        assert!(response.is_ok());
        let response = response.ok().unwrap_or_else(|| unreachable!());
        assert!(response.is_some());
        let response = response.unwrap_or(Value::Null);
        assert_eq!(
            response["hookSpecificOutput"]["permissionDecision"],
            Value::String("deny".to_owned())
        );
    }

    #[tokio::test]
    async fn pre_tool_use_passes_regex_queries_through() {
        let fixture = TestFixture::new("small_rust");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());
        write_config(fixture.root(), &stub_config());

        let payload = json!({
            "tool_name": "Grep",
            "tool_input": {
                "pattern": "^pub fn"
            }
        });
        let response = run(fixture.root(), HookEvent::PreToolUse, &payload.to_string()).await;
        assert!(response.is_ok());
        assert!(response.ok().unwrap_or_else(|| unreachable!()).is_none());
    }

    #[tokio::test]
    async fn pre_tool_use_passes_conceptual_bash_rg_when_search_returns_no_results() {
        let fixture = TestFixture::new("small_rust");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());
        write_config(fixture.root(), &stub_config());

        let claudix = Claudix::new(fixture.root().to_path_buf(), Arc::new(stub_config())).await;
        assert!(claudix.is_ok());
        assert!(
            claudix
                .ok()
                .unwrap_or_else(|| unreachable!())
                .index_full()
                .await
                .is_ok()
        );

        // "where is the config loaded" is conceptual but the small_rust fixture has no config code;
        // semantic search returns empty → must pass through, not block grep with no alternative.
        let payload = json!({
            "tool_name": "Bash",
            "tool_input": {
                "command": "rg \"where is the config loaded\""
            }
        });
        let response = run(fixture.root(), HookEvent::PreToolUse, &payload.to_string()).await;
        assert!(response.is_ok());
        assert!(
            response.ok().unwrap_or_else(|| unreachable!()).is_none(),
            "must pass through when semantic search has no results to offer"
        );
    }

    #[tokio::test]
    async fn pre_tool_use_passes_bash_rg_with_path_arg_through_when_no_conceptual_pattern() {
        let fixture = TestFixture::new("small_rust");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());
        write_config(fixture.root(), &stub_config());

        let claudix = Claudix::new(fixture.root().to_path_buf(), Arc::new(stub_config())).await;
        assert!(claudix.is_ok());
        assert!(
            claudix
                .ok()
                .unwrap_or_else(|| unreachable!())
                .index_full()
                .await
                .is_ok()
        );

        // Unquoted single-word pattern — extracted but token_count < 3 so passes through.
        let payload = json!({
            "tool_name": "Bash",
            "tool_input": {
                "command": "rg add src/"
            }
        });
        let response = run(fixture.root(), HookEvent::PreToolUse, &payload.to_string()).await;
        assert!(response.is_ok());
        assert!(
            response.ok().unwrap_or_else(|| unreachable!()).is_none(),
            "unquoted single-word rg command should pass through"
        );
    }

    #[tokio::test]
    async fn pre_tool_use_intercepts_unquoted_identifier_in_bash_rg() {
        let fixture = TestFixture::new("small_rust");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());
        write_config(fixture.root(), &stub_config());

        let claudix = Claudix::new(fixture.root().to_path_buf(), Arc::new(stub_config())).await;
        assert!(claudix.is_ok());
        assert!(
            claudix
                .ok()
                .unwrap_or_else(|| unreachable!())
                .index_full()
                .await
                .is_ok()
        );

        // Unquoted identifier with enough tokens — should be intercepted just like a quoted query.
        let payload = json!({
            "tool_name": "Bash",
            "tool_input": {
                "command": "rg add_two_numbers"
            }
        });
        let response = run(fixture.root(), HookEvent::PreToolUse, &payload.to_string()).await;
        assert!(response.is_ok());
        let response = response.ok().unwrap_or_else(|| unreachable!());
        assert!(
            response.is_some(),
            "unquoted multi-token identifier should be intercepted"
        );
        let response = response.unwrap_or(Value::Null);
        assert_eq!(
            response["hookSpecificOutput"]["permissionDecision"],
            Value::String("deny".to_owned())
        );
    }

    #[tokio::test]
    async fn pre_tool_use_passes_bash_rg_with_regex_pattern_through() {
        let fixture = TestFixture::new("small_rust");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());
        write_config(fixture.root(), &stub_config());

        let claudix = Claudix::new(fixture.root().to_path_buf(), Arc::new(stub_config())).await;
        assert!(claudix.is_ok());
        assert!(
            claudix
                .ok()
                .unwrap_or_else(|| unreachable!())
                .index_full()
                .await
                .is_ok()
        );

        let payload = json!({
            "tool_name": "Bash",
            "tool_input": {
                "command": "rg \"^pub fn\" src/"
            }
        });
        let response = run(fixture.root(), HookEvent::PreToolUse, &payload.to_string()).await;
        assert!(response.is_ok());
        assert!(
            response.ok().unwrap_or_else(|| unreachable!()).is_none(),
            "regex pattern in rg command should pass through"
        );
    }

    #[tokio::test]
    async fn pre_tool_use_passes_conceptual_bash_rg_with_path_arg_when_no_results() {
        let fixture = TestFixture::new("small_rust");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());
        write_config(fixture.root(), &stub_config());

        let claudix = Claudix::new(fixture.root().to_path_buf(), Arc::new(stub_config())).await;
        assert!(claudix.is_ok());
        assert!(
            claudix
                .ok()
                .unwrap_or_else(|| unreachable!())
                .index_full()
                .await
                .is_ok()
        );

        // Conceptual query with a path arg: semantic search still finds nothing for "config loaded"
        // in the small_rust fixture → must pass through.
        let payload = json!({
            "tool_name": "Bash",
            "tool_input": {
                "command": "rg \"where is the config loaded\" src/"
            }
        });
        let response = run(fixture.root(), HookEvent::PreToolUse, &payload.to_string()).await;
        assert!(response.is_ok());
        assert!(
            response.ok().unwrap_or_else(|| unreachable!()).is_none(),
            "must pass through when semantic search has no results to offer"
        );
    }

    #[tokio::test]
    async fn pre_tool_use_returns_search_results_in_context() {
        let fixture = TestFixture::new("small_rust");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());
        write_config(fixture.root(), &stub_config());

        let claudix = Claudix::new(fixture.root().to_path_buf(), Arc::new(stub_config())).await;
        assert!(claudix.is_ok());
        assert!(
            claudix
                .ok()
                .unwrap_or_else(|| unreachable!())
                .index_full()
                .await
                .is_ok()
        );
        assert!(
            std::fs::write(
                fixture.root().join("src/math.rs"),
                "pub fn subtract(left: i32, right: i32) -> i32 { left - right }\n",
            )
            .is_ok()
        );

        let payload = json!({
            "tool_name": "Grep",
            "tool_input": { "pattern": "add two numbers together" }
        });
        let response = run(fixture.root(), HookEvent::PreToolUse, &payload.to_string()).await;
        assert!(response.is_ok());
        let response = response.ok().unwrap_or_else(|| unreachable!());
        assert!(response.is_some());
        let response = response.unwrap_or(Value::Null);

        assert_eq!(
            response["hookSpecificOutput"]["permissionDecision"],
            Value::String("deny".to_owned())
        );
        let context = response["hookSpecificOutput"]["additionalContext"]
            .as_str()
            .unwrap_or_default();
        assert!(
            context.contains("claudix search results"),
            "hook must embed search results in context, got: {context}"
        );
        assert!(
            context.contains("src/"),
            "context must include file paths from search results"
        );
        assert!(
            context.contains("[STALE - file modified since index]"),
            "context must warn about stale hits, got: {context}"
        );
        assert!(
            context.contains("search_code MCP tool"),
            "context must include tip to use search_code directly, got: {context}"
        );
    }

    #[test]
    fn pending_index_marker_round_trips() {
        let dir = tempdir().ok().unwrap_or_else(|| unreachable!());
        let marker_path = dir.path().join("indexing-pending");
        let payload = format!("2026-04-20T00:00:00Z\n{}\n42\n", now_rfc3339());

        assert!(try_claim_pending_index_marker(&marker_path, &payload));
        let marker = read_pending_index_marker(&marker_path);
        assert!(marker.is_some());
        let marker = marker.unwrap_or_else(|| unreachable!());
        assert_eq!(marker.prior_ts, "2026-04-20T00:00:00Z");
        assert_eq!(marker.child_pid, Some(42));
    }

    #[test]
    fn pending_index_marker_treats_zero_pid_as_placeholder() {
        let dir = tempdir().ok().unwrap_or_else(|| unreachable!());
        let marker_path = dir.path().join("indexing-pending");
        let payload = format!("none\n{}\n0\n", now_rfc3339());
        assert!(try_claim_pending_index_marker(&marker_path, &payload));
        let marker = read_pending_index_marker(&marker_path)
            .unwrap_or_else(|| unreachable!());
        assert_eq!(marker.child_pid, None);
    }

    #[test]
    fn pending_index_marker_claim_is_atomic() {
        let dir = tempdir().ok().unwrap_or_else(|| unreachable!());
        let marker_path = dir.path().join("indexing-pending");
        let first = format!("none\n{}\n", now_rfc3339());
        assert!(try_claim_pending_index_marker(&marker_path, &first));

        // Second claim while the first is still fresh must fail.
        let second = format!("ts-2\n{}\n", now_rfc3339());
        assert!(!try_claim_pending_index_marker(&marker_path, &second));
        let marker = read_pending_index_marker(&marker_path)
            .unwrap_or_else(|| unreachable!());
        assert_eq!(marker.prior_ts, "none", "first claim must remain in place");
    }

    #[test]
    fn pending_index_marker_with_future_timestamp_is_not_fresh() {
        let dir = tempdir().ok().unwrap_or_else(|| unreachable!());
        let marker_path = dir.path().join("indexing-pending");
        // 1 hour in the future — simulates clock skew / backup restore.
        let future = SystemTime::now() + Duration::from_secs(3_600);
        let future_ts = crate::util::format_rfc3339(future);
        fs::write(&marker_path, format!("none\n{future_ts}\n0\n"))
            .unwrap_or_else(|_| unreachable!());

        assert!(
            !pending_index_marker_is_fresh(&marker_path),
            "future-timestamped marker must be reclaimable"
        );
    }

    #[test]
    fn pending_index_marker_claim_replaces_legacy_or_unparseable() {
        let dir = tempdir().ok().unwrap_or_else(|| unreachable!());
        let marker_path = dir.path().join("indexing-pending");
        assert!(fs::write(&marker_path, "legacy-single-line").is_ok());

        let payload = format!("none\n{}\n", now_rfc3339());
        assert!(try_claim_pending_index_marker(&marker_path, &payload));
        let marker = read_pending_index_marker(&marker_path);
        assert!(marker.is_some());
    }

    #[tokio::test]
    async fn check_index_ready_resurfaces_failure_for_first_index_sentinel() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let config = stub_config();
        write_config(fixture.root(), &config);
        let store = Store::new(fixture.root(), &config)?;
        store.ensure_layout()?;

        // Pre-ack the "none" sentinel so a naive dedup check would suppress.
        let ack_path = store.state_dir_path().join(PENDING_INDEX_ACK_FILE_NAME);
        fs::write(&ack_path, "none")?;

        let stale_created_at = "2025-01-01T00:00:00Z";
        let payload = format!("none\n{stale_created_at}\n0\n");
        fs::write(store.pending_index_marker_path(), payload)?;

        let response = check_index_ready(fixture.root(), &config);
        let response = response.unwrap_or(Value::Null);
        let context = response["hookSpecificOutput"]["additionalContext"]
            .as_str()
            .unwrap_or_default();
        assert!(
            context.contains("ended without updating the index"),
            "repeat failures before the first successful index must still surface, got: {context}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn check_index_ready_defers_failure_while_child_pid_alive() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let config = stub_config();
        write_config(fixture.root(), &config);
        let store = Store::new(fixture.root(), &config)?;
        store.ensure_layout()?;

        // Backdate the marker past the failure grace but record this test
        // process as the spawned child — `check_index_ready` must defer the
        // failure decision while that PID is still alive.
        let stale_created_at = "2025-01-01T00:00:00Z";
        let my_pid = std::process::id();
        let payload = format!("none\n{stale_created_at}\n{my_pid}\n");
        fs::write(store.pending_index_marker_path(), payload)?;

        let response = check_index_ready(fixture.root(), &config);
        assert!(
            response.is_none(),
            "must not declare failure while spawned child is alive"
        );
        assert!(
            store.pending_index_marker_path().exists(),
            "marker must survive the deferred decision"
        );
        Ok(())
    }

    #[tokio::test]
    async fn check_index_ready_surfaces_failure_when_manifest_is_missing() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let config = stub_config();
        write_config(fixture.root(), &config);
        let store = Store::new(fixture.root(), &config)?;
        store.ensure_layout()?;

        // Backdate the marker past the failure grace; leave the manifest absent
        // to simulate a first-time index that crashed before writing one.
        let stale_created_at = "2025-01-01T00:00:00Z";
        let payload = format!("none\n{stale_created_at}\n");
        fs::write(store.pending_index_marker_path(), payload)?;

        let response = check_index_ready(fixture.root(), &config);
        let response = response.unwrap_or(Value::Null);
        let context = response["hookSpecificOutput"]["additionalContext"]
            .as_str()
            .unwrap_or_default();
        assert!(
            context.contains("ended without updating the index"),
            "expected failure context, got: {context}"
        );
        assert!(
            !store.pending_index_marker_path().exists(),
            "marker must be cleaned up after surfacing failure"
        );
        Ok(())
    }

    #[test]
    fn stale_index_detection_respects_threshold() {
        let mut config = stub_config();
        config.indexing.reindex_after_hours = 24;
        let mut manifest = Manifest::new("stub-v1", 8);
        manifest.last_full_index_at = Some("2026-04-20T00:00:00Z".to_owned());
        assert!(index_is_stale(&manifest, &config));
    }

    #[test]
    fn fresh_index_detection_allows_recent_manifest() {
        let project_root = tempdir();
        assert!(project_root.is_ok());
        let _project_root = project_root.ok().unwrap_or_else(|| unreachable!());
        let mut config = stub_config();
        config.indexing.reindex_after_hours = 24 * 365 * 20;
        let mut manifest = Manifest::new("stub-v1", 8);
        manifest.last_full_index_at = Some(crate::util::now_rfc3339());
        assert!(!index_is_stale(&manifest, &config));
    }
}
