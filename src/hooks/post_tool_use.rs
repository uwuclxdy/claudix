use std::path::{Path, PathBuf};
use std::time::Duration;

use serde_json::{Value, json};

use crate::config::{self, Config};
use crate::enumeration::WatchFilter;
use crate::error::Result;
use crate::prompts;
use crate::search::neighbors::neighbors;
use crate::store::Store;
use crate::store::marker::change_neighbors;
use crate::types::RelativePath;

use super::payload::{HookPayload, ToolInput};
use super::ready_check::check_index_ready;
use super::spawn::spawn_background_drain_worker;
use crate::store::marker::WATCH_MARKER_STALE_SECS;
use crate::store::marker::reindex_queue;

/// Fail-open budget for the read-time surfacing load + cosine scan. The `Read`
/// hook must not stall the session, so the whole-corpus read and O(n²) scan are
/// each capped at this; on elapse the hook surfaces nothing.
const READ_SURFACING_TIMEOUT_MS: u64 = 2_000;

/// Corpus ceiling for read-time surfacing. Above this the O(n²) scan is too
/// expensive to run on the hot `Read` path, so surfacing is skipped entirely.
const READ_SURFACING_MAX_CHUNKS: usize = 50_000;

pub(super) async fn handle_post_tool_use(
    project_root: &Path,
    payload: HookPayload,
) -> Result<Option<Value>> {
    let config = config::load(project_root).ok();

    // `Read` rides this hook too (see hooks.json matcher) purely for read-time
    // surfacing — it must NEVER spawn a reindex. The edit tools below do.
    let tool_name = payload.tool_name.as_deref();

    if tool_name == Some("Read")
        && !config
            .as_ref()
            .is_some_and(|cfg| cfg.hooks.surface_related_on_read)
    {
        return Ok(None);
    }

    // One Store for the whole event: every branch below needs its paths, and
    // `Store::new` canonicalizes the root — building it per check multiplies
    // that syscall cost on the busiest hook. `None` fail-opens every branch.
    let store = config
        .as_ref()
        .and_then(|cfg| Store::new(project_root, cfg).ok());

    // Coalesce the no-watcher reindex path. The ready-check and
    // neighbor-surfacing run on every PostToolUse event regardless.
    //
    // Multi-file payloads (MultiEdit-style tools) may carry `files_modified` in
    // addition to or instead of `file_path`. Collect all distinct paths, dedupe,
    // and append each watchable file to the on-disk reindex queue, then ensure a
    // single drain worker owns the debounce loop. Rapid edits to the same file
    // pile up as queue lines that collapse to one reindex.
    //
    // When the live watcher owns reindexing (`watcher_alive`) or auto-reembed is
    // off, this whole branch is skipped and nothing touches the queue.
    let read_input = if let Some(name) = tool_name
        && matches!(name, "Edit" | "Write" | "NotebookEdit" | "MultiEdit")
        && let Some(cfg) = config.as_ref()
        && cfg.hooks.auto_reembed_on_edit
        && let Some(st) = store.as_ref()
        && !watcher_alive(st)
        && let Some(input) = payload.tool_input.as_ref()
    {
        let enqueued = enqueue_watchable_edits(st, input);
        // Append-before-ensure-worker: the worker must see the entry when it
        // reads the queue. Claim-or-skip, so calling it every time is safe.
        if enqueued > 0 {
            spawn_background_drain_worker(project_root, cfg);
            None
        } else {
            payload.tool_input
        }
    } else {
        payload.tool_input
    };

    // Prefer not to drop any message: index-ready, change-neighbors, and
    // read-neighbors are all surfaced together. Index-ready goes first (most
    // urgent); the rest append in order.
    let index_ready = match (store.as_ref(), config.as_ref()) {
        (Some(st), Some(cfg)) => check_index_ready(st, cfg, "PostToolUse"),
        _ => None,
    };

    let change_neighbors =
        take_change_neighbors_context(store.as_ref(), config.as_ref(), "PostToolUse");

    let read_neighbors = read_surfacing_context(
        store.as_ref(),
        config.as_ref(),
        tool_name,
        read_input.as_ref(),
        "PostToolUse",
    )
    .await;

    Ok(combine_hook_responses(
        "PostToolUse",
        [index_ready, change_neighbors, read_neighbors],
    ))
}

/// Surface code semantically related to a ranged `Read`.
///
/// Fast flag-off path: when `surface_related_on_read` is false (the default),
/// this returns before any store read or cosine scan — flag-off users pay only
/// the process spawn + config load this hook costs.
///
/// Range semantics: `start = offset.unwrap_or(1)`; `end` is `offset + limit - 1`
/// when `limit` is present, else open-ended (window runs to EOF). A chunk's
/// `[line_start, line_end]` overlaps the window when it starts at or before
/// `end` (if bounded) and ends at or after `start`. A full-file read (neither
/// `offset` nor `limit`) is a deliberate noop — surfacing is tied to a focused
/// region the agent narrowed to.
///
/// Reuses [`neighbors`] over the read file's stored vectors — no embedding call.
/// Fail-open: any error behaves as a noop.
async fn read_surfacing_context(
    store: Option<&Store>,
    config: Option<&Config>,
    tool_name: Option<&str>,
    tool_input: Option<&ToolInput>,
    event_name: &str,
) -> Option<Value> {
    if tool_name != Some("Read") {
        return None;
    }
    let cfg = config?;
    if !cfg.hooks.surface_related_on_read {
        return None;
    }

    let input = tool_input?;
    let file_path = input.file_path.as_deref()?;
    // Full-file read (no offset/limit) → noop.
    if input.offset.is_none() && input.limit.is_none() {
        return None;
    }

    let store = store?;
    let project_root = store.project_root();

    // Claude Code typically sends absolute paths. Strip the project root to get
    // a relative path so it matches what the store indexes. Reject anything that
    // escapes the project root (absolute path outside the root, or `..` traversal).
    let relative = project_relative(project_root, file_path)?;
    let read_path = RelativePath::from_path(&relative);
    read_path
        .reject_escape(prompts::hints::READ_INSIDE_PROJECT_DIR)
        .ok()?;

    let start = input.offset.unwrap_or(1);
    let end = input
        .limit
        .map(|count| start.saturating_add(count.saturating_sub(1)));
    // Read-time surfacing rides the hot `Read` path. `read_chunks` deserializes
    // every vector and the cosine scan is O(n²); on a large repo (or while a full
    // reindex holds the write lock) either can block for seconds and stall the
    // session. Bound the whole load+scan in a timeout and skip outright when the
    // corpus is too large to scan cheaply. Fail-open: any elapse/error → noop.
    let all_rows = match tokio::time::timeout(
        Duration::from_millis(READ_SURFACING_TIMEOUT_MS),
        store.read_chunks(),
    )
    .await
    {
        Ok(Ok(rows)) => rows,
        _ => return None,
    };
    if all_rows.len() > READ_SURFACING_MAX_CHUNKS {
        return None;
    }

    let query_vectors: Vec<Vec<f32>> = all_rows
        .iter()
        .filter(|row| row.file_path == read_path.as_str())
        .filter(|row| chunk_overlaps_window(row.line_start, row.line_end, start, end))
        .map(|row| row.vector.clone())
        .collect();
    if query_vectors.is_empty() {
        return None;
    }

    let exclude = read_path.clone();
    let top_k = cfg.hooks.related_top_k;
    let min_similarity = cfg.hooks.related_min_similarity;
    let hits = match tokio::time::timeout(
        Duration::from_millis(READ_SURFACING_TIMEOUT_MS),
        tokio::task::spawn_blocking(move || {
            neighbors(&all_rows, &query_vectors, &exclude, top_k, min_similarity)
        }),
    )
    .await
    {
        Ok(Ok(hits)) => hits,
        _ => return None,
    };
    // Defense in depth on top of index-time pruning: a file deleted out-of-band
    // may still have chunks in the store until the next reindex, so never offer
    // a now-missing file as related code. Cheap: one stat per hit (≤ top_k).
    let hits: Vec<_> = hits
        .into_iter()
        .filter(|n| neighbor_file_exists(project_root, &n.file_path))
        .collect();
    if hits.is_empty() {
        return None;
    }

    let locations: Vec<String> = hits
        .iter()
        .map(|n| {
            prompts::hooks::read_neighbor_line(
                &n.file_path,
                n.line_start,
                n.line_end,
                n.name.as_deref(),
                n.score,
            )
        })
        .collect();

    let context = prompts::hooks::read_related_context(read_path.as_str(), start, end, &locations);

    Some(json!({
        "hookSpecificOutput": {
            "hookEventName": event_name,
            "additionalContext": context,
        }
    }))
}

/// Whether a neighbor's repo-relative file still exists on disk. Guards against
/// surfacing stale chunks for files deleted since they were indexed, and rejects
/// a poisoned/legacy row whose stored path escapes the project root (an absolute
/// path or `..` traversal) so it never yields a filesystem existence oracle
/// outside the project or a misleading agent-visible "related code" path.
fn neighbor_file_exists(project_root: &Path, relative_path: &str) -> bool {
    let relative = RelativePath::new(relative_path);
    if relative
        .reject_escape(prompts::hints::READ_INSIDE_PROJECT_DIR)
        .is_err()
    {
        return false;
    }
    project_root.join(relative.as_str()).exists()
}

/// Whether a chunk's inclusive `[chunk_start, chunk_end]` line span overlaps the
/// read window `[window_start, window_end]`. `window_end == None` is open-ended
/// (runs to EOF), so only the lower bound constrains the chunk.
fn chunk_overlaps_window(
    chunk_start: u32,
    chunk_end: u32,
    window_start: u32,
    window_end: Option<u32>,
) -> bool {
    let starts_in_range = window_end.is_none_or(|end| chunk_start <= end);
    starts_in_range && chunk_end >= window_start
}

/// Read and ack the change-neighbors marker, returning formatted additionalContext.
/// Returns `None` when the marker is absent, feature is disabled, or the store
/// cannot be constructed (fail-open).
pub(super) fn take_change_neighbors_context(
    store: Option<&Store>,
    config: Option<&Config>,
    event_name: &str,
) -> Option<Value> {
    let cfg = config?;
    if !cfg.hooks.surface_related_on_edit {
        return None;
    }
    let store = store?;
    let project_root = store.project_root();
    let marker_path = store.change_neighbors_marker_path();
    let marker = change_neighbors::read_and_remove(&marker_path)?;

    // Per-session dedup: suppress an (edited file → neighbor) pair already
    // surfaced this session so re-editing the same file doesn't repeat the same
    // related code. The ledger is reset on SessionStart. Fail-open: an unreadable
    // ledger reads as empty, so nothing is wrongly suppressed.
    let seen_path = store.change_neighbors_seen_path();
    let seen = change_neighbors::read_seen(&seen_path);

    let mut fresh_keys: Vec<String> = Vec::new();
    let hits: Vec<String> = marker
        .neighbors
        .iter()
        .filter(|n| n.file_path != marker.edited_path)
        .filter(|n| neighbor_file_exists(project_root, &n.file_path))
        .filter(|n| {
            let key =
                change_neighbors::seen_key(&marker.edited_path, &n.file_path, n.name.as_deref());
            if seen.contains(&key) {
                return false;
            }
            fresh_keys.push(key);
            true
        })
        .map(|n| {
            prompts::hooks::edit_neighbor_line(
                &n.file_path,
                n.line_start,
                n.line_end,
                n.name.as_deref(),
                n.score,
            )
        })
        .collect();

    if hits.is_empty() {
        return None;
    }

    // Record only the pairs actually surfaced.
    change_neighbors::append_seen(&seen_path, &fresh_keys);

    let context = prompts::hooks::edit_related_context(&marker.edited_path, &hits);

    Some(json!({
        "hookSpecificOutput": {
            "hookEventName": event_name,
            "additionalContext": context,
        }
    }))
}

/// Merge several hook responses into one `additionalContext`.
///
/// Sources are joined in the order given, so callers pass the most urgent
/// first (index-ready before neighbor surfacing). Absent sources are skipped;
/// all-absent → `None`; a single present source is returned unchanged.
pub(super) fn combine_hook_responses(
    event_name: &str,
    sources: impl IntoIterator<Item = Option<Value>>,
) -> Option<Value> {
    let present: Vec<Value> = sources.into_iter().flatten().collect();
    match present.as_slice() {
        [] => None,
        [single] => Some(single.clone()),
        many => {
            let combined = many
                .iter()
                .map(|value| {
                    value["hookSpecificOutput"]["additionalContext"]
                        .as_str()
                        .unwrap_or("")
                })
                .collect::<Vec<_>>()
                .join("\n");
            Some(json!({
                "hookSpecificOutput": {
                    "hookEventName": event_name,
                    "additionalContext": combined,
                }
            }))
        }
    }
}

pub(super) fn watcher_alive(store: &Store) -> bool {
    crate::store::marker::is_alive(
        &store.watch_marker_path(),
        Duration::from_secs(WATCH_MARKER_STALE_SECS),
    )
}

/// Decide whether a Write/Edit target deserves a background reindex spawn.
///
/// Mirrors the watcher's `WatchFilter::is_watchable` check so PostToolUse
/// doesn't fire a detached `claudix reindex-file` for `.claudix/manifest.json`,
/// `.git/HEAD`, gitignored build artifacts, or paths outside the project root.
/// Fail-open: a failed filter load (`None`) returns `true` so a legitimate
/// edit is still reindexed if the filter setup itself fails.
fn reindex_target_is_watchable(
    canonical_root: &Path,
    filter: Option<&WatchFilter>,
    file_path: &str,
) -> bool {
    let Some(relative) = project_relative(canonical_root, file_path) else {
        return false;
    };
    if relative.as_os_str().is_empty() {
        return false;
    }
    match filter {
        Some(filter) => filter.is_watchable(&relative),
        None => true,
    }
}

/// Append every watchable edited path to the reindex queue, returning how many
/// were enqueued. Fail-open: a store/layout/append failure enqueues fewer (or
/// none) — the miss is caught by the next edit to that file or a manifest-age
/// full reindex — and never breaks the hook.
///
/// Paths are keyed on their canonical project-relative spelling (via
/// [`canonical_dedup_key`]) so the same file arriving as an absolute path in one
/// edit and a project-relative path in another coalesces to a single queue path,
/// and the drain worker's `reindex_file` consumes the relative spelling directly.
fn enqueue_watchable_edits(store: &Store, input: &ToolInput) -> usize {
    if store.ensure_layout().is_err() {
        return 0;
    }
    let root = store.project_root();
    // The filter parses three ignore files; load it once per event, not per
    // edited path. A failed load fail-opens inside the watchable check.
    let filter = WatchFilter::load(root).ok();
    let queue_path = store.reindex_queue_path();
    reindex_paths_from_input(root, input)
        .into_iter()
        .filter(|p| reindex_target_is_watchable(root, filter.as_ref(), p))
        .filter(|p| reindex_queue::append(&queue_path, &canonical_dedup_key(root, p)))
        .count()
}

/// Collect the distinct file paths that a tool input targets for reindexing.
///
/// Merges `file_path`, `notebook_path`, and the `files_modified` list (for
/// MultiEdit-style payloads). Duplicates are dropped; order is preserved.
///
/// Dedup keys on a canonical project-relative spelling (separators normalized,
/// absolute paths stripped to the project root), so the same file arriving as
/// both an absolute and a project-relative path spawns exactly one reindex
/// instead of racing two `drop_table`→`add` writers on the chunks table. The
/// original spelling is preserved in the output — the canonical form is only the
/// dedup key — so the downstream watchable check still sees what Claude sent.
fn reindex_paths_from_input(project_root: &Path, input: &ToolInput) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut paths = Vec::new();

    let singles = input
        .file_path
        .as_deref()
        .into_iter()
        .chain(input.notebook_path.as_deref());

    let multi = input
        .files_modified
        .as_deref()
        .unwrap_or(&[])
        .iter()
        .map(String::as_str);

    for p in singles.chain(multi) {
        if seen.insert(canonical_dedup_key(project_root, p)) {
            paths.push(p.to_owned());
        }
    }
    paths
}

/// Canonical project-relative dedup key for a tool-input path. Strips an
/// absolute path to the project root via [`project_relative`] and normalizes
/// separators via [`RelativePath`]. A path outside the project (or one that
/// fails to strip) keys on its own normalized spelling — it is never reindexed
/// anyway, so a unique key is harmless.
fn canonical_dedup_key(canonical_root: &Path, file_path: &str) -> String {
    let relative = project_relative(canonical_root, file_path)
        .unwrap_or_else(|| Path::new(file_path).to_path_buf());
    RelativePath::from_path(&relative).as_str().to_owned()
}

/// Project-relative form of a tool-input path against an already-canonical
/// project root (`Store::new` canonicalizes; hook entry canonicalizes too).
/// The raw path is canonicalized so a symlinked prefix (macOS `/tmp` →
/// `/private/tmp`) still strip-prefix-matches. `None` when the path is
/// absolute but outside the project root.
fn project_relative(canonical_root: &Path, file_path: &str) -> Option<PathBuf> {
    let raw = Path::new(file_path);
    if raw.is_absolute() {
        let raw_canonical = raw.canonicalize();
        let raw_absolute = raw_canonical.as_deref().unwrap_or(raw);
        raw_absolute
            .strip_prefix(canonical_root)
            .ok()
            .map(Path::to_path_buf)
    } else {
        Some(raw.to_path_buf())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::fs;

    use serde_json::json;

    use crate::config::Config;
    use crate::hooks::{HookEvent, run};
    use crate::store::{Manifest, Store};

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

    // ── reindex_paths_from_input ──────────────────────────────────────────────

    fn make_tool_input(file_path: Option<&str>, files_modified: Option<Vec<&str>>) -> ToolInput {
        let json = serde_json::json!({
            "file_path": file_path,
            "files_modified": files_modified.map(|v| v.into_iter().collect::<Vec<_>>()),
        });
        serde_json::from_value(json).expect("must parse")
    }

    #[test]
    fn reindex_paths_only_file_path() {
        let input = make_tool_input(Some("src/lib.rs"), None);
        assert_eq!(
            reindex_paths_from_input(Path::new("/proj"), &input),
            vec!["src/lib.rs"]
        );
    }

    #[test]
    fn reindex_paths_only_files_modified() {
        let input = make_tool_input(None, Some(vec!["src/lib.rs", "src/main.rs"]));
        assert_eq!(
            reindex_paths_from_input(Path::new("/proj"), &input),
            vec!["src/lib.rs", "src/main.rs"]
        );
    }

    #[test]
    fn reindex_paths_both_deduped() {
        // file_path appears in files_modified too — must appear exactly once.
        let input = make_tool_input(Some("src/lib.rs"), Some(vec!["src/lib.rs", "src/main.rs"]));
        let paths = reindex_paths_from_input(Path::new("/proj"), &input);
        assert_eq!(paths, vec!["src/lib.rs", "src/main.rs"]);
    }

    #[test]
    fn reindex_paths_neither_field_is_empty() {
        let input = make_tool_input(None, None);
        assert!(reindex_paths_from_input(Path::new("/proj"), &input).is_empty());
    }

    #[test]
    fn reindex_paths_abs_and_relative_spelling_dedup_to_one() {
        // The same file arriving as an absolute path in file_path and a
        // project-relative path in files_modified must spawn exactly once.
        let fixture = TestFixture::new("small_rust").unwrap_or_else(|_| unreachable!());
        let abs = fixture.root().join("src/math.rs");
        let abs = abs.to_string_lossy().into_owned();
        let input = make_tool_input(Some(&abs), Some(vec!["src/math.rs"]));
        let paths = reindex_paths_from_input(fixture.root(), &input);
        assert_eq!(
            paths.len(),
            1,
            "abs + project-relative spelling of one file must dedup, got: {paths:?}"
        );
    }

    #[tokio::test]
    async fn post_tool_use_enqueues_reindex_and_returns_none() {
        let fixture = TestFixture::new("small_rust");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());
        let config = stub_config();
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

        // The edited file must land in the queue under its project-relative
        // spelling, ready for the drain worker.
        let store = Store::new(fixture.root(), &config).unwrap_or_else(|_| unreachable!());
        let entries =
            reindex_queue::parse_entries(&reindex_queue::read_content(&store.reindex_queue_path()));
        let paths: Vec<&str> = entries.iter().map(|e| e.path.as_str()).collect();
        assert_eq!(paths, vec!["src/math.rs"]);
    }

    #[tokio::test]
    async fn repeated_edits_coalesce_to_one_queue_path() {
        let fixture = TestFixture::new("small_rust").unwrap_or_else(|_| unreachable!());
        let config = stub_config();
        write_config(fixture.root(), &config);

        let payload = json!({
            "tool_name": "Write",
            "tool_input": { "file_path": fixture.root().join("src/math.rs") }
        });
        // Three rapid edits to the same file.
        for _ in 0..3 {
            let response = run(fixture.root(), HookEvent::PostToolUse, &payload.to_string()).await;
            assert!(response.is_ok());
        }

        let store = Store::new(fixture.root(), &config).unwrap_or_else(|_| unreachable!());
        let entries =
            reindex_queue::parse_entries(&reindex_queue::read_content(&store.reindex_queue_path()));
        let windows = reindex_queue::collapse(&entries);
        assert_eq!(
            windows.len(),
            1,
            "three edits of one file must collapse to a single queued reindex"
        );
        assert!(windows.contains_key("src/math.rs"));
    }

    #[test]
    fn reindex_target_is_watchable_rejects_index_internal_paths() {
        let fixture = TestFixture::new("small_rust").unwrap_or_else(|_| unreachable!());
        // Callers pass the store's canonical root; mirror that here so a
        // symlinked temp prefix (macOS) still strip-prefix-matches.
        let root = fixture
            .root()
            .canonicalize()
            .unwrap_or_else(|_| fixture.root().to_path_buf());
        let filter = WatchFilter::load(&root).ok();
        assert!(reindex_target_is_watchable(
            &root,
            filter.as_ref(),
            "src/math.rs"
        ));
        assert!(!reindex_target_is_watchable(
            &root,
            filter.as_ref(),
            ".claudix/manifest.json"
        ));
        assert!(!reindex_target_is_watchable(
            &root,
            filter.as_ref(),
            ".git/HEAD"
        ));
        // Outside the project root: claude code generally resolves to absolute
        // paths inside CLAUDE_PROJECT_DIR, but defend in depth.
        let absolute_outside = std::env::temp_dir().join("nope.rs");
        assert!(!reindex_target_is_watchable(
            &root,
            filter.as_ref(),
            &absolute_outside.to_string_lossy(),
        ));
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

        // auto_reembed_on_edit = false must not touch the queue at all.
        let store = Store::new(fixture.root(), &config).unwrap_or_else(|_| unreachable!());
        assert!(
            reindex_queue::read_content(&store.reindex_queue_path()).is_empty(),
            "disabled auto-reembed must enqueue nothing"
        );
    }

    #[tokio::test]
    async fn watcher_alive_bypasses_the_queue() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let config = stub_config();
        write_config(fixture.root(), &config);
        let store = Store::new(fixture.root(), &config)?;
        store.ensure_layout()?;
        // A live watch marker (our own pid) means the watcher owns reindexing;
        // the hook must not append to the queue.
        fs::write(store.watch_marker_path(), std::process::id().to_string())?;

        let payload = json!({
            "tool_name": "Write",
            "tool_input": { "file_path": fixture.root().join("src/math.rs") }
        });
        let response = run(fixture.root(), HookEvent::PostToolUse, &payload.to_string()).await?;
        assert!(response.is_none());
        assert!(
            reindex_queue::read_content(&store.reindex_queue_path()).is_empty(),
            "a live watcher must bypass the queue entirely"
        );
        Ok(())
    }

    #[tokio::test]
    async fn user_prompt_submit_surfaces_indexing_completion() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let config = stub_config();
        write_config(fixture.root(), &config);
        let store = Store::new(fixture.root(), &config)?;
        store.ensure_layout()?;

        // Marker says the prior index timestamp was "none"; the manifest now
        // has a fresh successful timestamp, so the handler must surface the
        // completion message even though no write tool fired.
        let stale_created_at = "2025-01-01T00:00:00Z";
        let payload = format!("none\n{stale_created_at}\n0\n");
        fs::write(store.pending_index_marker_path(), payload)?;
        let mut manifest = Manifest::new(config.embedding.model.clone(), 8);
        manifest.file_count = 3;
        manifest.chunk_count = 12;
        manifest.last_full_index_at = Some(crate::util::now_rfc3339());
        store.write_manifest(&manifest)?;

        let response = run(fixture.root(), HookEvent::UserPromptSubmit, "{}").await?;
        let response = response.unwrap_or(Value::Null);
        assert_eq!(
            response["hookSpecificOutput"]["hookEventName"].as_str(),
            Some("UserPromptSubmit"),
            "response must carry the firing event name"
        );
        let context = response["hookSpecificOutput"]["additionalContext"]
            .as_str()
            .unwrap_or_default();
        assert!(
            context.contains("indexing complete"),
            "expected completion context, got: {context}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn watcher_alive_reports_live_marker() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let config = stub_config();
        let store = Store::new(fixture.root(), &config)?;
        store.ensure_layout()?;
        let marker_path = store.watch_marker_path();
        fs::write(&marker_path, std::process::id().to_string())?;

        assert!(
            watcher_alive(&store),
            "current-PID watch marker must register as alive"
        );
        Ok(())
    }

    #[tokio::test]
    async fn watcher_alive_returns_false_without_marker() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let config = stub_config();
        let store = Store::new(fixture.root(), &config)?;
        assert!(!watcher_alive(&store));
        Ok(())
    }

    // ── change-neighbors surfacing ──────────────────────────────────────────

    fn write_neighbors_marker(
        store: &Store,
        edited_path: &str,
        neighbors: Vec<crate::store::marker::change_neighbors::NeighborEntry>,
    ) {
        use crate::store::marker::change_neighbors::{ChangeNeighborsMarker, write};
        let marker = ChangeNeighborsMarker {
            edited_path: edited_path.to_owned(),
            neighbors,
        };
        write(&store.change_neighbors_marker_path(), &marker);
    }

    fn make_neighbor_entry(
        file_path: &str,
        name: &str,
        score: f32,
    ) -> crate::store::marker::change_neighbors::NeighborEntry {
        crate::store::marker::change_neighbors::NeighborEntry {
            file_path: file_path.to_owned(),
            line_start: 10,
            line_end: 25,
            name: Some(name.to_owned()),
            score,
        }
    }

    #[tokio::test]
    async fn neighbors_marker_surfaces_related_file_in_additional_context() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let config = stub_config();
        write_config(fixture.root(), &config);
        let store = Store::new(fixture.root(), &config)?;
        store.ensure_layout()?;

        write_neighbors_marker(
            &store,
            "src/lib.rs",
            vec![make_neighbor_entry("src/math.rs", "add", 0.82)],
        );

        let response = take_change_neighbors_context(Some(&store), Some(&config), "PostToolUse");
        let response = response.unwrap_or(serde_json::Value::Null);
        let context = response["hookSpecificOutput"]["additionalContext"]
            .as_str()
            .unwrap_or_default();

        assert!(
            context.contains("src/math.rs"),
            "neighbor file must appear in additionalContext, got: {context}"
        );
        assert!(
            !context.contains("src/lib.rs:") || context.contains("edit of `src/lib.rs`"),
            "edited file must not appear as a hit in context, got: {context}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn neighbors_marker_acked_on_read() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let config = stub_config();
        write_config(fixture.root(), &config);
        let store = Store::new(fixture.root(), &config)?;
        store.ensure_layout()?;

        write_neighbors_marker(
            &store,
            "src/lib.rs",
            vec![make_neighbor_entry("src/math.rs", "add", 0.80)],
        );

        assert!(
            store.change_neighbors_marker_path().exists(),
            "marker must exist before read"
        );

        let _ = take_change_neighbors_context(Some(&store), Some(&config), "PostToolUse");

        assert!(
            !store.change_neighbors_marker_path().exists(),
            "marker must be removed after being read (ack)"
        );
        Ok(())
    }

    #[tokio::test]
    async fn no_neighbors_marker_produces_no_context() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let config = stub_config();
        write_config(fixture.root(), &config);
        let store = Store::new(fixture.root(), &config)?;
        store.ensure_layout()?;

        // No marker written — must produce None.
        let response = take_change_neighbors_context(Some(&store), Some(&config), "PostToolUse");
        assert!(
            response.is_none(),
            "absent marker must produce no additionalContext"
        );
        Ok(())
    }

    #[tokio::test]
    async fn surface_related_on_edit_false_suppresses_context() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let mut config = stub_config();
        config.hooks.surface_related_on_edit = false;
        write_config(fixture.root(), &config);
        let store = Store::new(fixture.root(), &config)?;
        store.ensure_layout()?;

        write_neighbors_marker(
            &store,
            "src/lib.rs",
            vec![make_neighbor_entry("src/math.rs", "add", 0.82)],
        );

        let response = take_change_neighbors_context(Some(&store), Some(&config), "PostToolUse");
        assert!(
            response.is_none(),
            "surface_related_on_edit = false must suppress output even when marker is present"
        );
        Ok(())
    }

    #[tokio::test]
    async fn edited_file_never_surfaced_as_own_neighbor() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let config = stub_config();
        write_config(fixture.root(), &config);
        let store = Store::new(fixture.root(), &config)?;
        store.ensure_layout()?;

        // Simulate a marker where the only neighbor IS the edited file (must not happen in
        // practice but the hook layer must not surface it either way).
        write_neighbors_marker(
            &store,
            "src/lib.rs",
            vec![
                make_neighbor_entry("src/lib.rs", "greet", 0.99), // same as edited
                make_neighbor_entry("src/math.rs", "add", 0.80),
            ],
        );

        let response = take_change_neighbors_context(Some(&store), Some(&config), "PostToolUse");
        let response = response.unwrap_or(serde_json::Value::Null);
        let context = response["hookSpecificOutput"]["additionalContext"]
            .as_str()
            .unwrap_or_default();

        assert!(
            context.contains("src/math.rs"),
            "non-edited neighbor must be in context, got: {context}"
        );
        // The hook layer must filter out the edited file even if the marker
        // somehow carries it (defense in depth on top of neighbors() exclusion).
        assert!(
            !context.contains("src/lib.rs:"),
            "edited file must not appear as a hit in context, got: {context}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn repeated_identical_neighbor_pair_is_suppressed() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let config = stub_config();
        write_config(fixture.root(), &config);
        let store = Store::new(fixture.root(), &config)?;
        store.ensure_layout()?;

        // First take surfaces the (src/lib.rs → src/math.rs) pair.
        write_neighbors_marker(
            &store,
            "src/lib.rs",
            vec![make_neighbor_entry("src/math.rs", "add", 0.82)],
        );
        let response = take_change_neighbors_context(Some(&store), Some(&config), "PostToolUse");
        let response = response.unwrap_or(serde_json::Value::Null);
        let context = response["hookSpecificOutput"]["additionalContext"]
            .as_str()
            .unwrap_or_default();
        assert!(
            context.contains("src/math.rs"),
            "first take must surface the neighbor, got: {context}"
        );

        // read_and_remove deletes the marker on each take, so re-write the SAME
        // marker before the second take.
        write_neighbors_marker(
            &store,
            "src/lib.rs",
            vec![make_neighbor_entry("src/math.rs", "add", 0.82)],
        );
        let response = take_change_neighbors_context(Some(&store), Some(&config), "PostToolUse");
        assert!(
            response.is_none(),
            "identical (edited → neighbor) pair already surfaced this session must be suppressed"
        );
        Ok(())
    }

    #[tokio::test]
    async fn same_neighbor_via_different_edited_file_still_surfaces() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let config = stub_config();
        write_config(fixture.root(), &config);
        let store = Store::new(fixture.root(), &config)?;
        store.ensure_layout()?;

        // First: edited=src/lib.rs → src/math.rs, surfaces and records the key.
        write_neighbors_marker(
            &store,
            "src/lib.rs",
            vec![make_neighbor_entry("src/math.rs", "add", 0.82)],
        );
        let _ = take_change_neighbors_context(Some(&store), Some(&config), "PostToolUse");

        // Same neighbor, different edited file → different dedup key → surfaces.
        write_neighbors_marker(
            &store,
            "src/other.rs",
            vec![make_neighbor_entry("src/math.rs", "add", 0.82)],
        );
        let response = take_change_neighbors_context(Some(&store), Some(&config), "PostToolUse");
        let response = response.unwrap_or(serde_json::Value::Null);
        let context = response["hookSpecificOutput"]["additionalContext"]
            .as_str()
            .unwrap_or_default();
        assert!(
            context.contains("src/math.rs"),
            "same neighbor via a different edited file must still surface, got: {context}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn session_start_clears_seen_ledger() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let config = stub_config();
        write_config(fixture.root(), &config);
        let store = Store::new(fixture.root(), &config)?;
        store.ensure_layout()?;

        // Seed a dummy key into the per-session dedup ledger.
        let seen_path = store.change_neighbors_seen_path();
        fs::write(&seen_path, "src/lib.rs\tsrc/math.rs\tadd\n")?;
        assert!(seen_path.exists(), "ledger must exist before SessionStart");

        run(fixture.root(), HookEvent::SessionStart, "{}").await?;

        assert!(
            !seen_path.exists(),
            "SessionStart must reset the dedup ledger"
        );
        Ok(())
    }

    #[tokio::test]
    async fn combine_both_messages_when_both_present() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let config = stub_config();
        write_config(fixture.root(), &config);
        let store = Store::new(fixture.root(), &config)?;
        store.ensure_layout()?;

        // Set up index-ready marker.
        let stale_created_at = "2025-01-01T00:00:00Z";
        let payload = format!("none\n{stale_created_at}\n0\n");
        fs::write(store.pending_index_marker_path(), payload)?;
        let mut manifest = Manifest::new(config.embedding.model.clone(), 8);
        manifest.file_count = 1;
        manifest.chunk_count = 2;
        manifest.last_full_index_at = Some(crate::util::now_rfc3339());
        store.write_manifest(&manifest)?;

        // Set up neighbors marker.
        write_neighbors_marker(
            &store,
            "src/lib.rs",
            vec![make_neighbor_entry("src/math.rs", "add", 0.80)],
        );

        let response = run(fixture.root(), HookEvent::PostToolUse, "{}").await?;
        let response = response.unwrap_or(serde_json::Value::Null);
        let context = response["hookSpecificOutput"]["additionalContext"]
            .as_str()
            .unwrap_or_default();

        assert!(
            context.contains("indexing complete"),
            "combined context must include index-ready message, got: {context}"
        );
        assert!(
            context.contains("src/math.rs"),
            "combined context must include neighbor file, got: {context}"
        );
        Ok(())
    }

    // ── read-time surfacing ─────────────────────────────────────────────────

    /// Seed the store directly with chunks at explicit line ranges and vectors,
    /// bypassing embedding so cosine similarities are fully controlled.
    async fn seed_rows(store: &Store, config: &Config, rows: &[(&str, &str, u32, u32, Vec<f32>)]) {
        use crate::types::{
            ByteRange, Chunk, ChunkId, ChunkKind, EmbeddedChunk, FileHash, Language, LineRange,
            RelativePath,
        };
        // Materialize each referenced file on disk so the neighbor-existence
        // guard (which drops surfaced neighbors whose file was deleted) treats
        // them as live — these tests assert real files get surfaced.
        for (path, ..) in rows {
            let abs = store.project_root().join(path);
            if let Some(parent) = abs.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            let _ = std::fs::write(&abs, "// seeded\n");
        }
        let embedded: Vec<EmbeddedChunk> = rows
            .iter()
            .enumerate()
            .map(|(i, (path, name, start, end, vector))| {
                let chunk = Chunk {
                    id: ChunkId(i as u64 + 1),
                    file_path: RelativePath::new(*path),
                    language: Language::Rust,
                    kind: ChunkKind::Function,
                    name: Some((*name).to_owned()),
                    line_range: LineRange {
                        start: *start,
                        end: *end,
                    },
                    byte_range: ByteRange { start: 0, end: 50 },
                    file_hash: FileHash([0u8; 16]),
                    content: format!("pub fn {name}() {{}}"),
                };
                EmbeddedChunk {
                    chunk,
                    vector: vector.clone(),
                }
            })
            .collect();
        assert!(store.replace_chunks(&embedded, config).await.is_ok());
    }

    fn read_config_on() -> Config {
        let mut config = stub_config();
        config.hooks.surface_related_on_read = true;
        config.hooks.related_top_k = 5;
        config.hooks.related_min_similarity = 0.5;
        config
    }

    #[test]
    fn chunk_overlaps_window_respects_bounds() {
        // Bounded window [10, 20].
        assert!(
            chunk_overlaps_window(8, 12, 10, Some(20)),
            "straddles start"
        );
        assert!(chunk_overlaps_window(15, 18, 10, Some(20)), "inside window");
        assert!(chunk_overlaps_window(18, 25, 10, Some(20)), "straddles end");
        assert!(!chunk_overlaps_window(1, 9, 10, Some(20)), "ends before");
        assert!(!chunk_overlaps_window(21, 30, 10, Some(20)), "starts after");
        // Open-ended window [10, EOF]: only the lower bound constrains.
        assert!(chunk_overlaps_window(50, 60, 10, None), "far below EOF");
        assert!(!chunk_overlaps_window(1, 9, 10, None), "ends before start");
        assert!(
            chunk_overlaps_window(u32::MAX, u32::MAX, u32::MAX, Some(u32::MAX)),
            "saturating read windows still match the last line"
        );
    }

    #[test]
    fn read_surfacing_defaults_off() {
        assert!(!Config::default().hooks.surface_related_on_read);
    }

    #[tokio::test]
    async fn ranged_read_surfaces_related_neighbor_naming_region() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let config = read_config_on();
        write_config(fixture.root(), &config);
        let store = Store::new(fixture.root(), &config)?;
        store.ensure_layout()?;

        // The read file's chunk at lines 10-20 shares a near-duplicate vector
        // with a chunk in another file; an unrelated chunk sits orthogonal.
        seed_rows(
            &store,
            &config,
            &[
                (
                    "src/foo.rs",
                    "target",
                    10,
                    20,
                    vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ),
                (
                    "src/bar.rs",
                    "twin",
                    40,
                    58,
                    vec![0.99, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ),
                (
                    "src/other.rs",
                    "unrelated",
                    1,
                    5,
                    vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ),
            ],
        )
        .await;

        let payload = json!({
            "tool_name": "Read",
            "tool_input": { "file_path": "src/foo.rs", "offset": 12, "limit": 6 }
        });
        let response = run(fixture.root(), HookEvent::PostToolUse, &payload.to_string()).await?;
        let response = response.unwrap_or(Value::Null);
        let context = response["hookSpecificOutput"]["additionalContext"]
            .as_str()
            .unwrap_or_default();

        assert!(
            context.contains("src/bar.rs"),
            "neighbor file must be surfaced, got: {context}"
        );
        assert!(
            context.contains("lines 12-17"),
            "context must name the read region, got: {context}"
        );
        assert!(
            !context.contains("src/foo.rs:"),
            "read file must not be listed as its own neighbor, got: {context}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn ranged_read_absolute_path_surfaces_neighbor() -> Result<()> {
        // Claude Code sends absolute paths in tool_input.file_path; verify they
        // are resolved to relative before matching against the store.
        let fixture = TestFixture::new("small_rust")?;
        let config = read_config_on();
        write_config(fixture.root(), &config);
        let store = Store::new(fixture.root(), &config)?;
        store.ensure_layout()?;

        seed_rows(
            &store,
            &config,
            &[
                (
                    "src/foo.rs",
                    "target",
                    10,
                    20,
                    vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ),
                (
                    "src/bar.rs",
                    "twin",
                    40,
                    58,
                    vec![0.99, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ),
            ],
        )
        .await;

        // Use the absolute path as Claude Code would supply it.
        let abs_path = fixture.root().join("src/foo.rs");
        let payload = json!({
            "tool_name": "Read",
            "tool_input": { "file_path": abs_path, "offset": 12, "limit": 6 }
        });
        let response = run(fixture.root(), HookEvent::PostToolUse, &payload.to_string()).await?;
        let response = response.unwrap_or(Value::Null);
        let context = response["hookSpecificOutput"]["additionalContext"]
            .as_str()
            .unwrap_or_default();

        assert!(
            context.contains("src/bar.rs"),
            "absolute-path read must still surface the neighbor, got: {context}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn deleted_neighbor_is_not_surfaced_on_read() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let config = read_config_on();
        write_config(fixture.root(), &config);
        let store = Store::new(fixture.root(), &config)?;
        store.ensure_layout()?;

        // seed_rows materializes both files on disk; the neighbor (bar.rs) is
        // then deleted out-of-band so only its now-stale chunk remains indexed.
        seed_rows(
            &store,
            &config,
            &[
                (
                    "src/foo.rs",
                    "target",
                    10,
                    20,
                    vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ),
                (
                    "src/bar.rs",
                    "twin",
                    40,
                    58,
                    vec![0.99, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ),
            ],
        )
        .await;
        assert!(fs::remove_file(store.project_root().join("src/bar.rs")).is_ok());

        let payload = json!({
            "tool_name": "Read",
            "tool_input": { "file_path": "src/foo.rs", "offset": 12, "limit": 6 }
        });
        let response = run(fixture.root(), HookEvent::PostToolUse, &payload.to_string()).await?;
        assert!(
            response.is_none(),
            "a neighbor whose file was deleted on disk must not be surfaced"
        );
        Ok(())
    }

    #[tokio::test]
    async fn full_file_read_is_noop() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let config = read_config_on();
        write_config(fixture.root(), &config);
        let store = Store::new(fixture.root(), &config)?;
        store.ensure_layout()?;

        seed_rows(
            &store,
            &config,
            &[
                (
                    "src/foo.rs",
                    "target",
                    10,
                    20,
                    vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ),
                (
                    "src/bar.rs",
                    "twin",
                    40,
                    58,
                    vec![0.99, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ),
            ],
        )
        .await;

        // No offset or limit → whole-file read → noop.
        let payload = json!({
            "tool_name": "Read",
            "tool_input": { "file_path": "src/foo.rs" }
        });
        let response = run(fixture.root(), HookEvent::PostToolUse, &payload.to_string()).await?;
        assert!(
            response.is_none(),
            "full-file read must not surface neighbors"
        );
        Ok(())
    }

    #[tokio::test]
    async fn ranged_read_with_nothing_related_is_noop() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let config = read_config_on();
        write_config(fixture.root(), &config);
        let store = Store::new(fixture.root(), &config)?;
        store.ensure_layout()?;

        // The read chunk's neighbor sits below the 0.5 floor (orthogonal vector).
        seed_rows(
            &store,
            &config,
            &[
                (
                    "src/foo.rs",
                    "target",
                    10,
                    20,
                    vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ),
                (
                    "src/bar.rs",
                    "stranger",
                    40,
                    58,
                    vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ),
            ],
        )
        .await;

        let payload = json!({
            "tool_name": "Read",
            "tool_input": { "file_path": "src/foo.rs", "offset": 12, "limit": 6 }
        });
        let response = run(fixture.root(), HookEvent::PostToolUse, &payload.to_string()).await?;
        assert!(response.is_none(), "no neighbor clears the floor → noop");
        Ok(())
    }

    #[tokio::test]
    async fn window_missing_chunk_span_yields_no_query_vectors() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let config = read_config_on();
        write_config(fixture.root(), &config);
        let store = Store::new(fixture.root(), &config)?;
        store.ensure_layout()?;

        // Read file's only chunk spans 10-20; a near-duplicate exists elsewhere.
        // The read window 1-5 misses the chunk entirely → no query vectors → noop,
        // even though a strong neighbor exists.
        seed_rows(
            &store,
            &config,
            &[
                (
                    "src/foo.rs",
                    "target",
                    10,
                    20,
                    vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ),
                (
                    "src/bar.rs",
                    "twin",
                    40,
                    58,
                    vec![0.99, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ),
            ],
        )
        .await;

        let payload = json!({
            "tool_name": "Read",
            "tool_input": { "file_path": "src/foo.rs", "offset": 1, "limit": 5 }
        });
        let response = run(fixture.root(), HookEvent::PostToolUse, &payload.to_string()).await?;
        assert!(
            response.is_none(),
            "window missing the chunk span must surface nothing"
        );
        Ok(())
    }

    #[tokio::test]
    async fn read_surfacing_disabled_skips_store() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let mut config = stub_config();
        config.hooks.surface_related_on_read = false;
        write_config(fixture.root(), &config);
        let store = Store::new(fixture.root(), &config)?;
        store.ensure_layout()?;
        seed_rows(
            &store,
            &config,
            &[
                (
                    "src/foo.rs",
                    "target",
                    10,
                    20,
                    vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ),
                (
                    "src/bar.rs",
                    "twin",
                    40,
                    58,
                    vec![0.99, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ),
            ],
        )
        .await;

        let payload = json!({
            "tool_name": "Read",
            "tool_input": { "file_path": "src/foo.rs", "offset": 12, "limit": 6 }
        });
        let response = run(fixture.root(), HookEvent::PostToolUse, &payload.to_string()).await?;
        assert!(
            response.is_none(),
            "flag off must surface nothing on a ranged read"
        );
        // The reindex invariant must hold regardless of the read flag: a Read
        // never triggers a background reindex — no change-neighbors marker
        // appears (that is only written by the reindex path).
        assert!(
            !store.change_neighbors_marker_path().exists(),
            "Read must never trigger a reindex"
        );
        Ok(())
    }
}
