use std::path::{Path, PathBuf};
use std::time::Duration;

use serde_json::{Value, json};

use crate::config::{self, Config};
use crate::enumeration::WatchFilter;
use crate::error::Result;
use crate::prompts;
use crate::search::neighbors::{Neighbor, neighbors};
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

/// Claude Code's Read tool advertises a per-call cap of 2000 lines (its tool
/// description; the hook payload carries no truncation notice). The read ledger
/// never records beyond it: an over-long claim would suppress hints for lines
/// the session never saw. Under-claiming only costs a repeated hint (fail-open).
const READ_TOOL_MAX_LINES: u32 = 2000;

pub(super) async fn handle_post_tool_use(
    project_root: &Path,
    payload: HookPayload,
) -> Result<Option<Value>> {
    let config = config::load(project_root).ok();

    // `Read` rides this hook too (see hooks.json matcher) for read-time
    // surfacing and read-tracking — it must NEVER spawn a reindex. The edit
    // tools below do.
    let tool_name = payload.tool_name.as_deref();

    // With both surfacing flags off a Read event has nothing to record and
    // nothing to surface: keep the early return so those users pay only the
    // process spawn + config load. With either flag on, the event records what
    // the session read (the read ledger feeds both surfacing paths' dedupe).
    if tool_name == Some("Read")
        && !config.as_ref().is_some_and(|cfg| {
            cfg.hooks.surface_related_on_read || cfg.hooks.surface_related_on_edit
        })
    {
        return Ok(None);
    }

    // One Store for the whole event: every branch below needs its paths, and
    // `Store::new` canonicalizes the root — building it per check multiplies
    // that syscall cost on the busiest hook. `None` fail-opens every branch.
    let store = config
        .as_ref()
        .and_then(|cfg| Store::new(project_root, cfg).ok());

    let session_id = payload.session_id.as_deref();

    // Files this event itself edited (any spelling: file_path, notebook_path,
    // or the files_modified list). The change-neighbors ack compares these
    // against the marker's edited_path, so "your edit" is only ever claimed by
    // the event that actually edited that file.
    let event_files: Vec<String> = payload
        .tool_input
        .as_ref()
        .map(|input| {
            input
                .file_path
                .iter()
                .chain(input.notebook_path.iter())
                .chain(input.files_modified.iter().flatten())
                .cloned()
                .collect()
        })
        .unwrap_or_default();

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
        let enqueued = enqueue_watchable_edits(st, input, session_id);
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

    // Track what this Read event showed the agent, so both surfacing paths can
    // skip hints naming lines the session already has. Attributed events only:
    // an unattributed record could suppress another session's hints.
    if let (Some(st), Some(input), Some(session)) =
        (store.as_ref(), read_input.as_ref(), session_id)
        && tool_name == Some("Read")
    {
        record_read_range(st, session, input);
    }

    // Prefer not to drop any message: index-ready, change-neighbors, and
    // read-neighbors are all surfaced together. Index-ready goes first (most
    // urgent); the rest append in order.
    let index_ready = match (store.as_ref(), config.as_ref()) {
        (Some(st), Some(cfg)) => check_index_ready(
            st,
            cfg,
            "PostToolUse",
            session_id,
            payload.prompt_id.as_deref(),
        ),
        _ => None,
    };

    let change_neighbors = take_change_neighbors_context(
        store.as_ref(),
        config.as_ref(),
        "PostToolUse",
        session_id,
        &event_files,
    );

    let read_neighbors = read_surfacing_context(
        store.as_ref(),
        config.as_ref(),
        tool_name,
        read_input.as_ref(),
        "PostToolUse",
        session_id,
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
    session_id: Option<&str>,
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
    // Read-time surfacing rides the hot `Read` path. The scan needs vectors,
    // not chunk text, so it reads rows without the text column; the cosine scan
    // is still O(n²) and, on a large repo (or while a full reindex holds the
    // write lock), either the load or the scan can block for seconds and stall
    // the session. Bound the whole load+scan in a timeout and skip outright when
    // the corpus is too large to scan cheaply. Fail-open: any elapse/error → noop.
    let all_rows = match tokio::time::timeout(
        Duration::from_millis(READ_SURFACING_TIMEOUT_MS),
        store.read_chunks_without_content(),
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
    // Same effective-floor resolution as the edit path: the index's stored
    // corpus floor when present, else the configured floor.
    let stored_floor = store
        .read_manifest()
        .ok()
        .flatten()
        .and_then(|manifest| manifest.similarity_floor);
    let min_similarity =
        config::resolve_similarity_floor(stored_floor, cfg.hooks.related_min_similarity);
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
    //
    // Then the two line-aware dedupe layers: a hint the session was already
    // shown (seen ledger), and a hint pointing at lines the session already
    // Read (read ledger). Without a session id neither ledger is scoped, so
    // nothing is suppressed — fail-open toward a repeated hint.
    let seen_path = session_id.map(|id| store.change_neighbors_seen_path(id));
    let seen = seen_path
        .as_deref()
        .map_or_else(Vec::new, change_neighbors::read_seen);
    let read_ledger_path = session_id.map(|id| store.change_neighbors_read_path(id));
    let read = read_ledger_path
        .as_deref()
        .map_or_else(Vec::new, change_neighbors::read_ranges);

    let hits: Vec<_> = hits
        .into_iter()
        .filter(|n| neighbor_file_exists(project_root, &n.file_path))
        .filter(|n| {
            !change_neighbors::is_seen(
                &seen,
                &n.file_path,
                n.name.as_deref(),
                n.line_start,
                n.line_end,
            )
        })
        .filter(|n| !change_neighbors::is_read(&read, &n.file_path, n.line_start, n.line_end))
        .collect();
    if hits.is_empty() {
        return None;
    }

    // Record only the neighbors actually surfaced, so the edit-time ack's
    // seen-check skips them too. Best-effort: a write failure risks a repeat.
    if let Some(path) = seen_path.as_deref() {
        change_neighbors::append_seen(
            path,
            &hits
                .iter()
                .map(|n| {
                    change_neighbors::SeenEntry::new(
                        &n.file_path,
                        n.name.as_deref(),
                        n.line_start,
                        n.line_end,
                    )
                })
                .collect::<Vec<_>>(),
        );
    }

    // Source hits and doc hits render under separate labels; the dedupe layers
    // and the floor run on the combined set, so this split is presentational
    // only. Doc-ness is a property of the neighbor's path (see
    // `prompts::hooks::is_doc_file`) — the read-side hits carry no chunk
    // metadata beyond what `neighbors` returned.
    let (code_hits, doc_hits): (Vec<_>, Vec<_>) = hits
        .iter()
        .partition(|n| !prompts::hooks::is_doc_file(&n.file_path));
    let line = |n: &Neighbor| {
        prompts::hooks::read_neighbor_line(
            &n.file_path,
            n.line_start,
            n.line_end,
            n.name.as_deref(),
            n.score,
        )
    };
    let code_locations: Vec<String> = code_hits.into_iter().map(line).collect();
    let doc_locations: Vec<String> = doc_hits.into_iter().map(line).collect();

    let context = prompts::hooks::read_related_context(
        read_path.as_str(),
        start,
        end,
        &code_locations,
        &doc_locations,
    );

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

/// Append the Read event's window to the session's read ledger, so later
/// related-code hints naming lines the agent already has can be suppressed.
///
/// Range semantics match [`read_surfacing_context`]: `start = offset.unwrap_or(1)`,
/// and the window is clamped to the Read tool's per-call cap — a full-file read
/// (neither field) covers at most the first [`READ_TOOL_MAX_LINES`] lines, and
/// an explicit `limit` beyond the cap is cut to it. The tool shows no more per
/// call, so claiming more would suppress hints for lines the session never saw.
/// A zero `limit` shows nothing and records nothing: even a one-line claim
/// could suppress a hint at exactly `start`. Fail-open: a path that escapes the
/// project root or a write failure just means a hint may surface again.
fn record_read_range(store: &Store, session_id: &str, input: &ToolInput) {
    let Some(file_path) = input.file_path.as_deref() else {
        return;
    };
    let Some(relative) = project_relative(store.project_root(), file_path) else {
        return;
    };
    let relative = RelativePath::from_path(&relative);
    if relative
        .reject_escape(prompts::hints::READ_INSIDE_PROJECT_DIR)
        .is_err()
    {
        return;
    }
    if input.limit == Some(0) {
        return;
    }
    let start = input.offset.unwrap_or(1);
    let shown = input
        .limit
        .map_or(READ_TOOL_MAX_LINES, |count| count.min(READ_TOOL_MAX_LINES));
    let end = start.saturating_add(shown.saturating_sub(1));
    change_neighbors::append_read_range(
        &store.change_neighbors_read_path(session_id),
        &change_neighbors::ReadRange {
            file_path: relative.as_str().to_owned(),
            line_start: start,
            line_end: end,
        },
    );
}

/// Read and ack the change-neighbors marker, returning formatted additionalContext.
/// Returns `None` when the marker is absent, feature is disabled, or the store
/// cannot be constructed (fail-open).
///
/// Session attribution: the marker names the session whose edit produced it,
/// and it is acked only from an event carrying that same session id — both
/// sides must be present and equal. Anything else — a marker left by another
/// session, an unattributed one (legacy format, manual reindex), or an event
/// without a session id — is stale and dropped without surfacing, so a
/// related-code hint can never claim another session's edit, not even in a
/// session-less event. The hint claims "your edit" only when `event_files`
/// names the marker's edited file; otherwise the wording credits a recent
/// edit, since the acking event is not the edit that produced the marker.
pub(super) fn take_change_neighbors_context(
    store: Option<&Store>,
    config: Option<&Config>,
    event_name: &str,
    session_id: Option<&str>,
    event_files: &[String],
) -> Option<Value> {
    let cfg = config?;
    if !cfg.hooks.surface_related_on_edit {
        return None;
    }
    let store = store?;
    let project_root = store.project_root();
    let marker_path = store.change_neighbors_marker_path();
    let marker = change_neighbors::read_and_remove(&marker_path)?;

    // Both sides must be PRESENT and equal: two absences are not a match, and
    // an unattributed marker must never surface — not even to a session-less
    // event, which cannot prove it owns the edit either.
    let (Some(marker_session), Some(event_session)) = (marker.session_id.as_deref(), session_id)
    else {
        return None;
    };
    if marker_session != event_session {
        return None;
    }

    // Two per-session dedupe layers, both line-aware:
    // - the seen ledger: a neighbor whose lines this session was already shown
    //   is suppressed whatever file the edit touched, so a hub file can't be
    //   re-injected once per edited path. The same pair at different lines is
    //   new context and still surfaces.
    // - the read ledger: a neighbor naming lines the session already Read is
    //   suppressed — the agent already has that content in context.
    // Both ledgers are reset on SessionStart. Fail-open: an unreadable ledger
    // reads as empty, so nothing is wrongly suppressed.
    let seen_path = store.change_neighbors_seen_path(event_session);
    let seen = change_neighbors::read_seen(&seen_path);
    let read_path = store.change_neighbors_read_path(event_session);
    let read = change_neighbors::read_ranges(&read_path);

    let mut fresh: Vec<change_neighbors::SeenEntry> = Vec::new();
    let (code_hits, doc_hits): (Vec<_>, Vec<_>) = marker
        .neighbors
        .iter()
        .filter(|n| n.file_path != marker.edited_path)
        .filter(|n| neighbor_file_exists(project_root, &n.file_path))
        .filter(|n| {
            if change_neighbors::is_seen(
                &seen,
                &n.file_path,
                n.name.as_deref(),
                n.line_start,
                n.line_end,
            ) {
                return false;
            }
            if change_neighbors::is_read(&read, &n.file_path, n.line_start, n.line_end) {
                return false;
            }
            fresh.push(change_neighbors::SeenEntry::new(
                &n.file_path,
                n.name.as_deref(),
                n.line_start,
                n.line_end,
            ));
            true
        })
        // Lazily, so the filter above never runs for the tail this drops: it pushes
        // into `fresh`, and an entry recorded for a neighbor that was truncated
        // away would suppress a hint nobody ever saw. The marker is over-fetched
        // (see `write_change_neighbors_marker`), so this is where a hint budget of
        // top_k is actually applied — against unseen candidates rather than against
        // a pool the seen-filter has already eaten into.
        .take(cfg.hooks.related_top_k)
        // Presentational split, after the cap: source hits render under
        // `related code:`, doc hits under `related docs:`. Doc-ness is a
        // property of the neighbor's path (`prompts::hooks::is_doc_file`), so
        // a marker written by any binary version renders correctly.
        .partition(|n| !prompts::hooks::is_doc_file(&n.file_path));

    if code_hits.is_empty() && doc_hits.is_empty() {
        return None;
    }

    // Record only the neighbors actually surfaced.
    change_neighbors::append_seen(&seen_path, &fresh);

    let line = |n: &change_neighbors::NeighborEntry| {
        prompts::hooks::edit_neighbor_line(
            &n.file_path,
            n.line_start,
            n.line_end,
            n.name.as_deref(),
            n.score,
        )
    };
    let code_lines: Vec<String> = code_hits.into_iter().map(line).collect();
    let doc_lines: Vec<String> = doc_hits.into_iter().map(line).collect();

    let acked_own_edit = event_files
        .iter()
        .any(|file| canonical_dedup_key(project_root, file) == marker.edited_path);
    let context = if acked_own_edit {
        prompts::hooks::edit_related_context(&marker.edited_path, &code_lines, &doc_lines)
    } else {
        prompts::hooks::recent_edit_related_context(&marker.edited_path, &code_lines, &doc_lines)
    };

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
fn enqueue_watchable_edits(store: &Store, input: &ToolInput, session_id: Option<&str>) -> usize {
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
        .filter(|p| reindex_queue::append(&queue_path, &canonical_dedup_key(root, p), session_id))
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
    async fn non_git_dir_edit_creates_no_state() -> Result<()> {
        // A non-git working directory is not indexable, so an edit's PostToolUse
        // hook must passthrough without laying down `.claudix/` — the reindex
        // queue append used to `create_dir_all` it unconditionally.
        let fixture = TestFixture::without_git("small_rust")?;
        let config = stub_config();
        write_config(fixture.root(), &config);

        let payload = json!({
            "tool_name": "Write",
            "tool_input": { "file_path": fixture.root().join("src/math.rs") }
        });
        let response = run(fixture.root(), HookEvent::PostToolUse, &payload.to_string()).await?;

        assert!(response.is_none(), "non-git edit must passthrough");
        assert!(
            !fixture.root().join(".claudix").exists(),
            "no state dir may be created outside a git repo"
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

    /// Session every typed-marker helper writes and every direct ack call
    /// fires with; the run()-level attribution tests use their own literals.
    const TEST_SESSION: &str = "sess-A";

    fn write_neighbors_marker(
        store: &Store,
        edited_path: &str,
        neighbors: Vec<crate::store::marker::change_neighbors::NeighborEntry>,
    ) {
        use crate::store::marker::change_neighbors::{ChangeNeighborsMarker, write};
        let marker = ChangeNeighborsMarker {
            edited_path: edited_path.to_owned(),
            session_id: Some(TEST_SESSION.to_owned()),
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

    fn neighbor_entry_with_lines(
        file_path: &str,
        name: &str,
        line_start: u32,
        line_end: u32,
    ) -> crate::store::marker::change_neighbors::NeighborEntry {
        let mut entry = make_neighbor_entry(file_path, name, 0.82);
        entry.line_start = line_start;
        entry.line_end = line_end;
        entry
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

        let response = take_change_neighbors_context(
            Some(&store),
            Some(&config),
            "PostToolUse",
            Some(TEST_SESSION),
            &[],
        );
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

        let _ = take_change_neighbors_context(
            Some(&store),
            Some(&config),
            "PostToolUse",
            Some(TEST_SESSION),
            &[],
        );

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
        let response = take_change_neighbors_context(
            Some(&store),
            Some(&config),
            "PostToolUse",
            Some(TEST_SESSION),
            &[],
        );
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

        let response = take_change_neighbors_context(
            Some(&store),
            Some(&config),
            "PostToolUse",
            Some(TEST_SESSION),
            &[],
        );
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

        let response = take_change_neighbors_context(
            Some(&store),
            Some(&config),
            "PostToolUse",
            Some(TEST_SESSION),
            &[],
        );
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
        let response = take_change_neighbors_context(
            Some(&store),
            Some(&config),
            "PostToolUse",
            Some(TEST_SESSION),
            &[],
        );
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
        let response = take_change_neighbors_context(
            Some(&store),
            Some(&config),
            "PostToolUse",
            Some(TEST_SESSION),
            &[],
        );
        assert!(
            response.is_none(),
            "a neighbor already surfaced this session must be suppressed on re-edit"
        );
        Ok(())
    }

    #[tokio::test]
    async fn same_neighbor_via_different_edited_file_is_suppressed() -> Result<()> {
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
        let first = take_change_neighbors_context(
            Some(&store),
            Some(&config),
            "PostToolUse",
            Some(TEST_SESSION),
            &[],
        )
        .unwrap_or(serde_json::Value::Null);
        assert!(
            first["hookSpecificOutput"]["additionalContext"]
                .as_str()
                .unwrap_or_default()
                .contains("src/math.rs"),
            "first take must surface the neighbor, got: {first}"
        );

        // Same neighbor, different edited file: the agent has already been shown
        // this symbol, so the hint has no value left to pay for its context.
        write_neighbors_marker(
            &store,
            "src/other.rs",
            vec![make_neighbor_entry("src/math.rs", "add", 0.82)],
        );
        let response = take_change_neighbors_context(
            Some(&store),
            Some(&config),
            "PostToolUse",
            Some(TEST_SESSION),
            &[],
        );
        assert!(
            response.is_none(),
            "a neighbor surfaced once must not resurface via a different edited file, got: {response:?}"
        );
        Ok(())
    }

    // ── code/docs group rendering ──────────────────────────────────────────

    /// The one line a rendered context devotes to a label, or `None` when the
    /// group is empty (the label is omitted, never left bare).
    fn labeled_line<'a>(context: &'a str, label: &str) -> Option<&'a str> {
        context.lines().find(|l| l.starts_with(label))
    }

    /// Create a doc neighbor file on disk so the existence guard passes.
    fn write_doc_neighbor_file(root: &Path, path: &str) {
        let abs = root.join(path);
        if let Some(parent) = abs.parent() {
            assert!(fs::create_dir_all(parent).is_ok());
        }
        assert!(fs::write(&abs, "# seeded doc\n").is_ok());
    }

    /// Entry with no symbol name, the shape of a doc or nameless fallback chunk.
    fn make_unnamed_entry(
        file_path: &str,
        score: f32,
    ) -> crate::store::marker::change_neighbors::NeighborEntry {
        let mut entry = make_neighbor_entry(file_path, "drop", score);
        entry.name = None;
        entry
    }

    #[tokio::test]
    async fn edit_surfacing_splits_hits_into_code_and_doc_groups() -> Result<()> {
        // The hq-5 verify line: an edit whose marker carries both kinds shows
        // two labeled groups, each holding only its own hits.
        let fixture = TestFixture::new("small_rust")?;
        let config = stub_config();
        write_config(fixture.root(), &config);
        let store = Store::new(fixture.root(), &config)?;
        store.ensure_layout()?;

        write_doc_neighbor_file(fixture.root(), "docs/guide.md");
        write_neighbors_marker(
            &store,
            "src/lib.rs",
            vec![
                make_neighbor_entry("src/math.rs", "add", 0.82),
                make_unnamed_entry("docs/guide.md", 0.81),
            ],
        );

        let payload = json!({
            "session_id": TEST_SESSION,
            "tool_name": "Write",
            "tool_input": { "file_path": fixture.root().join("src/lib.rs") }
        });
        let response = run(fixture.root(), HookEvent::PostToolUse, &payload.to_string()).await?;
        let context = response.unwrap_or(Value::Null)["hookSpecificOutput"]["additionalContext"]
            .as_str()
            .unwrap_or_default()
            .to_owned();

        let code_line = labeled_line(&context, "related code:").unwrap_or_default();
        assert!(
            code_line.contains("src/math.rs"),
            "the code group must carry the source hit, got: {context}"
        );
        assert!(
            !code_line.contains("docs/guide.md"),
            "the code group must not carry the doc hit, got: {context}"
        );
        let doc_line = labeled_line(&context, "related docs:").unwrap_or_default();
        assert!(
            doc_line.contains("docs/guide.md"),
            "the docs group must carry the doc hit, got: {context}"
        );
        assert!(
            !doc_line.contains("src/math.rs"),
            "the docs group must not carry the source hit, got: {context}"
        );
        assert!(
            context.find("related code:").unwrap_or(usize::MAX)
                < context.find("related docs:").unwrap_or(usize::MAX),
            "the code group must precede the docs group, got: {context}"
        );
        assert!(
            context.contains("your edit of `src/lib.rs`"),
            "the own-edit wording must survive the split, got: {context}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn source_only_hits_render_under_the_code_label() -> Result<()> {
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

        let response = take_change_neighbors_context(
            Some(&store),
            Some(&config),
            "PostToolUse",
            Some(TEST_SESSION),
            &[],
        );
        let context = response.unwrap_or(Value::Null)["hookSpecificOutput"]["additionalContext"]
            .as_str()
            .unwrap_or_default()
            .to_owned();
        assert!(
            labeled_line(&context, "related code:").is_some_and(|l| l.contains("src/math.rs")),
            "a source-only edit must render the hit under the code label, got: {context}"
        );
        assert!(
            !context.contains("related docs:"),
            "a source-only edit must not render a docs label, got: {context}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn doc_only_hits_render_under_the_docs_label() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let config = stub_config();
        write_config(fixture.root(), &config);
        let store = Store::new(fixture.root(), &config)?;
        store.ensure_layout()?;

        write_doc_neighbor_file(fixture.root(), "docs/guide.md");
        write_neighbors_marker(
            &store,
            "src/lib.rs",
            vec![make_unnamed_entry("docs/guide.md", 0.82)],
        );

        let response = take_change_neighbors_context(
            Some(&store),
            Some(&config),
            "PostToolUse",
            Some(TEST_SESSION),
            &[],
        );
        let context = response.unwrap_or(Value::Null)["hookSpecificOutput"]["additionalContext"]
            .as_str()
            .unwrap_or_default()
            .to_owned();
        assert!(
            labeled_line(&context, "related docs:").is_some_and(|l| l.contains("docs/guide.md")),
            "a doc-only edit must render the hit under the docs label, got: {context}"
        );
        assert!(
            !context.contains("related code:"),
            "a doc-only edit must not render a code label, got: {context}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn unnamed_hits_still_render_under_their_label() -> Result<()> {
        // A neighbor without a symbol name (doc chunks, nameless fallback
        // chunks) must still group and render — the name is optional in the
        // marker and the line format.
        let fixture = TestFixture::new("small_rust")?;
        let config = stub_config();
        write_config(fixture.root(), &config);
        let store = Store::new(fixture.root(), &config)?;
        store.ensure_layout()?;

        write_doc_neighbor_file(fixture.root(), "docs/guide.md");
        write_neighbors_marker(
            &store,
            "src/lib.rs",
            vec![
                make_unnamed_entry("src/math.rs", 0.82),
                make_unnamed_entry("docs/guide.md", 0.81),
            ],
        );

        let response = take_change_neighbors_context(
            Some(&store),
            Some(&config),
            "PostToolUse",
            Some(TEST_SESSION),
            &[],
        );
        let context = response.unwrap_or(Value::Null)["hookSpecificOutput"]["additionalContext"]
            .as_str()
            .unwrap_or_default()
            .to_owned();
        assert!(
            labeled_line(&context, "related code:")
                .is_some_and(|l| l.contains("src/math.rs:10-25")),
            "an unnamed source hit must render its location under the code label, got: {context}"
        );
        assert!(
            labeled_line(&context, "related docs:")
                .is_some_and(|l| l.contains("docs/guide.md:10-25")),
            "an unnamed doc hit must render its location under the docs label, got: {context}"
        );
        Ok(())
    }

    // ── change-neighbors session attribution ────────────────────────────────

    /// Plant a session-attributed marker as raw JSON so these tests exercise the
    /// public `run` layer against whatever the marker schema accepts today.
    fn write_raw_neighbors_marker(store: &Store, edited_path: &str, session_id: &str) {
        let json = json!({
            "edited_path": edited_path,
            "session_id": session_id,
            "neighbors": [{
                "file_path": "src/math.rs",
                "line_start": 10,
                "line_end": 25,
                "name": "add",
                "score": 0.82,
            }],
        });
        fs::write(store.change_neighbors_marker_path(), json.to_string())
            .expect("marker write must succeed");
    }

    #[tokio::test]
    async fn foreign_session_marker_is_dropped_without_surfacing() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let config = stub_config();
        write_config(fixture.root(), &config);
        let store = Store::new(fixture.root(), &config)?;
        store.ensure_layout()?;

        // A marker left by another session's edit (the triage shape: an edit
        // acking a marker a different session's write left behind).
        write_raw_neighbors_marker(&store, "src/lib.rs", "session-foreign");

        let payload = json!({
            "session_id": "session-mine",
            "tool_name": "Write",
            "tool_input": { "file_path": fixture.root().join("src/math.rs") },
        });
        let response = run(fixture.root(), HookEvent::PostToolUse, &payload.to_string()).await?;

        assert!(
            response.is_none(),
            "a foreign-session marker must never surface, got: {response:?}"
        );
        assert!(
            !store.change_neighbors_marker_path().exists(),
            "a foreign-session marker must be dropped from disk"
        );
        Ok(())
    }

    #[tokio::test]
    async fn matching_session_marker_surfaces_your_edit_hint() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let config = stub_config();
        write_config(fixture.root(), &config);
        let store = Store::new(fixture.root(), &config)?;
        store.ensure_layout()?;

        write_raw_neighbors_marker(&store, "src/lib.rs", "sess-S");

        // The acking event edits the very file the marker records.
        let payload = json!({
            "session_id": "sess-S",
            "tool_name": "Write",
            "tool_input": { "file_path": fixture.root().join("src/lib.rs") },
        });
        let response = run(fixture.root(), HookEvent::PostToolUse, &payload.to_string()).await?;
        let response = response.unwrap_or(Value::Null);
        let context = response["hookSpecificOutput"]["additionalContext"]
            .as_str()
            .unwrap_or_default();

        assert!(
            context.contains("src/math.rs"),
            "a same-session marker must surface its neighbors, got: {context}"
        );
        assert!(
            context.contains("your edit of `src/lib.rs`"),
            "an event that edited the recorded file may claim 'your edit', got: {context}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn your_edit_wording_requires_the_acking_event_to_be_that_edit() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let config = stub_config();
        write_config(fixture.root(), &config);
        let store = Store::new(fixture.root(), &config)?;
        store.ensure_layout()?;

        // Same session, but the acking event edits a different file than the
        // marker records (the triage shape: a todo.md Write attributed to
        // docs/handoff-state.md).
        write_raw_neighbors_marker(&store, "src/lib.rs", "sess-S");

        let payload = json!({
            "session_id": "sess-S",
            "tool_name": "Write",
            "tool_input": { "file_path": fixture.root().join("src/math.rs") },
        });
        let response = run(fixture.root(), HookEvent::PostToolUse, &payload.to_string()).await?;
        let response = response.unwrap_or(Value::Null);
        let context = response["hookSpecificOutput"]["additionalContext"]
            .as_str()
            .unwrap_or_default();

        assert!(
            context.contains("src/math.rs"),
            "a same-session marker must still surface its neighbors, got: {context}"
        );
        assert!(
            !context.contains("your edit"),
            "an event that did not edit the recorded file must not claim 'your edit', got: {context}"
        );
        assert!(
            context.contains("recent edit of `src/lib.rs`"),
            "the hint must attribute to a recent edit instead, got: {context}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn unattributed_marker_never_surfaces_not_even_to_a_sessionless_event() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let config = stub_config();
        write_config(fixture.root(), &config);
        let store = Store::new(fixture.root(), &config)?;
        store.ensure_layout()?;

        // A legacy marker (no session_id key) must not surface even when the
        // acking event also carries no session: two absences are not a match,
        // and an unowned hint must never surface.
        let json = json!({
            "edited_path": "src/lib.rs",
            "neighbors": [{
                "file_path": "src/math.rs",
                "line_start": 10,
                "line_end": 25,
                "name": "add",
                "score": 0.82,
            }],
        });
        fs::write(store.change_neighbors_marker_path(), json.to_string())
            .expect("marker write must succeed");

        let payload = json!({
            "tool_name": "Write",
            "tool_input": { "file_path": fixture.root().join("src/math.rs") },
        });
        let response = run(fixture.root(), HookEvent::PostToolUse, &payload.to_string()).await?;

        assert!(
            response.is_none(),
            "an unattributed marker must never surface, got: {response:?}"
        );
        assert!(
            !store.change_neighbors_marker_path().exists(),
            "an unattributed marker must be dropped from disk"
        );
        Ok(())
    }

    #[tokio::test]
    async fn distinct_symbols_in_one_neighbor_file_stay_separate_keys() -> Result<()> {
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
        let _ = take_change_neighbors_context(
            Some(&store),
            Some(&config),
            "PostToolUse",
            Some(TEST_SESSION),
            &[],
        );

        // Same neighbor file, different symbol → still unknown to the agent.
        write_neighbors_marker(
            &store,
            "src/lib.rs",
            vec![make_neighbor_entry("src/math.rs", "multiply", 0.82)],
        );
        let response = take_change_neighbors_context(
            Some(&store),
            Some(&config),
            "PostToolUse",
            Some(TEST_SESSION),
            &[],
        )
        .unwrap_or(serde_json::Value::Null);
        let context = response["hookSpecificOutput"]["additionalContext"]
            .as_str()
            .unwrap_or_default();
        assert!(
            context.contains("multiply"),
            "a different symbol in an already-surfaced file must still surface, got: {context}"
        );
        Ok(())
    }

    // ── line-aware dedupe (seen + read ledgers) ─────────────────────────────

    #[tokio::test]
    async fn same_symbol_at_different_lines_still_surfaces() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let config = stub_config();
        write_config(fixture.root(), &config);
        let store = Store::new(fixture.root(), &config)?;
        store.ensure_layout()?;

        // First take surfaces (src/math.rs, add) at lines 10-25 and records it.
        write_neighbors_marker(
            &store,
            "src/lib.rs",
            vec![make_neighbor_entry("src/math.rs", "add", 0.82)],
        );
        let _ = take_change_neighbors_context(
            Some(&store),
            Some(&config),
            "PostToolUse",
            Some(TEST_SESSION),
            &[],
        );

        // Same file + symbol, different lines: new context the agent has not
        // seen, so it must surface — the old whole-file key would swallow it.
        write_neighbors_marker(
            &store,
            "src/lib.rs",
            vec![neighbor_entry_with_lines("src/math.rs", "add", 60, 80)],
        );
        let response = take_change_neighbors_context(
            Some(&store),
            Some(&config),
            "PostToolUse",
            Some(TEST_SESSION),
            &[],
        )
        .unwrap_or(serde_json::Value::Null);
        let context = response["hookSpecificOutput"]["additionalContext"]
            .as_str()
            .unwrap_or_default();
        assert!(
            context.contains("src/math.rs"),
            "the same symbol at different lines is new context, got: {context}"
        );
        assert!(
            context.contains("60-80"),
            "the hint must name the new line range, got: {context}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn seen_dedupe_state_is_per_session() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let config = stub_config();
        write_config(fixture.root(), &config);
        let store = Store::new(fixture.root(), &config)?;
        store.ensure_layout()?;

        // Session A surfaces the (src/lib.rs → src/math.rs add) pair once.
        write_neighbors_marker(
            &store,
            "src/lib.rs",
            vec![make_neighbor_entry("src/math.rs", "add", 0.82)],
        );
        let _ = take_change_neighbors_context(
            Some(&store),
            Some(&config),
            "PostToolUse",
            Some("sess-A"),
            &[],
        );

        // Session B's own marker for the identical pair: B has never been shown
        // it, so B must surface it — a shared ledger would wrongly suppress it.
        write_raw_neighbors_marker(&store, "src/lib.rs", "sess-B");
        let payload = json!({
            "session_id": "sess-B",
            "tool_name": "Write",
            "tool_input": { "file_path": fixture.root().join("src/lib.rs") }
        });
        let response = run(fixture.root(), HookEvent::PostToolUse, &payload.to_string()).await?;
        let context = response.unwrap_or(Value::Null)["hookSpecificOutput"]["additionalContext"]
            .as_str()
            .unwrap_or_default()
            .to_owned();
        assert!(
            context.contains("src/math.rs"),
            "session B's ledger is empty and must surface the hint, got: {context}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn read_then_edit_suppresses_hint_inside_read_lines() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let config = stub_config(); // edit surfacing on (default), read surfacing off
        write_config(fixture.root(), &config);
        let store = Store::new(fixture.root(), &config)?;
        store.ensure_layout()?;

        // The session reads lines 1-50 of src/math.rs.
        let read_payload = json!({
            "session_id": TEST_SESSION,
            "tool_name": "Read",
            "tool_input": {
                "file_path": fixture.root().join("src/math.rs"),
                "offset": 1,
                "limit": 50
            }
        });
        let response = run(
            fixture.root(),
            HookEvent::PostToolUse,
            &read_payload.to_string(),
        )
        .await?;
        assert!(
            response.is_none(),
            "a ranged read with read-surfacing off surfaces nothing, got: {response:?}"
        );

        // An edit of src/lib.rs leaves a marker whose hint points at lines 10-25
        // of the file the session just read: inside the read window.
        write_neighbors_marker(
            &store,
            "src/lib.rs",
            vec![neighbor_entry_with_lines("src/math.rs", "add", 10, 25)],
        );
        let edit_payload = json!({
            "session_id": TEST_SESSION,
            "tool_name": "Write",
            "tool_input": { "file_path": fixture.root().join("src/lib.rs") }
        });
        let response = run(
            fixture.root(),
            HookEvent::PostToolUse,
            &edit_payload.to_string(),
        )
        .await?;
        assert!(
            response.is_none(),
            "a hint pointing inside lines the session read must be suppressed, got: {response:?}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn read_then_edit_hint_outside_read_lines_still_fires() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let config = stub_config();
        write_config(fixture.root(), &config);
        let store = Store::new(fixture.root(), &config)?;
        store.ensure_layout()?;

        let read_payload = json!({
            "session_id": TEST_SESSION,
            "tool_name": "Read",
            "tool_input": {
                "file_path": fixture.root().join("src/math.rs"),
                "offset": 1,
                "limit": 50
            }
        });
        let _ = run(
            fixture.root(),
            HookEvent::PostToolUse,
            &read_payload.to_string(),
        )
        .await?;

        // The hint points at lines 60-80: outside the read window.
        write_neighbors_marker(
            &store,
            "src/lib.rs",
            vec![neighbor_entry_with_lines("src/math.rs", "add", 60, 80)],
        );
        let edit_payload = json!({
            "session_id": TEST_SESSION,
            "tool_name": "Write",
            "tool_input": { "file_path": fixture.root().join("src/lib.rs") }
        });
        let response = run(
            fixture.root(),
            HookEvent::PostToolUse,
            &edit_payload.to_string(),
        )
        .await?;
        let context = response.unwrap_or(Value::Null)["hookSpecificOutput"]["additionalContext"]
            .as_str()
            .unwrap_or_default()
            .to_owned();
        assert!(
            context.contains("src/math.rs"),
            "a hint outside the read lines must still fire, got: {context}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn read_then_edit_hint_partially_overlapping_read_lines_still_fires() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let config = stub_config();
        write_config(fixture.root(), &config);
        let store = Store::new(fixture.root(), &config)?;
        store.ensure_layout()?;

        let read_payload = json!({
            "session_id": TEST_SESSION,
            "tool_name": "Read",
            "tool_input": {
                "file_path": fixture.root().join("src/math.rs"),
                "offset": 1,
                "limit": 50
            }
        });
        let _ = run(
            fixture.root(),
            HookEvent::PostToolUse,
            &read_payload.to_string(),
        )
        .await?;

        // Lines 40-60 overlap the read window but are not contained in it: part
        // of the hinted region is new, so the hint must fire.
        write_neighbors_marker(
            &store,
            "src/lib.rs",
            vec![neighbor_entry_with_lines("src/math.rs", "add", 40, 60)],
        );
        let edit_payload = json!({
            "session_id": TEST_SESSION,
            "tool_name": "Write",
            "tool_input": { "file_path": fixture.root().join("src/lib.rs") }
        });
        let response = run(
            fixture.root(),
            HookEvent::PostToolUse,
            &edit_payload.to_string(),
        )
        .await?;
        let context = response.unwrap_or(Value::Null)["hookSpecificOutput"]["additionalContext"]
            .as_str()
            .unwrap_or_default()
            .to_owned();
        assert!(
            context.contains("src/math.rs"),
            "a hint that only partially overlaps the read lines must still fire, got: {context}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn read_dedupe_state_is_per_session() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let config = stub_config();
        write_config(fixture.root(), &config);
        let store = Store::new(fixture.root(), &config)?;
        store.ensure_layout()?;

        // Session A reads lines 1-50 of src/math.rs.
        let read_payload = json!({
            "session_id": "sess-reader",
            "tool_name": "Read",
            "tool_input": {
                "file_path": fixture.root().join("src/math.rs"),
                "offset": 1,
                "limit": 50
            }
        });
        let _ = run(
            fixture.root(),
            HookEvent::PostToolUse,
            &read_payload.to_string(),
        )
        .await?;

        // Session B edits src/lib.rs and its own marker points inside those
        // lines. B has read nothing, so the hint must fire — a shared read
        // ledger would wrongly suppress it.
        write_raw_neighbors_marker(&store, "src/lib.rs", "sess-editor");
        let edit_payload = json!({
            "session_id": "sess-editor",
            "tool_name": "Write",
            "tool_input": { "file_path": fixture.root().join("src/lib.rs") }
        });
        let response = run(
            fixture.root(),
            HookEvent::PostToolUse,
            &edit_payload.to_string(),
        )
        .await?;
        let context = response.unwrap_or(Value::Null)["hookSpecificOutput"]["additionalContext"]
            .as_str()
            .unwrap_or_default()
            .to_owned();
        assert!(
            context.contains("src/math.rs"),
            "session B has not read those lines; the hint must fire, got: {context}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn full_file_read_suppresses_hints_into_that_file() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let config = stub_config();
        write_config(fixture.root(), &config);
        let store = Store::new(fixture.root(), &config)?;
        store.ensure_layout()?;

        // No offset/limit: the Read tool shows the file up to its per-call cap,
        // and the hint sits far inside it.
        let read_payload = json!({
            "session_id": TEST_SESSION,
            "tool_name": "Read",
            "tool_input": { "file_path": fixture.root().join("src/math.rs") }
        });
        let response = run(
            fixture.root(),
            HookEvent::PostToolUse,
            &read_payload.to_string(),
        )
        .await?;
        assert!(
            response.is_none(),
            "a full-file read with read-surfacing off surfaces nothing, got: {response:?}"
        );

        write_neighbors_marker(
            &store,
            "src/lib.rs",
            vec![make_neighbor_entry("src/math.rs", "add", 0.82)],
        );
        let edit_payload = json!({
            "session_id": TEST_SESSION,
            "tool_name": "Write",
            "tool_input": { "file_path": fixture.root().join("src/lib.rs") }
        });
        let response = run(
            fixture.root(),
            HookEvent::PostToolUse,
            &edit_payload.to_string(),
        )
        .await?;
        assert!(
            response.is_none(),
            "the hint sits inside the first READ_TOOL_MAX_LINES lines the full-file read covers, so it is redundant, got: {response:?}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn full_file_read_does_not_cover_lines_beyond_the_read_tool_cap() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let config = stub_config();
        write_config(fixture.root(), &config);
        let store = Store::new(fixture.root(), &config)?;
        store.ensure_layout()?;

        // A 2600-line file: the Read tool shows at most 2000 lines per call,
        // so the ledger must not claim beyond that.
        let big: String = (0..2600).map(|i| format!("line {i}\n")).collect();
        fs::write(fixture.root().join("src/big.rs"), big)?;

        let read_payload = json!({
            "session_id": TEST_SESSION,
            "tool_name": "Read",
            "tool_input": { "file_path": fixture.root().join("src/big.rs") }
        });
        let _ = run(
            fixture.root(),
            HookEvent::PostToolUse,
            &read_payload.to_string(),
        )
        .await?;

        // A hint at lines 2500-2510: beyond what the tool could have shown, so
        // the session never read them and the hint must fire.
        write_neighbors_marker(
            &store,
            "src/lib.rs",
            vec![neighbor_entry_with_lines("src/big.rs", "far", 2500, 2510)],
        );
        let edit_payload = json!({
            "session_id": TEST_SESSION,
            "tool_name": "Write",
            "tool_input": { "file_path": fixture.root().join("src/lib.rs") }
        });
        let response = run(
            fixture.root(),
            HookEvent::PostToolUse,
            &edit_payload.to_string(),
        )
        .await?;
        let context = response.unwrap_or(Value::Null)["hookSpecificOutput"]["additionalContext"]
            .as_str()
            .unwrap_or_default()
            .to_owned();
        assert!(
            context.contains("src/big.rs"),
            "lines beyond the Read tool's cap were never shown; the hint must fire, got: {context}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn zero_limit_read_records_nothing() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let config = stub_config();
        write_config(fixture.root(), &config);
        let store = Store::new(fixture.root(), &config)?;
        store.ensure_layout()?;

        // limit=0 shows no lines, so nothing may be recorded: a hint at exactly
        // the offset line must still fire.
        let read_payload = json!({
            "session_id": TEST_SESSION,
            "tool_name": "Read",
            "tool_input": {
                "file_path": fixture.root().join("src/math.rs"),
                "offset": 10,
                "limit": 0
            }
        });
        let _ = run(
            fixture.root(),
            HookEvent::PostToolUse,
            &read_payload.to_string(),
        )
        .await?;

        write_neighbors_marker(
            &store,
            "src/lib.rs",
            vec![neighbor_entry_with_lines("src/math.rs", "add", 10, 10)],
        );
        let edit_payload = json!({
            "session_id": TEST_SESSION,
            "tool_name": "Write",
            "tool_input": { "file_path": fixture.root().join("src/lib.rs") }
        });
        let response = run(
            fixture.root(),
            HookEvent::PostToolUse,
            &edit_payload.to_string(),
        )
        .await?;
        let context = response.unwrap_or(Value::Null)["hookSpecificOutput"]["additionalContext"]
            .as_str()
            .unwrap_or_default()
            .to_owned();
        assert!(
            context.contains("src/math.rs"),
            "a zero-limit read showed nothing; the hint at the offset line must fire, got: {context}"
        );
        Ok(())
    }

    /// Create `count` real neighbor files (so `neighbor_file_exists` passes) plus
    /// the marker entries pointing at them, descending by score so marker order is
    /// rank order.
    fn seed_neighbor_files(
        root: &Path,
        count: usize,
    ) -> Vec<crate::store::marker::change_neighbors::NeighborEntry> {
        (0..count)
            .map(|i| {
                let file = format!("nbr{i:02}.rs");
                assert!(fs::write(root.join(&file), "fn placeholder() {}").is_ok());
                let score = 0.99 - (i as f32) * 0.01;
                make_neighbor_entry(&file, &format!("sym{i:02}"), score)
            })
            .collect()
    }

    #[tokio::test]
    async fn surfaced_hints_are_capped_at_top_k_and_ledger_records_only_those() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let config = stub_config();
        write_config(fixture.root(), &config);
        let store = Store::new(fixture.root(), &config)?;
        store.ensure_layout()?;

        // An over-fetched marker: far more candidates than the hint budget.
        let top_k = config.hooks.related_top_k;
        let entries = seed_neighbor_files(fixture.root(), top_k * 3);
        write_neighbors_marker(&store, "src/lib.rs", entries);

        let response = take_change_neighbors_context(
            Some(&store),
            Some(&config),
            "PostToolUse",
            Some(TEST_SESSION),
            &[],
        )
        .unwrap_or(serde_json::Value::Null);
        let context = response["hookSpecificOutput"]["additionalContext"]
            .as_str()
            .unwrap_or_default();
        assert_eq!(
            context.matches("nbr").count(),
            top_k,
            "over-fetching must not widen how many hints an edit shows, got: {context}"
        );

        // A key recorded for a neighbor that was truncated away would suppress a
        // hint nobody ever saw — the failure mode of truncating after collecting.
        let ledger =
            fs::read_to_string(store.change_neighbors_seen_path(TEST_SESSION)).unwrap_or_default();
        let recorded = ledger.lines().filter(|line| !line.is_empty()).count();
        assert_eq!(
            recorded, top_k,
            "ledger must record exactly the surfaced hints, got: {ledger}"
        );
        assert!(
            !ledger.contains(&format!("nbr{top_k:02}")),
            "a truncated-away neighbor must never reach the ledger, got: {ledger}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn over_fetched_tail_surfaces_when_leading_candidates_are_seen() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let config = stub_config();
        write_config(fixture.root(), &config);
        let store = Store::new(fixture.root(), &config)?;
        store.ensure_layout()?;

        let top_k = config.hooks.related_top_k;
        let entries = seed_neighbor_files(fixture.root(), top_k * 3);

        // The whole leading rank is already spent by earlier edits this session.
        let spent: Vec<crate::store::marker::change_neighbors::SeenEntry> = entries
            .iter()
            .take(top_k)
            .map(|e| {
                crate::store::marker::change_neighbors::SeenEntry::new(
                    &e.file_path,
                    e.name.as_deref(),
                    e.line_start,
                    e.line_end,
                )
            })
            .collect();
        crate::store::marker::change_neighbors::append_seen(
            &store.change_neighbors_seen_path(TEST_SESSION),
            &spent,
        );

        write_neighbors_marker(&store, "src/lib.rs", entries);
        let response = take_change_neighbors_context(
            Some(&store),
            Some(&config),
            "PostToolUse",
            Some(TEST_SESSION),
            &[],
        )
        .unwrap_or(serde_json::Value::Null);
        let context = response["hookSpecificOutput"]["additionalContext"]
            .as_str()
            .unwrap_or_default();

        assert!(
            context.contains(&format!("nbr{top_k:02}")),
            "an edit whose leading candidates are all seen must fall through to the \
             unseen tail instead of going silent, got: {context}"
        );
        assert_eq!(
            context.matches("nbr").count(),
            top_k,
            "the tail must still be capped at the hint budget, got: {context}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn unreadable_seen_ledger_suppresses_nothing() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let config = stub_config();
        write_config(fixture.root(), &config);
        let store = Store::new(fixture.root(), &config)?;
        store.ensure_layout()?;

        // A directory where the ledger file belongs: every read AND every append
        // fails, exercising both fail-open branches at once.
        let seen_path = store.change_neighbors_seen_path(TEST_SESSION);
        fs::create_dir_all(&seen_path)?;

        for take in 1..=2 {
            write_neighbors_marker(
                &store,
                "src/lib.rs",
                vec![make_neighbor_entry("src/math.rs", "add", 0.82)],
            );
            let response = take_change_neighbors_context(
                Some(&store),
                Some(&config),
                "PostToolUse",
                Some(TEST_SESSION),
                &[],
            )
            .unwrap_or(serde_json::Value::Null);
            let context = response["hookSpecificOutput"]["additionalContext"]
                .as_str()
                .unwrap_or_default();
            assert!(
                context.contains("src/math.rs"),
                "take {take}: an unreadable ledger must suppress nothing, got: {context}"
            );
        }
        Ok(())
    }

    #[tokio::test]
    async fn session_start_resets_only_the_starting_sessions_ledgers() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let config = stub_config();
        write_config(fixture.root(), &config);
        let store = Store::new(fixture.root(), &config)?;
        store.ensure_layout()?;

        // Seed dummy state for the starting session and for another one.
        let seen_path = store.change_neighbors_seen_path("sess-start");
        let read_path = store.change_neighbors_read_path("sess-start");
        let other_seen = store.change_neighbors_seen_path("sess-other");
        let other_read = store.change_neighbors_read_path("sess-other");
        for path in [&seen_path, &read_path, &other_seen, &other_read] {
            fs::write(path, "src/math.rs\tadd\t10\t25\n")?;
        }

        let payload = json!({ "session_id": "sess-start" });
        run(
            fixture.root(),
            HookEvent::SessionStart,
            &payload.to_string(),
        )
        .await?;

        assert!(
            !seen_path.exists(),
            "SessionStart must reset the starting session's seen ledger"
        );
        assert!(
            !read_path.exists(),
            "SessionStart must reset the starting session's read ledger"
        );
        assert!(
            other_seen.exists(),
            "another session's seen ledger must survive"
        );
        assert!(
            other_read.exists(),
            "another session's read ledger must survive"
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

        let response = run(
            fixture.root(),
            HookEvent::PostToolUse,
            &json!({ "session_id": TEST_SESSION }).to_string(),
        )
        .await?;
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
    async fn ranged_read_groups_code_and_doc_neighbors() -> Result<()> {
        // Read-time surfacing shares the edit path's two-label rendering: a
        // read whose neighbors mix source and doc files shows both groups.
        let fixture = TestFixture::new("small_rust")?;
        let config = read_config_on();
        write_config(fixture.root(), &config);
        let store = Store::new(fixture.root(), &config)?;
        store.ensure_layout()?;

        // seed_rows classifies by path at render time, so the doc row groups
        // under `related docs:` regardless of the seeded chunk metadata.
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
                    "docs/guide.md",
                    "guide",
                    1,
                    5,
                    vec![0.98, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ),
            ],
        )
        .await;

        let payload = json!({
            "tool_name": "Read",
            "tool_input": { "file_path": "src/foo.rs", "offset": 12, "limit": 6 }
        });
        let response = run(fixture.root(), HookEvent::PostToolUse, &payload.to_string()).await?;
        let context = response.unwrap_or(Value::Null)["hookSpecificOutput"]["additionalContext"]
            .as_str()
            .unwrap_or_default()
            .to_owned();

        assert!(
            labeled_line(&context, "related code:").is_some_and(|l| l.contains("src/bar.rs")),
            "the source neighbor must group under the code label, got: {context}"
        );
        assert!(
            labeled_line(&context, "related docs:").is_some_and(|l| l.contains("docs/guide.md")),
            "the doc neighbor must group under the docs label, got: {context}"
        );
        assert!(
            context.contains("lines 12-17"),
            "the read region naming must survive the split, got: {context}"
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

    /// The stored corpus floor gates read-time surfacing, not just the config
    /// value: a neighbor above the config floor but below the stored one is
    /// suppressed once the index records a floor.
    #[tokio::test]
    async fn read_surfacing_honors_the_stored_corpus_floor() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let config = read_config_on(); // related_min_similarity = 0.5
        write_config(fixture.root(), &config);
        let store = Store::new(fixture.root(), &config)?;
        store.ensure_layout()?;

        // Read chunk and a neighbor whose cosine to it is exactly 0.9.
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
                    vec![0.9, 0.436, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ),
            ],
        )
        .await;

        let payload = json!({
            "tool_name": "Read",
            "tool_input": { "file_path": "src/foo.rs", "offset": 12, "limit": 6 }
        });

        // No stored floor: the 0.9 neighbor clears the 0.5 config floor.
        let before = run(fixture.root(), HookEvent::PostToolUse, &payload.to_string()).await?;
        assert!(
            before.unwrap_or(Value::Null)["hookSpecificOutput"]["additionalContext"]
                .as_str()
                .unwrap_or_default()
                .contains("src/bar.rs"),
            "a 0.9 neighbor must surface when no corpus floor is stored"
        );

        // A stored floor of 0.95 sits above the neighbor's 0.9 score, so it must
        // now be suppressed. If resolution ignored the stored floor this would
        // still surface — that is the mutation this pins.
        store.update_similarity_floor(Some(0.95))?;
        let after = run(fixture.root(), HookEvent::PostToolUse, &payload.to_string()).await?;
        assert!(
            after.is_none(),
            "a stored corpus floor above the neighbor's score must suppress it, got: {after:?}"
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

    #[tokio::test]
    async fn read_hint_already_surfaced_this_session_is_suppressed() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let config = read_config_on();
        write_config(fixture.root(), &config);
        let store = Store::new(fixture.root(), &config)?;
        store.ensure_layout()?;

        // foo and c both neighbor bar's chunk at lines 40-58.
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
                    "src/c.rs",
                    "twin-c",
                    70,
                    80,
                    vec![0.98, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ),
            ],
        )
        .await;

        let read_payload = |path: &str, offset: u32, limit: u32| {
            json!({
                "session_id": TEST_SESSION,
                "tool_name": "Read",
                "tool_input": { "file_path": path, "offset": offset, "limit": limit }
            })
            .to_string()
        };

        // Reading foo surfaces bar and c, and records both in the seen ledger.
        let first = run(
            fixture.root(),
            HookEvent::PostToolUse,
            &read_payload("src/foo.rs", 12, 6),
        )
        .await?;
        let context = first.unwrap_or(Value::Null)["hookSpecificOutput"]["additionalContext"]
            .as_str()
            .unwrap_or_default()
            .to_owned();
        assert!(
            context.contains("src/bar.rs"),
            "first read must surface bar, got: {context}"
        );
        assert!(
            context.contains("src/c.rs"),
            "first read must surface c, got: {context}"
        );

        // Reading c now queries its own chunk: bar neighbors it too but was
        // already surfaced at these same lines, so it is suppressed; foo is new.
        let second = run(
            fixture.root(),
            HookEvent::PostToolUse,
            &read_payload("src/c.rs", 70, 11),
        )
        .await?;
        let context = second.unwrap_or(Value::Null)["hookSpecificOutput"]["additionalContext"]
            .as_str()
            .unwrap_or_default()
            .to_owned();
        assert!(
            !context.contains("src/bar.rs"),
            "bar was already surfaced at these lines, got: {context}"
        );
        assert!(
            context.contains("src/foo.rs"),
            "foo is new context and must surface, got: {context}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn read_hint_inside_previously_read_lines_is_suppressed() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let config = read_config_on();
        write_config(fixture.root(), &config);
        let store = Store::new(fixture.root(), &config)?;
        store.ensure_layout()?;

        // bar's only neighbor is foo.
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

        // The session reads all of bar's chunk (lines 40-58) — which surfaces
        // foo — and the read is recorded in the read ledger.
        let bar_payload = json!({
            "session_id": TEST_SESSION,
            "tool_name": "Read",
            "tool_input": { "file_path": "src/bar.rs", "offset": 40, "limit": 19 }
        });
        let first = run(
            fixture.root(),
            HookEvent::PostToolUse,
            &bar_payload.to_string(),
        )
        .await?;
        assert!(
            first.unwrap_or(Value::Null)["hookSpecificOutput"]["additionalContext"]
                .as_str()
                .unwrap_or_default()
                .contains("src/foo.rs"),
            "reading bar must surface foo"
        );

        // Reading foo would now hint at bar's lines 40-58 — but the session
        // already read exactly those lines, so the hint is redundant.
        let foo_payload = json!({
            "session_id": TEST_SESSION,
            "tool_name": "Read",
            "tool_input": { "file_path": "src/foo.rs", "offset": 12, "limit": 6 }
        });
        let second = run(
            fixture.root(),
            HookEvent::PostToolUse,
            &foo_payload.to_string(),
        )
        .await?;
        assert!(
            second.is_none(),
            "a hint pointing at lines the session read must be suppressed, got: {second:?}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn read_hint_partially_outside_previously_read_lines_still_fires() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let config = read_config_on();
        write_config(fixture.root(), &config);
        let store = Store::new(fixture.root(), &config)?;
        store.ensure_layout()?;

        // bar's only neighbor is foo.
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

        // The session read only lines 45-46 of bar — inside the chunk, but not
        // covering the whole 40-58 range the hint would name.
        let bar_payload = json!({
            "session_id": TEST_SESSION,
            "tool_name": "Read",
            "tool_input": { "file_path": "src/bar.rs", "offset": 45, "limit": 2 }
        });
        let _ = run(
            fixture.root(),
            HookEvent::PostToolUse,
            &bar_payload.to_string(),
        )
        .await?;

        let foo_payload = json!({
            "session_id": TEST_SESSION,
            "tool_name": "Read",
            "tool_input": { "file_path": "src/foo.rs", "offset": 12, "limit": 6 }
        });
        let response = run(
            fixture.root(),
            HookEvent::PostToolUse,
            &foo_payload.to_string(),
        )
        .await?;
        let context = response.unwrap_or(Value::Null)["hookSpecificOutput"]["additionalContext"]
            .as_str()
            .unwrap_or_default()
            .to_owned();
        assert!(
            context.contains("src/bar.rs"),
            "the read covered only part of the hinted range, so the hint must fire, got: {context}"
        );
        Ok(())
    }
}
