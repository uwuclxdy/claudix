use std::path::Path;
use std::time::Duration;

use serde_json::{Value, json};

use crate::config::{self, Config};
use crate::enumeration::WatchFilter;
use crate::error::Result;
use crate::search::neighbors::neighbors;
use crate::store::Store;
use crate::store::marker::change_neighbors;
use crate::types::RelativePath;

use super::payload::{HookPayload, ToolInput};
use super::ready_check::check_index_ready;
use super::spawn::spawn_background_reindex_file;
use crate::store::marker::WATCH_MARKER_STALE_SECS;

const READ_NEIGHBOR_RECOVERY: &str = "Read a path inside the project root";

pub(super) async fn handle_post_tool_use(
    project_root: &Path,
    payload: HookPayload,
) -> Result<Option<Value>> {
    let config = config::load(project_root).ok();

    // `Read` rides this hook too (see hooks.json matcher) purely for read-time
    // surfacing — it must NEVER spawn a reindex. The edit tools below do.
    let tool_name = payload.tool_name.as_deref();

    if tool_name == Some("Read")
        && config
            .as_ref()
            .is_some_and(|cfg| !cfg.hooks.surface_related_on_read)
    {
        return Ok(None);
    }

    // Spawn a background reindex only when an edit tool fired on a real file.
    // The ready-check and neighbor-surfacing runs on every PostToolUse event
    // regardless of whether a spawn happened.
    let read_input = if let Some(name) = tool_name
        && matches!(name, "Edit" | "Write" | "NotebookEdit" | "MultiEdit")
        && let Some(cfg) = config.as_ref()
        && cfg.hooks.auto_reembed_on_edit
        && !watcher_alive(project_root, cfg)
        && let Some(input) = payload.tool_input.as_ref()
        && let Some(file_path) = input
            .file_path
            .as_deref()
            .or(input.notebook_path.as_deref())
        && reindex_target_is_watchable(project_root, file_path)
    {
        spawn_background_reindex_file(project_root, file_path);
        None
    } else {
        payload.tool_input
    };

    // Prefer not to drop any message: index-ready, change-neighbors, and
    // read-neighbors are all surfaced together. Index-ready goes first (most
    // urgent); the rest append in order.
    let index_ready = config
        .as_ref()
        .and_then(|cfg| check_index_ready(project_root, cfg, "PostToolUse"));

    let change_neighbors =
        take_change_neighbors_context(project_root, config.as_ref(), "PostToolUse");

    let read_neighbors = read_surfacing_context(
        project_root,
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
    project_root: &Path,
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

    // Claude Code typically sends absolute paths. Strip the project root to get
    // a relative path so it matches what the store indexes. Reject anything that
    // escapes the project root (absolute path outside the root, or `..` traversal).
    let raw = Path::new(file_path);
    let relative = if raw.is_absolute() {
        let raw_canonical = raw.canonicalize();
        let raw_absolute = raw_canonical.as_deref().unwrap_or(raw);
        let root_canonical = project_root.canonicalize();
        let root_absolute = root_canonical.as_deref().unwrap_or(project_root);
        raw_absolute.strip_prefix(root_absolute).ok()?.to_path_buf()
    } else {
        raw.to_path_buf()
    };
    let read_path = RelativePath::from_path(&relative);
    read_path.reject_escape(READ_NEIGHBOR_RECOVERY).ok()?;

    let start = input.offset.unwrap_or(1);
    let end = input
        .limit
        .map(|count| start.saturating_add(count.saturating_sub(1)));

    let store = Store::new(project_root, cfg).ok()?;
    let all_rows = store.read_chunks().await.ok()?;

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
    let hits = tokio::task::spawn_blocking(move || {
        neighbors(&all_rows, &query_vectors, &exclude, top_k, min_similarity)
    })
    .await
    .ok()?;
    if hits.is_empty() {
        return None;
    }

    let locations: Vec<String> = hits
        .iter()
        .map(|n| {
            let name_part = n
                .name
                .as_deref()
                .map(|name| format!(" `{name}`"))
                .unwrap_or_default();
            format!(
                "{}:{}-{}{} ({:.2})",
                n.file_path, n.line_start, n.line_end, name_part, n.score
            )
        })
        .collect();

    let region = match end {
        Some(end) => format!("lines {start}-{end}"),
        None => format!("lines {start}+"),
    };
    let context = format!(
        "claudix: code related to {region} of `{}`: {}",
        read_path.as_str(),
        locations.join("; "),
    );

    Some(json!({
        "hookSpecificOutput": {
            "hookEventName": event_name,
            "additionalContext": context,
        }
    }))
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
    project_root: &Path,
    config: Option<&Config>,
    event_name: &str,
) -> Option<Value> {
    let cfg = config?;
    if !cfg.hooks.surface_related_on_edit {
        return None;
    }
    let store = Store::new(project_root, cfg).ok()?;
    let marker_path = store.change_neighbors_marker_path();
    let marker = change_neighbors::read_and_remove(&marker_path)?;

    let hits: Vec<String> = marker
        .neighbors
        .iter()
        .filter(|n| n.file_path != marker.edited_path)
        .map(|n| {
            let name_part = n
                .name
                .as_deref()
                .map(|name| format!(" `{name}`"))
                .unwrap_or_default();
            format!(
                "{}:{}-{}{}  ({:.2})",
                n.file_path, n.line_start, n.line_end, name_part, n.score
            )
        })
        .collect();

    if hits.is_empty() {
        return None;
    }

    let context = format!(
        "claudix: code related to your edit of `{}` (may need matching changes): {}",
        marker.edited_path,
        hits.join("; "),
    );

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

pub(super) fn watcher_alive(project_root: &Path, config: &Config) -> bool {
    let Ok(store) = Store::new(project_root, config) else {
        return false;
    };
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
/// Fail-open: any error during the check returns `true` so a legitimate edit
/// is still reindexed if the filter setup itself fails.
fn reindex_target_is_watchable(project_root: &Path, file_path: &str) -> bool {
    let raw = Path::new(file_path);
    let relative = if raw.is_absolute() {
        // Canonicalise both sides so a project root on a symlinked prefix
        // (macOS `/tmp` → `/private/tmp`) still strip-prefix-matches the
        // canonical raw path Claude Code hands us.
        let raw_canonical = raw.canonicalize();
        let raw_absolute = raw_canonical.as_deref().unwrap_or(raw);
        let root_canonical = project_root.canonicalize();
        let root_absolute = root_canonical.as_deref().unwrap_or(project_root);
        match raw_absolute.strip_prefix(root_absolute) {
            Ok(relative) => relative.to_path_buf(),
            Err(_) => return false,
        }
    } else {
        raw.to_path_buf()
    };
    if relative.as_os_str().is_empty() {
        return false;
    }
    match WatchFilter::load(project_root) {
        Ok(filter) => filter.is_watchable(&relative),
        Err(_) => true,
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

    #[test]
    fn reindex_target_is_watchable_rejects_index_internal_paths() {
        let fixture = TestFixture::new("small_rust").unwrap_or_else(|_| unreachable!());
        assert!(reindex_target_is_watchable(fixture.root(), "src/math.rs"));
        assert!(!reindex_target_is_watchable(
            fixture.root(),
            ".claudix/manifest.json"
        ));
        assert!(!reindex_target_is_watchable(fixture.root(), ".git/HEAD"));
        // Outside the project root: claude code generally resolves to absolute
        // paths inside CLAUDE_PROJECT_DIR, but defend in depth.
        let absolute_outside = std::env::temp_dir().join("nope.rs");
        assert!(!reindex_target_is_watchable(
            fixture.root(),
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
            watcher_alive(fixture.root(), &config),
            "current-PID watch marker must register as alive"
        );
        Ok(())
    }

    #[tokio::test]
    async fn watcher_alive_returns_false_without_marker() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let config = stub_config();
        assert!(!watcher_alive(fixture.root(), &config));
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

        let response = take_change_neighbors_context(fixture.root(), Some(&config), "PostToolUse");
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

        let _ = take_change_neighbors_context(fixture.root(), Some(&config), "PostToolUse");

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
        let response = take_change_neighbors_context(fixture.root(), Some(&config), "PostToolUse");
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

        let response = take_change_neighbors_context(fixture.root(), Some(&config), "PostToolUse");
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

        let response = take_change_neighbors_context(fixture.root(), Some(&config), "PostToolUse");
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
    fn read_surfacing_defaults_on() {
        assert!(Config::default().hooks.surface_related_on_read);
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
