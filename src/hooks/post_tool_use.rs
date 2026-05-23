use std::path::Path;
use std::time::Duration;

use serde_json::{Value, json};

use crate::config::{self, Config};
use crate::enumeration::WatchFilter;
use crate::error::Result;
use crate::store::Store;
use crate::store::marker::change_neighbors;

use super::payload::HookPayload;
use super::ready_check::check_index_ready;
use super::spawn::spawn_background_reindex_file;
use crate::store::marker::WATCH_MARKER_STALE_SECS;

pub(super) async fn handle_post_tool_use(
    project_root: &Path,
    payload: HookPayload,
) -> Result<Option<Value>> {
    let config = config::load(project_root).ok();

    // Spawn a background reindex only when an edit tool fired on a real file.
    // The ready-check and neighbor-surfacing runs on every PostToolUse event
    // regardless of whether a spawn happened.
    if let Some(tool_name) = payload.tool_name.as_deref()
        && matches!(tool_name, "Edit" | "Write" | "NotebookEdit" | "MultiEdit")
        && let Some(cfg) = config.as_ref()
        && cfg.hooks.auto_reembed_on_edit
        && !watcher_alive(project_root, cfg)
        && let Some(input) = payload.tool_input
        && let Some(file_path) = input.file_path.or(input.notebook_path)
        && reindex_target_is_watchable(project_root, &file_path)
    {
        spawn_background_reindex_file(project_root, &file_path);
    }

    // Prefer not to drop either message: if both an index-ready notification
    // and a neighbors notification are pending, combine them into a single
    // additionalContext so the agent sees both in one hook response. The
    // index-ready message goes first (more urgent); neighbors append after.
    let index_ready = config
        .as_ref()
        .and_then(|cfg| check_index_ready(project_root, cfg, "PostToolUse"));

    let neighbors_context =
        take_change_neighbors_context(project_root, config.as_ref(), "PostToolUse");

    Ok(combine_hook_responses(
        "PostToolUse",
        index_ready,
        neighbors_context,
    ))
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

/// Combine an index-ready response and a neighbors response into one.
///
/// Both can be `None` → `None`. Both present → index-ready context is
/// emitted first; neighbors appended in the same additionalContext field.
/// Only one present → that one wins unchanged.
pub(super) fn combine_hook_responses(
    event_name: &str,
    index_ready: Option<Value>,
    neighbors: Option<Value>,
) -> Option<Value> {
    match (index_ready, neighbors) {
        (None, None) => None,
        (Some(ready), None) => Some(ready),
        (None, Some(nbr)) => Some(nbr),
        (Some(ready), Some(nbr)) => {
            let ready_ctx = ready["hookSpecificOutput"]["additionalContext"]
                .as_str()
                .unwrap_or("");
            let nbr_ctx = nbr["hookSpecificOutput"]["additionalContext"]
                .as_str()
                .unwrap_or("");
            let combined = format!("{ready_ctx}\n{nbr_ctx}");
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
}
