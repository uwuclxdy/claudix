use std::path::Path;
use std::time::Duration;

use serde_json::Value;

use crate::config::{self, Config};
use crate::enumeration::WatchFilter;
use crate::error::Result;
use crate::store::Store;

use super::WATCH_MARKER_STALE_SECS;
use super::payload::{HookPayload, is_write_tool};
use super::ready_check::check_index_ready;
use super::spawn::spawn_background_reindex_file;

pub(super) async fn handle_post_tool_use(
    project_root: &Path,
    payload: HookPayload,
) -> Result<Option<Value>> {
    let Some(tool_name) = payload.tool_name.as_deref() else {
        return Ok(None);
    };

    let config = config::load(project_root).ok();

    // Skip the per-edit spawn when a live watcher already covers file changes;
    // otherwise rapid edits fan out N detached `reindex-file` processes that
    // each cold-load ONNX before losing the in-process lock to the watcher.
    // Also skip ignored paths (.claudix/, .git/, gitignored) — spawning a
    // reindex for `.claudix/manifest.json` would round-trip the index's own
    // metadata back through the embedder.
    if is_write_tool(tool_name)
        && let Some(cfg) = config.as_ref()
        && cfg.hooks.auto_reembed_on_edit
        && !watcher_alive(project_root, cfg)
        && let Some(input) = payload.tool_input
        && let Some(file_path) = input.file_path.or(input.notebook_path)
        && reindex_target_is_watchable(project_root, &file_path)
    {
        spawn_background_reindex_file(project_root, &file_path);
    }

    Ok(config
        .as_ref()
        .and_then(|cfg| check_index_ready(project_root, cfg, "PostToolUse")))
}

pub(super) async fn handle_user_prompt_submit(project_root: &Path) -> Result<Option<Value>> {
    let config = config::load(project_root).ok();
    Ok(config
        .as_ref()
        .and_then(|cfg| check_index_ready(project_root, cfg, "UserPromptSubmit")))
}

pub(super) fn watcher_alive(project_root: &Path, config: &Config) -> bool {
    let Ok(store) = Store::new(project_root, config) else {
        return false;
    };
    crate::store::marker::live_owner(
        &store.watch_marker_path(),
        Duration::from_secs(WATCH_MARKER_STALE_SECS),
    )
    .is_some()
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
}
