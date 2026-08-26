use std::path::Path;

use serde_json::Value;

use crate::cli;
use crate::config;
use crate::error::Result;
use crate::prompts::hooks::{session_start_message, session_start_response};
use crate::store::Store;

use super::payload::HookPayload;
use super::spawn::{spawn_background_index, spawn_background_watch};

pub(super) async fn handle_session_start(
    project_root: &Path,
    payload: HookPayload,
) -> Result<Option<Value>> {
    let config = config::load(project_root).ok();

    let indexing_spawned = if let Some(ref config) = config
        && config.hooks.auto_index_on_session_start
    {
        spawn_background_index(project_root, config)
    } else {
        false
    };
    if let Some(ref config) = config {
        let _ = spawn_background_watch(project_root, config, payload.session_id.as_deref());
    }

    let store = config
        .as_ref()
        .and_then(|config| Store::new(project_root, config).ok());
    // New session → forget which related-code pairs were already surfaced and
    // which lines it has Read, so both dedupe ledgers start clean. Only the
    // starting session's own files are reset outright; other sessions' files
    // are reaped only once the age-gated sweep below proves them stale. Fail-open.
    if let Some(store) = store.as_ref() {
        if let Some(session_id) = payload.session_id.as_deref() {
            let _ = std::fs::remove_file(store.change_neighbors_seen_path(session_id));
            let _ = std::fs::remove_file(store.change_neighbors_read_path(session_id));
        }
        // Reap ledgers no live session has touched within the retention window:
        // dead sessions' per-session files, plus the pre-hq-4 shared file —
        // which a concurrently running old binary may still append, so the age
        // gate keeps it while it is live rather than deleting on sight.
        crate::store::marker::change_neighbors::sweep_stale_ledgers(store.state_dir_path());
    }
    let manifest = store
        .as_ref()
        .and_then(|store| store.read_manifest().ok().flatten());
    // A spawn that lost the claim (another session already started one) still
    // counts as "in flight" for the user-facing message — otherwise we'd tell
    // them the index is empty while it's actively being built.
    // A failed run's marker now survives its surfaced failure (each turn
    // resurfaces it), so marker existence alone reads as "building" forever on
    // a broken repo. Gate on liveness instead: fresh claim or a live child.
    let indexing_in_flight = indexing_spawned
        || store.as_ref().is_some_and(|store| {
            store.full_index_running()
                || crate::store::marker::pending_index::is_in_flight(
                    &store.pending_index_marker_path(),
                )
        });

    let indexed_file_count = manifest.as_ref().map(|m| m.file_count).unwrap_or(0);
    let indexed_chunk_count = manifest.as_ref().map(|m| m.chunk_count).unwrap_or(0);
    let index_stale = match (&manifest, &config) {
        (Some(m), Some(c)) => m.is_stale(c),
        _ => indexed_chunk_count == 0,
    };
    let model_mismatch = match (&manifest, &config) {
        (Some(m), Some(c)) => m.embedding_model != c.embedding.model,
        _ => false,
    };

    let log_hint = indexing_in_flight
        .then_some(config.as_ref())
        .flatten()
        .map(|c| {
            c.paths
                .log_dir
                .join("index.log")
                .to_string_lossy()
                .into_owned()
        });
    let mut response = session_start_response(
        indexed_file_count,
        indexed_chunk_count,
        index_stale,
        model_mismatch,
        indexing_in_flight,
        log_hint.as_deref(),
    );
    let user_message = match cli::setup_state(project_root).await {
        cli::SetupState::Ready => String::new(),
        cli::SetupState::Missing(missing) => session_start_message(&missing),
    };
    if !user_message.is_empty() {
        response["systemMessage"] = Value::String(user_message);
    }
    Ok(Some(response))
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::fs;

    use crate::config::Config;
    use crate::hooks::{HookEvent, run};
    use crate::store::{Manifest, Store};
    use crate::util::now_rfc3339;

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
        let mut config = stub_config();
        // Disable auto-index so the additionalContext assertion stays focused on
        // the empty-index state rather than the in-flight indexing message.
        config.hooks.auto_index_on_session_start = false;
        write_config(fixture.root(), &config);

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
            "claudix is installed but the index is empty. Call the reindex tool (or run `claudix index`) to build it; until then use Grep or Read for code discovery."
        );
    }

    #[test]
    fn session_start_message_reports_ready_setup() {
        assert_eq!(session_start_message(&[]), "");
        assert_eq!(
            session_start_message(&["bin"]),
            "claudix setup incomplete (missing bin); run the install script again"
        );
    }

    #[test]
    fn session_start_context_guides_tool_choice() {
        let response = session_start_response(42, 683, false, false, false, None);
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

    #[test]
    fn session_start_context_reports_background_rebuild_on_model_mismatch() {
        let in_flight = session_start_response(42, 683, false, true, true, None);
        let context = in_flight["hookSpecificOutput"]["additionalContext"]
            .as_str()
            .unwrap_or_default();
        assert!(
            context.contains("different embedding model"),
            "mismatch must be named, got: {context}"
        );
        assert!(
            context.contains("rebuild is in progress"),
            "an in-flight mismatch rebuild must be announced, got: {context}"
        );
        assert!(
            !context.contains("Call the reindex tool"),
            "an in-flight rebuild must not ask the user to act, got: {context}"
        );

        let not_in_flight = session_start_response(42, 683, false, true, false, None);
        let context = not_in_flight["hookSpecificOutput"]["additionalContext"]
            .as_str()
            .unwrap_or_default();
        assert!(
            context.contains("Call the reindex tool"),
            "a mismatch without a running rebuild must keep the manual guidance, got: {context}"
        );
    }

    #[tokio::test]
    async fn session_start_reports_indexing_in_flight_when_marker_exists()
    -> crate::error::Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let mut config = stub_config();
        config.hooks.auto_index_on_session_start = false;
        write_config(fixture.root(), &config);

        let store = Store::new(fixture.root(), &config)?;
        store.ensure_layout()?;
        let payload = format!("none\n{}\n0\n", now_rfc3339());
        fs::write(store.pending_index_marker_path(), payload)?;

        let response = run(fixture.root(), HookEvent::SessionStart, "{}").await?;
        let response = response.unwrap_or(Value::Null);
        let context = response["hookSpecificOutput"]["additionalContext"]
            .as_str()
            .unwrap_or_default();
        assert!(
            context.contains("building its first index"),
            "expected in-flight message for empty manifest with pending marker, got: {context}"
        );
        Ok(())
    }

    /// SessionStart on a fresh, populated, model-mismatched store must spawn
    /// the background rebuild (plain `claudix index` auto-clears) and say so —
    /// not tell the user to act (ruling 2026-08-26).
    #[tokio::test]
    async fn session_start_spawns_background_rebuild_on_model_mismatch() -> crate::error::Result<()>
    {
        let fixture = TestFixture::new("small_rust")?;
        let config = stub_config();
        write_config(fixture.root(), &config);

        let store = Store::new(fixture.root(), &config)?;
        store.ensure_layout()?;
        let mut manifest = Manifest::new("other-model", config.embedding.dimensions);
        manifest.chunk_count = 42;
        manifest.file_count = 7;
        // Fresh so the spawn must happen because of the mismatch alone.
        manifest.last_full_index_at = Some(now_rfc3339());
        store.write_manifest(&manifest)?;

        let response = run(fixture.root(), HookEvent::SessionStart, "{}").await?;
        let response = response.unwrap_or(Value::Null);
        let context = response["hookSpecificOutput"]["additionalContext"]
            .as_str()
            .unwrap_or_default();
        assert!(
            context.contains("rebuild is in progress"),
            "a model-mismatched store must start a rebuild at SessionStart, got: {context}"
        );
        Ok(())
    }
}
