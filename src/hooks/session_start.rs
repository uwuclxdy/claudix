use std::path::Path;

use serde_json::{Value, json};

use crate::cli;
use crate::config;
use crate::error::Result;
use crate::store::Store;

use super::payload::HookPayload;
use super::spawn::{spawn_background_index, spawn_background_watch};

pub(super) async fn handle_session_start(
    project_root: &Path,
    _payload: HookPayload,
) -> Result<Option<Value>> {
    let config = config::load(project_root).ok();

    let indexing_spawned = if let Some(ref config) = config
        && config.hooks.auto_index_on_session_start
        && crate::enumeration::is_git_repo(project_root)
    {
        spawn_background_index(project_root, config)
    } else {
        false
    };
    if let Some(ref config) = config
        && crate::enumeration::is_git_repo(project_root)
    {
        let _ = spawn_background_watch(project_root, config);
    }

    let store = config
        .as_ref()
        .and_then(|config| Store::new(project_root, config).ok());
    let manifest = store
        .as_ref()
        .and_then(|store| store.read_manifest().ok().flatten());
    // A spawn that lost the claim (another session already started one) still
    // counts as "in flight" for the user-facing message — otherwise we'd tell
    // them the index is empty while it's actively being built.
    let indexing_in_flight = indexing_spawned
        || store.as_ref().is_some_and(|store| {
            store.full_index_running() || store.pending_index_marker_path().exists()
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
    let user_message = session_start_message(cli::setup_state(project_root).await);
    if !user_message.is_empty() {
        response["systemMessage"] = Value::String(user_message);
    }
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

fn session_start_response(
    file_count: u64,
    chunk_count: u64,
    stale: bool,
    model_mismatch: bool,
    indexing_in_flight: bool,
    log_hint: Option<&str>,
) -> Value {
    let progress_suffix = log_hint
        .map(|p| format!(" Tail `{p}` to follow progress."))
        .unwrap_or_default();
    let additional_context = if model_mismatch {
        "claudix semantic search unavailable — embedding model mismatch. Run `claudix clear && claudix index` to rebuild.".to_owned()
    } else if chunk_count == 0 && indexing_in_flight {
        format!(
            "claudix is building its first index in the background — search_code will report when ready. \
             Use Grep or Read in the meantime.{progress_suffix}"
        )
    } else if chunk_count == 0 {
        "claudix is installed but the index is empty. Run /claudix:index to build it; until then use Grep or Read for code discovery.".to_owned()
    } else if indexing_in_flight {
        format!(
            "claudix semantic search ready — {file_count} files, {chunk_count} chunks (reindexing in background; you'll be notified when complete). \
             Use search_code for fast semantic search: conceptual queries, identifier lookups, cross-file discovery. \
             Use Grep for exact literals, regexes, or path-filtered scans.{progress_suffix}"
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

#[cfg(test)]
mod tests {
    use super::*;

    use std::fs;

    use crate::config::Config;
    use crate::hooks::{HookEvent, run};
    use crate::store::Store;
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
}
