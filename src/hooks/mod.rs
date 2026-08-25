use std::path::Path;

use serde_json::Value;

use crate::config;
use crate::error::Result;

mod grep;
mod payload;
mod post_tool_use;
mod pre_tool_use;
mod ready_check;
mod session_start;
pub(crate) mod spawn;

use payload::HookPayload;
use post_tool_use::{combine_hook_responses, take_change_neighbors_context};
use ready_check::check_index_ready;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HookEvent {
    SessionStart,
    PostToolUse,
    PreToolUse,
    UserPromptSubmit,
}

pub async fn run(project_root: &Path, event: HookEvent, payload: &str) -> Result<Option<Value>> {
    // claudix only indexes git repositories: outside one there is nothing to
    // index and no reason to lay down `.claudix/`. Passthrough (fail open) so no
    // hook touches disk in a non-git directory.
    if !crate::enumeration::is_git_repo(project_root) {
        return Ok(None);
    }

    let payload: HookPayload = if payload.trim().is_empty() {
        HookPayload {
            tool_name: None,
            tool_input: None,
            session_id: None,
            prompt_id: None,
        }
    } else {
        serde_json::from_str(payload)?
    };

    match event {
        HookEvent::SessionStart => session_start::handle_session_start(project_root, payload).await,
        HookEvent::PostToolUse => post_tool_use::handle_post_tool_use(project_root, payload).await,
        HookEvent::PreToolUse => pre_tool_use::handle_pre_tool_use(project_root, payload).await,
        HookEvent::UserPromptSubmit => {
            let config = config::load(project_root).ok();
            let store = config
                .as_ref()
                .and_then(|cfg| crate::store::Store::new(project_root, cfg).ok());
            let index_ready = match (store.as_ref(), config.as_ref()) {
                (Some(st), Some(cfg)) => check_index_ready(
                    st,
                    cfg,
                    "UserPromptSubmit",
                    payload.session_id.as_deref(),
                    payload.prompt_id.as_deref(),
                ),
                _ => None,
            };
            let neighbors = take_change_neighbors_context(
                store.as_ref(),
                config.as_ref(),
                "UserPromptSubmit",
                payload.session_id.as_deref(),
                &[],
            );
            Ok(combine_hook_responses(
                "UserPromptSubmit",
                [index_ready, neighbors],
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    use serde_json::{Value, json};

    use super::{HookEvent, run};
    use crate::config::Config;
    use crate::store::Store;

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

    fn failure_copies(response: &Option<Value>) -> usize {
        response
            .as_ref()
            .and_then(|v| v["hookSpecificOutput"]["additionalContext"].as_str())
            .map(|context| usize::from(context.contains("ended without updating the index")))
            .unwrap_or(0)
    }

    /// Broken index: a stale pending-index marker (first-index sentinel, dead
    /// child) and no manifest. `check_index_ready` then reads every submit past
    /// the failure grace as a failed background index.
    fn plant_broken_index(fixture: &TestFixture) -> crate::error::Result<()> {
        let mut config = stub_config();
        // No reindex queue / drain-worker spawns; these tests exercise the
        // ready-check arms only.
        config.hooks.auto_reembed_on_edit = false;
        write_config(fixture.root(), &config);
        let store = Store::new(fixture.root(), &config)?;
        store.ensure_layout()?;
        fs::write(
            store.pending_index_marker_path(),
            "none\n2025-01-01T00:00:00Z\n0\n",
        )?;
        Ok(())
    }

    /// The in-process panic guard `run_hook_command` relies on: a panic inside
    /// the spawned hook task surfaces as a `JoinError` (its `Ok(Err(_))` arm),
    /// not a process abort. Requires `panic = "unwind"` — guards against the
    /// release profile re-acquiring `panic = "abort"`.
    #[tokio::test]
    #[allow(clippy::panic)]
    async fn tokio_spawn_surfaces_hook_panic_as_join_error() {
        let handle = tokio::spawn(async { panic!("synthetic hook panic") });
        let result = handle.await;
        assert!(
            result.is_err(),
            "panicking spawn must return Err(JoinError)"
        );
        let is_panic = result.err().map(|error| error.is_panic()).unwrap_or(false);
        assert!(is_panic, "JoinError must report is_panic() = true");
    }

    #[tokio::test]
    async fn user_prompt_submit_attaches_index_failure_once_per_turn() -> crate::error::Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        plant_broken_index(&fixture)?;

        let submit = json!({ "session_id": "turn-session", "prompt_id": "prompt-1" }).to_string();
        let mut copies = 0;
        for _ in 0..3 {
            let response = run(fixture.root(), HookEvent::UserPromptSubmit, &submit).await?;
            copies += failure_copies(&response);
        }
        assert_eq!(
            copies, 1,
            "three submits in one turn must attach the index-failure context exactly once, got {copies}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn index_failure_attaches_fresh_copy_on_new_turn() -> crate::error::Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        plant_broken_index(&fixture)?;

        let turn1 = json!({ "session_id": "turn-session", "prompt_id": "prompt-1" }).to_string();
        let mut copies = 0;
        for _ in 0..3 {
            let response = run(fixture.root(), HookEvent::UserPromptSubmit, &turn1).await?;
            copies += failure_copies(&response);
        }
        assert_eq!(
            copies, 1,
            "turn 1 must attach the index-failure context exactly once"
        );

        // New turn, same session, index still broken: a fresh copy must attach
        // (the dedupe must not become once-per-session).
        let turn2 = json!({ "session_id": "turn-session", "prompt_id": "prompt-2" }).to_string();
        let response = run(fixture.root(), HookEvent::UserPromptSubmit, &turn2).await?;
        assert_eq!(
            failure_copies(&response),
            1,
            "a new turn must attach a fresh index-failure copy while the index stays broken"
        );
        Ok(())
    }

    #[tokio::test]
    async fn index_failure_dedupe_state_is_per_session() -> crate::error::Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        plant_broken_index(&fixture)?;

        // Same prompt id in both sessions: if the record were shared (one file
        // for every session), session B would read session A's prompt id as its
        // own and be wrongly suppressed. B must still surface its own copy.
        let session_a = json!({ "session_id": "sess-A", "prompt_id": "prompt-A" }).to_string();
        let response = run(fixture.root(), HookEvent::UserPromptSubmit, &session_a).await?;
        assert_eq!(
            failure_copies(&response),
            1,
            "the first session's submit must attach the index-failure context"
        );

        let session_b = json!({ "session_id": "sess-B", "prompt_id": "prompt-A" }).to_string();
        let response = run(fixture.root(), HookEvent::UserPromptSubmit, &session_b).await?;
        assert_eq!(
            failure_copies(&response),
            1,
            "another session must surface its own copy even with the same prompt id; the dedupe record is session-scoped"
        );
        Ok(())
    }

    #[tokio::test]
    async fn index_failure_dedupes_across_submit_and_tool_events_in_one_turn()
    -> crate::error::Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        plant_broken_index(&fixture)?;

        let submit = json!({ "session_id": "turn-session", "prompt_id": "prompt-1" }).to_string();
        let response = run(fixture.root(), HookEvent::UserPromptSubmit, &submit).await?;
        assert_eq!(
            failure_copies(&response),
            1,
            "the turn's submit must attach the context"
        );

        let write = json!({
            "session_id": "turn-session",
            "prompt_id": "prompt-1",
            "tool_name": "Write",
            "tool_input": { "file_path": "src/lib.rs" }
        })
        .to_string();
        let response = run(fixture.root(), HookEvent::PostToolUse, &write).await?;
        assert_eq!(
            failure_copies(&response),
            0,
            "a tool event in the same turn must not reattach the index-failure context"
        );
        Ok(())
    }
}
