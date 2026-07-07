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
mod spawn;

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
    let payload: HookPayload = if payload.trim().is_empty() {
        HookPayload {
            tool_name: None,
            tool_input: None,
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
                (Some(st), Some(cfg)) => check_index_ready(st, cfg, "UserPromptSubmit"),
                _ => None,
            };
            let neighbors =
                take_change_neighbors_context(store.as_ref(), config.as_ref(), "UserPromptSubmit");
            Ok(combine_hook_responses(
                "UserPromptSubmit",
                [index_ready, neighbors],
            ))
        }
    }
}

#[cfg(test)]
mod tests {
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
}
