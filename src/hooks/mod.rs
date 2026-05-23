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
            let index_ready = config
                .as_ref()
                .and_then(|cfg| check_index_ready(project_root, cfg, "UserPromptSubmit"));
            let neighbors =
                take_change_neighbors_context(project_root, config.as_ref(), "UserPromptSubmit");
            Ok(combine_hook_responses(
                "UserPromptSubmit",
                [index_ready, neighbors],
            ))
        }
    }
}
