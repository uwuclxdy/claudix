use std::path::Path;

use serde_json::Value;

use crate::error::Result;

mod grep;
mod payload;
mod post_tool_use;
mod pre_tool_use;
mod ready_check;
mod session_start;
mod spawn;

pub(super) const WATCH_MARKER_STALE_SECS: u64 = 120;

use payload::HookPayload;

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
        HookEvent::UserPromptSubmit => post_tool_use::handle_user_prompt_submit(project_root).await,
    }
}
