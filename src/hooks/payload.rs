use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Deserialize)]
pub(super) struct HookPayload {
    pub tool_name: Option<String>,
    pub tool_input: Option<ToolInput>,
}

#[derive(Debug, Deserialize)]
pub(super) struct ToolInput {
    pub file_path: Option<String>,
    pub notebook_path: Option<String>,
    pub pattern: Option<String>,
    pub command: Option<String>,
    pub path: Option<String>,
    pub include: Option<String>,
    pub glob: Option<String>,
    #[serde(rename = "type")]
    pub file_type: Option<String>,
    pub output_mode: Option<String>,
    pub head_limit: Option<Value>,
    #[serde(rename = "-A")]
    pub after_lines: Option<Value>,
    #[serde(rename = "-B")]
    pub before_lines: Option<Value>,
    #[serde(rename = "-C")]
    pub context_lines: Option<Value>,
    pub multiline: Option<bool>,
}

pub(super) fn grep_input_has_scoping_flag(input: &ToolInput) -> bool {
    input.path.is_some()
        || input.include.is_some()
        || input.glob.is_some()
        || input.file_type.is_some()
        || input.output_mode.is_some()
        || input.head_limit.is_some()
        || input.after_lines.is_some()
        || input.before_lines.is_some()
        || input.context_lines.is_some()
        || input.multiline.is_some()
}
