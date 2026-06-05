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
    /// MultiEdit-style tools may carry a list of modified paths in addition to,
    /// or instead of, a single `file_path`. Unknown/missing is always `None`.
    #[serde(default)]
    pub files_modified: Option<Vec<String>>,
    /// Read tool 1-based start line.
    pub offset: Option<u32>,
    /// Read tool line count starting at `offset`.
    pub limit: Option<u32>,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_tool_input(json: &str) -> ToolInput {
        serde_json::from_str(json).expect("payload must parse")
    }

    #[test]
    fn tool_input_parses_single_file_path() {
        let input = parse_tool_input(r#"{"file_path": "src/lib.rs"}"#);
        assert_eq!(input.file_path.as_deref(), Some("src/lib.rs"));
        assert!(input.files_modified.is_none());
    }

    #[test]
    fn tool_input_parses_files_modified_array() {
        let input = parse_tool_input(r#"{"files_modified": ["src/lib.rs", "src/main.rs"]}"#);
        assert!(input.file_path.is_none());
        assert_eq!(
            input.files_modified.as_deref(),
            Some(&["src/lib.rs".to_owned(), "src/main.rs".to_owned()][..])
        );
    }

    #[test]
    fn tool_input_parses_both_file_path_and_files_modified_present() {
        // When both fields are present the consumer dedupes; parsing must succeed.
        let input = parse_tool_input(
            r#"{"file_path": "src/lib.rs", "files_modified": ["src/lib.rs", "src/main.rs"]}"#,
        );
        assert_eq!(input.file_path.as_deref(), Some("src/lib.rs"));
        let modified = input.files_modified.as_deref().unwrap_or(&[]);
        assert!(modified.contains(&"src/lib.rs".to_owned()));
        assert!(modified.contains(&"src/main.rs".to_owned()));
    }

    #[test]
    fn tool_input_parses_neither_field_present() {
        // Payload with neither field — noop on the consumer side.
        let input = parse_tool_input(r#"{"pattern": "foo"}"#);
        assert!(input.file_path.is_none());
        assert!(input.files_modified.is_none());
    }

    #[test]
    fn tool_input_unknown_fields_do_not_error() {
        // Extra keys (future Claude Code additions) must not break parsing.
        let input =
            parse_tool_input(r#"{"file_path": "src/lib.rs", "unknown_future_field": true}"#);
        assert_eq!(input.file_path.as_deref(), Some("src/lib.rs"));
    }
}
