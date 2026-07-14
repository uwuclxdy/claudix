//! Every agent-visible string claudix emits, in one place.
//!
//! - [`mcp`] — MCP tool definitions (descriptions + input schemas).
//! - [`hooks`] — hook `additionalContext` / `systemMessage` builders.
//! - [`hints`] — error recovery hints surfaced via `ClaudixError::recovery_hint`.
//!
//! Edit wording here; callers only supply data. Dependency direction is
//! one-way: `prompts` reaches down into lower-level domain types (`search`)
//! for rendering, and the rest of the tree reaches in here for text — never
//! the reverse.
//!
//! Not here, by necessity: `commands/*.md` (slash-command text, read off disk
//! by Claude Code — the binary never serves it) and `hooks/hooks.json`
//! (matcher config, not prose).

pub mod hints;
pub mod hooks;
pub mod mcp;

/// Line cap for a code snippet handed to the agent, shared by the MCP
/// `search_code` payload and the PreToolUse grep-intercept context. Tree-sitter
/// chunks are whole functions or impl blocks with no upper bound, so an uncapped
/// snippet lets one fat chunk cost more context than the rest of the response.
pub const SNIPPET_MAX_LINES: usize = 20;

/// First `max_lines` of `content`, with an ellipsis line when anything was cut.
pub fn truncate_snippet(content: &str, max_lines: usize) -> String {
    let mut lines = content.lines();
    let taken: Vec<&str> = lines.by_ref().take(max_lines).collect();
    if lines.next().is_some() {
        format!("{}\n…", taken.join("\n"))
    } else {
        taken.join("\n")
    }
}

/// Tool names the catalog no longer serves. A hook string or tool description
/// still naming one points the agent at a call that fails as unknown-tool, and
/// nothing else in the build catches it — the strings are data, not symbols, so
/// dropping the handler leaves them compiling and wrong.
#[cfg(test)]
const RETIRED_TOOL_NAMES: [&str; 3] = ["get_index_status", "clear_index", "reindex_file"];

#[cfg(test)]
mod tests {
    use super::*;

    /// Render every branch of every agent-visible hook builder.
    fn rendered_agent_strings() -> Vec<String> {
        let mut rendered = vec![
            hooks::session_start_message(&["binary", "hooks"]),
            hooks::session_start_response(0, 0, false, true, false, None).to_string(),
            hooks::session_start_response(0, 0, false, false, true, Some("log")).to_string(),
            hooks::session_start_response(0, 0, false, false, false, None).to_string(),
            hooks::session_start_response(4, 9, false, false, true, Some("log")).to_string(),
            hooks::session_start_response(4, 9, true, false, false, None).to_string(),
            hooks::session_start_response(4, 9, false, false, false, None).to_string(),
            hooks::indexing_complete_response("SessionStart", 4, 9).to_string(),
            hooks::indexing_failed_response("SessionStart", "log", Some("boom")).to_string(),
            hooks::read_related_context("a.rs", 1, Some(9), &["b.rs:1-2".to_owned()]),
            hooks::edit_related_context("a.rs", &["b.rs:1-2".to_owned()]),
        ];
        rendered.extend(
            mcp::tool_definitions(true)
                .iter()
                .map(|tool| tool["description"].to_string()),
        );
        rendered
    }

    #[test]
    fn no_agent_visible_string_names_a_retired_tool() {
        let served: Vec<String> = mcp::tool_definitions(true)
            .iter()
            .filter_map(|tool| tool.get("name")?.as_str().map(str::to_owned))
            .collect();

        // Keeps the list honest: a name that came back must leave this array,
        // or the check below silently forbids a tool that legitimately exists.
        for retired in RETIRED_TOOL_NAMES {
            assert!(
                !served.iter().any(|name| name == retired),
                "{retired} is served again — remove it from RETIRED_TOOL_NAMES"
            );
        }

        for text in rendered_agent_strings() {
            for retired in RETIRED_TOOL_NAMES {
                assert!(
                    !text.contains(retired),
                    "an agent-visible string still names the retired tool {retired}: {text}"
                );
            }
        }
    }

    #[test]
    fn truncate_snippet_keeps_short_content_verbatim() {
        assert_eq!(truncate_snippet("a\nb", 20), "a\nb");
    }

    #[test]
    fn truncate_snippet_cuts_and_marks_overlong_content() {
        let content = (1..=25)
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        let truncated = truncate_snippet(&content, 20);

        assert!(truncated.starts_with("1\n2\n"));
        assert!(truncated.ends_with("20\n…"));
        assert_eq!(truncated.lines().count(), 21);
    }

    #[test]
    fn truncate_snippet_at_exact_cap_adds_no_ellipsis() {
        let content = (1..=20)
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join("\n");

        assert_eq!(truncate_snippet(&content, 20), content);
    }
}
