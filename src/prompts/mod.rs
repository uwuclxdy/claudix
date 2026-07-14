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
///
/// `lines()` + `\n` rejoin LF-normalizes CRLF content — deliberate: snippets
/// feed a model, where a `\r` is token noise, and the agent Reads the file
/// when it needs true bytes.
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

    /// Every source file whose string literals reach the agent, scanned as raw
    /// text. Rendering each builder instead would mean enumerating them, and an
    /// enumeration is itself a list someone forgets to extend — the exact failure
    /// this test exists to catch. `hints` alone is ~60 standalone consts that
    /// nothing enumerates. Raw text also flags a doc comment naming a dead tool,
    /// which is rot too. `mod.rs` is excluded: it holds RETIRED_TOOL_NAMES and
    /// would match itself. `cli/` is excluded: the retired MCP names live on
    /// there as legitimate CLI symbols (`run_reindex_file`), not rot. The two
    /// markdown files ship to the agent verbatim — the slash command auto-runs
    /// and the skill loads into context — so they scan like prompt sources.
    const PROMPT_SOURCES: [(&str, &str); 5] = [
        ("prompts/hints.rs", include_str!("hints.rs")),
        ("prompts/hooks.rs", include_str!("hooks.rs")),
        ("prompts/mcp.rs", include_str!("mcp.rs")),
        ("commands/doctor.md", include_str!("../../commands/doctor.md")),
        (
            "skills/claudix/SKILL.md",
            include_str!("../../skills/claudix/SKILL.md"),
        ),
    ];

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

        for (file, source) in PROMPT_SOURCES {
            for retired in RETIRED_TOOL_NAMES {
                assert!(
                    !source.contains(retired),
                    "{file} still names the retired tool {retired} — \
                     the agent will be pointed at a tool that no longer exists"
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
