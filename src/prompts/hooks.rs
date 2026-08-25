//! Hook `additionalContext` / `systemMessage` builders for every hook event.

use serde_json::{Value, json};

use crate::search::SearchResult;

// --- SessionStart ----------------------------------------------------------

/// User-facing `systemMessage` shown when plugin assets are missing.
///
/// `missing` lists the absent asset labels; an empty slice means setup is
/// complete and yields an empty message. The caller maps its setup state into
/// this slice, so this module stays free of `cli` types.
pub fn session_start_message(missing: &[&str]) -> String {
    if missing.is_empty() {
        return String::new();
    }
    format!(
        "claudix setup incomplete (missing {}); run the install script again",
        missing.join(", ")
    )
}

pub fn session_start_response(
    file_count: u64,
    chunk_count: u64,
    stale: bool,
    model_mismatch: bool,
    indexing_in_flight: bool,
    log_hint: Option<&str>,
) -> Value {
    let progress_suffix = log_hint
        .map(|p| format!(" Tail `{p}` to check progress."))
        .unwrap_or_default();
    let additional_context = if model_mismatch {
        "claudix semantic search unavailable: the index was built with a different embedding model. Call the reindex tool (or run `claudix index`) to rebuild.".to_owned()
    } else if chunk_count == 0 && indexing_in_flight {
        format!(
            "claudix is building its first index in the background; you'll be notified here when it's ready, so just carry on with Grep or Read until then.{progress_suffix}"
        )
    } else if chunk_count == 0 {
        "claudix is installed but the index is empty. Call the reindex tool (or run `claudix index`) to build it; until then use Grep or Read for code discovery.".to_owned()
    } else if indexing_in_flight {
        format!(
            "claudix semantic search ready: {file_count} files, {chunk_count} chunks (reindexing in background; you'll be notified when complete). \
             Use search_code for fast semantic search by what the code does: conceptual queries, identifier lookups, cross-file discovery. \
             Use Grep for exact literals or regexes.{progress_suffix}"
        )
    } else if stale {
        format!(
            "claudix semantic search ready: {file_count} files, {chunk_count} chunks (index stale: files changed since the last index, line numbers may drift; call reindex to refresh). \
             Use search_code for fast semantic search by what the code does: conceptual queries, identifier lookups, cross-file discovery. \
             Use Grep for exact literals or regexes."
        )
    } else {
        format!(
            "claudix semantic search ready: {file_count} files, {chunk_count} chunks. \
             Use search_code for fast semantic search by what the code does: conceptual queries, identifier lookups, cross-file discovery. \
             Use Grep for exact literals or regexes."
        )
    };
    json!({
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": additional_context,
        }
    })
}

// --- Index readiness (SessionStart / PostToolUse / UserPromptSubmit) -------

pub fn indexing_complete_response(event_name: &str, file_count: u64, chunk_count: u64) -> Value {
    json!({
        "hookSpecificOutput": {
            "hookEventName": event_name,
            "additionalContext": format!(
                "claudix indexing complete: {} files, {} chunks. Semantic search is now ready: \
                 use search_code for conceptual queries by what the code does, identifier lookups, and cross-file discovery.",
                file_count, chunk_count
            ),
        }
    })
}

pub fn indexing_failed_response(
    event_name: &str,
    log_path: &str,
    last_error: Option<&str>,
) -> Value {
    let error_suffix = last_error
        .map(|line| format!(" Last error: {line}."))
        .unwrap_or_default();
    json!({
        "hookSpecificOutput": {
            "hookEventName": event_name,
            "additionalContext": format!(
                "claudix background indexing ended without updating the index.{error_suffix} \
                 See `{log_path}` for the full log, or run /claudix:doctor to diagnose."
            ),
        }
    })
}

// --- PreToolUse grep intercept ----------------------------------------------

pub fn pre_tool_use_search_response(query: &str, results: Vec<SearchResult>) -> Value {
    let mut lines = vec![
        format!(
            "claudix search results for '{query}' (this Grep was intercepted and answered semantically):"
        ),
        String::new(),
    ];
    for result in &results {
        let chunk = &result.chunk;
        let name_part = chunk
            .name
            .as_deref()
            .map(|n| format!(" {n}"))
            .unwrap_or_default();
        let stale_warning = if result.stale {
            " [STALE - file modified since index]"
        } else {
            ""
        };
        lines.push(format!(
            "{}:{}-{} [{}] {}{name_part} (score {:.3}){}",
            chunk.file_path,
            chunk.line_range.start,
            chunk.line_range.end,
            chunk.language,
            chunk.kind,
            result.score,
            stale_warning,
        ));
        if !chunk.content.is_empty() {
            lines.push(super::truncate_snippet(
                &chunk.content,
                super::SNIPPET_MAX_LINES,
            ));
        }
        lines.push(String::new());
    }
    lines.push(
        "Tip: call the search_code MCP tool directly next time to skip this round-trip. For an exact literal/regex match, re-run Grep with an anchored pattern (^/$) or a file glob; those pass through untouched."
            .to_owned(),
    );
    let context = lines.join("\n");
    json!({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": format!("claudix found {} semantic matches for '{query}'", results.len()),
            "additionalContext": context,
        }
    })
}

// --- PostToolUse neighbor surfacing -----------------------------------------

fn neighbor_name_part(name: Option<&str>) -> String {
    name.map(|n| format!(" `{n}`")).unwrap_or_default()
}

/// Doc-file extensions, matched case-insensitively: `README.MD` is a markdown
/// file on any platform, and on the case-insensitive filesystems (Windows,
/// macOS) the uppercase spelling is the same file as the lowercase.
const DOC_EXTENSIONS: [&str; 8] = [
    "md", "markdown", "mdown", "rst", "txt", "adoc", "asciidoc", "org",
];

/// Conventional basenames of extensionless documentation files.
const DOC_BASENAMES: [&str; 4] = ["README", "LICENSE", "CHANGELOG", "CONTRIBUTING"];

/// Whether a repo-relative path is a documentation file, by its extension or
/// its conventional doc basename. Docs are classified by the file's own type,
/// not its directory: a `.rs` under `docs/` is source, a `README.md` at the
/// root is a doc. The store's language column cannot make this split — every
/// grammarless file (markdown, toml, yaml, json, shell) stores `unknown` — and
/// the change-neighbors marker carries only the path, so the check runs at
/// render time, where the path is all the caller has. A doc neither list
/// covers renders under `related code:` (loss-only: unrecognized, and every
/// spelling these lists match is a doc type, so no source file is mislabeled).
pub(crate) fn is_doc_file(file_path: &str) -> bool {
    let name = file_path
        .rsplit_once('/')
        .map_or(file_path, |(_, name)| name);
    match name.rsplit_once('.') {
        Some((_, ext)) => DOC_EXTENSIONS
            .iter()
            .any(|cand| ext.eq_ignore_ascii_case(cand)),
        None => DOC_BASENAMES
            .iter()
            .any(|cand| name.eq_ignore_ascii_case(cand)),
    }
}

/// One location line in the read-surfacing context.
pub fn read_neighbor_line(
    file_path: &str,
    line_start: u32,
    line_end: u32,
    name: Option<&str>,
    score: f32,
) -> String {
    format!(
        "{file_path}:{line_start}-{line_end}{} ({score:.2})",
        neighbor_name_part(name)
    )
}

/// One location line in the change-neighbors context.
pub fn edit_neighbor_line(
    file_path: &str,
    line_start: u32,
    line_end: u32,
    name: Option<&str>,
    score: f32,
) -> String {
    format!(
        "{file_path}:{line_start}-{line_end}{}  ({score:.2})",
        neighbor_name_part(name)
    )
}

/// One labeled group line, present only while the group is non-empty — the
/// label is omitted, never left bare.
fn group_line(label: &str, locations: &[String]) -> Option<String> {
    (!locations.is_empty()).then(|| format!("{label} {}", locations.join("; ")))
}

/// The preamble line plus one labeled line per non-empty group, so source
/// hits land under `related code:` and doc hits under `related docs:`.
fn with_groups(preamble: String, code_locations: &[String], doc_locations: &[String]) -> String {
    let mut lines = vec![preamble];
    lines.extend(group_line("related code:", code_locations));
    lines.extend(group_line("related docs:", doc_locations));
    lines.join("\n")
}

pub fn read_related_context(
    read_path: &str,
    window_start: u32,
    window_end: Option<u32>,
    code_locations: &[String],
    doc_locations: &[String],
) -> String {
    let region = match window_end {
        Some(end) => format!("lines {window_start}-{end}"),
        None => format!("lines {window_start}+"),
    };
    with_groups(
        format!("claudix: related to {region} of `{read_path}`:"),
        code_locations,
        doc_locations,
    )
}

pub fn edit_related_context(
    edited_path: &str,
    code_locations: &[String],
    doc_locations: &[String],
) -> String {
    with_groups(
        format!("claudix: related to your edit of `{edited_path}` (may need matching changes):"),
        code_locations,
        doc_locations,
    )
}

/// Variant for an ack whose event did not edit the recorded file (a later
/// event, or one without a file like UserPromptSubmit). Never claims "your
/// edit" — that wording is reserved for [`edit_related_context`], which the
/// caller uses only when the acking event's own file is the edited file.
pub fn recent_edit_related_context(
    edited_path: &str,
    code_locations: &[String],
    doc_locations: &[String],
) -> String {
    with_groups(
        format!(
            "claudix: related to a recent edit of `{edited_path}` (may need matching changes):"
        ),
        code_locations,
        doc_locations,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn doc_extensions_classify_docs() {
        for path in [
            "README.md",
            "docs/architecture.markdown",
            "docs/guide.mdown",
            "wiki/reference.rst",
            "LICENSE.txt",
            "docs/manual.adoc",
            "notes.asciidoc",
            "todo.org",
            // Uppercase spellings are the same file type, and on the
            // case-insensitive filesystems they are the same file.
            "README.MD",
            "docs/GUIDE.RST",
            // Conventional extensionless doc names.
            "README",
            "LICENSE",
            "CHANGELOG",
            "CONTRIBUTING",
        ] {
            assert!(is_doc_file(path), "{path} must classify as a doc");
        }
    }

    #[test]
    fn source_and_unknown_extensions_are_not_docs() {
        // A source file inside a docs directory is still source: the file's
        // own type decides, never its location. Unrecognized extensions and
        // basenames default to the code group (loss-only).
        for path in [
            "src/lib.rs",
            "SRC/MATH.RS",
            "Cargo.toml",
            ".gitignore",
            "docs/example.py",
            "src/notes.md.bak",
        ] {
            assert!(!is_doc_file(path), "{path} must not classify as a doc");
        }
    }

    #[test]
    fn empty_groups_render_no_label_line() {
        let code = vec!["src/math.rs:10-25 `add` (0.82)".to_owned()];
        let none: Vec<String> = Vec::new();

        let context = edit_related_context("src/lib.rs", &code, &none);
        assert!(
            context.contains("related code:"),
            "a source-only edit must carry the code label, got: {context}"
        );
        assert!(
            !context.contains("related docs:"),
            "an empty docs group must not render a bare label, got: {context}"
        );

        let context = edit_related_context("src/lib.rs", &none, &code);
        assert!(
            !context.contains("related code:"),
            "an empty code group must not render a bare label, got: {context}"
        );
        assert!(
            context.contains("related docs:"),
            "a doc-only edit must carry the docs label, got: {context}"
        );
    }

    #[test]
    fn both_groups_render_in_order() {
        let code = vec!["src/math.rs:10-25 `add` (0.82)".to_owned()];
        let docs = vec!["docs/guide.md:1-5 (0.81)".to_owned()];
        let context = edit_related_context("src/lib.rs", &code, &docs);
        assert!(
            context.find("related code:").is_some_and(|code_pos| {
                context
                    .find("related docs:")
                    .is_some_and(|docs_pos| code_pos < docs_pos)
            }),
            "the code group must precede the docs group, got: {context}"
        );
    }
}
