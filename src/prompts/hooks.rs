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
        "claudix semantic search unavailable: the index was built with a different embedding model. Call the reindex tool with force: true (or run /claudix:index) to rebuild.".to_owned()
    } else if chunk_count == 0 && indexing_in_flight {
        format!(
            "claudix is building its first index in the background; you'll be notified here when it's ready. Don't poll get_index_status.{progress_suffix}"
        )
    } else if chunk_count == 0 {
        "claudix is installed but the index is empty. Run /claudix:index to build it; until then use Grep or Read for code discovery.".to_owned()
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

pub fn indexing_failed_response(event_name: &str) -> Value {
    json!({
        "hookSpecificOutput": {
            "hookEventName": event_name,
            "additionalContext":
                "claudix background indexing ended without updating the index. \
                 Run /claudix:doctor to diagnose.",
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
            lines.push(truncate_snippet(&chunk.content, 20));
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

fn truncate_snippet(content: &str, max_lines: usize) -> String {
    let mut lines = content.lines();
    let taken: Vec<&str> = lines.by_ref().take(max_lines).collect();
    if lines.next().is_some() {
        format!("{}\n…", taken.join("\n"))
    } else {
        taken.join("\n")
    }
}

// --- PostToolUse neighbor surfacing -----------------------------------------

fn neighbor_name_part(name: Option<&str>) -> String {
    name.map(|n| format!(" `{n}`")).unwrap_or_default()
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

pub fn read_related_context(
    read_path: &str,
    window_start: u32,
    window_end: Option<u32>,
    locations: &[String],
) -> String {
    let region = match window_end {
        Some(end) => format!("lines {window_start}-{end}"),
        None => format!("lines {window_start}+"),
    };
    format!(
        "claudix: code related to {region} of `{read_path}`: {}",
        locations.join("; "),
    )
}

pub fn edit_related_context(edited_path: &str, locations: &[String]) -> String {
    format!(
        "claudix: code related to your edit of `{edited_path}` (may need matching changes): {}",
        locations.join("; "),
    )
}
