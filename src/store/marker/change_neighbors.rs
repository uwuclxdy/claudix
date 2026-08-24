//! The "change-neighbor surfacing" marker written by the detached reindex-file
//! child after computing semantic neighbors of the just-embedded chunks.
//!
//! Layout — single-line JSON file:
//!
//! ```json
//! {"edited_path":"src/foo.rs","session_id":"abc123","neighbors":[{"file_path":"src/bar.rs","line_start":1,"line_end":10,"name":"fn_name","score":0.82}]}
//! ```
//!
//! The hook reads, surfaces, and atomically deletes this file on the next
//! `PostToolUse` or `UserPromptSubmit` event. If it does not exist the hook
//! produces no neighbors output.
//!
//! `session_id` names the session whose edit produced the marker; the acking
//! hook drops (without surfacing) any marker whose session is not the acking
//! event's own. Absent on legacy markers — `#[serde(default)]`, and those read
//! as `None`, which no event matches, so a pre-attribution marker never
//! surfaces either.

use std::collections::HashSet;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

/// A single serialized neighbor entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct NeighborEntry {
    pub file_path: String,
    pub line_start: u32,
    pub line_end: u32,
    pub name: Option<String>,
    pub score: f32,
}

/// The full marker payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ChangeNeighborsMarker {
    pub edited_path: String,
    /// Session whose edit produced this marker; the ack only ever surfaces it
    /// to an event of that same session. `None` = unattributed (legacy format,
    /// or a writer that refused to guess) and never surfaces.
    #[serde(default)]
    pub session_id: Option<String>,
    pub neighbors: Vec<NeighborEntry>,
}

/// Write the marker. Silently overwrites any previous one (only the latest
/// edit matters; stale markers from a rapid sequence of edits are harmless).
pub(crate) fn write(marker_path: &Path, marker: &ChangeNeighborsMarker) {
    if let Ok(json) = serde_json::to_string(marker) {
        let _ = fs::write(marker_path, json);
    }
}

/// Read the marker. Returns `None` when absent or malformed (fail-open).
pub(crate) fn read(marker_path: &Path) -> Option<ChangeNeighborsMarker> {
    let content = fs::read_to_string(marker_path).ok()?;
    serde_json::from_str(&content).ok()
}

/// Read the marker and atomically remove it (ack). Returns `None` when absent.
pub(crate) fn read_and_remove(marker_path: &Path) -> Option<ChangeNeighborsMarker> {
    let marker = read(marker_path)?;
    let _ = fs::remove_file(marker_path);
    Some(marker)
}

/// Per-session dedup key for a surfaced neighbor: once shown, it is never shown
/// again this session. The edited path is deliberately not part of the key — a
/// related-code hint is only worth its context while the symbol is still unknown
/// to the agent, and a hub file that neighbors most of the tree would otherwise
/// resurface once per distinct file edited.
pub(crate) fn seen_key(neighbor_file: &str, neighbor_name: Option<&str>) -> String {
    format!("{neighbor_file}\t{}", neighbor_name.unwrap_or(""))
}

/// Read the "already surfaced this session" ledger into a key set. Absent or
/// unreadable → empty set (fail-open).
pub(crate) fn read_seen(seen_path: &Path) -> HashSet<String> {
    fs::read_to_string(seen_path)
        .map(|content| content.lines().map(str::to_owned).collect())
        .unwrap_or_default()
}

/// Append newly-surfaced keys to the ledger (one per line). Best-effort: a write
/// failure just means a neighbor may surface again (fail-open).
pub(crate) fn append_seen(seen_path: &Path, keys: &[String]) {
    if keys.is_empty() {
        return;
    }
    use std::io::Write;
    if let Ok(mut file) = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(seen_path)
    {
        for key in keys {
            let _ = writeln!(file, "{key}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn seen_key_separates_distinct_symbols_in_one_file() {
        assert_ne!(
            seen_key("src/math.rs", Some("add")),
            seen_key("src/math.rs", Some("multiply"))
        );
    }

    #[test]
    fn seen_key_is_stable_for_the_same_neighbor() {
        assert_eq!(
            seen_key("src/math.rs", Some("add")),
            seen_key("src/math.rs", Some("add"))
        );
        assert_eq!(seen_key("src/math.rs", Some("add")), "src/math.rs\tadd");
    }

    #[test]
    fn stale_three_field_ledger_lines_never_match_a_key() {
        let dir = tempdir().ok().unwrap_or_else(|| unreachable!());
        let seen_path = dir.path().join("change-neighbors-seen");
        // What the previous build wrote: {edited}\t{neighbor}\t{name}.
        let _ = fs::write(&seen_path, "src/lib.rs\tsrc/math.rs\tadd\n");

        let seen = read_seen(&seen_path);
        assert!(
            !seen.contains(&seen_key("src/math.rs", Some("add"))),
            "a stale line must simply miss, never be treated as a match"
        );
    }

    #[test]
    fn absent_ledger_reads_as_empty() {
        let dir = tempdir().ok().unwrap_or_else(|| unreachable!());
        assert!(read_seen(&dir.path().join("nope")).is_empty());
    }
}
