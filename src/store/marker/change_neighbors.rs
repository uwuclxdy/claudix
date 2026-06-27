//! The "change-neighbor surfacing" marker written by the detached reindex-file
//! child after computing semantic neighbors of the just-embedded chunks.
//!
//! Layout — single-line JSON file:
//!
//! ```json
//! {"edited_path":"src/foo.rs","neighbors":[{"file_path":"src/bar.rs","line_start":1,"line_end":10,"name":"fn_name","score":0.82}]}
//! ```
//!
//! The hook reads, surfaces, and atomically deletes this file on the next
//! `PostToolUse` or `UserPromptSubmit` event. If it does not exist the hook
//! produces no neighbors output.

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

/// Per-session dedup key for a surfaced (edited file → neighbor) pair. Keyed on
/// edited path + neighbor file + symbol so re-editing the same file won't repeat
/// the same neighbor, while a different edited file still can.
pub(crate) fn seen_key(
    edited_path: &str,
    neighbor_file: &str,
    neighbor_name: Option<&str>,
) -> String {
    format!(
        "{edited_path}\t{neighbor_file}\t{}",
        neighbor_name.unwrap_or("")
    )
}

/// Read the "already surfaced this session" ledger into a key set. Absent or
/// unreadable → empty set (fail-open).
pub(crate) fn read_seen(seen_path: &Path) -> HashSet<String> {
    fs::read_to_string(seen_path)
        .map(|content| content.lines().map(str::to_owned).collect())
        .unwrap_or_default()
}

/// Append newly-surfaced keys to the ledger (one per line). Best-effort: a write
/// failure just means a pair may surface again (fail-open).
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
