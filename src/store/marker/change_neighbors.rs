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
//!
//! Besides the marker, this module owns the two per-session dedupe ledgers the
//! surfacing paths consult. Both are append-only newline text in the state dir
//! (paths via `Store`), reset on SessionStart, and never touched by an
//! unattributed event:
//!
//! - the seen ledger (`change-neighbors-seen-<session>`): every hint the
//!   session was shown, one `file\tname\tstart\tend` line per hint (empty name
//!   when the chunk carries none). A hint repeats only when a recorded line for
//!   the same file + symbol fully contains its range — the same pair at
//!   different lines is new context.
//! - the read ledger (`change-neighbors-read-<session>`): every file + line
//!   range the session Read, one `file\tstart\tend` line per Read, clamped to
//!   the Read tool's per-call cap and compacted on append (a window already
//!   contained in the ledger is not appended again). A hint is suppressed when
//!   a recorded range fully contains the lines it points at.
//!
//! Both ledgers fail toward a repeated hint, never a wrongly suppressed one:
//! an unreadable ledger reads as empty, an unparseable line is skipped, and a
//! failed append is silently dropped. A path or symbol containing a tab or
//! newline is rejected at write time and never recorded — it could not
//! round-trip through the line format, and a re-parsed fragment would forge a
//! valid record for a different file, which is the one failure mode that
//! WOULD wrongly suppress. Stale ledgers of dead sessions are age-reaped by
//! [`sweep_stale_ledgers`] on SessionStart.

use std::fs;
use std::path::Path;
use std::time::{Duration, SystemTime};

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

// ── per-session dedupe ledgers ───────────────────────────────────────────────

/// One surfaced hint recorded in the seen ledger: where it pointed. The edited
/// path is deliberately not part of the identity — a related-code hint is only
/// worth its context while the symbol is still unknown to the agent, and a hub
/// file that neighbors most of the tree would otherwise resurface once per
/// distinct file edited.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SeenEntry {
    pub file_path: String,
    /// `None` stored as the empty name, matching the pre-line-aware ledger.
    pub name: String,
    pub line_start: u32,
    pub line_end: u32,
}

impl SeenEntry {
    pub(crate) fn new(file_path: &str, name: Option<&str>, line_start: u32, line_end: u32) -> Self {
        Self {
            file_path: file_path.to_owned(),
            name: name.unwrap_or("").to_owned(),
            line_start,
            line_end,
        }
    }
}

/// One file + line range the session Read, recorded in the read ledger.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReadRange {
    pub file_path: String,
    pub line_start: u32,
    /// Inclusive end. Producers clamp to what the Read tool can have shown;
    /// the containment check accepts any end, so a wider producer stays correct.
    pub line_end: u32,
}

/// Read the seen ledger. Absent or unreadable → empty; unparseable lines are
/// skipped (fail-open: a missed match only ever repeats a hint).
pub(crate) fn read_seen(seen_path: &Path) -> Vec<SeenEntry> {
    fs::read_to_string(seen_path)
        .map(|content| content.lines().filter_map(parse_seen_line).collect())
        .unwrap_or_default()
}

fn parse_seen_line(line: &str) -> Option<SeenEntry> {
    let mut fields = line.split('\t');
    let file_path = fields.next()?;
    let name = fields.next()?;
    // Both separators are consumed by the splitting above, so this cannot fire
    // for any line this format's own writer produced — the write side rejects
    // such values, and the check documents that round-trip invariant at the
    // read site in case a future writer stops honoring it.
    if !ledger_safe(file_path) || !ledger_safe(name) {
        return None;
    }
    let line_start: u32 = fields.next()?.parse().ok()?;
    let line_end: u32 = fields.next()?.parse().ok()?;
    // A fifth field means the line was written by a newer shape; skip it rather
    // than guess. Lines from the older shapes have fewer fields and fail above.
    if fields.next().is_some() {
        return None;
    }
    Some(SeenEntry {
        file_path: file_path.to_owned(),
        name: name.to_owned(),
        line_start,
        line_end,
    })
}

/// Append newly-surfaced hints to the seen ledger (one line per entry).
/// Entries whose path or name cannot round-trip through the line format are
/// skipped. Best-effort: a write failure just means a hint may surface again
/// (fail-open).
pub(crate) fn append_seen(seen_path: &Path, entries: &[SeenEntry]) {
    if entries.is_empty() {
        return;
    }
    use std::io::Write;
    if let Ok(mut file) = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(seen_path)
    {
        for entry in entries {
            if !ledger_safe(&entry.file_path) || !ledger_safe(&entry.name) {
                continue;
            }
            let _ = writeln!(
                file,
                "{}\t{}\t{}\t{}",
                entry.file_path, entry.name, entry.line_start, entry.line_end
            );
        }
    }
}

/// Whether a hint at `[line_start, line_end]` of `file_path`/`name` was already
/// surfaced this session: a recorded entry for the same file + symbol whose
/// range fully contains the hint's. A partial overlap is not a repeat — the
/// hint names lines the agent has not all seen.
pub(crate) fn is_seen(
    seen: &[SeenEntry],
    file_path: &str,
    name: Option<&str>,
    line_start: u32,
    line_end: u32,
) -> bool {
    seen.iter().any(|entry| {
        entry.file_path == file_path
            && entry.name == name.unwrap_or("")
            && range_contains(entry.line_start, entry.line_end, line_start, line_end)
    })
}

/// Read the session's read ledger. Absent or unreadable → empty; unparseable
/// lines are skipped (fail-open: a missed match only ever repeats a hint).
pub(crate) fn read_ranges(read_path: &Path) -> Vec<ReadRange> {
    fs::read_to_string(read_path)
        .map(|content| content.lines().filter_map(parse_range_line).collect())
        .unwrap_or_default()
}

fn parse_range_line(line: &str) -> Option<ReadRange> {
    let mut fields = line.split('\t');
    let file_path = fields.next()?;
    // Same round-trip invariant as [`parse_seen_line`]: unreachable for lines
    // the current writer produces, kept as defense-in-depth at the read site.
    if !ledger_safe(file_path) {
        return None;
    }
    let line_start: u32 = fields.next()?.parse().ok()?;
    let line_end: u32 = fields.next()?.parse().ok()?;
    if fields.next().is_some() {
        return None;
    }
    Some(ReadRange {
        file_path: file_path.to_owned(),
        line_start,
        line_end,
    })
}

/// Append one Read window to the read ledger, skipping windows the ledger
/// already contains (the common case: re-reading the same or a narrower
/// window), so the ledger does not grow one line per repeat. A path that cannot
/// round-trip through the line format is never recorded. Best-effort
/// throughout: a failed read or write only costs a future suppression, never a
/// hint (fail-open); two concurrent hook processes may still append the same
/// window once each, and a duplicate line is harmless.
pub(crate) fn append_read_range(read_path: &Path, range: &ReadRange) {
    if !ledger_safe(&range.file_path) {
        return;
    }
    let existing = read_ranges(read_path);
    if is_read(
        &existing,
        &range.file_path,
        range.line_start,
        range.line_end,
    ) {
        return;
    }
    use std::io::Write;
    if let Ok(mut file) = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(read_path)
    {
        let _ = writeln!(
            file,
            "{}\t{}\t{}",
            range.file_path, range.line_start, range.line_end
        );
    }
}

/// Whether the session Read a range fully containing the hint's
/// `[line_start, line_end]`. A partial overlap still surfaces — the hint names
/// lines the agent has not all seen.
pub(crate) fn is_read(
    ranges: &[ReadRange],
    file_path: &str,
    line_start: u32,
    line_end: u32,
) -> bool {
    ranges.iter().any(|range| {
        range.file_path == file_path
            && range_contains(range.line_start, range.line_end, line_start, line_end)
    })
}

/// The containment rule both ledgers share: `outer` covers `inner` only when it
/// holds every line of it.
fn range_contains(outer_start: u32, outer_end: u32, inner_start: u32, inner_end: u32) -> bool {
    outer_start <= inner_start && outer_end >= inner_end
}

/// Whether a value can round-trip through the newline-delimited, tab-separated
/// ledger line it would be written to. A tab or newline would forge or split
/// entries on re-parse, and the forged record would wrongly suppress a real
/// hint — the one failure direction the ledgers otherwise exclude — so such
/// values are never recorded: a missed suppression at worst.
fn ledger_safe(value: &str) -> bool {
    !value.contains(['\t', '\n'])
}

/// How long a dedupe ledger survives without a touch before SessionStart reaps
/// it. Sized far past any live session's idle gaps: reclaiming a live-but-idle
/// session's ledger only means hints repeat (fail-open), never a lost one.
const DEDUPE_LEDGER_RETENTION: Duration = Duration::from_secs(7 * 24 * 60 * 60);

/// Reap dedupe ledgers no live session has touched within
/// [`DEDUPE_LEDGER_RETENTION`]: the per-session seen/read files of dead
/// sessions, plus the pre-hq-4 shared `change-neighbors-seen` file — the age
/// gate keeps a concurrently running old binary's file out of the sweep while
/// it is still being appended. Best-effort: an unreadable entry, a future
/// mtime, or a failed removal is skipped, and a skip only postpones reaping.
pub(crate) fn sweep_stale_ledgers(state_dir: &Path) {
    let Ok(entries) = fs::read_dir(state_dir) else {
        return;
    };
    for entry in entries.flatten() {
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        if !(name == "change-neighbors-seen"
            || name.starts_with("change-neighbors-seen-")
            || name.starts_with("change-neighbors-read-"))
        {
            continue;
        }
        let Ok(modified) = entry.metadata().and_then(|m| m.modified()) else {
            continue;
        };
        let stale = SystemTime::now()
            .duration_since(modified)
            .is_ok_and(|age| age > DEDUPE_LEDGER_RETENTION);
        if stale {
            let _ = fs::remove_file(entry.path());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn seen_matches_only_inside_the_recorded_range() {
        let seen = vec![SeenEntry::new("src/math.rs", Some("add"), 10, 25)];
        assert!(is_seen(&seen, "src/math.rs", Some("add"), 10, 25));
        assert!(
            is_seen(&seen, "src/math.rs", Some("add"), 12, 20),
            "a hint inside the recorded range is a repeat"
        );
        assert!(
            !is_seen(&seen, "src/math.rs", Some("add"), 15, 30),
            "partial overlap names unshown lines"
        );
        assert!(
            !is_seen(&seen, "src/math.rs", Some("add"), 30, 40),
            "disjoint ranges are new context"
        );
        assert!(
            !is_seen(&seen, "src/math.rs", Some("multiply"), 12, 20),
            "a different symbol is new context"
        );
        assert!(
            !is_seen(&seen, "src/other.rs", Some("add"), 12, 20),
            "a different file is new context"
        );
    }

    #[test]
    fn nameless_hint_matches_only_the_empty_name() {
        let seen = vec![SeenEntry::new("src/math.rs", None, 10, 25)];
        assert!(is_seen(&seen, "src/math.rs", None, 10, 25));
        assert!(
            !is_seen(&seen, "src/math.rs", Some("add"), 10, 25),
            "a named hint must not match an unnamed record"
        );
    }

    #[test]
    fn stale_ledger_lines_never_match() {
        let dir = tempdir().ok().unwrap_or_else(|| unreachable!());
        let seen_path = dir.path().join("change-neighbors-seen");
        // Lines the previous builds wrote: three-field
        // {edited}\t{neighbor}\t{name} and two-field {neighbor}\t{name}.
        let _ = fs::write(
            &seen_path,
            "src/lib.rs\tsrc/math.rs\tadd\nsrc/math.rs\tadd\nsrc/math.rs\tadd\t10\n",
        );

        let seen = read_seen(&seen_path);
        assert!(
            seen.is_empty(),
            "stale lines must parse to nothing, got: {seen:?}"
        );
        assert!(
            !is_seen(&seen, "src/math.rs", Some("add"), 10, 25),
            "a stale line must simply miss, never be treated as a match"
        );
    }

    #[test]
    fn pre_hq4_newline_split_lines_never_match() {
        // A pre-hq-4 writer recording a file literally named "b\nc" wrote a
        // line that splits on re-parse; neither fragment may forge a record
        // for file "c". Both shipped pre-hq-4 shapes (two- and three-field)
        // are planted.
        let dir = tempdir().ok().unwrap_or_else(|| unreachable!());
        let seen_path = dir.path().join("change-neighbors-seen");
        let _ = fs::write(&seen_path, "b\nc\tadd\nsrc/lib.rs\tsrc/b\nc\tadd\n");

        let seen = read_seen(&seen_path);
        assert!(
            seen.is_empty(),
            "no fragment may parse as a record, got: {seen:?}"
        );
        assert!(
            !is_seen(&seen, "c", Some("add"), 10, 25),
            "the fragment must never forge a record for c"
        );
    }

    #[test]
    fn seen_round_trips_through_the_ledger_file() {
        let dir = tempdir().ok().unwrap_or_else(|| unreachable!());
        let seen_path = dir.path().join("change-neighbors-seen");
        let entry = SeenEntry::new("src/math.rs", Some("add"), 10, 25);
        append_seen(&seen_path, std::slice::from_ref(&entry));

        let seen = read_seen(&seen_path);
        assert_eq!(seen, vec![entry]);
        assert!(is_seen(&seen, "src/math.rs", Some("add"), 12, 20));
    }

    #[test]
    fn absent_ledger_reads_as_empty() {
        let dir = tempdir().ok().unwrap_or_else(|| unreachable!());
        assert!(read_seen(&dir.path().join("nope")).is_empty());
        assert!(read_ranges(&dir.path().join("nope")).is_empty());
    }

    #[test]
    fn read_ranges_cover_only_fully_contained_hints() {
        let ranges = vec![ReadRange {
            file_path: "src/math.rs".to_owned(),
            line_start: 1,
            line_end: 50,
        }];
        assert!(is_read(&ranges, "src/math.rs", 10, 25));
        assert!(is_read(&ranges, "src/math.rs", 1, 50));
        assert!(
            !is_read(&ranges, "src/math.rs", 40, 60),
            "partial overlap names unread lines"
        );
        assert!(
            !is_read(&ranges, "src/math.rs", 60, 80),
            "disjoint ranges still fire"
        );
        assert!(!is_read(&ranges, "src/other.rs", 10, 25));
    }

    #[test]
    fn open_ended_read_range_covers_every_bounded_hint() {
        let ranges = vec![ReadRange {
            file_path: "src/math.rs".to_owned(),
            line_start: 40,
            line_end: u32::MAX,
        }];
        assert!(is_read(&ranges, "src/math.rs", 41, 1000));
        assert!(is_read(&ranges, "src/math.rs", u32::MAX, u32::MAX));
        assert!(
            !is_read(&ranges, "src/math.rs", 39, 1000),
            "starts before the read window"
        );
    }

    #[test]
    fn read_ledger_round_trips_and_skips_malformed_lines() {
        use std::io::Write;

        let dir = tempdir().ok().unwrap_or_else(|| unreachable!());
        let read_path = dir.path().join("change-neighbors-read");
        append_read_range(
            &read_path,
            &ReadRange {
                file_path: "src/math.rs".to_owned(),
                line_start: 1,
                line_end: 50,
            },
        );
        let mut file = fs::OpenOptions::new()
            .append(true)
            .open(&read_path)
            .unwrap_or_else(|_| unreachable!());
        let _ = writeln!(file, "src/one.rs\t10"); // too few fields
        let _ = writeln!(file, "src/two.rs\t10\t20\t30"); // too many fields
        let _ = writeln!(file, "src/three.rs\t10\toops"); // end not a number
        drop(file);

        let ranges = read_ranges(&read_path);
        assert_eq!(
            ranges,
            vec![ReadRange {
                file_path: "src/math.rs".to_owned(),
                line_start: 1,
                line_end: 50,
            }],
            "only the well-formed line parses"
        );
    }

    #[test]
    fn ledger_writers_reject_unroundtrippable_paths() {
        let dir = tempdir().ok().unwrap_or_else(|| unreachable!());
        let read_path = dir.path().join("change-neighbors-read");
        // A newline in the name would split one line into two on re-parse, and
        // the fragment forges a valid record for a different file — the wrong
        // suppression the ledgers otherwise exclude.
        append_read_range(
            &read_path,
            &ReadRange {
                file_path: "b\nc".to_owned(),
                line_start: 10,
                line_end: 25,
            },
        );
        let ranges = read_ranges(&read_path);
        assert!(
            ranges.is_empty(),
            "an unroundtrippable path must never be recorded, got: {ranges:?}"
        );
        assert!(
            !is_read(&ranges, "c", 10, 25),
            "the fragment must not forge a record for file c"
        );

        let seen_path = dir.path().join("change-neighbors-seen");
        append_seen(
            &seen_path,
            &[SeenEntry::new("src/a\tb.rs", Some("add"), 10, 25)],
        );
        assert!(
            read_seen(&seen_path).is_empty(),
            "a tab in the path must never be recorded"
        );
    }

    #[test]
    fn read_ledger_skips_windows_already_contained() {
        let dir = tempdir().ok().unwrap_or_else(|| unreachable!());
        let read_path = dir.path().join("change-neighbors-read");
        let window = |start: u32, end: u32| ReadRange {
            file_path: "src/math.rs".to_owned(),
            line_start: start,
            line_end: end,
        };
        append_read_range(&read_path, &window(10, 25));
        append_read_range(&read_path, &window(10, 25)); // identical repeat
        append_read_range(&read_path, &window(12, 20)); // narrower repeat
        append_read_range(&read_path, &window(1, 5)); // new region
        assert_eq!(
            read_ranges(&read_path),
            vec![window(10, 25), window(1, 5)],
            "repeated windows must not grow the ledger"
        );
    }

    #[test]
    fn sweep_stale_ledgers_reaps_only_untouched_files() {
        use std::fs::File;

        let dir = tempdir().ok().unwrap_or_else(|| unreachable!());
        let old = SystemTime::now() - Duration::from_secs(8 * 24 * 60 * 60);
        let names = [
            "change-neighbors-seen", // pre-hq-4 shared file
            "change-neighbors-seen-0123456789abcdef",
            "change-neighbors-read-0123456789abcdef",
            "change-neighbors-seen-fedcba9876543210", // fresh: must survive
            "index-failure-seen-0123456789abcdef.json", // not ours: must survive
        ];
        for (i, name) in names.iter().enumerate() {
            let path = dir.path().join(name);
            fs::write(&path, "seed\n").unwrap_or_else(|_| unreachable!());
            if i < 3 {
                // A backdated mtime needs a write handle (read-only opens no-op
                // on windows).
                let file = File::options()
                    .write(true)
                    .open(&path)
                    .unwrap_or_else(|_| unreachable!());
                file.set_modified(old).unwrap_or_else(|_| unreachable!());
            }
        }

        sweep_stale_ledgers(dir.path());

        assert!(
            !dir.path().join(names[0]).exists(),
            "the aged shared file must be reaped"
        );
        assert!(
            !dir.path().join(names[1]).exists(),
            "an aged seen file must be reaped"
        );
        assert!(
            !dir.path().join(names[2]).exists(),
            "an aged read file must be reaped"
        );
        assert!(
            dir.path().join(names[3]).exists(),
            "a fresh session's file must survive"
        );
        assert!(
            dir.path().join(names[4]).exists(),
            "unrelated state must survive"
        );
    }
}
