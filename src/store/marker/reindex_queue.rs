//! Append-only on-disk queue coalescing the no-watcher reindex path.
//!
//! One-shot `PostToolUse` hooks share no memory, so rapid edits to the same
//! file would each spawn a fresh detached reindex. Instead every edit appends a
//! `millis<TAB>relative-path<TAB>session-id` line here (the session id is
//! omitted when the enqueueing event carried none, so legacy two-field lines
//! parse as `session: None`), and a single drain worker (claimed via a pid
//! marker) owns the debounce timing loop and reindexes each file once.
//!
//! Layout — one line per edit, append-only:
//!
//! ```text
//! 1717430400123\tsrc/foo.rs\tsess-abc
//! 1717430400512\tsrc/foo.rs
//! ```
//!
//! Timestamps are epoch **milliseconds**, but no-loss rests on ordering, not
//! timing. `PostToolUse` fires only after the edit tool's file write, and the
//! worker reads a file (inside `reindex_file`) strictly after it read the queue
//! and fixed `last_seen`. So an edit whose new content a running reindex misses
//! must have written the file, then appended its queue line, after that read: its
//! line is a later append, so [`remove_drained`]'s `ts <= drained_last_seen`
//! filter keeps it. Ties at the same millisecond among the edits already queued
//! fold into `last_seen` harmlessly. Independent of how long the reindex takes
//! and of the up-to-30-min queue-lock gap.
//!
//! Concurrency: hooks only ever append; the worker's read-modify-write in
//! [`remove_drained`] is the sole rewrite. Both take a short-lived queue lock
//! (`reindex-queue.lock`, distinct from the 30-min `index.lock`) so a line
//! appended between the worker's read and its rewrite is never lost. The lock is
//! never held across the slow reindex.

use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use super::{process_running, read_pid};

/// How long the queue lock is chased before an append fails open. Appends hold
/// it for microseconds and the worker never holds it across a reindex, so real
/// contention clears far inside this; a dropped append just misses coalescing and
/// is picked up on the next edit to that file, or by a full reindex once the
/// manifest ages past `reindex_after_hours`, rather than stalling the hook.
const QUEUE_LOCK_WAIT_MS: u64 = 2_000;
const QUEUE_LOCK_POLL_MS: u64 = 10;

/// Epoch milliseconds now. `0` only if the clock predates the Unix epoch.
pub(crate) fn now_epoch_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}

/// One parsed queue line.
pub(crate) struct QueueEntry {
    pub ts: u64,
    pub path: String,
    /// Session of the hook event that enqueued this edit; `None` on legacy
    /// two-field lines or events that carried no session id.
    pub session: Option<String>,
}

/// Per-distinct-path first/last edit timestamps (epoch millis).
pub(crate) struct PathWindow {
    pub first_seen: u64,
    pub last_seen: u64,
}

/// Paths due to reindex now, plus the nearest future deadline to sleep until.
pub(crate) struct DuePaths {
    /// Distinct due path → the `last_seen` that made it due (its drain cutoff).
    pub due: BTreeMap<String, u64>,
    /// Earliest epoch-millis at which a not-yet-due path becomes due, if any.
    pub next_deadline: Option<u64>,
}

/// Parse raw queue content into entries, skipping malformed lines (fail-open).
/// A line is `millis<TAB>path` or `millis<TAB>path<TAB>session`; the session
/// field is everything after the path's first tab, so a session id containing a
/// tab cannot forge an extra field — it just never matches a real event session
/// and the marker it would attribute gets dropped unread (loss, never a wrong
/// hint). A newline in a session id splits the line: the edit's own line still
/// parses, with a truncated session that never matches a real event (the hint
/// is lost, never misattributed), and the stray remainder fails the parse.
pub(crate) fn parse_entries(content: &str) -> Vec<QueueEntry> {
    content
        .lines()
        .filter_map(|line| {
            let (ts, rest) = line.split_once('\t')?;
            let ts = ts.trim().parse::<u64>().ok()?;
            let (path, session) = match rest.split_once('\t') {
                Some((path, session)) => (path, Some(session)),
                None => (rest, None),
            };
            if path.is_empty() {
                return None;
            }
            Some(QueueEntry {
                ts,
                path: path.to_owned(),
                session: session
                    .filter(|session| !session.is_empty())
                    .map(str::to_owned),
            })
        })
        .collect()
}

/// Collapse entries to one window per distinct path (min first, max last seen).
pub(crate) fn collapse(entries: &[QueueEntry]) -> BTreeMap<String, PathWindow> {
    let mut map: BTreeMap<String, PathWindow> = BTreeMap::new();
    for entry in entries {
        map.entry(entry.path.clone())
            .and_modify(|window| {
                window.first_seen = window.first_seen.min(entry.ts);
                window.last_seen = window.last_seen.max(entry.ts);
            })
            .or_insert(PathWindow {
                first_seen: entry.ts,
                last_seen: entry.ts,
            });
    }
    map
}

/// Sliding-debounce decision over the whole queue.
///
/// A path is due when `now >= last_seen + debounce` (idle window elapsed) OR
/// `now >= first_seen + max_wait` (hard cap so a continuously-edited file never
/// starves). For not-yet-due paths the nearest such deadline is returned so the
/// worker can sleep exactly until the next thing becomes due.
pub(crate) fn due_paths(
    entries: &[QueueEntry],
    now: u64,
    debounce: u64,
    max_wait: u64,
) -> DuePaths {
    let mut due = BTreeMap::new();
    let mut next_deadline: Option<u64> = None;
    for (path, window) in collapse(entries) {
        let idle_deadline = window.last_seen.saturating_add(debounce);
        let cap_deadline = window.first_seen.saturating_add(max_wait);
        let deadline = idle_deadline.min(cap_deadline);
        if now >= deadline {
            due.insert(path, window.last_seen);
        } else {
            next_deadline = Some(next_deadline.map_or(deadline, |current| current.min(deadline)));
        }
    }
    DuePaths { due, next_deadline }
}

/// Raw queue content, or empty when absent/unreadable (fail-open). Read without
/// the lock — a partial read of an in-flight append fails-open on parse and the
/// line (persisted on disk) is picked up next iteration, never dropped.
pub(crate) fn read_content(queue_path: &Path) -> String {
    fs::read_to_string(queue_path).unwrap_or_default()
}

/// Append one edited path (and its editing session, when known) to the queue.
/// Fail-open: a lock miss or write error returns `false` (the edit is simply
/// not coalesced; it is caught by the next edit to that file or a manifest-age
/// full reindex). Serialized against [`remove_drained`] via the queue lock so
/// an append is never lost to a concurrent worker rewrite.
pub(crate) fn append(queue_path: &Path, relative_path: &str, session_id: Option<&str>) -> bool {
    if let Some(parent) = queue_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let Some(_lock) = acquire_queue_lock(queue_path) else {
        return false;
    };
    let Ok(mut file) = OpenOptions::new()
        .create(true)
        .append(true)
        .open(queue_path)
    else {
        return false;
    };
    let written = match session_id {
        Some(session) => writeln!(file, "{}\t{relative_path}\t{session}", now_epoch_millis()),
        None => writeln!(file, "{}\t{relative_path}", now_epoch_millis()),
    };
    written.is_ok()
}

/// Session of the newest queue line for `path` (ties resolve to the
/// later-appended line), or `None` when the path has no entries or its newest
/// edit carried no session. The drain worker attributes the reindexed edit's
/// change-neighbors marker to this session.
pub(crate) fn latest_session<'a>(entries: &'a [QueueEntry], path: &str) -> Option<&'a str> {
    let mut newest: Option<&QueueEntry> = None;
    for entry in entries {
        if entry.path != path {
            continue;
        }
        if newest.is_none_or(|newest| entry.ts >= newest.ts) {
            newest = Some(entry);
        }
    }
    newest.and_then(|entry| entry.session.as_deref())
}

/// Remove the entries drained by a completed reindex, keeping every other line.
///
/// Re-reads the queue under the lock so an append that landed during the (slow,
/// unlocked) reindex is preserved: a line is dropped only when its path is in
/// `drained` AND its `ts <= drained[path]` — a newer edit for the same path
/// survives. Returns `false` on a lock miss or write error (fail-open; the stale
/// entries just get reprocessed next iteration).
pub(crate) fn remove_drained(queue_path: &Path, drained: &BTreeMap<String, u64>) -> bool {
    let Some(_lock) = acquire_queue_lock(queue_path) else {
        return false;
    };
    let content = read_content(queue_path);
    let mut kept = String::new();
    for entry in parse_entries(&content) {
        let is_drained = drained
            .get(&entry.path)
            .is_some_and(|&cutoff| entry.ts <= cutoff);
        if !is_drained {
            match entry.session.as_deref() {
                Some(session) => {
                    kept.push_str(&format!("{}\t{}\t{session}\n", entry.ts, entry.path));
                }
                None => kept.push_str(&format!("{}\t{}\n", entry.ts, entry.path)),
            }
        }
    }
    // Temp+rename so the lock-free loop-top read never sees a truncated file.
    let temp_path = queue_path.with_extension("tmp");
    if fs::write(&temp_path, &kept).is_err() {
        return false;
    }
    fs::rename(&temp_path, queue_path).is_ok()
}

fn queue_lock_path(queue_path: &Path) -> PathBuf {
    queue_path.with_extension("lock")
}

/// RAII guard for the short-lived queue lock; removes the file on drop.
struct QueueLock {
    path: PathBuf,
}

impl Drop for QueueLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

/// Claim the queue lock, reclaiming a dead holder's file, giving up after
/// [`QUEUE_LOCK_WAIT_MS`]. `None` means the caller must fail open.
fn acquire_queue_lock(queue_path: &Path) -> Option<QueueLock> {
    let lock_path = queue_lock_path(queue_path);
    let deadline = Instant::now() + Duration::from_millis(QUEUE_LOCK_WAIT_MS);
    loop {
        if let Ok(mut file) = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&lock_path)
        {
            let _ = writeln!(file, "{}", std::process::id());
            return Some(QueueLock { path: lock_path });
        }
        if let Some(pid) = read_pid(&lock_path)
            && !process_running(pid)
        {
            let _ = fs::remove_file(&lock_path);
            continue;
        }
        if Instant::now() >= deadline {
            return None;
        }
        thread::sleep(Duration::from_millis(QUEUE_LOCK_POLL_MS));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    const SEC: u64 = 1_000; // one second in the millis unit the queue uses

    fn entry(ts: u64, path: &str) -> QueueEntry {
        QueueEntry {
            ts,
            path: path.to_owned(),
            session: None,
        }
    }

    #[test]
    fn parse_skips_malformed_lines() {
        let content = "100\tsrc/a.rs\nnot-a-line\n\t\n200\t\n300\tsrc/b.rs\n";
        let entries = parse_entries(content);
        let paths: Vec<&str> = entries.iter().map(|e| e.path.as_str()).collect();
        assert_eq!(paths, vec!["src/a.rs", "src/b.rs"]);
    }

    #[test]
    fn collapse_dedups_same_path_to_min_first_and_max_last() {
        let entries = vec![
            entry(500, "src/a.rs"),
            entry(100, "src/a.rs"),
            entry(300, "src/a.rs"),
        ];
        let windows = collapse(&entries);
        assert_eq!(windows.len(), 1, "same path must collapse to one window");
        let window = windows.get("src/a.rs").unwrap_or_else(|| unreachable!());
        assert_eq!(window.first_seen, 100);
        assert_eq!(window.last_seen, 500);
    }

    #[test]
    fn sliding_window_not_due_until_debounce_after_latest_edit() {
        // A newer edit at 5s pushes the idle deadline out to 5s + debounce.
        let entries = vec![entry(SEC, "src/a.rs"), entry(5 * SEC, "src/a.rs")];
        let debounce = 10 * SEC;
        let max_wait = 60 * SEC;

        // One millisecond before the idle deadline: not due.
        let before = due_paths(&entries, 5 * SEC + debounce - 1, debounce, max_wait);
        assert!(before.due.is_empty(), "must wait out the full idle window");
        assert_eq!(before.next_deadline, Some(5 * SEC + debounce));

        // At the idle deadline: due.
        let at = due_paths(&entries, 5 * SEC + debounce, debounce, max_wait);
        assert!(at.due.contains_key("src/a.rs"));
        assert_eq!(
            at.due.get("src/a.rs").copied(),
            Some(5 * SEC),
            "drain cutoff is the latest seen ts"
        );
    }

    #[test]
    fn hard_cap_forces_reindex_under_continuous_edits() {
        // first_seen anchored at 0; the file keeps being edited (last_seen rides
        // near `now`) so the idle window never elapses — only the max_wait cap
        // can fire.
        let now = 60 * SEC;
        let debounce = 10 * SEC;
        let max_wait = 60 * SEC;
        let entries = vec![entry(0, "src/hot.rs"), entry(now, "src/hot.rs")];

        let result = due_paths(&entries, now, debounce, max_wait);
        assert!(
            result.due.contains_key("src/hot.rs"),
            "first_seen + max_wait must force the reindex despite a live idle window"
        );

        // One millisecond earlier the cap has not yet elapsed → not due.
        let earlier = due_paths(&entries, now - 1, debounce, max_wait);
        assert!(earlier.due.is_empty());
    }

    #[test]
    fn next_deadline_is_the_earliest_pending() {
        let debounce = 10 * SEC;
        let max_wait = 60 * SEC;
        let entries = vec![entry(SEC, "src/a.rs"), entry(4 * SEC, "src/b.rs")];
        let result = due_paths(&entries, 2 * SEC, debounce, max_wait);
        assert!(result.due.is_empty());
        // a becomes due at 1s+10s = 11s, b at 4s+10s = 14s → nearest is 11s.
        assert_eq!(result.next_deadline, Some(SEC + debounce));
    }

    #[test]
    fn append_writes_a_parseable_line() {
        let dir = tempdir().ok().unwrap_or_else(|| unreachable!());
        let queue = dir.path().join("reindex-queue");
        assert!(append(&queue, "src/a.rs", None));

        let entries = parse_entries(&read_content(&queue));
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].path, "src/a.rs");
    }

    #[test]
    fn append_then_parse_collapses_repeated_path_to_one() {
        let dir = tempdir().ok().unwrap_or_else(|| unreachable!());
        let queue = dir.path().join("reindex-queue");
        assert!(append(&queue, "src/a.rs", None));
        assert!(append(&queue, "src/a.rs", None));
        assert!(append(&queue, "src/a.rs", None));

        let windows = collapse(&parse_entries(&read_content(&queue)));
        assert_eq!(windows.len(), 1, "three edits of one file → one path");
        let window = windows.get("src/a.rs").unwrap_or_else(|| unreachable!());
        assert!(
            window.last_seen >= window.first_seen,
            "last_seen must not precede first_seen"
        );
    }

    #[test]
    fn append_with_session_round_trips_and_latest_session_wins_ties() {
        let dir = tempdir().ok().unwrap_or_else(|| unreachable!());
        let queue = dir.path().join("reindex-queue");
        assert!(append(&queue, "src/a.rs", Some("sess-1")));
        // Same path later, no session (legacy-shaped line).
        assert!(append(&queue, "src/a.rs", None));
        assert!(append(&queue, "src/b.rs", Some("sess-2")));

        let entries = parse_entries(&read_content(&queue));
        assert_eq!(entries.len(), 3);
        assert_eq!(
            entries[0].session.as_deref(),
            Some("sess-1"),
            "the session field must survive the append/parse round trip"
        );
        assert_eq!(entries[1].session, None, "a two-field line parses as None");

        // The newest line for src/a.rs is the later-appended one, whose session
        // is None — latest_session must report None, not the stale sess-1.
        assert_eq!(latest_session(&entries, "src/a.rs"), None);
        assert_eq!(latest_session(&entries, "src/b.rs"), Some("sess-2"));
        assert_eq!(latest_session(&entries, "src/nope.rs"), None);
    }

    #[test]
    fn remove_drained_preserves_the_session_field() {
        let dir = tempdir().ok().unwrap_or_else(|| unreachable!());
        let queue = dir.path().join("reindex-queue");
        let seeded = "100\tsrc/a.rs\tsess-1\n200\tsrc/a.rs\n150\tsrc/b.rs\tsess-2\n";
        assert!(fs::write(&queue, seeded).is_ok());

        let mut drained = BTreeMap::new();
        drained.insert("src/a.rs".to_owned(), 200);
        drained.insert("src/b.rs".to_owned(), 100);
        assert!(remove_drained(&queue, &drained));

        let entries = parse_entries(&read_content(&queue));
        assert_eq!(entries.len(), 1, "only the undrained b@150 line survives");
        assert_eq!(entries[0].path, "src/b.rs");
        assert_eq!(
            entries[0].session.as_deref(),
            Some("sess-2"),
            "the rewrite must keep the surviving line's session field"
        );
    }

    #[test]
    fn remove_drained_keeps_newer_and_other_paths() {
        let dir = tempdir().ok().unwrap_or_else(|| unreachable!());
        let queue = dir.path().join("reindex-queue");
        // a@100, a@200 drained; a@300 arrived during the reindex (must survive);
        // b@150 is an unrelated path (must survive).
        let seeded = "100\tsrc/a.rs\n200\tsrc/a.rs\n300\tsrc/a.rs\n150\tsrc/b.rs\n";
        assert!(fs::write(&queue, seeded).is_ok());

        let mut drained = BTreeMap::new();
        drained.insert("src/a.rs".to_owned(), 200);
        assert!(remove_drained(&queue, &drained));

        let windows = collapse(&parse_entries(&read_content(&queue)));
        assert!(
            windows.contains_key("src/a.rs"),
            "the newer a@300 edit must survive the drain"
        );
        let a = windows.get("src/a.rs").unwrap_or_else(|| unreachable!());
        assert_eq!(a.first_seen, 300, "a@100 and a@200 must be removed");
        assert!(
            windows.contains_key("src/b.rs"),
            "an unrelated path must not be touched by the drain"
        );
    }

    #[test]
    fn acquire_queue_lock_reclaims_dead_pid() {
        let dir = tempdir().ok().unwrap_or_else(|| unreachable!());
        let queue = dir.path().join("reindex-queue");
        // A dead pid holds the lock; the next claimant must reclaim it.
        assert!(fs::write(queue_lock_path(&queue), "9999999\n").is_ok());
        let lock = acquire_queue_lock(&queue);
        assert!(lock.is_some(), "a dead-pid queue lock must be reclaimable");
        drop(lock);
        assert!(
            !queue_lock_path(&queue).exists(),
            "dropping the guard must release the lock"
        );
    }
}
