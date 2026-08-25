//! The "background index is running" marker — distinct from the watcher's
//! single-line pid marker because the index spawn path needs to remember
//! the manifest timestamp it raced against and the spawned child's pid.
//!
//! Layout (newline-separated):
//!
//! ```text
//! <prior_ts>          // manifest's last_full_index_at when we claimed, or "none"
//! <created_at>        // RFC3339, anchors the failure-grace clock
//! <child_pid>         // spawned child pid, or "0" placeholder
//! ```
//!
//! Stale recovery is age-based on `created_at` rather than mtime: a surfaced
//! failure leaves the marker in place (each turn resurfaces it), so the file's
//! mtime cannot say when the failed run was claimed.

use std::fs;
use std::path::Path;
use std::time::{Duration, SystemTime};

use crate::util::parse_rfc3339;

/// How long a fresh marker blocks new claims and defers failure decisions.
/// Long enough to outlast a cold ONNX load + first embed batch on a slow
/// laptop; short enough that a genuinely crashed spawn surfaces a failure
/// to the user within a minute or so.
pub(crate) const FAILURE_GRACE_SECS: u64 = 60;

pub(crate) struct PendingIndexMarker {
    pub prior_ts: String,
    pub created_at: SystemTime,
    /// Raw RFC3339 string as written in the marker file. Distinct from
    /// `created_at` (the parsed value) and refreshed on every claim, so it
    /// identifies one failed run for the session-less ack ledger.
    pub created_at_raw: String,
    /// PID of the spawned `claudix index` child, or `None` if the marker is
    /// still the placeholder written before the child was forked (or a legacy
    /// marker missing this line entirely).
    pub child_pid: Option<u32>,
}

pub(crate) fn read(marker_path: &Path) -> Option<PendingIndexMarker> {
    let content = fs::read_to_string(marker_path).ok()?;
    let mut lines = content.lines();
    let prior_ts = lines.next()?.to_owned();
    let created_at_raw = lines.next()?.to_owned();
    let created_at = parse_rfc3339(&created_at_raw).ok()?;
    let child_pid = lines
        .next()
        .and_then(|line| line.trim().parse::<u32>().ok())
        .filter(|pid| *pid != 0);
    Some(PendingIndexMarker {
        prior_ts,
        created_at,
        created_at_raw,
        child_pid,
    })
}

pub(crate) fn try_claim(marker_path: &Path, payload: &str) -> bool {
    use std::fs::OpenOptions;
    use std::io::Write;

    for _ in 0..2 {
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(marker_path)
        {
            Ok(mut file) => return file.write_all(payload.as_bytes()).is_ok(),
            Err(_) => {
                if is_fresh(marker_path) {
                    return false;
                }
                if fs::remove_file(marker_path).is_err() {
                    return false;
                }
            }
        }
    }
    false
}

fn is_fresh(marker_path: &Path) -> bool {
    let Some(marker) = read(marker_path) else {
        return false;
    };
    // `duration_since` errors when `created_at` is in the future — clock skew
    // or restore-from-backup. Treat that as expired so a single bad timestamp
    // can't permanently jam the auto-indexer.
    SystemTime::now()
        .duration_since(marker.created_at)
        .map(|age| age < Duration::from_secs(FAILURE_GRACE_SECS))
        .unwrap_or(false)
}

/// A pending marker reads as "index in flight" while it is still fresh (inside
/// the failure-grace window) or its recorded child is alive. A stale marker
/// with a dead child is a failed run: `check_index_ready` leaves those in place
/// so each turn can resurface the failure, and SessionStart must not report
/// them as still building.
pub(crate) fn is_in_flight(marker_path: &Path) -> bool {
    let Some(marker) = read(marker_path) else {
        return false;
    };
    is_fresh(marker_path)
        || marker
            .child_pid
            .is_some_and(crate::store::marker::process_running)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::now_rfc3339;
    use tempfile::tempdir;

    #[test]
    fn round_trips() {
        let dir = tempdir().ok().unwrap_or_else(|| unreachable!());
        let marker_path = dir.path().join("indexing-pending");
        let payload = format!("2026-04-20T00:00:00Z\n{}\n42\n", now_rfc3339());

        assert!(try_claim(&marker_path, &payload));
        let marker = read(&marker_path);
        assert!(marker.is_some());
        let marker = marker.unwrap_or_else(|| unreachable!());
        assert_eq!(marker.prior_ts, "2026-04-20T00:00:00Z");
        assert_eq!(marker.child_pid, Some(42));
    }

    #[test]
    fn treats_zero_pid_as_placeholder() {
        let dir = tempdir().ok().unwrap_or_else(|| unreachable!());
        let marker_path = dir.path().join("indexing-pending");
        let payload = format!("none\n{}\n0\n", now_rfc3339());
        assert!(try_claim(&marker_path, &payload));
        let marker = read(&marker_path).unwrap_or_else(|| unreachable!());
        assert_eq!(marker.child_pid, None);
    }

    #[test]
    fn claim_is_atomic() {
        let dir = tempdir().ok().unwrap_or_else(|| unreachable!());
        let marker_path = dir.path().join("indexing-pending");
        let first = format!("none\n{}\n", now_rfc3339());
        assert!(try_claim(&marker_path, &first));

        // Second claim while the first is still fresh must fail.
        let second = format!("ts-2\n{}\n", now_rfc3339());
        assert!(!try_claim(&marker_path, &second));
        let marker = read(&marker_path).unwrap_or_else(|| unreachable!());
        assert_eq!(marker.prior_ts, "none", "first claim must remain in place");
    }

    #[test]
    fn future_timestamp_is_not_fresh() {
        let dir = tempdir().ok().unwrap_or_else(|| unreachable!());
        let marker_path = dir.path().join("indexing-pending");
        // 1 hour in the future — simulates clock skew / backup restore.
        let future = SystemTime::now() + Duration::from_secs(3_600);
        let future_ts = crate::util::format_rfc3339(future);
        fs::write(&marker_path, format!("none\n{future_ts}\n0\n"))
            .unwrap_or_else(|_| unreachable!());

        assert!(
            !is_fresh(&marker_path),
            "future-timestamped marker must be reclaimable"
        );
    }

    #[test]
    fn claim_replaces_legacy_or_unparseable() {
        let dir = tempdir().ok().unwrap_or_else(|| unreachable!());
        let marker_path = dir.path().join("indexing-pending");
        assert!(fs::write(&marker_path, "legacy-single-line").is_ok());

        let payload = format!("none\n{}\n", now_rfc3339());
        assert!(try_claim(&marker_path, &payload));
        let marker = read(&marker_path);
        assert!(marker.is_some());
    }

    #[test]
    fn is_in_flight_reads_stale_dead_child_as_not_building() {
        let dir = tempdir().ok().unwrap_or_else(|| unreachable!());
        let marker_path = dir.path().join("indexing-pending");
        // Backdated past the failure grace with a "0" (dead) child: a failed run
        // whose marker now persists so each turn can resurface it.
        fs::write(&marker_path, "none\n2025-01-01T00:00:00Z\n0\n")
            .unwrap_or_else(|_| unreachable!());

        assert!(
            !is_in_flight(&marker_path),
            "a stale marker with a dead child must not read as building"
        );
    }

    #[test]
    fn is_in_flight_reads_fresh_marker_as_building() {
        let dir = tempdir().ok().unwrap_or_else(|| unreachable!());
        let marker_path = dir.path().join("indexing-pending");
        fs::write(&marker_path, format!("none\n{}\n0\n", now_rfc3339()))
            .unwrap_or_else(|_| unreachable!());

        assert!(
            is_in_flight(&marker_path),
            "a fresh claim must read as building"
        );
    }
}
