//! Cross-process locking for chunk-table writes.
//!
//! Both the full index and per-file reindex paths take the same on-disk
//! lock so the LanceDB table is only ever rewritten by one writer at a
//! time. Dead pids are reclaimed; live pids block.

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant, SystemTime};

use crate::error::{ClaudixError, Result};

use super::Store;
use super::marker::{process_running, read_pid};

pub(super) const LOCK_FILE_NAME: &str = "index.lock";
// Long enough to outlast a full index on a large repo; the per-file reindex
// is a background subprocess, so a generous deadline is preferable to giving
// up and silently losing the user's edit.
const REINDEX_LOCK_WAIT_MS: u64 = 1_800_000;
const REINDEX_LOCK_POLL_MS: u64 = 50;
const LOCK_TERMINATION_GRACE_MS: u64 = 2_000;
const LOCK_TERMINATION_POLL_MS: u64 = 100;

pub struct IndexLockGuard {
    path: PathBuf,
}

impl Drop for IndexLockGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

impl Store {
    /// Block until the shared chunk-writer lock is available, then claim it.
    ///
    /// Shares [`LOCK_FILE_NAME`] with [`Self::acquire_index_lock`] so a full
    /// index and a per-file reindex can never rewrite the chunk table at the
    /// same time. The deadline is long enough to outlast a full index on a
    /// large repo; if the holder dies or never wrote its PID, the dead-lock
    /// recovery branch reclaims it on the next poll.
    pub fn acquire_reindex_lock(&self) -> Result<IndexLockGuard> {
        fs::create_dir_all(self.state_dir_path())?;
        let lock_path = self.state_dir_path().join(LOCK_FILE_NAME);
        let deadline = Instant::now() + Duration::from_millis(REINDEX_LOCK_WAIT_MS);

        loop {
            if let Ok(mut file) = fs::File::create_new(&lock_path) {
                let _ = writeln!(file, "{}", std::process::id());
                return Ok(IndexLockGuard { path: lock_path });
            }
            if let Some(pid) = read_pid(&lock_path)
                && !process_running(pid)
            {
                let _ = fs::remove_file(&lock_path);
                continue;
            }
            if Instant::now() >= deadline {
                return Err(ClaudixError::Store(
                    "reindex lock contention: another reindex job is in progress".to_owned(),
                ));
            }
            thread::sleep(Duration::from_millis(REINDEX_LOCK_POLL_MS));
        }
    }

    pub fn acquire_index_lock(&self) -> Option<IndexLockGuard> {
        fs::create_dir_all(self.state_dir_path()).ok()?;
        let lock_path = self.state_dir_path().join(LOCK_FILE_NAME);

        if let Ok(mut file) = fs::File::create_new(&lock_path) {
            writeln!(file, "{}", std::process::id()).ok()?;
            return Some(IndexLockGuard { path: lock_path });
        }

        if self.full_index_running() {
            return None;
        }
        let _ = fs::remove_file(&lock_path);
        if let Ok(mut file) = fs::File::create_new(&lock_path) {
            writeln!(file, "{}", std::process::id()).ok()?;
            return Some(IndexLockGuard { path: lock_path });
        }
        None
    }

    pub fn full_index_running(&self) -> bool {
        let lock_path = self.state_dir_path().join(LOCK_FILE_NAME);
        let Ok(content) = fs::read_to_string(&lock_path) else {
            return false;
        };
        match content.trim().parse::<u32>() {
            // Lock has a parseable PID — defer to the OS.
            Ok(pid) => process_running(pid),
            // Lock exists but has no PID yet. This is the small window between
            // `create_new` and `writeln!` in `acquire_index_lock`; the writer
            // is racing to fill it in. Anything older than this window is
            // corrupt and should not block recovery.
            Err(_) if content.trim().is_empty() => fs::metadata(&lock_path)
                .ok()
                .and_then(|m| m.modified().ok())
                .and_then(|t| SystemTime::now().duration_since(t).ok())
                .is_some_and(|age| age < Duration::from_secs(5)),
            // Lock has non-empty unparseable contents — almost certainly a
            // crash mid-write or unrelated debris. Treat as dead so the next
            // acquire can replace it instead of waiting hours.
            Err(_) => false,
        }
    }

    pub fn stop_index_lock_holder(&self) {
        let lock_path = self.state_dir_path().join(LOCK_FILE_NAME);
        let Some(pid) = read_pid(&lock_path) else {
            let _ = fs::remove_file(lock_path);
            return;
        };

        if pid == std::process::id() {
            return;
        }

        terminate_process(pid);
        wait_for_process_exit(pid, Duration::from_millis(LOCK_TERMINATION_GRACE_MS));
        if process_running(pid) {
            kill_process(pid);
        }
        let _ = fs::remove_file(lock_path);
    }
}

fn wait_for_process_exit(pid: u32, timeout: Duration) {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if !process_running(pid) {
            return;
        }
        thread::sleep(Duration::from_millis(LOCK_TERMINATION_POLL_MS));
    }
}

#[cfg(unix)]
fn terminate_process(pid: u32) {
    let _ = Command::new("kill")
        .args(["-TERM", &pid.to_string()])
        .stderr(Stdio::null())
        .status();
}

#[cfg(unix)]
fn kill_process(pid: u32) {
    let _ = Command::new("kill")
        .args(["-KILL", &pid.to_string()])
        .stderr(Stdio::null())
        .status();
}

#[cfg(windows)]
fn terminate_process(pid: u32) {
    let _ = Command::new("taskkill")
        .args(["/PID", &pid.to_string()])
        .status();
}

#[cfg(windows)]
fn kill_process(pid: u32) {
    let _ = Command::new("taskkill")
        .args(["/PID", &pid.to_string(), "/F"])
        .status();
}
