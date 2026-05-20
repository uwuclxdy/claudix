//! Pid-marker primitives shared by the watch process and the
//! background-index spawn path. Each marker is a single file under the
//! store's state directory whose contents are the pid of the live owner
//! (plus optional payload lines).

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, SystemTime};

/// How long an unparseable or otherwise-pending marker stays "young" before
/// the next claimant is allowed to overwrite it. Matches the watcher's
/// heartbeat cadence so a stalled watch process can't outlive a session.
pub(crate) const WATCH_MARKER_STALE_SECS: u64 = 120;

#[cfg(unix)]
pub(crate) fn process_running(pid: u32) -> bool {
    // `kill -0 0` targets the current process group, so an explicit guard
    // keeps "pid 0" from being mistaken for a live process.
    if pid == 0 {
        return false;
    }
    Command::new("kill")
        .args(["-0", &pid.to_string()])
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

#[cfg(windows)]
pub(crate) fn process_running(pid: u32) -> bool {
    Command::new("tasklist")
        .args(["/FI", &format!("PID eq {pid}"), "/NH"])
        .output()
        .is_ok_and(|output| {
            output.status.success()
                && String::from_utf8_lossy(&output.stdout).contains(&pid.to_string())
        })
}

/// Atomically claim a pid-bearing marker file. Replaces dead-process
/// markers; refuses live-foreign markers; treats unparseable young
/// markers as still-claiming (handles the create-then-write race).
///
/// `stale_after` bounds how long an unparseable marker stays "young"
/// before it can be reclaimed.
///
/// Returns `Ok(())` if we now own the marker. Every other outcome —
/// live foreign owner, racing placeholder, failed reclaim — folds into
/// `Err(AlreadyHeld)` because no caller acts on the distinction.
pub fn try_claim(path: &Path, stale_after: Duration) -> Result<(), AlreadyHeld> {
    use std::fs::OpenOptions;

    for _ in 0..2 {
        if let Ok(mut file) = OpenOptions::new().write(true).create_new(true).open(path) {
            let _ = writeln!(file, "{}", std::process::id());
            return Ok(());
        }

        match read_pid(path) {
            Some(pid) if pid == std::process::id() => {
                // Adopt our own placeholder. Used when the parent process
                // pre-wrote our pid before exec'ing into the watcher.
                let _ = fs::write(path, format!("{}\n", std::process::id()));
                return Ok(());
            }
            Some(pid) if process_running(pid) => return Err(AlreadyHeld),
            Some(_) => {
                if fs::remove_file(path).is_err() {
                    return Err(AlreadyHeld);
                }
            }
            None => {
                if marker_age(path).is_some_and(|age| age < stale_after) {
                    return Err(AlreadyHeld);
                }
                if fs::remove_file(path).is_err() {
                    return Err(AlreadyHeld);
                }
            }
        }
    }
    Err(AlreadyHeld)
}

/// Marker was held by a live foreign process, or a placeholder claim
/// hadn't aged out yet. Returned by [`try_claim`] and [`PidMarker::install`].
#[derive(Debug, Clone, Copy)]
pub struct AlreadyHeld;

/// `true` if the marker still exists and either parses as a pid whose
/// process is alive, or — for unparseable content — was modified within
/// `stale_after`. The unparseable branch covers the brief window between
/// `create_new` and the owner writing its pid; without it a placeholder
/// marker would be declared dead before its owner finishes claiming it.
pub fn is_alive(path: &Path, stale_after: Duration) -> bool {
    match read_pid(path) {
        Some(pid) => process_running(pid),
        None => marker_age(path).is_some_and(|age| age < stale_after),
    }
}

/// `None` when the marker is absent or its content can't parse as a `u32`.
/// `Some(0)` is preserved here so callers can route it through
/// `process_running`, which knows that pid 0 is never a real process.
pub fn read_pid(path: &Path) -> Option<u32> {
    fs::read_to_string(path).ok()?.trim().parse::<u32>().ok()
}

fn marker_age(path: &Path) -> Option<Duration> {
    let modified = fs::metadata(path).ok()?.modified().ok()?;
    SystemTime::now().duration_since(modified).ok()
}

/// RAII wrapper for a claimed marker. The marker is removed on drop only
/// if our pid still owns it — another process may have reclaimed it after
/// a heartbeat stall (SIGSTOP, etc).
pub struct PidMarker {
    path: PathBuf,
}

impl PidMarker {
    /// Claim the marker and wrap it in an RAII guard. Errors if the
    /// marker is held by a live foreign process, the claim races out,
    /// or the parent directory cannot be created.
    pub fn install(path: PathBuf) -> Result<Self, InstallError> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|_| InstallError::Setup)?;
        }
        // PidMarker's contract is "we are the live owner" — bound the
        // stale window to the shared WATCH_MARKER_STALE_SECS so callers
        // and the marker's own claim path agree on what "stale" means.
        try_claim(&path, Duration::from_secs(WATCH_MARKER_STALE_SECS))
            .map_err(|AlreadyHeld| InstallError::AlreadyHeld)?;
        Ok(Self { path })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn heartbeat(&self) {
        let _ = fs::write(&self.path, std::process::id().to_string());
    }
}

impl Drop for PidMarker {
    fn drop(&mut self) {
        if read_pid(&self.path) == Some(std::process::id()) {
            let _ = fs::remove_file(&self.path);
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum InstallError {
    AlreadyHeld,
    Setup,
}

impl std::fmt::Display for InstallError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlreadyHeld => f.write_str("marker already held by a live process"),
            Self::Setup => f.write_str("failed to prepare marker directory"),
        }
    }
}

impl std::error::Error for InstallError {}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[cfg(unix)]
    #[test]
    fn install_refuses_live_foreign_pid() {
        let dir = tempdir().ok().unwrap_or_else(|| unreachable!());
        let path = dir.path().join("watch.pid");
        let mut child = std::process::Command::new("sleep")
            .arg("60")
            .spawn()
            .ok()
            .unwrap_or_else(|| unreachable!());
        let foreign_pid = child.id();
        assert!(std::fs::write(&path, foreign_pid.to_string()).is_ok());

        let result = PidMarker::install(path.clone());
        let stored_after = std::fs::read_to_string(&path).ok();
        let _ = child.kill();
        let _ = child.wait();

        assert!(
            matches!(result, Err(InstallError::AlreadyHeld)),
            "expected AlreadyHeld when a live foreign PID owns the marker"
        );
        assert_eq!(
            stored_after.as_deref().map(str::trim),
            Some(foreign_pid.to_string().as_str()),
            "foreign marker contents must be untouched"
        );
    }

    #[test]
    fn install_clears_malformed_marker() {
        let dir = tempdir().ok().unwrap_or_else(|| unreachable!());
        let path = dir.path().join("watch.pid");
        assert!(std::fs::write(&path, "not-a-pid").is_ok());
        // Backdate so the malformed-marker stale window has elapsed.
        let stale = SystemTime::now() - Duration::from_secs(3 * 60);
        let _ = fs::File::open(&path).and_then(|f| f.set_modified(stale));

        let marker = PidMarker::install(path.clone());
        assert!(marker.is_ok(), "malformed marker must be reclaimable");
    }

    #[test]
    fn install_takes_over_dead_pid() {
        let dir = tempdir().ok().unwrap_or_else(|| unreachable!());
        let path = dir.path().join("watch.pid");
        let dead_pid = pick_dead_pid();
        assert!(std::fs::write(&path, dead_pid.to_string()).is_ok());

        let marker = PidMarker::install(path.clone());
        assert!(marker.is_ok(), "must reclaim a stale marker");
        let stored = std::fs::read_to_string(&path).ok();
        assert_eq!(
            stored.as_deref().map(str::trim),
            Some(std::process::id().to_string().as_str())
        );
    }

    #[test]
    fn install_adopts_handoff_with_own_pid() {
        let dir = tempdir().ok().unwrap_or_else(|| unreachable!());
        let path = dir.path().join("watch.pid");
        assert!(std::fs::write(&path, std::process::id().to_string()).is_ok());

        let marker = PidMarker::install(path.clone());
        assert!(marker.is_ok(), "must adopt parent's hand-off claim");
    }

    #[test]
    fn drop_leaves_foreign_pid_alone() {
        let dir = tempdir().ok().unwrap_or_else(|| unreachable!());
        let path = dir.path().join("watch.pid");
        let marker = PidMarker::install(path.clone())
            .ok()
            .unwrap_or_else(|| unreachable!());
        // Simulate another process reclaiming the marker before our drop runs.
        let foreign_pid = if std::process::id() == 1 { 2 } else { 1 };
        assert!(std::fs::write(&path, foreign_pid.to_string()).is_ok());

        drop(marker);
        assert!(
            path.exists(),
            "drop must not remove a marker reclaimed by another process"
        );
    }

    fn pick_dead_pid() -> u32 {
        for candidate in [9_999_999u32, 8_888_888, 7_777_777] {
            if !process_running(candidate) {
                return candidate;
            }
        }
        9_999_999
    }
}
