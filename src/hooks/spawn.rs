use std::ffi::OsStr;
use std::fs;
use std::path::Path;
use std::time::Duration;

use crate::config::Config;
use crate::store::Store;
use crate::store::marker::WATCH_MARKER_STALE_SECS;
use crate::store::marker::pending_index;
use crate::util::now_rfc3339;

pub(super) fn spawn_background_index(project_root: &Path, config: &Config) -> bool {
    let Ok(store) = Store::new(project_root, config) else {
        return false;
    };
    if store.full_index_running() {
        return false;
    }
    if store.ensure_layout().is_err() {
        return false;
    }

    let marker_path = store.pending_index_marker_path();
    let manifest = store.read_manifest().ok().flatten();

    // Mismatched embedding model means the existing chunks have the wrong
    // dimension. Spawning `claudix index` without `--force` would either fail
    // or append vectors of a different shape — the user has to run
    // `claudix clear && claudix index` (or equivalent) themselves; the
    // session-start additionalContext already tells them so.
    if let Some(ref manifest) = manifest
        && manifest.chunk_count > 0
        && manifest.embedding_model != config.embedding.model
    {
        return false;
    }

    // Skip respawning when the existing index is fresh and populated. Without
    // this guard every SessionStart after the 60s marker window triggers a
    // full reindex of an unchanged repo.
    if let Some(ref manifest) = manifest
        && manifest.chunk_count > 0
        && !manifest.is_stale(config)
    {
        return false;
    }

    let prior_ts = manifest
        .as_ref()
        .and_then(|m| m.last_full_index_at.as_deref())
        .unwrap_or("none");

    // The marker doubles as the "spawn in progress" sentinel. `create_new`
    // serializes concurrent SessionStarts so we don't double-spawn before
    // the child has taken the full-index lock. A fresh marker from a prior
    // spawn means an index is already running (or just failed and hasn't
    // aged out yet); either way, leave it alone so its `created_at` keeps
    // anchoring the failure-grace clock.
    let placeholder = format!("{prior_ts}\n{}\n0\n", now_rfc3339());
    // Leave the ack file alone — `check_index_ready` rewrites it on success and
    // on each surfaced failure. Wiping it here would defeat dedup, so an index
    // that keeps failing with the same `prior_ts` would re-notify every session.
    if !pending_index::try_claim(&marker_path, &placeholder) {
        return false;
    }

    let Some(child_pid) = spawn_detached_claudix(project_root, [OsStr::new("index")]) else {
        let _ = fs::remove_file(&marker_path);
        return false;
    };
    // Overwrite the placeholder with the spawned PID so `check_index_ready`
    // can gate the failure decision on the child still being alive — slow
    // ONNX cold loads or large config parses must not be misclassified as
    // a crashed index.
    let payload = format!("{prior_ts}\n{}\n{child_pid}\n", now_rfc3339());
    let _ = fs::write(&marker_path, payload);
    true
}

/// Set on every background child we spawn. A process that finds it in its own
/// environment must never spawn another background claudix: this caps the spawn
/// chain at depth 1 and turns a binary that misparses its argv (e.g. a libtest
/// harness reading `index` as a test-name filter and re-running the suite)
/// from an unbounded fork bomb into a no-op.
const BACKGROUND_SENTINEL: &str = "CLAUDIX_BACKGROUND";

pub(super) fn spawn_detached_claudix<const N: usize, S>(
    project_root: &Path,
    args: [S; N],
) -> Option<u32>
where
    S: AsRef<OsStr>,
{
    // Re-entry guard: background workers (`index`, `watch`, `reindex-file`)
    // never legitimately spawn further background claudix processes.
    if std::env::var_os(BACKGROUND_SENTINEL).is_some() {
        return None;
    }
    let binary = std::env::current_exe().ok()?;

    // In the unit-test build `current_exe()` is the libtest harness, which
    // would treat `index`/`watch` as a test-name filter and re-run the suite
    // from every spawn site — recursively. Shadow the args with a filter that
    // matches no test: callers still get a real detached PID, and the child
    // exits after running zero tests.
    #[cfg(test)]
    let args = {
        let _ = args;
        [
            OsStr::new("__claudix_no_such_test__"),
            OsStr::new("--exact"),
        ]
    };

    spawn_detached_command(project_root, binary.as_os_str(), args)
}

pub(super) fn spawn_background_watch(project_root: &Path, config: &Config) -> bool {
    if !config.watch || !config.hooks.auto_reembed_on_edit {
        return false;
    }
    let Ok(store) = Store::new(project_root, config) else {
        return false;
    };
    if store.ensure_layout().is_err() {
        return false;
    }
    let marker_path = store.watch_marker_path();
    let stale_after = Duration::from_secs(WATCH_MARKER_STALE_SECS);
    if crate::store::marker::try_claim(&marker_path, stale_after).is_err() {
        return false;
    }

    let Some(child_pid) = spawn_detached_claudix(project_root, [OsStr::new("watch")]) else {
        let _ = fs::remove_file(&marker_path);
        return false;
    };
    // Replace our (parent) pid with the spawned child PID so concurrent
    // SessionStarts see the watcher as live before the child finishes booting.
    let _ = fs::write(&marker_path, child_pid.to_string());
    true
}

/// Ensure the single no-watcher reindex drain worker is running.
///
/// Mirrors [`spawn_background_watch`] but claims the DRAIN marker and spawns the
/// internal `drain-reindex-queue` subcommand. Claim-or-skip: safe to call after
/// every enqueue — it no-ops when a worker already owns the marker. Returns
/// `false` (and cleans up the marker) when already claimed or the spawn fails.
pub(super) fn spawn_background_drain_worker(project_root: &Path, config: &Config) -> bool {
    if config.watch || !config.hooks.auto_reembed_on_edit {
        return false;
    }
    let Ok(store) = Store::new(project_root, config) else {
        return false;
    };
    if store.ensure_layout().is_err() {
        return false;
    }
    let marker_path = store.reindex_drain_marker_path();
    let stale_after = Duration::from_secs(WATCH_MARKER_STALE_SECS);
    if crate::store::marker::try_claim(&marker_path, stale_after).is_err() {
        return false;
    }

    let Some(child_pid) = spawn_detached_claudix(project_root, [OsStr::new("drain-reindex-queue")])
    else {
        let _ = fs::remove_file(&marker_path);
        return false;
    };
    // Replace our (parent) pid with the spawned child PID so concurrent hooks
    // see the worker as live before the child finishes booting.
    let _ = fs::write(&marker_path, child_pid.to_string());
    true
}

#[cfg(unix)]
fn spawn_detached_command<const N: usize, S>(
    project_root: &Path,
    binary: &OsStr,
    args: [S; N],
) -> Option<u32>
where
    S: AsRef<OsStr>,
{
    use std::os::unix::process::CommandExt;

    // `nohup` re-execs the binary, so its PID is the nohup process itself;
    // for our liveness checks that's fine because the wrapper stays alive
    // for the whole runtime of the child it execs into.
    std::process::Command::new("nohup")
        .arg(binary)
        .args(args.iter().map(AsRef::as_ref))
        .current_dir(project_root)
        .env("CLAUDE_PROJECT_DIR", project_root)
        .env(BACKGROUND_SENTINEL, "1")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .process_group(0)
        .spawn()
        .ok()
        .map(|child| child.id())
}

#[cfg(windows)]
fn spawn_detached_command<const N: usize, S>(
    project_root: &Path,
    binary: &OsStr,
    args: [S; N],
) -> Option<u32>
where
    S: AsRef<OsStr>,
{
    use std::os::windows::process::CommandExt;

    const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
    const DETACHED_PROCESS: u32 = 0x0000_0008;

    std::process::Command::new(binary)
        .args(args.iter().map(AsRef::as_ref))
        .current_dir(project_root)
        .env("CLAUDE_PROJECT_DIR", project_root)
        .env(BACKGROUND_SENTINEL, "1")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .creation_flags(DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP)
        .spawn()
        .ok()
        .map(|child| child.id())
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::fs;
    use std::sync::Arc;
    use std::time::SystemTime;
    use tempfile::tempdir;

    use crate::Claudix;
    use crate::config::Config;

    mod fixture {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/common/fixture.rs"
        ));
    }

    mod config_support {
        use crate as claudix;

        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/common/config_support.rs"
        ));
    }

    use config_support::stub_config;
    use fixture::TestFixture;

    fn write_config(project_root: &Path, config: &Config) {
        let claude_dir = project_root.join(".claude");
        assert!(fs::create_dir_all(&claude_dir).is_ok());
        let config_text = toml::to_string(config);
        assert!(config_text.is_ok());
        assert!(
            fs::write(
                claude_dir.join("claudix.toml"),
                config_text.ok().unwrap_or_default()
            )
            .is_ok()
        );
    }

    #[test]
    fn watch_marker_with_dead_pid_is_not_alive() {
        let dir = tempdir().ok().unwrap_or_else(|| unreachable!());
        let marker_path = dir.path().join("watch.pid");
        // PID 0 never refers to a real process on Unix or Windows, so this
        // exercises the "parsed PID but process is gone" branch.
        fs::write(&marker_path, "0").unwrap_or_else(|_| unreachable!());

        assert!(
            !crate::store::marker::is_alive(
                &marker_path,
                Duration::from_secs(WATCH_MARKER_STALE_SECS)
            ),
            "watch marker with dead PID must be reclaimable"
        );
    }

    #[test]
    fn drain_marker_with_dead_pid_is_reclaimable() {
        // A crashed drain worker leaves a dead pid in the drain marker; the next
        // edit's claim must reclaim it so a burst never gets stuck without a
        // worker.
        let dir = tempdir().ok().unwrap_or_else(|| unreachable!());
        let store = crate::store::Store::new(dir.path(), &Config::default())
            .ok()
            .unwrap_or_else(|| unreachable!());
        let marker_path = store.reindex_drain_marker_path();
        assert!(store.ensure_layout().is_ok());
        fs::write(&marker_path, "9999999\n").unwrap_or_else(|_| unreachable!());

        let marker = crate::store::marker::PidMarker::install(marker_path.clone());
        assert!(marker.is_ok(), "dead-pid drain marker must be reclaimable");
        let stored = fs::read_to_string(&marker_path).ok();
        assert_eq!(
            stored.as_deref().map(str::trim),
            Some(std::process::id().to_string().as_str()),
            "reclaiming worker must own the marker"
        );
    }

    #[test]
    fn watch_marker_with_live_pid_ignores_mtime() {
        let dir = tempdir().ok().unwrap_or_else(|| unreachable!());
        let marker_path = dir.path().join("watch.pid");
        fs::write(&marker_path, std::process::id().to_string()).unwrap_or_else(|_| unreachable!());

        let file = fs::OpenOptions::new()
            .write(true)
            .open(&marker_path)
            .unwrap_or_else(|_| unreachable!());
        // Past mtime that would have flunked the old stale-window gate — PID
        // liveness is now authoritative so the watcher must still register alive.
        let stale = SystemTime::now() - Duration::from_secs(WATCH_MARKER_STALE_SECS * 10);
        file.set_modified(stale).unwrap_or_else(|_| unreachable!());
        drop(file);

        assert!(
            crate::store::marker::is_alive(
                &marker_path,
                Duration::from_secs(WATCH_MARKER_STALE_SECS)
            ),
            "watch marker with live PID must stay alive regardless of mtime"
        );
    }

    // ── Part A: watcher boot-window visibility ────────────────────────────────
    //
    // The sequence in spawn_background_watch is:
    //   1. try_claim writes parent PID → marker exists with live PID
    //   2. spawn_detached_claudix forks child C
    //   3. fs::write(marker, child_pid) → marker has child PID (C is running)
    //
    // In run_watch the child does:
    //   4. PidMarker::install → adopts its own PID (handoff from parent)
    //   5. early_heartbeat task starts (fires every 30s, first tick consumed)
    //   6. Claudix::new / ONNX cold load (can take minutes)
    //
    // is_alive uses process_running(pid) which is true as long as the process
    // is alive — mtime is only consulted for unparseable marker content. Both
    // the parent (steps 1-3) and the child (steps 4+) are live processes, so
    // watcher_alive() must return true throughout the entire boot sequence.
    //
    // These tests exercise the marker/liveness functions directly to confirm
    // the invariant holds without needing a real child process spawn.

    #[test]
    fn watcher_boot_window_marker_with_parent_pid_is_alive() {
        // Simulates step 1: parent wrote its own PID before spawning the child.
        // Parent is the current process, which is definitely alive.
        let dir = tempdir().ok().unwrap_or_else(|| unreachable!());
        let marker_path = dir.path().join("watch.pid");
        fs::write(&marker_path, std::process::id().to_string()).unwrap_or_else(|_| unreachable!());

        assert!(
            crate::store::marker::is_alive(
                &marker_path,
                Duration::from_secs(WATCH_MARKER_STALE_SECS),
            ),
            "marker with parent PID must be alive during pre-spawn window"
        );
    }

    #[test]
    fn watcher_boot_window_fresh_unparseable_marker_is_alive() {
        // Simulates the brief window between create_new and the PID write
        // (e.g. empty file or partial write). The mtime-fallback branch of
        // is_alive must keep the marker alive within WATCH_MARKER_STALE_SECS.
        let dir = tempdir().ok().unwrap_or_else(|| unreachable!());
        let marker_path = dir.path().join("watch.pid");
        fs::write(&marker_path, "").unwrap_or_else(|_| unreachable!());
        // mtime is now — well within the stale window.

        assert!(
            crate::store::marker::is_alive(
                &marker_path,
                Duration::from_secs(WATCH_MARKER_STALE_SECS),
            ),
            "fresh unparseable marker must be treated as alive (boot-window cover)"
        );
    }

    #[test]
    fn watcher_boot_window_stale_unparseable_marker_is_dead() {
        // A malformed marker older than WATCH_MARKER_STALE_SECS must be
        // reclaimable so stale-marker recovery still works.
        let dir = tempdir().ok().unwrap_or_else(|| unreachable!());
        let marker_path = dir.path().join("watch.pid");
        fs::write(&marker_path, "not-a-pid").unwrap_or_else(|_| unreachable!());
        let stale = SystemTime::now() - Duration::from_secs(WATCH_MARKER_STALE_SECS + 1);
        let _ = fs::File::open(&marker_path).and_then(|f| f.set_modified(stale));

        assert!(
            !crate::store::marker::is_alive(
                &marker_path,
                Duration::from_secs(WATCH_MARKER_STALE_SECS),
            ),
            "stale unparseable marker must be reclaimable after timeout"
        );
    }

    #[tokio::test]
    async fn spawn_background_index_skips_when_manifest_fresh_and_populated()
    -> crate::error::Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let config = stub_config();
        write_config(fixture.root(), &config);

        Claudix::new(fixture.root().to_path_buf(), Arc::new(config.clone()))
            .await?
            .index_full(&mut ())
            .await?;

        let store = crate::store::Store::new(fixture.root(), &config)?;
        let marker_path = store.pending_index_marker_path();
        let _ = fs::remove_file(&marker_path);

        assert!(
            !spawn_background_index(fixture.root(), &config),
            "fresh non-empty matching-model index must not respawn"
        );
        assert!(
            !marker_path.exists(),
            "no pending marker should be written when spawn is skipped"
        );
        Ok(())
    }

    #[tokio::test]
    async fn spawn_background_index_runs_when_manifest_missing() -> crate::error::Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let config = stub_config();
        write_config(fixture.root(), &config);

        let store = crate::store::Store::new(fixture.root(), &config)?;
        store.ensure_layout()?;

        assert!(
            spawn_background_index(fixture.root(), &config),
            "missing manifest must trigger a spawn"
        );
        Ok(())
    }
}
