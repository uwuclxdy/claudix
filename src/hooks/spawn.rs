use std::ffi::OsStr;
use std::fs;
use std::path::Path;
use std::time::{Duration, SystemTime};

use crate::config::Config;
use crate::store::Store;
use crate::store::marker::WATCH_MARKER_STALE_SECS;
use crate::util::{now_rfc3339, parse_rfc3339};

use super::ready_check::PENDING_INDEX_FAILURE_GRACE_SECS;

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
    if !try_claim_pending_index_marker(&marker_path, &placeholder) {
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

pub(super) fn try_claim_pending_index_marker(marker_path: &Path, payload: &str) -> bool {
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
                if pending_index_marker_is_fresh(marker_path) {
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

fn pending_index_marker_is_fresh(marker_path: &Path) -> bool {
    let Some(marker) = read_pending_index_marker(marker_path) else {
        return false;
    };
    // `duration_since` errors when `created_at` is in the future — clock skew
    // or restore-from-backup. Treat that as expired so a single bad timestamp
    // can't permanently jam the auto-indexer.
    SystemTime::now()
        .duration_since(marker.created_at)
        .map(|age| age < Duration::from_secs(PENDING_INDEX_FAILURE_GRACE_SECS))
        .unwrap_or(false)
}

pub(super) struct PendingIndexMarker {
    pub prior_ts: String,
    pub created_at: SystemTime,
    /// PID of the spawned `claudix index` child, or `None` if the marker is
    /// still the placeholder written before the child was forked (or a legacy
    /// marker missing this line entirely).
    pub child_pid: Option<u32>,
}

pub(super) fn read_pending_index_marker(marker_path: &Path) -> Option<PendingIndexMarker> {
    let content = fs::read_to_string(marker_path).ok()?;
    let mut lines = content.lines();
    let prior_ts = lines.next()?.to_owned();
    let created_at = parse_rfc3339(lines.next()?).ok()?;
    let child_pid = lines
        .next()
        .and_then(|line| line.trim().parse::<u32>().ok())
        .filter(|pid| *pid != 0);
    Some(PendingIndexMarker {
        prior_ts,
        created_at,
        child_pid,
    })
}

pub(super) fn spawn_detached_claudix<const N: usize, S>(
    project_root: &Path,
    args: [S; N],
) -> Option<u32>
where
    S: AsRef<OsStr>,
{
    let binary = std::env::current_exe().ok()?;
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
    if !matches!(
        crate::store::marker::try_claim(&marker_path, stale_after),
        crate::store::marker::MarkerClaim::Acquired
    ) {
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

pub(super) fn spawn_background_reindex_file(project_root: &Path, file_path: &str) {
    spawn_detached_claudix(
        project_root,
        [OsStr::new("reindex-file"), OsStr::new(file_path)],
    );
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
    fn pending_index_marker_round_trips() {
        let dir = tempdir().ok().unwrap_or_else(|| unreachable!());
        let marker_path = dir.path().join("indexing-pending");
        let payload = format!("2026-04-20T00:00:00Z\n{}\n42\n", now_rfc3339());

        assert!(try_claim_pending_index_marker(&marker_path, &payload));
        let marker = read_pending_index_marker(&marker_path);
        assert!(marker.is_some());
        let marker = marker.unwrap_or_else(|| unreachable!());
        assert_eq!(marker.prior_ts, "2026-04-20T00:00:00Z");
        assert_eq!(marker.child_pid, Some(42));
    }

    #[test]
    fn pending_index_marker_treats_zero_pid_as_placeholder() {
        let dir = tempdir().ok().unwrap_or_else(|| unreachable!());
        let marker_path = dir.path().join("indexing-pending");
        let payload = format!("none\n{}\n0\n", now_rfc3339());
        assert!(try_claim_pending_index_marker(&marker_path, &payload));
        let marker = read_pending_index_marker(&marker_path).unwrap_or_else(|| unreachable!());
        assert_eq!(marker.child_pid, None);
    }

    #[test]
    fn pending_index_marker_claim_is_atomic() {
        let dir = tempdir().ok().unwrap_or_else(|| unreachable!());
        let marker_path = dir.path().join("indexing-pending");
        let first = format!("none\n{}\n", now_rfc3339());
        assert!(try_claim_pending_index_marker(&marker_path, &first));

        // Second claim while the first is still fresh must fail.
        let second = format!("ts-2\n{}\n", now_rfc3339());
        assert!(!try_claim_pending_index_marker(&marker_path, &second));
        let marker = read_pending_index_marker(&marker_path).unwrap_or_else(|| unreachable!());
        assert_eq!(marker.prior_ts, "none", "first claim must remain in place");
    }

    #[test]
    fn watch_marker_with_dead_pid_is_not_alive() {
        let dir = tempdir().ok().unwrap_or_else(|| unreachable!());
        let marker_path = dir.path().join("watch.pid");
        // PID 0 never refers to a real process on Unix or Windows, so this
        // exercises the "parsed PID but process is gone" branch.
        fs::write(&marker_path, "0").unwrap_or_else(|_| unreachable!());

        assert!(
            crate::store::marker::live_owner(
                &marker_path,
                Duration::from_secs(WATCH_MARKER_STALE_SECS)
            )
            .is_none(),
            "watch marker with dead PID must be reclaimable"
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
            crate::store::marker::live_owner(
                &marker_path,
                Duration::from_secs(WATCH_MARKER_STALE_SECS)
            )
            .is_some(),
            "watch marker with live PID must stay alive regardless of mtime"
        );
    }

    #[test]
    fn pending_index_marker_with_future_timestamp_is_not_fresh() {
        let dir = tempdir().ok().unwrap_or_else(|| unreachable!());
        let marker_path = dir.path().join("indexing-pending");
        // 1 hour in the future — simulates clock skew / backup restore.
        let future = SystemTime::now() + Duration::from_secs(3_600);
        let future_ts = crate::util::format_rfc3339(future);
        fs::write(&marker_path, format!("none\n{future_ts}\n0\n"))
            .unwrap_or_else(|_| unreachable!());

        assert!(
            !pending_index_marker_is_fresh(&marker_path),
            "future-timestamped marker must be reclaimable"
        );
    }

    #[test]
    fn pending_index_marker_claim_replaces_legacy_or_unparseable() {
        let dir = tempdir().ok().unwrap_or_else(|| unreachable!());
        let marker_path = dir.path().join("indexing-pending");
        assert!(fs::write(&marker_path, "legacy-single-line").is_ok());

        let payload = format!("none\n{}\n", now_rfc3339());
        assert!(try_claim_pending_index_marker(&marker_path, &payload));
        let marker = read_pending_index_marker(&marker_path);
        assert!(marker.is_some());
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
