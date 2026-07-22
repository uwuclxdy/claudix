use std::collections::{HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use notify::{EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use tokio::sync::mpsc;

use crate::Claudix;
use crate::config;
use crate::enumeration::WatchFilter;
use crate::error::{ClaudixError, Result};
use crate::store::Store;
use crate::store::marker::{InstallError, PidMarker};

use super::canonical_project_root;

pub(super) const WATCH_HEARTBEAT_SECS: u64 = 30;

pub async fn run_watch(project_root: impl AsRef<Path>) -> Result<()> {
    let project_root = canonical_project_root(project_root.as_ref())?;
    super::require_git_repo(&project_root)?;
    let config = config::load(&project_root)?;
    if !config.watch {
        return Ok(());
    }

    let store = Store::new(&project_root, &config)?;
    store.ensure_layout()?;
    let marker = Arc::new(PidMarker::install(store.watch_marker_path()).map_err(
        |error| match error {
            InstallError::AlreadyHeld => {
                ClaudixError::Store("another claudix watch process is already running".to_owned())
            }
            InstallError::Setup => ClaudixError::Store("watch marker setup failed".to_owned()),
        },
    )?);

    // Cold ONNX loads can exceed the marker stale window; refresh the marker
    // from a side task while the watcher itself is still booting so concurrent
    // SessionStarts do not misclassify us as dead and spawn a duplicate.
    let early_heartbeat = {
        let marker = Arc::clone(&marker);
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(Duration::from_secs(WATCH_HEARTBEAT_SECS));
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            tick.tick().await; // consume the immediate first tick
            loop {
                tick.tick().await;
                marker.heartbeat();
            }
        })
    };

    let (event_tx, mut event_rx) = mpsc::unbounded_channel();
    let mut watcher = RecommendedWatcher::new(
        move |event| {
            let _ = event_tx.send(event);
        },
        notify::Config::default(),
    )
    .map_err(|error| ClaudixError::Store(format!("file watcher failed: {error}")))?;
    watcher
        .watch(&project_root, RecursiveMode::Recursive)
        .map_err(|error| ClaudixError::Store(format!("file watcher failed: {error}")))?;

    let filter = WatchFilter::load(&project_root)?;
    let claudix = Claudix::new(project_root.clone(), Arc::new(config)).await?;

    early_heartbeat.abort();
    let _ = early_heartbeat.await;

    let mut pending = VecDeque::new();
    let mut debounce_deadline: Option<tokio::time::Instant> = None;
    let mut heartbeat = tokio::time::interval(Duration::from_secs(WATCH_HEARTBEAT_SECS));
    heartbeat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    heartbeat.tick().await; // first tick fires immediately; consume it before the loop
    loop {
        tokio::select! {
            event = event_rx.recv() => {
                let Some(event) = event else {
                    return Ok(());
                };
                queue_reindex_paths(&project_root, &filter, event, &mut pending);
                // Anchor the debounce window when the first event lands; further
                // events do NOT extend it so a continuous file-event stream
                // still gets drained on schedule instead of starving.
                if debounce_deadline.is_none() && !pending.is_empty() {
                    debounce_deadline =
                        Some(tokio::time::Instant::now() + Duration::from_millis(250));
                }
            }
            _ = async {
                match debounce_deadline {
                    Some(deadline) => tokio::time::sleep_until(deadline).await,
                    None => std::future::pending::<()>().await,
                }
            } => {
                debounce_deadline = None;
                let paths = drain_unique_paths(&mut pending);
                for path in paths {
                    // Serialize against concurrent reindex-file CLI/MCP calls and
                    // any duplicate watcher that slipped through the marker claim.
                    let _reindex_lock = match store.acquire_reindex_lock() {
                        Ok(lock) => lock,
                        Err(error) => {
                            tracing::warn!(
                                "claudix watch skipped reindex of {}: {error}",
                                path.display()
                            );
                            continue;
                        }
                    };
                    if let Err(error) = claudix.reindex_file(&path).await {
                        tracing::warn!("claudix watch failed to reindex {}: {error}", path.display());
                    }
                }
            }
            _ = heartbeat.tick() => {
                marker.heartbeat();
            }
        }
    }
}

fn queue_reindex_paths(
    project_root: &Path,
    filter: &WatchFilter,
    event: notify::Result<notify::Event>,
    pending: &mut VecDeque<PathBuf>,
) {
    let Ok(event) = event else {
        return;
    };
    if !is_reindex_event(&event.kind) {
        return;
    }

    pending.extend(event.paths.into_iter().filter_map(|path| {
        let relative = path.strip_prefix(project_root).ok()?;
        relative.components().next()?;
        if !filter.is_watchable(relative) {
            return None;
        }
        Some(relative.to_path_buf())
    }));
}

fn is_reindex_event(kind: &EventKind) -> bool {
    matches!(
        kind,
        EventKind::Create(_) | EventKind::Modify(_) | EventKind::Remove(_)
    )
}

fn drain_unique_paths(pending: &mut VecDeque<PathBuf>) -> Vec<PathBuf> {
    let mut seen = HashSet::new();
    let mut paths = Vec::new();
    while let Some(path) = pending.pop_front() {
        if seen.insert(path.clone()) {
            paths.push(path);
        }
    }
    paths
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queue_reindex_paths_ignores_internal_state_paths() {
        let tmp = tempfile::tempdir();
        assert!(tmp.is_ok());
        let tmp = tmp.ok().unwrap_or_else(|| unreachable!());
        let root = tmp.path();
        let event = notify::Event {
            kind: EventKind::Modify(notify::event::ModifyKind::Data(
                notify::event::DataChange::Content,
            )),
            paths: vec![
                root.join("src/lib.rs"),
                root.join(".claudix/manifest.json"),
                root.join(".git/HEAD"),
            ],
            attrs: notify::event::EventAttributes::new(),
        };
        let filter = WatchFilter::load(root);
        assert!(filter.is_ok());
        let filter = filter.ok().unwrap_or_else(|| unreachable!());
        let mut pending = VecDeque::new();

        queue_reindex_paths(root, &filter, Ok(event), &mut pending);

        assert_eq!(
            pending.into_iter().collect::<Vec<_>>(),
            vec![PathBuf::from("src/lib.rs")]
        );
    }

    #[test]
    fn queue_reindex_paths_respects_project_gitignore() {
        let tmp = tempfile::tempdir();
        assert!(tmp.is_ok());
        let tmp = tmp.ok().unwrap_or_else(|| unreachable!());
        let root = tmp.path();
        assert!(std::fs::write(root.join(".gitignore"), "target/\nnode_modules/\n").is_ok());

        let event = notify::Event {
            kind: EventKind::Modify(notify::event::ModifyKind::Data(
                notify::event::DataChange::Content,
            )),
            paths: vec![
                root.join("src/lib.rs"),
                root.join("target/debug/build/foo"),
                root.join("node_modules/pkg/index.js"),
            ],
            attrs: notify::event::EventAttributes::new(),
        };
        let filter = WatchFilter::load(root);
        assert!(filter.is_ok());
        let filter = filter.ok().unwrap_or_else(|| unreachable!());
        let mut pending = VecDeque::new();

        queue_reindex_paths(root, &filter, Ok(event), &mut pending);

        assert_eq!(
            pending.into_iter().collect::<Vec<_>>(),
            vec![PathBuf::from("src/lib.rs")]
        );
    }

    #[test]
    fn drain_unique_paths_deduplicates_in_order() {
        let mut pending = VecDeque::from(vec![
            PathBuf::from("src/lib.rs"),
            PathBuf::from("src/lib.rs"),
            PathBuf::from("src/main.rs"),
        ]);

        assert_eq!(
            drain_unique_paths(&mut pending),
            vec![PathBuf::from("src/lib.rs"), PathBuf::from("src/main.rs")]
        );
    }
}
