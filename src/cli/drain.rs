//! The no-watcher reindex drain worker.
//!
//! Spawned detached by the `PostToolUse` hook after it appends an edited path to
//! the on-disk reindex queue. Exactly one worker runs at a time (claimed via the
//! drain pid marker); it owns the sliding-debounce timing loop and reindexes each
//! queued file once, no matter how many edits piled up. Mirrors [`super::watch`]
//! but reads its work from the queue instead of filesystem events.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use crate::Claudix;
use crate::config::{self, Config};
use crate::error::{ClaudixError, Result};
use crate::store::Store;
use crate::store::marker::{InstallError, PidMarker, reindex_queue};

use super::canonical_project_root;
use super::watch::WATCH_HEARTBEAT_SECS;

/// Re-evaluation ceiling for a debounce sleep: never sleep longer than this
/// without re-reading the queue, so a freshly-appended path with a nearer
/// deadline is not slept past by more than one heartbeat window.
const DRAIN_POLL_CAP_MS: u64 = WATCH_HEARTBEAT_SECS * 1_000;
/// Floor so a deadline that is essentially `now` never busy-spins the loop.
const DRAIN_MIN_SLEEP_MS: u64 = 50;

pub async fn run_drain_reindex_queue(project_root: impl AsRef<Path>) -> Result<()> {
    let project_root = canonical_project_root(project_root.as_ref())?;
    super::require_git_repo(&project_root)?;
    // Write path: clean up any pre-fix nested store below this resolved root
    // before `ensure_layout` (ruling 2026-08-25); fail-open.
    crate::enumeration::delete_nested_stores(&project_root);
    let config = config::load(&project_root)?;
    // The worker is only ever spawned in the no-watcher path; guard anyway so a
    // stale spawn under a flipped config exits cleanly instead of fighting the
    // live watcher for the reindex lock.
    if config.watch || !config.hooks.auto_reembed_on_edit {
        return Ok(());
    }

    let store = Store::new(&project_root, &config)?;
    store.ensure_layout()?;

    // AlreadyHeld → another worker owns the loop; exit quietly so a burst of
    // edits never runs two drainers against the same queue.
    let marker = match PidMarker::install(store.reindex_drain_marker_path()) {
        Ok(marker) => Arc::new(marker),
        Err(InstallError::AlreadyHeld) => return Ok(()),
        Err(InstallError::Setup) => {
            return Err(ClaudixError::Store(
                "reindex drain marker setup failed".to_owned(),
            ));
        }
    };

    // Keep the marker live across the cold ONNX load and every debounce sleep so
    // a concurrent hook never misclassifies this worker as dead and spawns a
    // duplicate.
    let heartbeat = {
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

    let result = drain_loop(&project_root, &config, &store).await;

    heartbeat.abort();
    let _ = heartbeat.await;
    result
}

async fn drain_loop(project_root: &Path, config: &Config, store: &Store) -> Result<()> {
    let debounce_ms = config.hooks.reindex_debounce_secs.saturating_mul(1_000);
    let max_wait_ms = config.hooks.reindex_max_wait_secs.saturating_mul(1_000);
    let queue_path = store.reindex_queue_path();

    // Built once — reloading ONNX per file is exactly the waste this coalescing
    // exists to avoid.
    let claudix = Claudix::new(project_root.to_path_buf(), Arc::new(config.clone())).await?;

    loop {
        let now = reindex_queue::now_epoch_millis();
        let entries = reindex_queue::parse_entries(&reindex_queue::read_content(&queue_path));
        let due = reindex_queue::due_paths(&entries, now, debounce_ms, max_wait_ms);

        if !due.due.is_empty() {
            // Forward-progress guard: only a successful `remove_drained` proves a
            // due entry actually left the queue. If every due path fails to leave
            // (the reindex lock is unreachable for all of them, or a persistent
            // state-dir write failure blocks the rewrite) the entries stay due, so
            // a bare `continue` would hot-loop this detached worker at full tilt.
            // Sleep before retrying in that case.
            let mut progressed = false;
            for (path, last_seen) in &due.due {
                // Serialize each single-file reindex against a full index and any
                // stray reindex-file call, exactly like the watcher does.
                let reindex_lock = match store.acquire_reindex_lock() {
                    Ok(lock) => lock,
                    Err(error) => {
                        tracing::warn!("claudix drain skipped reindex of {path}: {error}");
                        continue;
                    }
                };
                // The marker the reindex writes is attributed to the session of
                // this path's newest queue line (the last edit the reindex will
                // see). Unattributed edits never write a marker: the reader
                // only surfaces a marker whose session matches the event's, so
                // an unowned marker would be dropped unread.
                let session = reindex_queue::latest_session(&entries, path);
                if let Err(error) = claudix.reindex_file(Path::new(path), session).await {
                    tracing::warn!("claudix drain failed to reindex {path}: {error}");
                }
                drop(reindex_lock);

                // The queue lock is never held across the slow reindex above.
                // Take it now to drop only what we drained: a newer edit for the
                // same path that landed mid-reindex has a ts past `last_seen` and
                // survives for its own debounce window. Drain even on a reindex
                // Err (fail-open, mirrors the watcher) so a genuinely-unindexable
                // file cannot hot-loop either.
                let mut drained = BTreeMap::new();
                drained.insert(path.clone(), *last_seen);
                if reindex_queue::remove_drained(&queue_path, &drained) {
                    progressed = true;
                }
            }
            if !progressed {
                tokio::time::sleep(Duration::from_millis(DRAIN_POLL_CAP_MS)).await;
            }
            continue;
        }

        match due.next_deadline {
            Some(deadline) => {
                let sleep_ms = deadline
                    .saturating_sub(now)
                    .clamp(DRAIN_MIN_SLEEP_MS, DRAIN_POLL_CAP_MS);
                tokio::time::sleep(Duration::from_millis(sleep_ms)).await;
            }
            None => {
                // Nothing actionable. An append may have raced in after our read;
                // re-check once before exiting. An append landing in the tiny
                // window between this recheck and the marker drop is not lost: it
                // persists on disk and is picked up on the next edit to that file
                // (which spawns a worker), or by a full reindex once the manifest
                // ages past reindex_after_hours, whichever comes first.
                let recheck =
                    reindex_queue::parse_entries(&reindex_queue::read_content(&queue_path));
                let again = reindex_queue::due_paths(
                    &recheck,
                    reindex_queue::now_epoch_millis(),
                    debounce_ms,
                    max_wait_ms,
                );
                if again.due.is_empty() && again.next_deadline.is_none() {
                    return Ok(());
                }
            }
        }
    }
}
