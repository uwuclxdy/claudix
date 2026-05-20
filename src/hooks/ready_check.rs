use std::fs;
use std::path::Path;
use std::time::{Duration, SystemTime};

use serde_json::{Value, json};

use crate::config::Config;
use crate::store::Store;

use super::spawn::read_pending_index_marker;

pub(super) const PENDING_INDEX_READY_GRACE_SECS: u64 = 3;
pub(super) const PENDING_INDEX_FAILURE_GRACE_SECS: u64 = 60;
pub(super) const PENDING_INDEX_ACK_FILE_NAME: &str = "indexing-pending-acked";

pub(super) fn check_index_ready(
    project_root: &Path,
    config: &Config,
    event_name: &str,
) -> Option<Value> {
    let store = Store::new(project_root, config).ok()?;
    let marker_path = store.pending_index_marker_path();
    let ack_path = store.state_dir_path().join(PENDING_INDEX_ACK_FILE_NAME);
    let marker = read_pending_index_marker(&marker_path)?;

    // `created_at` in the future (clock skew, restored backup) would make
    // `duration_since` error forever; treat that as "past every grace window"
    // so the marker can be cleaned up instead of jamming the auto-indexer.
    let now = SystemTime::now();
    let age = match now.duration_since(marker.created_at) {
        Ok(age) => age,
        Err(_) => Duration::from_secs(PENDING_INDEX_FAILURE_GRACE_SECS),
    };
    if age < Duration::from_secs(PENDING_INDEX_READY_GRACE_SECS) {
        return None;
    }

    if store.full_index_running() {
        return None;
    }

    let manifest = store.read_manifest().ok().flatten();
    let current_ts = manifest
        .as_ref()
        .and_then(|m| m.last_full_index_at.as_deref())
        .unwrap_or("none");

    if current_ts != marker.prior_ts {
        let _ = fs::remove_file(&marker_path);
        let _ = fs::remove_file(&ack_path);
        let Some(manifest) = manifest else {
            // ts changed but the manifest is gone — treat as a failed run so
            // the user is informed instead of silently dropping the signal.
            return Some(indexing_failed_response(event_name));
        };
        return Some(json!({
            "hookSpecificOutput": {
                "hookEventName": event_name,
                "additionalContext": format!(
                    "claudix indexing complete — {} files, {} chunks. Semantic search is now ready: \
                     use search_code for conceptual queries, identifier lookups, and cross-file discovery.",
                    manifest.file_count, manifest.chunk_count
                ),
            }
        }));
    }

    // Manifest timestamp unchanged AND no index lock holder. If the spawned
    // child is still alive we're inside the slow-boot window (cold ONNX
    // load, large config parse) — never declare failure yet. Only after the
    // PID exits and the failure grace has elapsed do we surface a failure.
    if marker.child_pid.is_some_and(crate::store::marker::process_running) {
        return None;
    }
    if age < Duration::from_secs(PENDING_INDEX_FAILURE_GRACE_SECS) {
        return None;
    }

    // Pre-first-index state has `prior_ts == "none"` indefinitely, so suppressing
    // duplicate-ts acks would silence every consecutive failure on a fresh repo.
    // Always surface failure for that sentinel; gate dedup on real timestamps.
    let is_first_index = marker.prior_ts == "none";
    let already_acked = !is_first_index
        && fs::read_to_string(&ack_path)
            .ok()
            .map(|content| content.trim() == marker.prior_ts)
            .unwrap_or(false);
    let _ = fs::write(&ack_path, &marker.prior_ts);
    let _ = fs::remove_file(&marker_path);

    if already_acked {
        return None;
    }

    Some(indexing_failed_response(event_name))
}

fn indexing_failed_response(event_name: &str) -> Value {
    json!({
        "hookSpecificOutput": {
            "hookEventName": event_name,
            "additionalContext":
                "claudix background indexing ended without updating the index — it may have failed. \
                 Run /claudix:doctor to diagnose.",
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::fs;

    use crate::config::Config;
    use crate::store::Store;

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

    #[tokio::test]
    async fn check_index_ready_resurfaces_failure_for_first_index_sentinel()
    -> crate::error::Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let config = stub_config();
        write_config(fixture.root(), &config);
        let store = Store::new(fixture.root(), &config)?;
        store.ensure_layout()?;

        // Pre-ack the "none" sentinel so a naive dedup check would suppress.
        let ack_path = store.state_dir_path().join(PENDING_INDEX_ACK_FILE_NAME);
        fs::write(&ack_path, "none")?;

        let stale_created_at = "2025-01-01T00:00:00Z";
        let payload = format!("none\n{stale_created_at}\n0\n");
        fs::write(store.pending_index_marker_path(), payload)?;

        let response = check_index_ready(fixture.root(), &config, "PostToolUse");
        let response = response.unwrap_or(Value::Null);
        let context = response["hookSpecificOutput"]["additionalContext"]
            .as_str()
            .unwrap_or_default();
        assert!(
            context.contains("ended without updating the index"),
            "repeat failures before the first successful index must still surface, got: {context}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn check_index_ready_defers_failure_while_child_pid_alive() -> crate::error::Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let config = stub_config();
        write_config(fixture.root(), &config);
        let store = Store::new(fixture.root(), &config)?;
        store.ensure_layout()?;

        // Backdate the marker past the failure grace but record this test
        // process as the spawned child — `check_index_ready` must defer the
        // failure decision while that PID is still alive.
        let stale_created_at = "2025-01-01T00:00:00Z";
        let my_pid = std::process::id();
        let payload = format!("none\n{stale_created_at}\n{my_pid}\n");
        fs::write(store.pending_index_marker_path(), payload)?;

        let response = check_index_ready(fixture.root(), &config, "PostToolUse");
        assert!(
            response.is_none(),
            "must not declare failure while spawned child is alive"
        );
        assert!(
            store.pending_index_marker_path().exists(),
            "marker must survive the deferred decision"
        );
        Ok(())
    }

    #[tokio::test]
    async fn check_index_ready_surfaces_failure_when_manifest_is_missing()
    -> crate::error::Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let config = stub_config();
        write_config(fixture.root(), &config);
        let store = Store::new(fixture.root(), &config)?;
        store.ensure_layout()?;

        // Backdate the marker past the failure grace; leave the manifest absent
        // to simulate a first-time index that crashed before writing one.
        let stale_created_at = "2025-01-01T00:00:00Z";
        let payload = format!("none\n{stale_created_at}\n");
        fs::write(store.pending_index_marker_path(), payload)?;

        let response = check_index_ready(fixture.root(), &config, "PostToolUse");
        let response = response.unwrap_or(Value::Null);
        let context = response["hookSpecificOutput"]["additionalContext"]
            .as_str()
            .unwrap_or_default();
        assert!(
            context.contains("ended without updating the index"),
            "expected failure context, got: {context}"
        );
        assert!(
            !store.pending_index_marker_path().exists(),
            "marker must be cleaned up after surfacing failure"
        );
        Ok(())
    }
}
