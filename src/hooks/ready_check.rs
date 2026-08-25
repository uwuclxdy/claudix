use std::fs;
use std::path::Path;
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::Config;
use crate::prompts::hooks::{indexing_complete_response, indexing_failed_response};
use crate::store::Store;
use crate::store::marker::pending_index::{self, FAILURE_GRACE_SECS};

pub(super) const PENDING_INDEX_READY_GRACE_SECS: u64 = 3;
pub(super) const PENDING_INDEX_ACK_FILE_NAME: &str = "indexing-pending-acked";

pub(super) fn check_index_ready(
    store: &Store,
    config: &Config,
    event_name: &str,
    session_id: Option<&str>,
    prompt_id: Option<&str>,
) -> Option<Value> {
    let marker_path = store.pending_index_marker_path();
    let ack_path = store.state_dir_path().join(PENDING_INDEX_ACK_FILE_NAME);
    let marker = pending_index::read(&marker_path)?;

    // `created_at` in the future (clock skew, restored backup) would make
    // `duration_since` error forever; treat that as "past every grace window"
    // so the marker can be cleaned up instead of jamming the auto-indexer.
    let now = SystemTime::now();
    let age = match now.duration_since(marker.created_at) {
        Ok(age) => age,
        Err(_) => Duration::from_secs(FAILURE_GRACE_SECS),
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
            return Some(build_failed_response(
                store.project_root(),
                config,
                event_name,
            ));
        };
        return Some(indexing_complete_response(
            event_name,
            manifest.file_count,
            manifest.chunk_count,
        ));
    }

    // Manifest timestamp unchanged AND no index lock holder. If the spawned
    // child is still alive we're inside the slow-boot window (cold ONNX
    // load, large config parse) — never declare failure yet. Only after the
    // PID exits and the failure grace has elapsed do we surface a failure.
    if marker
        .child_pid
        .is_some_and(crate::store::marker::process_running)
    {
        return None;
    }
    if age < Duration::from_secs(FAILURE_GRACE_SECS) {
        return None;
    }

    // A failed run is not consumed here: the marker stays so each new turn can
    // resurface the notice while the index stays broken. For an attributed event
    // the dedupe key is the prompt id: the first decision in a turn records it
    // and later decisions with the same id are suppressed, so re-fired submits
    // in one turn attach once and a new prompt (a new turn) attaches a fresh
    // copy. The record is per-session so another session's events can never
    // rewrite this one's dedupe state.
    let Some(session_id) = session_id else {
        // No session to scope a record to: fall back to the failed-run ledger
        // (the ack file), keyed on the marker's raw `created_at` so each
        // re-claim is a new failed run — keying on `prior_ts` would collapse a
        // fresh repo's consecutive "none" failures into one and silently
        // swallow every run after the first. A legacy client degrades to one
        // copy per failed run instead of one per decision.
        let already_acked = fs::read_to_string(&ack_path)
            .ok()
            .map(|content| content.trim() == marker.created_at_raw)
            .unwrap_or(false);
        if already_acked {
            return None;
        }
        let _ = fs::write(&ack_path, &marker.created_at_raw);
        return Some(build_failed_response(
            store.project_root(),
            config,
            event_name,
        ));
    };
    let prompt_key = prompt_id.unwrap_or("");
    let record_path = store.index_failure_seen_path(session_id);
    if read_seen_failure(&record_path).is_some_and(|record| record.prompt_id == prompt_key) {
        return None;
    }
    let record = SeenFailure {
        prompt_id: prompt_key.to_owned(),
    };
    if let Ok(json) = serde_json::to_string(&record) {
        let _ = fs::write(&record_path, json);
    }

    Some(build_failed_response(
        store.project_root(),
        config,
        event_name,
    ))
}

/// Last surfaced prompt id for this session, if any. Fail-open: an unreadable
/// or malformed record reads as none, so the failure surfaces rather than being
/// wrongly suppressed.
fn read_seen_failure(path: &Path) -> Option<SeenFailure> {
    let content = fs::read_to_string(path).ok()?;
    serde_json::from_str(&content).ok()
}

/// Per-session, per-turn dedupe record for the index-failure context. Keyed on
/// the prompt id only: a new prompt (a new turn) resurfaces. The failed run's
/// `prior_ts` is constant for a session's marker lifetime — only a new
/// SessionStart re-claims the marker, and that is a new session with its own
/// record — so it carries no dedupe signal here.
#[derive(Debug, Serialize, Deserialize)]
struct SeenFailure {
    prompt_id: String,
}

/// Build the indexing-failed notice with a pointer to `index.log` and, when
/// cheap to read, the last error line the failed run left behind. The log path
/// is shown as the project-root-joined absolute path so the user can open it
/// from any working directory; the last error is read from that same path.
fn build_failed_response(project_root: &Path, config: &Config, event_name: &str) -> Value {
    let absolute = project_root.join(&config.paths.log_dir).join("index.log");
    let last_error = last_index_error(&absolute);
    indexing_failed_response(
        event_name,
        &absolute.to_string_lossy(),
        last_error.as_deref(),
    )
}

/// Last `error:` line a failed index appended to `index.log`, if any. Benign
/// progress lines (`indexed …`/`verified …`) are ignored so only a real error
/// is surfaced. Delegates to the shared log scanner in `util`.
fn last_index_error(log_path: &Path) -> Option<String> {
    crate::util::last_error_line(log_path)
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
    async fn check_index_ready_dedupes_within_a_prompt_and_resurfaces_on_a_new_prompt()
    -> crate::error::Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let config = stub_config();
        write_config(fixture.root(), &config);
        let store = Store::new(fixture.root(), &config)?;
        store.ensure_layout()?;

        let stale_created_at = "2025-01-01T00:00:00Z";
        let payload = format!("none\n{stale_created_at}\n0\n");
        fs::write(store.pending_index_marker_path(), payload)?;

        let first = check_index_ready(
            &store,
            &config,
            "PostToolUse",
            Some("sess-A"),
            Some("prompt-1"),
        );
        assert!(
            first.is_some(),
            "the first decision in a turn must surface the failure"
        );
        assert!(
            store.pending_index_marker_path().exists(),
            "the failed marker must survive its surfaced failure so later turns can resurface it"
        );

        let repeat = check_index_ready(
            &store,
            &config,
            "PostToolUse",
            Some("sess-A"),
            Some("prompt-1"),
        );
        assert!(
            repeat.is_none(),
            "a repeat decision in the same prompt must not resurface the failure"
        );

        let next_turn = check_index_ready(
            &store,
            &config,
            "PostToolUse",
            Some("sess-A"),
            Some("prompt-2"),
        );
        assert!(
            next_turn.is_some(),
            "a new prompt must resurface the failure while the index stays broken"
        );
        Ok(())
    }

    #[tokio::test]
    async fn session_less_does_not_silence_a_second_failed_run() -> crate::error::Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let config = stub_config();
        write_config(fixture.root(), &config);
        let store = Store::new(fixture.root(), &config)?;
        store.ensure_layout()?;

        // Two distinct failed runs on a fresh repo: same "none" prior_ts (the
        // manifest never advances) but different created_at (each re-claim
        // refreshes it). Both must surface; keying the ack ledger on prior_ts
        // would swallow the second.
        fs::write(
            store.pending_index_marker_path(),
            "none\n2025-01-01T00:00:00Z\n0\n",
        )?;
        let first = check_index_ready(&store, &config, "UserPromptSubmit", None, None);
        assert!(first.is_some(), "the first failed run must surface");

        fs::write(
            store.pending_index_marker_path(),
            "none\n2025-01-02T00:00:00Z\n0\n",
        )?;
        let second = check_index_ready(&store, &config, "UserPromptSubmit", None, None);
        assert!(
            second.is_some(),
            "a second failed run (new created_at, same 'none' prior_ts) must surface, not be swallowed by the ack ledger"
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

        let response = check_index_ready(&store, &config, "PostToolUse", None, None);
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
    async fn check_index_ready_failure_points_at_log_and_last_error() -> crate::error::Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let config = stub_config();
        write_config(fixture.root(), &config);
        let store = Store::new(fixture.root(), &config)?;
        store.ensure_layout()?;

        // A failed background index appended its error to index.log.
        let log_dir = fixture.root().join(&config.paths.log_dir);
        fs::create_dir_all(&log_dir)?;
        fs::write(
            log_dir.join("index.log"),
            "indexed src/lib.rs\nerror: embedding provider unreachable\n",
        )?;

        let stale_created_at = "2025-01-01T00:00:00Z";
        let payload = format!("none\n{stale_created_at}\n");
        fs::write(store.pending_index_marker_path(), payload)?;

        let response = check_index_ready(&store, &config, "UserPromptSubmit", None, None);
        let response = response.unwrap_or(Value::Null);
        let context = response["hookSpecificOutput"]["additionalContext"]
            .as_str()
            .unwrap_or_default();
        assert!(
            context.contains("ended without updating the index"),
            "must still carry the failure phrase, got: {context}"
        );
        assert!(
            context.contains("index.log"),
            "failure notice must point at the log path, got: {context}"
        );
        assert!(
            context.contains("embedding provider unreachable"),
            "failure notice must surface the last error line, got: {context}"
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

        let response = check_index_ready(&store, &config, "PostToolUse", None, None);
        let response = response.unwrap_or(Value::Null);
        let context = response["hookSpecificOutput"]["additionalContext"]
            .as_str()
            .unwrap_or_default();
        assert!(
            context.contains("ended without updating the index"),
            "expected failure context, got: {context}"
        );
        assert!(
            store.pending_index_marker_path().exists(),
            "marker must survive the surfaced failure so later turns can resurface it"
        );
        Ok(())
    }

    #[tokio::test]
    async fn build_failed_response_uses_absolute_log_path() -> crate::error::Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let config = stub_config();
        write_config(fixture.root(), &config);
        let store = Store::new(fixture.root(), &config)?;
        store.ensure_layout()?;

        let stale_created_at = "2025-01-01T00:00:00Z";
        let payload = format!("none\n{stale_created_at}\n");
        fs::write(store.pending_index_marker_path(), payload)?;

        let response = check_index_ready(&store, &config, "PostToolUse", None, None);
        let response = response.unwrap_or(Value::Null);
        let context = response["hookSpecificOutput"]["additionalContext"]
            .as_str()
            .unwrap_or_default();

        // The notice must include the project-root-joined absolute path so the
        // user can open it from any working directory, not just the project root.
        let expected = fixture
            .root()
            .join(&config.paths.log_dir)
            .join("index.log")
            .to_string_lossy()
            .to_string();
        assert!(
            context.contains(&expected),
            "failure notice must contain the absolute log path ({expected}), got: {context}"
        );
        Ok(())
    }
}
