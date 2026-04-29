use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use serde::Deserialize;
use serde_json::{Value, json};

use crate::Claudix;
use crate::config::{self, Config};
use crate::error::Result;
use crate::store::{Manifest, Store};
use crate::util::parse_rfc3339;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HookEvent {
    SessionStart,
    PostToolUse,
    PreToolUse,
}

pub async fn run(project_root: &Path, event: HookEvent, payload: &str) -> Result<Option<Value>> {
    let payload: HookPayload = serde_json::from_str(payload)?;

    match event {
        HookEvent::SessionStart => handle_session_start(project_root, payload).await,
        HookEvent::PostToolUse => handle_post_tool_use(project_root, payload).await,
        HookEvent::PreToolUse => handle_pre_tool_use(project_root, payload).await,
    }
}

async fn handle_session_start(project_root: &Path, _payload: HookPayload) -> Result<Option<Value>> {
    let pending_update = consume_pending_restart().await;

    let config = config::load(project_root)?;
    let store = Store::new(project_root, &config)?;
    let manifest = store.read_manifest()?;
    let stats = store.chunk_stats().await?;
    let mut notes = Vec::new();

    notes.push(format!(
        "claudix index: {} chunks across {} files",
        stats.chunk_count, stats.file_count
    ));

    if let Some(manifest) = manifest.as_ref() {
        notes.push(format!(
            "model {} ({} dims)",
            manifest.embedding_model, manifest.dimensions
        ));
        if let Some(last_full_index_at) = manifest.last_full_index_at.as_deref() {
            notes.push(format!("last full index {last_full_index_at}"));
        }
        if index_is_stale(manifest, &config) {
            notes.push(format!(
                "index is stale; run `claudix index` to refresh it (threshold {}h)",
                config.indexing.reindex_after_hours
            ));
        }
    } else {
        notes.push("index missing; run `claudix index` to build it".to_owned());
    }

    if config.hooks.session_start_warmup {
        let claudix = Claudix::new(project_root.to_path_buf(), Arc::new(config)).await;
        match claudix {
            Ok(claudix) => {
                if let Err(error) = claudix.embedder_health_check().await {
                    notes.push(format!("embedding health check failed: {error}"));
                }
            }
            Err(error) => notes.push(format!("embedding unavailable: {error}")),
        }
    }

    let mut response = session_start_response(notes.join(". "));
    if let Some(msg) = pending_update {
        response["systemMessage"] = Value::String(msg);
    }
    Ok(Some(response))
}

async fn handle_post_tool_use(project_root: &Path, payload: HookPayload) -> Result<Option<Value>> {
    let Some(tool_input) = payload.tool_input else {
        return Ok(None);
    };
    let Some(file_path) = tool_input.file_path else {
        return Ok(None);
    };

    let config = config::load(project_root)?;
    if !config.hooks.auto_reembed_on_edit {
        return Ok(None);
    }

    let claudix = Claudix::new(project_root.to_path_buf(), Arc::new(config)).await?;
    let _ = claudix.reindex_file(Path::new(&file_path)).await?;
    Ok(None)
}

async fn handle_pre_tool_use(project_root: &Path, payload: HookPayload) -> Result<Option<Value>> {
    let config = config::load(project_root)?;
    if !config.hooks.intercept_grep {
        return Ok(None);
    }

    let Some(tool_name) = payload.tool_name.as_deref() else {
        return Ok(None);
    };
    let Some(tool_input) = payload.tool_input else {
        return Ok(None);
    };

    let query = match tool_name {
        "Grep" => tool_input.pattern,
        "Bash" => extract_search_command(tool_input.command.as_deref()),
        _ => None,
    };

    let Some(query) = query else {
        return Ok(None);
    };
    if should_passthrough(&query) {
        return Ok(None);
    }

    let store = Store::new(project_root, &config)?;
    let manifest = store.read_manifest()?;
    let Some(manifest) = manifest else {
        return Ok(None);
    };
    if index_is_stale(&manifest, &config) {
        return Ok(None);
    }

    let stats = store.chunk_stats().await?;
    if stats.chunk_count == 0 {
        return Ok(None);
    }

    Ok(Some(pre_tool_use_deny_response(
        &query,
        stats.chunk_count,
        stats.file_count,
    )))
}

async fn consume_pending_restart() -> Option<String> {
    let data_dir = pending_restart_path()?;
    let content = tokio::fs::read_to_string(&data_dir).await.ok()?;
    let version = content.trim();
    if version.is_empty() {
        return None;
    }
    let msg = format!("claudix updated to v{version} — restart Claude Code to activate");
    let _ = tokio::fs::remove_file(&data_dir).await;
    Some(msg)
}

fn pending_restart_path() -> Option<std::path::PathBuf> {
    let base = std::env::var_os("CLAUDIX_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|| {
            std::env::var_os("XDG_DATA_HOME")
                .map(|p| std::path::PathBuf::from(p).join("claudix"))
        })
        .or_else(|| dirs::home_dir().map(|h| h.join(".local").join("share").join("claudix")))?;
    Some(base.join("pending-restart"))
}

fn session_start_response(additional_context: String) -> Value {
    json!({
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": additional_context,
        }
    })
}

fn pre_tool_use_deny_response(query: &str, chunk_count: usize, file_count: usize) -> Value {
    json!({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": "Use the claudix.search_code MCP tool for semantic queries; this query looks conceptual.",
            "additionalContext": format!(
                "Original query was '{query}'. The claudix search index has {chunk_count} chunks across {file_count} files."
            ),
        }
    })
}

fn extract_search_command(command: Option<&str>) -> Option<String> {
    let command = command?.trim();
    if !(command.starts_with("rg ") || command.starts_with("grep ") || command.starts_with("ag ")) {
        return None;
    }

    Some(command.to_owned())
}

fn should_passthrough(query: &str) -> bool {
    let trimmed = query.trim();
    trimmed.is_empty()
        || looks_like_regex(trimmed)
        || looks_like_file_target(trimmed)
        || token_count(trimmed) < 3
}

fn looks_like_regex(query: &str) -> bool {
    query.contains('^')
        || query.contains('$')
        || query.contains("\\")
        || query.contains('[')
        || query.contains(']')
        || query.contains(".*")
}

fn looks_like_file_target(query: &str) -> bool {
    query.contains("--glob")
        || query.contains("--include")
        || query.contains("*.")
        || query.contains("src/")
        || query.contains(".rs")
}

fn token_count(query: &str) -> usize {
    query
        .split(|character: char| !character.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .count()
}

fn index_is_stale(manifest: &Manifest, config: &Config) -> bool {
    let Some(last_full_index_at) = manifest.last_full_index_at.as_deref() else {
        return true;
    };
    let Ok(last_full_index_at) = parse_rfc3339(last_full_index_at) else {
        return true;
    };
    let Ok(age) = SystemTime::now().duration_since(last_full_index_at) else {
        return false;
    };

    age > Duration::from_secs(config.indexing.reindex_after_hours.saturating_mul(3_600))
}

#[derive(Debug, Deserialize)]
struct HookPayload {
    tool_name: Option<String>,
    tool_input: Option<ToolInput>,
}

#[derive(Debug, Deserialize)]
struct ToolInput {
    file_path: Option<String>,
    pattern: Option<String>,
    command: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::store::Manifest;
    use std::fs;
    use tempfile::tempdir;

    mod fixture {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/common/fixture.rs"
        ));
    }

    use fixture::TestFixture;

    fn stub_config() -> Config {
        let mut config = Config::default();
        config.embedding.model = "stub-v1".to_owned();
        config.embedding.dimensions = 8;
        config
    }

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
    async fn session_start_reports_missing_index() {
        let fixture = TestFixture::new("small_rust");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());
        let mut config = stub_config();
        config.hooks.session_start_warmup = false;
        write_config(fixture.root(), &config);

        let response = run(fixture.root(), HookEvent::SessionStart, "{}").await;
        assert!(response.is_ok());
        let response = response.ok().unwrap_or_else(|| unreachable!());
        assert!(response.is_some());
        let response = response.unwrap_or(Value::Null);
        let message = response["hookSpecificOutput"]["additionalContext"]
            .as_str()
            .unwrap_or_default();
        assert!(message.contains("index missing"));
    }

    #[tokio::test]
    async fn post_tool_use_reindexes_changed_file() {
        let fixture = TestFixture::new("small_rust");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());
        write_config(fixture.root(), &stub_config());

        let claudix = Claudix::new(fixture.root().to_path_buf(), Arc::new(stub_config())).await;
        assert!(claudix.is_ok());
        let claudix = claudix.ok().unwrap_or_else(|| unreachable!());
        assert!(claudix.index_full().await.is_ok());
        assert!(
            tokio::fs::write(
                fixture.root().join("src/math.rs"),
                "pub fn multiply(left: i32, right: i32) -> i32 {\n    left * right\n}\n",
            )
            .await
            .is_ok()
        );

        let payload = json!({
            "tool_name": "Write",
            "tool_input": {
                "file_path": fixture.root().join("src/math.rs"),
            }
        });
        let response = run(fixture.root(), HookEvent::PostToolUse, &payload.to_string()).await;
        assert!(response.is_ok());
        assert!(response.ok().unwrap_or_else(|| unreachable!()).is_none());

        let store = Store::new(fixture.root(), &stub_config());
        assert!(store.is_ok());
        let rows = store
            .ok()
            .unwrap_or_else(|| unreachable!())
            .read_chunks()
            .await;
        assert!(rows.is_ok());
        let rows = rows.ok().unwrap_or_else(|| unreachable!());
        assert!(
            rows.iter()
                .any(|row| row.name.as_deref() == Some("multiply"))
        );
        assert!(!rows.iter().any(|row| row.name.as_deref() == Some("add")));
    }

    #[tokio::test]
    async fn pre_tool_use_denies_conceptual_grep_when_index_ready() {
        let fixture = TestFixture::new("small_rust");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());
        write_config(fixture.root(), &stub_config());

        let claudix = Claudix::new(fixture.root().to_path_buf(), Arc::new(stub_config())).await;
        assert!(claudix.is_ok());
        assert!(
            claudix
                .ok()
                .unwrap_or_else(|| unreachable!())
                .index_full()
                .await
                .is_ok()
        );

        let payload = json!({
            "tool_name": "Grep",
            "tool_input": {
                "pattern": "where is config loaded"
            }
        });
        let response = run(fixture.root(), HookEvent::PreToolUse, &payload.to_string()).await;
        assert!(response.is_ok());
        let response = response.ok().unwrap_or_else(|| unreachable!());
        assert!(response.is_some());
        let response = response.unwrap_or(Value::Null);
        assert_eq!(
            response["hookSpecificOutput"]["permissionDecision"],
            Value::String("deny".to_owned())
        );
    }

    #[tokio::test]
    async fn pre_tool_use_passes_regex_queries_through() {
        let fixture = TestFixture::new("small_rust");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());
        write_config(fixture.root(), &stub_config());

        let payload = json!({
            "tool_name": "Grep",
            "tool_input": {
                "pattern": "^pub fn"
            }
        });
        let response = run(fixture.root(), HookEvent::PreToolUse, &payload.to_string()).await;
        assert!(response.is_ok());
        assert!(response.ok().unwrap_or_else(|| unreachable!()).is_none());
    }

    #[test]
    fn stale_index_detection_respects_threshold() {
        let mut config = stub_config();
        config.indexing.reindex_after_hours = 24;
        let mut manifest = Manifest::new("stub-v1", 8);
        manifest.last_full_index_at = Some("2026-04-20T00:00:00Z".to_owned());
        assert!(index_is_stale(&manifest, &config));
    }

    #[test]
    fn fresh_index_detection_allows_recent_manifest() {
        let project_root = tempdir();
        assert!(project_root.is_ok());
        let _project_root = project_root.ok().unwrap_or_else(|| unreachable!());
        let mut config = stub_config();
        config.indexing.reindex_after_hours = 24 * 365 * 20;
        let mut manifest = Manifest::new("stub-v1", 8);
        manifest.last_full_index_at = Some(crate::util::now_rfc3339());
        assert!(!index_is_stale(&manifest, &config));
    }
}
