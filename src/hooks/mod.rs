use std::path::Path;
use std::time::{Duration, SystemTime};

use serde::Deserialize;
use serde_json::{Value, json};

use crate::cli;
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

fn is_git_repo(path: &Path) -> bool {
    path.join(".git").exists()
}

fn spawn_background_index(project_root: &Path, config: &crate::config::Config) -> bool {
    let Ok(store) = Store::new(project_root, config) else {
        return false;
    };
    let needs_index = store
        .read_manifest()
        .ok()
        .flatten()
        .as_ref()
        .map(|m| index_is_stale(m, config))
        .unwrap_or(true);
    if !needs_index {
        return false;
    }
    let Ok(binary) = std::env::current_exe() else {
        return false;
    };
    std::process::Command::new(binary)
        .arg("index")
        .current_dir(project_root)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .is_ok()
}

async fn handle_session_start(project_root: &Path, _payload: HookPayload) -> Result<Option<Value>> {
    let config = config::load(project_root).ok();

    let indexing = if let Some(ref config) = config
        && config.hooks.auto_index_on_session_start
        && is_git_repo(project_root)
    {
        spawn_background_index(project_root, config)
    } else {
        false
    };

    let indexed_file_count = config
        .as_ref()
        .and_then(|config| Store::new(project_root, config).ok())
        .and_then(|store| store.read_manifest().ok().flatten())
        .map(|manifest| manifest.file_count)
        .unwrap_or(0);

    let mut response =
        session_start_response("claudix semantic search available".to_owned());
    let user_message = match consume_pending_restart().await {
        Some(message) => message,
        None => session_start_message(cli::setup_state(project_root).await, indexed_file_count, indexing),
    };
    response["systemMessage"] = Value::String(user_message);
    Ok(Some(response))
}

fn session_start_message(setup_state: cli::SetupState, indexed_file_count: u64, indexing: bool) -> String {
    match setup_state {
        cli::SetupState::Ready => {
            if indexing {
                format!("claudix indexed {indexed_file_count} files (indexing in background...)")
            } else {
                format!("claudix indexed {indexed_file_count} files")
            }
        }
        cli::SetupState::Missing(parts) => format!(
            "claudix setup incomplete (missing {}); run the install script again",
            parts.join(", ")
        ),
    }
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

    spawn_background_reindex_file(project_root, &file_path);
    Ok(None)
}

fn spawn_background_reindex_file(project_root: &Path, file_path: &str) {
    let Ok(binary) = std::env::current_exe() else {
        return;
    };
    let _ = std::process::Command::new(binary)
        .args(["reindex-file", file_path])
        .current_dir(project_root)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn();
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
        "Grep" => {
            if tool_input.path.is_some() || tool_input.include.is_some() {
                return Ok(None);
            }
            tool_input.pattern
        }
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
            std::env::var_os("XDG_DATA_HOME").map(|p| std::path::PathBuf::from(p).join("claudix"))
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
    let args = command
        .strip_prefix("rg ")
        .or_else(|| command.strip_prefix("grep "))
        .or_else(|| command.strip_prefix("ag "))?;
    extract_quoted_pattern(args.trim())
}

fn extract_quoted_pattern(args: &str) -> Option<String> {
    for &quote in b"\"'" {
        let bytes = args.as_bytes();
        if let Some(start) = bytes.iter().position(|&b| b == quote) {
            let after = &args[start + 1..];
            if let Some(end) = after.as_bytes().iter().position(|&b| b == quote) {
                let pattern = after[..end].trim();
                if !pattern.is_empty() {
                    return Some(pattern.to_owned());
                }
            }
        }
    }
    None
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
    path: Option<String>,
    include: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::Arc;
    use tempfile::tempdir;

    use crate::Claudix;
    use crate::config::Config;
    use crate::store::Manifest;

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
    async fn session_start_reports_incomplete_setup() {
        let fixture = TestFixture::new("small_rust");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());
        write_config(fixture.root(), &stub_config());

        let response = run(fixture.root(), HookEvent::SessionStart, "{}").await;
        assert!(response.is_ok());
        let response = response.ok().unwrap_or_else(|| unreachable!());
        assert!(response.is_some());
        let response = response.unwrap_or(Value::Null);
        let user_message = response["systemMessage"].as_str().unwrap_or_default();
        assert!(user_message.contains("run the install script again"));

        let model_context = response["hookSpecificOutput"]["additionalContext"]
            .as_str()
            .unwrap_or_default();
        assert_eq!(model_context, "claudix semantic search available");
    }

    #[test]
    fn session_start_message_reports_ready_setup() {
        assert_eq!(
            session_start_message(cli::SetupState::Ready, 0, false),
            "claudix indexed 0 files"
        );
        assert_eq!(
            session_start_message(cli::SetupState::Ready, 42, false),
            "claudix indexed 42 files"
        );
        assert_eq!(
            session_start_message(cli::SetupState::Ready, 42, true),
            "claudix indexed 42 files (indexing in background...)"
        );
    }

    #[tokio::test]
    async fn post_tool_use_spawns_background_reindex_and_returns_none() {
        let fixture = TestFixture::new("small_rust");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());
        write_config(fixture.root(), &stub_config());

        let payload = json!({
            "tool_name": "Write",
            "tool_input": {
                "file_path": fixture.root().join("src/math.rs"),
            }
        });
        let response = run(fixture.root(), HookEvent::PostToolUse, &payload.to_string()).await;
        assert!(response.is_ok());
        assert!(response.ok().unwrap_or_else(|| unreachable!()).is_none());
    }

    #[tokio::test]
    async fn post_tool_use_passes_through_when_auto_reembed_disabled() {
        let fixture = TestFixture::new("small_rust");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());
        let mut config = stub_config();
        config.hooks.auto_reembed_on_edit = false;
        write_config(fixture.root(), &config);

        let payload = json!({
            "tool_name": "Write",
            "tool_input": {
                "file_path": fixture.root().join("src/math.rs"),
            }
        });
        let response = run(fixture.root(), HookEvent::PostToolUse, &payload.to_string()).await;
        assert!(response.is_ok());
        assert!(response.ok().unwrap_or_else(|| unreachable!()).is_none());
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

    #[tokio::test]
    async fn pre_tool_use_denies_conceptual_bash_rg_when_index_ready() {
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
            "tool_name": "Bash",
            "tool_input": {
                "command": "rg \"where is the config loaded\""
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
        let context = response["hookSpecificOutput"]["additionalContext"]
            .as_str()
            .unwrap_or_default();
        assert!(
            context.contains("where is the config loaded"),
            "deny message should reference the extracted pattern, not the full command"
        );
        assert!(
            !context.contains("rg "),
            "deny message must not expose the raw tool invocation"
        );
    }

    #[tokio::test]
    async fn pre_tool_use_passes_bash_rg_with_path_arg_through_when_no_conceptual_pattern() {
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

        // Unquoted single-word pattern — passes through (token_count < 3 after extraction fails)
        let payload = json!({
            "tool_name": "Bash",
            "tool_input": {
                "command": "rg add src/"
            }
        });
        let response = run(fixture.root(), HookEvent::PreToolUse, &payload.to_string()).await;
        assert!(response.is_ok());
        assert!(
            response.ok().unwrap_or_else(|| unreachable!()).is_none(),
            "unquoted single-word rg command should pass through"
        );
    }

    #[tokio::test]
    async fn pre_tool_use_passes_bash_rg_with_regex_pattern_through() {
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
            "tool_name": "Bash",
            "tool_input": {
                "command": "rg \"^pub fn\" src/"
            }
        });
        let response = run(fixture.root(), HookEvent::PreToolUse, &payload.to_string()).await;
        assert!(response.is_ok());
        assert!(
            response.ok().unwrap_or_else(|| unreachable!()).is_none(),
            "regex pattern in rg command should pass through"
        );
    }

    #[tokio::test]
    async fn pre_tool_use_denies_conceptual_bash_rg_regardless_of_path_arg() {
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

        // Even with src/ path argument, a conceptual quoted pattern must be denied
        let payload = json!({
            "tool_name": "Bash",
            "tool_input": {
                "command": "rg \"where is the config loaded\" src/"
            }
        });
        let response = run(fixture.root(), HookEvent::PreToolUse, &payload.to_string()).await;
        assert!(response.is_ok());
        let response = response.ok().unwrap_or_else(|| unreachable!());
        assert!(
            response.is_some(),
            "conceptual rg query must be denied even when a path arg is present"
        );
        assert_eq!(
            response.unwrap_or(Value::Null)["hookSpecificOutput"]["permissionDecision"],
            Value::String("deny".to_owned())
        );
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
