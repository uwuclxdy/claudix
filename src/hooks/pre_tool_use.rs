use std::fs;
use std::future::Future;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use serde_json::Value;

use crate::Claudix;
use crate::config::{self, Config, EmbeddingProvider};
use crate::embedding::bundled::{BUNDLED_DIMENSIONS, BUNDLED_MODEL_ID};
use crate::embedding::side_channel;
use crate::error::Result;
use crate::prompts::hooks::pre_tool_use_search_response;
use crate::search::{SearchQuery, Searcher};
use crate::store::Store;
use crate::types::Dimension;

use super::grep::{extract_search_command, should_passthrough};
use super::payload::{HookPayload, grep_input_has_scoping_flag};

const HEALTH_CACHE_FILE_NAME: &str = "embedder-health";
const HEALTH_CACHE_HEALTHY_TTL_SECS: u64 = 60;
const HEALTH_CACHE_UNHEALTHY_TTL_SECS: u64 = 30;
const PRE_TOOL_USE_TIMEOUT_MS: u64 = 1_500;

/// Budget for the warm-embed exchange against the MCP server. Deliberately a
/// fraction of the hook budget: a miss (server still cold-building, no server)
/// falls back immediately, and the server finishes warming in the background
/// so the next interception hits.
const WARM_EMBED_TIMEOUT_MS: u64 = 400;

pub(super) async fn handle_pre_tool_use(
    project_root: &Path,
    payload: HookPayload,
) -> Result<Option<Value>> {
    // Hooks must fail open: a corrupt config or unreadable store yields a
    // passthrough, not a propagated error that could break the session.
    let Ok(config) = config::load(project_root) else {
        return Ok(None);
    };
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
            if grep_input_has_scoping_flag(&tool_input) {
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

    let Ok(store) = Store::new(project_root, &config) else {
        return Ok(None);
    };
    let Ok(Some(manifest)) = store.read_manifest() else {
        return Ok(None);
    };
    if manifest.is_stale(&config) {
        return Ok(None);
    }
    if manifest.chunk_count == 0 {
        return Ok(None);
    }

    let health_cache_path = store.state_dir_path().join(HEALTH_CACHE_FILE_NAME);
    if matches!(read_cached_health(&health_cache_path), Some(false)) {
        return Ok(None);
    }

    let search_query = SearchQuery {
        query: query.clone(),
        top_k: config.search.top_k,
        language_filter: None,
        path_prefix: None,
        // Interception is active-only; cross-repo search is opt-in at the
        // tool boundary, not when redirecting a Grep that mentions nothing.
        repos: Vec::new(),
    };

    // Warm path: embed via the MCP server's already-built provider. The vector
    // is only usable when it comes from the exact model identity the corpus
    // was indexed with.
    let warm = side_channel::request_embedding(
        &store.embed_port_marker_path(),
        &query,
        Duration::from_millis(WARM_EMBED_TIMEOUT_MS),
    )
    .await
    .filter(|embed| {
        embed.model == manifest.embedding_model && embed.dimensions == manifest.dimensions
    });

    let project_root = project_root.to_path_buf();
    let work = match warm {
        Some(embed) => {
            let searcher =
                Searcher::without_embedder(project_root, store.clone(), config.search.clone());
            let dims = Dimension(embed.dimensions);
            futures_work(async move {
                searcher
                    .search_with_query_vector(search_query, embed.vector, dims)
                    .await
            })
        }
        // No warm channel and the provider build means a bundled ONNX cold
        // load: more expensive than this hook's entire budget, and thrown away
        // with the process. Pass the grep through instead — the request above
        // has already primed the MCP server's cache for the next interception.
        None if cold_provider_is_bundled(&config) => return Ok(None),
        None => {
            let config_arc = Arc::new(config.clone());
            futures_work(async move {
                let claudix = Claudix::new(project_root, config_arc).await?;
                claudix.search(search_query).await
            })
        }
    };

    match tokio::time::timeout(Duration::from_millis(PRE_TOOL_USE_TIMEOUT_MS), work).await {
        Ok(Ok(found)) => {
            write_cached_health(&health_cache_path, true);
            if found.results.is_empty() {
                Ok(None)
            } else {
                Ok(Some(pre_tool_use_search_response(&query, found.results)))
            }
        }
        Ok(Err(_)) => {
            write_cached_health(&health_cache_path, false);
            Ok(None)
        }
        // Timeout doesn't necessarily mean the embedder is unhealthy — a slow
        // store read or endpoint can exceed the budget — so pass through
        // without poisoning the cache.
        Err(_) => Ok(None),
    }
}

/// Boxed unification of the two work futures (warm vector vs cold provider),
/// which have distinct concrete types.
fn futures_work(
    future: impl Future<Output = Result<crate::search::SearchResults>> + 'static,
) -> std::pin::Pin<Box<dyn Future<Output = Result<crate::search::SearchResults>>>> {
    Box::pin(future)
}

/// Whether `build_provider` for this config would construct the bundled ONNX
/// model — either directly, or as the http provider's eager fallback. A hook
/// process must never pay that cold load.
fn cold_provider_is_bundled(config: &Config) -> bool {
    #[cfg(any(test, feature = "test-stub"))]
    if config.embedding.model.starts_with("stub") {
        return false;
    }
    match config.embedding.provider {
        EmbeddingProvider::Bundled => true,
        EmbeddingProvider::Http => {
            config.embedding.model == BUNDLED_MODEL_ID
                && Dimension(config.embedding.dimensions) == BUNDLED_DIMENSIONS
        }
    }
}

fn read_cached_health(marker_path: &Path) -> Option<bool> {
    let metadata = fs::metadata(marker_path).ok()?;
    let modified = metadata.modified().ok()?;
    let age = SystemTime::now().duration_since(modified).ok()?;
    let content = fs::read_to_string(marker_path).ok()?;
    let healthy = match content.trim() {
        "1" => true,
        "0" => false,
        _ => return None,
    };
    let ttl = if healthy {
        HEALTH_CACHE_HEALTHY_TTL_SECS
    } else {
        HEALTH_CACHE_UNHEALTHY_TTL_SECS
    };
    (age < Duration::from_secs(ttl)).then_some(healthy)
}

fn write_cached_health(marker_path: &Path, healthy: bool) {
    let _ = fs::write(marker_path, if healthy { "1" } else { "0" });
}

#[cfg(test)]
mod tests {
    use super::*;

    use serde_json::json;

    use std::fs;
    use std::sync::Arc;

    use crate::Claudix;
    use crate::config::Config;
    use crate::hooks::{HookEvent, run};

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
                .index_full(&mut ())
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
    async fn pre_tool_use_passes_conceptual_bash_rg_when_search_returns_no_results() {
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
                .index_full(&mut ())
                .await
                .is_ok()
        );

        // "where is the config loaded" is conceptual but the small_rust fixture has no config code;
        // semantic search returns empty → must pass through, not block grep with no alternative.
        let payload = json!({
            "tool_name": "Bash",
            "tool_input": {
                "command": "rg \"where is the config loaded\""
            }
        });
        let response = run(fixture.root(), HookEvent::PreToolUse, &payload.to_string()).await;
        assert!(response.is_ok());
        assert!(
            response.ok().unwrap_or_else(|| unreachable!()).is_none(),
            "must pass through when semantic search has no results to offer"
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
                .index_full(&mut ())
                .await
                .is_ok()
        );

        // Unquoted single-word pattern — extracted but token_count < 3 so passes through.
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
    async fn pre_tool_use_intercepts_unquoted_identifier_in_bash_rg() {
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
                .index_full(&mut ())
                .await
                .is_ok()
        );

        // Unquoted identifier with enough tokens — should be intercepted just like a quoted query.
        let payload = json!({
            "tool_name": "Bash",
            "tool_input": {
                "command": "rg add_two_numbers"
            }
        });
        let response = run(fixture.root(), HookEvent::PreToolUse, &payload.to_string()).await;
        assert!(response.is_ok());
        let response = response.ok().unwrap_or_else(|| unreachable!());
        assert!(
            response.is_some(),
            "unquoted multi-token identifier should be intercepted"
        );
        let response = response.unwrap_or(Value::Null);
        assert_eq!(
            response["hookSpecificOutput"]["permissionDecision"],
            Value::String("deny".to_owned())
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
                .index_full(&mut ())
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
    async fn pre_tool_use_passes_conceptual_bash_rg_with_path_arg_when_no_results() {
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
                .index_full(&mut ())
                .await
                .is_ok()
        );

        // Conceptual query with a path arg: semantic search still finds nothing for "config loaded"
        // in the small_rust fixture → must pass through.
        let payload = json!({
            "tool_name": "Bash",
            "tool_input": {
                "command": "rg \"where is the config loaded\" src/"
            }
        });
        let response = run(fixture.root(), HookEvent::PreToolUse, &payload.to_string()).await;
        assert!(response.is_ok());
        assert!(
            response.ok().unwrap_or_else(|| unreachable!()).is_none(),
            "must pass through when semantic search has no results to offer"
        );
    }

    #[tokio::test]
    async fn pre_tool_use_returns_search_results_in_context() {
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
                .index_full(&mut ())
                .await
                .is_ok()
        );
        assert!(
            std::fs::write(
                fixture.root().join("src/math.rs"),
                "pub fn subtract(left: i32, right: i32) -> i32 { left - right }\n",
            )
            .is_ok()
        );

        let payload = json!({
            "tool_name": "Grep",
            "tool_input": { "pattern": "add two numbers together" }
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
            context.contains("claudix search results"),
            "hook must embed search results in context, got: {context}"
        );
        assert!(
            context.contains("src/"),
            "context must include file paths from search results"
        );
        assert!(
            context.contains("[STALE - file modified since index]"),
            "context must warn about stale hits, got: {context}"
        );
        assert!(
            context.contains("search_code MCP tool"),
            "context must include tip to use search_code directly, got: {context}"
        );
    }

    /// With the real bundled provider configured and no warm channel, the hook
    /// must pass through rather than cold-load ONNX inside its budget.
    #[tokio::test]
    async fn pre_tool_use_passes_through_for_bundled_without_warm_channel() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let stub = stub_config();
        write_config(fixture.root(), &stub);
        let claudix = Claudix::new(fixture.root().to_path_buf(), Arc::new(stub.clone())).await?;
        claudix.index_full(&mut ()).await?;

        // Switch the on-disk config to the real bundled model. The index is
        // fresh and non-empty, so only the never-cold-load rule can produce a
        // passthrough here.
        let mut bundled = stub;
        bundled.embedding.provider = crate::config::EmbeddingProvider::Bundled;
        bundled.embedding.model = BUNDLED_MODEL_ID.to_owned();
        bundled.embedding.dimensions = BUNDLED_DIMENSIONS.0;
        write_config(fixture.root(), &bundled);

        let payload = json!({
            "tool_name": "Grep",
            "tool_input": { "pattern": "where is config loaded" }
        });
        let response = run(fixture.root(), HookEvent::PreToolUse, &payload.to_string()).await?;
        assert!(
            response.is_none(),
            "bundled provider without a warm channel must pass through"
        );
        Ok(())
    }

    /// With a live warm channel serving the manifest's model identity, the
    /// hook intercepts even though its own cold path (bundled) is forbidden —
    /// proving the warm vector was used.
    #[tokio::test]
    async fn pre_tool_use_intercepts_via_warm_channel_with_bundled_config() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let stub = stub_config();
        write_config(fixture.root(), &stub);
        let claudix = Claudix::new(fixture.root().to_path_buf(), Arc::new(stub.clone())).await?;
        claudix.index_full(&mut ()).await?;

        // Warm listener owning the fixture's marker but loading its provider
        // config from a second root, so it serves the stub identity the corpus
        // was indexed with while the fixture's own config goes bundled below.
        let store = Store::new(fixture.root(), &stub)?;
        let provider_root = tempfile::tempdir()?;
        write_config(provider_root.path(), &stub);
        let listener = side_channel::bind_and_advertise(&store.embed_port_marker_path()).await?;
        let server = tokio::spawn(side_channel::serve_embed_requests(
            listener,
            provider_root.path().to_path_buf(),
            Arc::new(crate::embedding::ProviderCache::new()),
        ));

        let mut bundled = stub;
        bundled.embedding.provider = crate::config::EmbeddingProvider::Bundled;
        bundled.embedding.model = BUNDLED_MODEL_ID.to_owned();
        bundled.embedding.dimensions = BUNDLED_DIMENSIONS.0;
        write_config(fixture.root(), &bundled);

        let payload = json!({
            "tool_name": "Grep",
            "tool_input": { "pattern": "where is config loaded" }
        });
        let response = run(fixture.root(), HookEvent::PreToolUse, &payload.to_string()).await;
        server.abort();
        let response = response?;

        let response = response.unwrap_or(Value::Null);
        assert_eq!(
            response["hookSpecificOutput"]["permissionDecision"],
            Value::String("deny".to_owned()),
            "warm channel must enable interception despite a bundled cold path"
        );
        Ok(())
    }

    #[tokio::test]
    async fn pre_tool_use_passes_through_on_corrupt_config() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let claude_dir = fixture.root().join(".claude");
        fs::create_dir_all(&claude_dir)?;
        // Malformed TOML — config::load would error; the hook must fail open.
        fs::write(
            claude_dir.join("claudix.toml"),
            "this is = not = valid toml [",
        )?;

        let payload = json!({
            "tool_name": "Grep",
            "tool_input": { "pattern": "where is config loaded" }
        });
        let response = run(fixture.root(), HookEvent::PreToolUse, &payload.to_string()).await?;
        assert!(
            response.is_none(),
            "corrupt config must passthrough, not error"
        );
        Ok(())
    }
}
