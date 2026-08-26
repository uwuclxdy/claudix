use std::borrow::Cow;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{
    CallToolResult, ContentBlock, Implementation, JsonObject, ListToolsResult,
    PaginatedRequestParams, ServerCapabilities, ServerInfo, Tool,
};
use rmcp::service::RequestContext;
use rmcp::transport::io::stdio;
use rmcp::{ErrorData, RoleServer, ServerHandler, ServiceExt, tool, tool_handler, tool_router};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::Value;

use crate::cli;
use crate::config;
use crate::embedding::{ProviderCache, side_channel};
use crate::error::{ClaudixError, RecoveryHint};
use crate::prompts::hints;
use crate::store::Store;

/// What every `#[tool]` handler returns. The `Ok` always carries a
/// `CallToolResult`; a failed tool run is an `Ok(CallToolResult::error(..))`, not
/// an `Err`, so the message reaches the caller instead of being rendered opaque.
type ToolOutcome = std::result::Result<CallToolResult, ErrorData>;

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, JsonSchema)]
pub struct SearchCodeRequest {
    pub query: String,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub language_filter: Option<Vec<String>>,
    #[serde(default)]
    pub path_prefix: Option<String>,
    #[serde(default)]
    pub repos: Option<Vec<String>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, JsonSchema, Default)]
struct ReindexRequest {
    /// Present = reindex this file only; absent = sweep the whole project.
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    force: bool,
}

#[derive(Debug, Clone, PartialEq, Deserialize, JsonSchema, Default)]
struct FindDuplicatesRequest {
    #[serde(default)]
    min_similarity: Option<f32>,
    #[serde(default)]
    limit: Option<u32>,
    #[serde(default)]
    repos: Option<Vec<String>>,
}

/// The claudix MCP server. Holds the active project root and the generated tool
/// router; each `#[tool]` handler validates then delegates to the same
/// `cli::run_*` function the CLI uses.
#[derive(Clone)]
pub struct ClaudixServer {
    project_root: PathBuf,
    /// Warm provider shared with the embed side channel: the first search or
    /// hook embed pays the build, the rest of the session reuses it.
    provider_cache: Arc<ProviderCache>,
    /// Latched once a degraded-search notice (endpoint-down or reindex hint)
    /// has been surfaced, so the notice bills at most once per session. Shared
    /// across the per-call clones of this server.
    warned_degraded: Arc<AtomicBool>,
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl ClaudixServer {
    fn new(project_root: PathBuf, provider_cache: Arc<ProviderCache>) -> Self {
        Self {
            project_root,
            provider_cache,
            warned_degraded: Arc::new(AtomicBool::new(false)),
            tool_router: Self::tool_router(),
        }
    }

    /// Semantic code search over the active project plus optional cross-repos.
    #[tool(name = "search_code")]
    async fn search_code(&self, Parameters(request): Parameters<SearchCodeRequest>) -> ToolOutcome {
        if request.query.trim().is_empty() {
            return Ok(error_result(ClaudixError::ConfigInvalid {
                message: "query cannot be empty".to_owned(),
                recovery: RecoveryHint(hints::QUERY_NON_EMPTY),
            }));
        }
        let outcome = cli::run_search_cached(
            &self.project_root,
            &self.provider_cache,
            request.query,
            request.top_k.map(|value| value as usize),
            request.language_filter,
            request.path_prefix,
            request.repos,
        )
        .await
        .map(|mut output| {
            throttle_degraded_notice(&mut output, &self.warned_degraded);
            output
        })
        .and_then(to_value);
        Ok(into_result(outcome))
    }

    /// Whole-project sweep, or one file when `path` is set. `force` wipes first
    /// and is meaningless for a single file, so the two are mutually exclusive
    /// rather than silently letting `force` nuke the index on a file reindex.
    #[tool(name = "reindex")]
    async fn reindex(&self, Parameters(request): Parameters<ReindexRequest>) -> ToolOutcome {
        if let Some(path) = request.path {
            if path.trim().is_empty() {
                return Ok(error_result(ClaudixError::ConfigInvalid {
                    message: "path cannot be empty".to_owned(),
                    recovery: RecoveryHint(hints::PATH_NON_EMPTY),
                }));
            }
            let outcome = cli::run_reindex_file(&self.project_root, Path::new(&path))
                .await
                .and_then(to_value);
            return Ok(into_result(outcome));
        }
        let outcome = async {
            if request.force {
                cli::run_clear_index(&self.project_root).await?;
            }
            let output = cli::run_index(&self.project_root, false).await?;
            to_value(output)
        }
        .await;
        Ok(into_result(outcome))
    }

    /// Near-identical code chunks across files using stored embeddings.
    #[tool(name = "find_duplicates")]
    async fn find_duplicates(
        &self,
        Parameters(request): Parameters<FindDuplicatesRequest>,
    ) -> ToolOutcome {
        let outcome = cli::run_find_duplicates(
            &self.project_root,
            request.min_similarity,
            request.limit.map(|value| value as usize),
            request.repos,
        )
        .await
        .and_then(to_value);
        Ok(into_result(outcome))
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for ClaudixServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build()).with_server_info(
            Implementation::new(env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION")),
        )
    }

    /// Serve the centralized catalog from `prompts::mcp`, not the macro-derived
    /// per-tool schemas, so descriptions and order stay the single source.
    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> std::result::Result<ListToolsResult, ErrorData> {
        Ok(ListToolsResult::with_all_items(tool_catalog()?))
    }
}

/// The served tool catalog, built from `prompts::mcp` in declared order. Single
/// source of truth for `tools/list`; the macro-derived per-tool schemas are not
/// served.
fn tool_catalog() -> std::result::Result<Vec<Tool>, ErrorData> {
    crate::prompts::mcp::tool_definitions()
        .into_iter()
        .map(tool_from_definition)
        .collect()
}

pub async fn run(project_root: impl AsRef<Path>) -> crate::error::Result<()> {
    let project_root = project_root.as_ref().to_path_buf();
    let provider_cache = Arc::new(ProviderCache::new());
    let server = ClaudixServer::new(project_root.clone(), Arc::clone(&provider_cache));

    // Warm-embed side channel for hook processes: fail-open, the MCP server
    // must serve normally when the listener or marker cannot be set up. The
    // marker path needs a Store (config + canonical root); any failure along
    // the way just means hooks stay on their cold path. Skipped outside a git
    // repo, where there is nothing to index and `ensure_layout` would create a
    // stray `.claudix/`.
    let embed_marker = crate::enumeration::is_git_repo(&project_root)
        .then(|| {
            // `ensure_layout` below is a write: clean up any pre-fix nested
            // store between the session's start dir and this resolved root
            // first (ruling 2026-08-25); fail-open.
            crate::enumeration::delete_nested_stores(&project_root);
            config::load(&project_root).ok()
        })
        .flatten()
        .and_then(|config| Store::new(&project_root, &config).ok())
        // On a never-indexed repo the state dir doesn't exist yet and the
        // marker write would silently fail, muting the warm channel for the
        // whole session.
        .filter(|store| store.ensure_layout().is_ok())
        .map(|store| store.embed_port_marker_path());
    let mut advertisement: Option<tokio::task::JoinHandle<()>> = None;
    if let Some(marker_path) = embed_marker.clone()
        && let Ok(listener) = side_channel::bind_and_advertise(&marker_path).await
    {
        let port = listener.local_addr().ok().map(|addr| addr.port());
        tokio::spawn(side_channel::serve_embed_requests(
            listener,
            project_root,
            provider_cache,
        ));
        // Re-advertise on a cadence so the marker heals if another same-repo
        // server withdrew it on exit, leaving this one unadvertised.
        if let Some(port) = port {
            advertisement = Some(tokio::spawn(side_channel::maintain_advertisement(
                marker_path,
                port,
            )));
        }
    }

    // The service driver is spawned with `spawn_local` under the `local` feature,
    // so it must run inside a `LocalSet`.
    let local = tokio::task::LocalSet::new();
    let outcome = local
        .run_until(async move {
            let running = server
                .serve(stdio())
                .await
                .map_err(|error| ClaudixError::Mcp(error.to_string()))?;
            running
                .waiting()
                .await
                .map_err(|error| ClaudixError::Mcp(error.to_string()))?;
            Ok(())
        })
        .await;

    // Stop re-advertising before withdrawing so the marker isn't re-written for
    // this exiting pid right after we clear it.
    if let Some(advertisement) = advertisement {
        advertisement.abort();
    }
    if let Some(marker_path) = embed_marker {
        side_channel::withdraw(&marker_path);
    }
    outcome
}

/// Convert one `prompts::mcp` JSON tool definition into an rmcp `Tool`. Keeps
/// rmcp model types out of the prompts module.
fn tool_from_definition(definition: Value) -> std::result::Result<Tool, ErrorData> {
    let name = definition
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| ErrorData::internal_error("tool definition missing name", None))?
        .to_owned();
    let description = definition
        .get("description")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let input_schema: JsonObject = match definition.get("inputSchema") {
        Some(Value::Object(map)) => map.clone(),
        _ => JsonObject::new(),
    };
    Ok(Tool::new(
        Cow::Owned(name),
        Cow::Owned(description),
        Arc::new(input_schema),
    ))
}

fn to_value<T: serde::Serialize>(value: T) -> crate::error::Result<Value> {
    serde_json::to_value(value).map_err(ClaudixError::from)
}

/// Keep a degraded-search notice (endpoint-down or reindex hint) only the
/// first time a degraded search occurs this session; drop it on later ones.
/// The lexical results still return every time — only the notice is throttled.
/// A healthy search (`degraded_hint` is `None`) leaves the latch untouched
/// thanks to the short-circuit, so it can never consume the one-shot ahead of
/// a real degradation.
fn throttle_degraded_notice(output: &mut cli::SearchOutput, warned: &AtomicBool) {
    if output.degraded_hint.is_some() && warned.swap(true, Ordering::Relaxed) {
        output.degraded_hint = None;
    }
}

/// Successful payload → structured content; error → a tool-level error result the
/// caller can read. A serialize failure for the payload degrades to an error
/// result rather than tearing down the server loop.
fn into_result(outcome: crate::error::Result<Value>) -> CallToolResult {
    match outcome {
        Ok(payload) => CallToolResult::structured(payload),
        Err(error) => error_result(error),
    }
}

/// Map a `ClaudixError` to a tool-level error result, embedding the recovery hint
/// as `"{message}. Recovery: {hint}"` (no suffix when the error carries no hint).
fn error_result(error: ClaudixError) -> CallToolResult {
    let message = error.to_string();
    let text = match error.recovery_hint() {
        Some(recovery) => format!("{message}. Recovery: {recovery}"),
        None => message,
    };
    CallToolResult::error(vec![ContentBlock::text(text)])
}

#[cfg(test)]
mod tests {
    use super::*;

    mod fixture {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/common/fixture.rs"
        ));
    }

    // Each `include!` is its own module copy, so the helpers this one doesn't
    // need (the indexing path) are dead here but live in `cli`'s copy.
    #[allow(dead_code)]
    mod test_support {
        use crate as claudix;

        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/common/test_support.rs"
        ));
    }

    use fixture::TestFixture;
    use test_support::{stub_config, write_fixture_config};

    #[test]
    fn get_info_advertises_server_info() {
        let info =
            ClaudixServer::new(PathBuf::from("."), Arc::new(ProviderCache::new())).get_info();

        assert_eq!(info.server_info.name, env!("CARGO_PKG_NAME"));
        assert_eq!(info.server_info.version, env!("CARGO_PKG_VERSION"));
        assert!(info.capabilities.tools.is_some());
    }

    #[test]
    fn list_tools_returns_documented_tool_names_in_order() {
        let tools = tool_catalog().unwrap_or_default();
        let names = tools
            .iter()
            .map(|tool| tool.name.as_ref())
            .collect::<Vec<_>>();

        assert_eq!(names, vec!["search_code", "reindex", "find_duplicates"]);
    }

    /// Every served tool must have a handler the router can dispatch to, or the
    /// catalog advertises a call that fails at runtime.
    #[test]
    fn every_served_tool_has_a_handler() {
        let served = tool_catalog().unwrap_or_default();
        let routed = ClaudixServer::tool_router();

        for tool in &served {
            assert!(
                routed.has_route(tool.name.as_ref()),
                "{} is served but has no handler",
                tool.name
            );
        }
    }

    #[test]
    fn error_result_embeds_recovery_hint_with_is_error() {
        let result = error_result(ClaudixError::PathTraversal {
            path: "../escape.rs".into(),
            recovery: RecoveryHint("Use a path inside $CLAUDE_PROJECT_DIR"),
        });

        assert_eq!(result.is_error, Some(true));
        let text = result
            .content
            .iter()
            .filter_map(|content| content.as_text().map(|text| text.text.as_str()))
            .collect::<String>();
        assert!(text.contains("Use a path inside $CLAUDE_PROJECT_DIR"));
        assert!(text.contains("Recovery:"));
    }

    fn degraded_output() -> cli::SearchOutput {
        cli::SearchOutput {
            groups: Vec::new(),
            repo_errors: Vec::new(),
            stale_hint: None,
            degraded_hint: Some(crate::prompts::mcp::ENDPOINT_DOWN_NOTE),
        }
    }

    #[test]
    fn degraded_notice_surfaces_once_then_stops() {
        let warned = AtomicBool::new(false);

        let mut first = degraded_output();
        throttle_degraded_notice(&mut first, &warned);
        assert!(
            first.degraded_hint.is_some(),
            "first degraded search must carry the notice"
        );

        let mut second = degraded_output();
        throttle_degraded_notice(&mut second, &warned);
        assert!(
            second.degraded_hint.is_none(),
            "a repeat degraded search must not re-bill the notice"
        );
    }

    /// A healthy search must not latch the one-shot: the first degraded search
    /// after any number of healthy ones still gets the notice.
    #[test]
    fn healthy_search_does_not_consume_the_degraded_one_shot() {
        let warned = AtomicBool::new(false);

        let mut healthy = cli::SearchOutput {
            groups: Vec::new(),
            repo_errors: Vec::new(),
            stale_hint: None,
            degraded_hint: None,
        };
        throttle_degraded_notice(&mut healthy, &warned);

        let mut degraded = degraded_output();
        throttle_degraded_notice(&mut degraded, &warned);
        assert!(
            degraded.degraded_hint.is_some(),
            "a healthy search must not latch the degraded one-shot"
        );
    }

    #[tokio::test]
    async fn reindex_rejects_blank_path_rather_than_sweeping_the_project() {
        let server = ClaudixServer::new(PathBuf::from("."), Arc::new(ProviderCache::new()));
        let outcome = server
            .reindex(Parameters(ReindexRequest {
                path: Some("   ".to_owned()),
                force: false,
            }))
            .await;
        assert!(outcome.is_ok());
        let result = outcome.unwrap_or_else(|_| CallToolResult::success(vec![]));

        assert_eq!(result.is_error, Some(true));
        let text = result
            .content
            .iter()
            .filter_map(|content| content.as_text().map(|text| text.text.as_str()))
            .collect::<String>();
        assert!(text.contains("Pass a non-empty path to reindex"));
    }

    #[test]
    fn reindex_request_defaults_to_a_whole_project_sweep() -> serde_json::Result<()> {
        let request: ReindexRequest = serde_json::from_str("{}")?;
        assert_eq!(request.path, None);
        assert!(!request.force);
        Ok(())
    }

    #[test]
    fn reindex_request_parses_force_and_path() -> serde_json::Result<()> {
        let forced: ReindexRequest = serde_json::from_str(r#"{"force":true}"#)?;
        assert!(forced.force);
        assert_eq!(forced.path, None);

        let single: ReindexRequest = serde_json::from_str(r#"{"path":"src/lib.rs"}"#)?;
        assert_eq!(single.path.as_deref(), Some("src/lib.rs"));
        assert!(!single.force);
        Ok(())
    }

    /// `force` wipes the whole index. Reaching it on a single-file reindex would
    /// destroy every other file's chunks to re-embed one, so the `path` branch
    /// must return before `force` is read. Hoisting the force check above the
    /// path branch leaves every other test green, hence this one: a real index,
    /// a `path` + `force: true` call, and the other file's chunks still there.
    #[tokio::test]
    async fn reindex_with_path_and_force_does_not_wipe_the_index() {
        let fixture = TestFixture::new("small_rust");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());
        let config = stub_config();
        assert!(write_fixture_config(fixture.root(), &config).is_ok());

        let indexed = cli::run_index(fixture.root(), false).await;
        assert!(indexed.is_ok(), "fixture index failed: {indexed:?}");
        let before = indexed.ok().unwrap_or_else(|| unreachable!());
        assert!(before.chunk_count > 0, "fixture must index something");

        let server =
            ClaudixServer::new(fixture.root().to_path_buf(), Arc::new(ProviderCache::new()));
        let outcome = server
            .reindex(Parameters(ReindexRequest {
                path: Some("src/math.rs".to_owned()),
                force: true,
            }))
            .await;
        assert!(outcome.is_ok());
        let result = outcome.unwrap_or_else(|_| CallToolResult::success(vec![]));
        assert_ne!(
            result.is_error,
            Some(true),
            "single-file reindex errored: {result:?}"
        );

        let after = cli::run_status(fixture.root()).await;
        assert!(after.is_ok());
        let after = after.ok().unwrap_or_else(|| unreachable!());
        assert_eq!(
            after.chunk_count, before.chunk_count,
            "force wiped the index during a single-file reindex"
        );
    }
}
