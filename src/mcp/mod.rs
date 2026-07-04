use std::borrow::Cow;
use std::path::{Path, PathBuf};
use std::sync::Arc;

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
use crate::error::{ClaudixError, RecoveryHint};
use crate::prompts::hints;

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
    #[serde(default)]
    force: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, JsonSchema, Default)]
struct OverviewRequest {
    #[serde(default)]
    path_prefix: Option<String>,
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

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, JsonSchema)]
struct ReindexFileRequest {
    path: String,
}

/// The claudix MCP server. Holds the active project root and the generated tool
/// router; each `#[tool]` handler validates then delegates to the same
/// `cli::run_*` function the CLI uses.
#[derive(Clone)]
pub struct ClaudixServer {
    project_root: PathBuf,
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl ClaudixServer {
    fn new(project_root: PathBuf) -> Self {
        Self {
            project_root,
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
        let outcome = cli::run_search(
            &self.project_root,
            request.query,
            request.top_k.map(|value| value as usize),
            request.language_filter,
            request.path_prefix,
            request.repos,
        )
        .await
        .and_then(to_value);
        Ok(into_result(outcome))
    }

    /// Chunk count, file count, model, and staleness for the active index.
    #[tool(name = "get_index_status")]
    async fn get_index_status(&self) -> ToolOutcome {
        let outcome = cli::run_status(&self.project_root).await.and_then(to_value);
        Ok(into_result(outcome))
    }

    /// Rebuild the active project index; `force` wipes before rebuilding.
    #[tool(name = "reindex")]
    async fn reindex(&self, Parameters(request): Parameters<ReindexRequest>) -> ToolOutcome {
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

    /// Delete all stored chunks and the manifest for the active project.
    #[tool(name = "clear_index")]
    async fn clear_index(&self) -> ToolOutcome {
        let outcome = cli::run_clear_index(&self.project_root)
            .await
            .and_then(to_value);
        Ok(into_result(outcome))
    }

    /// Re-embed one file in the active project without touching other chunks.
    #[tool(name = "reindex_file")]
    async fn reindex_file(
        &self,
        Parameters(request): Parameters<ReindexFileRequest>,
    ) -> ToolOutcome {
        if request.path.trim().is_empty() {
            return Ok(error_result(ClaudixError::ConfigInvalid {
                message: "path cannot be empty".to_owned(),
                recovery: RecoveryHint(hints::PATH_NON_EMPTY),
            }));
        }
        let outcome = cli::run_reindex_file(&self.project_root, Path::new(&request.path))
            .await
            .and_then(to_value);
        Ok(into_result(outcome))
    }

    /// Per-directory map of the indexed repo.
    #[tool(name = "overview")]
    async fn overview(&self, Parameters(request): Parameters<OverviewRequest>) -> ToolOutcome {
        let outcome = cli::run_overview(&self.project_root, request.path_prefix)
            .await
            .and_then(to_value);
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
    let server = ClaudixServer::new(project_root.as_ref().to_path_buf());
    // The service driver is spawned with `spawn_local` under the `local` feature,
    // so it must run inside a `LocalSet`.
    let local = tokio::task::LocalSet::new();
    local
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
        .await
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

    #[test]
    fn get_info_advertises_server_info() {
        let info = ClaudixServer::new(PathBuf::from(".")).get_info();

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

        assert_eq!(
            names,
            vec![
                "search_code",
                "get_index_status",
                "reindex",
                "clear_index",
                "reindex_file",
                "overview",
                "find_duplicates",
            ]
        );
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

    #[tokio::test]
    async fn reindex_file_rejects_empty_path() {
        let server = ClaudixServer::new(PathBuf::from("."));
        let outcome = server
            .reindex_file(Parameters(ReindexFileRequest {
                path: "   ".to_owned(),
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
        assert!(text.contains("Pass a non-empty path to reindex_file"));
    }

    #[test]
    fn reindex_request_defaults_force_to_false() -> serde_json::Result<()> {
        let request: ReindexRequest = serde_json::from_str("{}")?;
        assert!(!request.force);
        Ok(())
    }

    #[test]
    fn reindex_request_parses_force_true() -> serde_json::Result<()> {
        let request: ReindexRequest = serde_json::from_str(r#"{"force":true}"#)?;
        assert!(request.force);
        Ok(())
    }
}
