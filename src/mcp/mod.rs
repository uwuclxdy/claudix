use std::path::Path;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use tokio::io::{self, AsyncBufReadExt, AsyncWriteExt, BufReader};

use crate::cli;
use crate::error::{ClaudixError, RecoveryHint, Result};

const JSONRPC_VERSION: &str = "2.0";
const PROTOCOL_VERSION: &str = "2024-11-05";

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
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

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Default)]
struct ReindexRequest {
    #[serde(default)]
    force: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Default)]
struct OverviewRequest {
    #[serde(default)]
    path_prefix: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Default)]
struct FindDuplicatesRequest {
    #[serde(default)]
    min_similarity: Option<f32>,
    #[serde(default)]
    limit: Option<u32>,
    #[serde(default)]
    repos: Option<Vec<String>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct ReindexFileRequest {
    path: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct JsonRpcRequest {
    jsonrpc: String,
    #[serde(default)]
    id: Option<Value>,
    method: String,
    #[serde(default)]
    params: Value,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct CallToolRequest {
    name: String,
    #[serde(default)]
    arguments: Value,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct TextContent {
    #[serde(rename = "type")]
    kind: &'static str,
    text: String,
}

pub async fn run(project_root: impl AsRef<Path>) -> Result<()> {
    let project_root = project_root.as_ref().to_path_buf();
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut lines = BufReader::new(stdin).lines();
    let mut writer = stdout;

    while let Some(line) = lines.next_line().await? {
        if line.trim().is_empty() {
            continue;
        }

        if let Some(response) = handle_line(&project_root, &line).await? {
            write_message(&mut writer, &response).await?;
        }
    }

    Ok(())
}

async fn handle_line(project_root: &Path, line: &str) -> Result<Option<Value>> {
    let request: JsonRpcRequest = match serde_json::from_str(line) {
        Ok(request) => request,
        Err(error) => {
            return Ok(Some(error_response(
                None,
                -32700,
                format!("parse error: {error}"),
                None,
            )));
        }
    };

    // Notifications have no id and must receive no response.
    if request.id.is_none() {
        return Ok(None);
    }

    if request.jsonrpc != JSONRPC_VERSION {
        return Ok(Some(error_response(
            request.id,
            -32600,
            "jsonrpc must be 2.0".to_owned(),
            None,
        )));
    }

    let id = request.id.clone();
    let response = match request.method.as_str() {
        "initialize" => success_response(id, initialize_result(&request.params)),
        "tools/list" => success_response(id, tools_list_result()),
        "tools/call" => handle_tools_call(project_root, id, request.params).await?,
        _ => error_response(
            id,
            -32601,
            format!("method not found: {}", request.method),
            None,
        ),
    };

    Ok(Some(response))
}

async fn handle_tools_call(project_root: &Path, id: Option<Value>, params: Value) -> Result<Value> {
    let call: CallToolRequest = match serde_json::from_value(params) {
        Ok(call) => call,
        Err(error) => {
            return Ok(error_response(
                id,
                -32602,
                format!("invalid tools/call params: {error}"),
                None,
            ));
        }
    };

    let tool_result = match call.name.as_str() {
        "search_code" => match search_code(project_root, call.arguments).await {
            Ok(result) => tool_success_result(result)?,
            Err(error) => tool_error_result(error),
        },
        "get_index_status" => match get_index_status(project_root).await {
            Ok(result) => tool_success_result(result)?,
            Err(error) => tool_error_result(error),
        },
        "reindex" => match reindex(project_root, call.arguments).await {
            Ok(result) => tool_success_result(result)?,
            Err(error) => tool_error_result(error),
        },
        "clear_index" => match clear_index(project_root).await {
            Ok(result) => tool_success_result(result)?,
            Err(error) => tool_error_result(error),
        },
        "reindex_file" => match reindex_file(project_root, call.arguments).await {
            Ok(result) => tool_success_result(result)?,
            Err(error) => tool_error_result(error),
        },
        "overview" => match overview(project_root, call.arguments).await {
            Ok(result) => tool_success_result(result)?,
            Err(error) => tool_error_result(error),
        },
        "find_duplicates" => match find_duplicates(project_root, call.arguments).await {
            Ok(result) => tool_success_result(result)?,
            Err(error) => tool_error_result(error),
        },
        _ => {
            return Ok(error_response(
                id,
                -32602,
                format!("unknown tool: {}", call.name),
                None,
            ));
        }
    };

    Ok(success_response(id, tool_result))
}

fn to_value<T: Serialize>(value: T) -> Result<Value> {
    serde_json::to_value(value).map_err(ClaudixError::from)
}

async fn search_code(project_root: &Path, arguments: Value) -> Result<Value> {
    let request: SearchCodeRequest = parse_tool_arguments(
        arguments,
        "search_code",
        "Pass query plus optional top_k, language_filter, path_prefix, and repos",
    )?;
    if request.query.trim().is_empty() {
        return Err(ClaudixError::ConfigInvalid {
            message: "query cannot be empty".to_owned(),
            recovery: RecoveryHint("Pass a non-empty query string to search_code"),
        });
    }
    let output = cli::run_search(
        project_root,
        request.query,
        request.top_k.map(|value| value as usize),
        request.language_filter,
        request.path_prefix,
        request.repos,
    )
    .await?;
    to_value(output)
}

async fn get_index_status(project_root: &Path) -> Result<Value> {
    let output = cli::run_status(project_root).await?;
    to_value(output)
}

async fn reindex(project_root: &Path, arguments: Value) -> Result<Value> {
    let request: ReindexRequest =
        parse_tool_arguments(arguments, "reindex", "Pass force as an optional boolean")?;
    if request.force {
        cli::run_clear_index(project_root).await?;
    }
    let output = cli::run_index(project_root, false).await?;
    to_value(output)
}

async fn clear_index(project_root: &Path) -> Result<Value> {
    let output = cli::run_clear_index(project_root).await?;
    to_value(output)
}

async fn reindex_file(project_root: &Path, arguments: Value) -> Result<Value> {
    let request: ReindexFileRequest = parse_tool_arguments(
        arguments,
        "reindex_file",
        "Pass path for a file inside $CLAUDE_PROJECT_DIR",
    )?;
    if request.path.trim().is_empty() {
        return Err(ClaudixError::ConfigInvalid {
            message: "path cannot be empty".to_owned(),
            recovery: RecoveryHint("Pass a non-empty path to reindex_file"),
        });
    }
    let output = cli::run_reindex_file(project_root, Path::new(&request.path)).await?;
    to_value(output)
}

async fn overview(project_root: &Path, arguments: Value) -> Result<Value> {
    let request: OverviewRequest =
        parse_tool_arguments(arguments, "overview", "Pass an optional path_prefix string")?;
    let output = cli::run_overview(project_root, request.path_prefix).await?;
    to_value(output)
}

async fn find_duplicates(project_root: &Path, arguments: Value) -> Result<Value> {
    let request: FindDuplicatesRequest = parse_tool_arguments(
        arguments,
        "find_duplicates",
        "Pass optional min_similarity (number), limit (integer), repos (array of strings)",
    )?;
    let output = cli::run_find_duplicates(
        project_root,
        request.min_similarity,
        request.limit.map(|n| n as usize),
        request.repos,
    )
    .await?;
    to_value(output)
}

fn parse_tool_arguments<T>(
    arguments: Value,
    tool_name: &'static str,
    recovery: &'static str,
) -> Result<T>
where
    T: DeserializeOwned,
{
    let arguments = match arguments {
        Value::Null => Value::Object(Map::new()),
        value => value,
    };

    serde_json::from_value(arguments).map_err(|error| ClaudixError::ConfigInvalid {
        message: format!("invalid arguments for {tool_name}: {error}"),
        recovery: RecoveryHint(recovery),
    })
}

fn initialize_result(_params: &Value) -> Value {
    json!({
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": {
            "tools": {}
        },
        "serverInfo": {
            "name": env!("CARGO_PKG_NAME"),
            "version": env!("CARGO_PKG_VERSION")
        }
    })
}

fn tools_list_result() -> Value {
    json!({
        "tools": tool_definitions()
    })
}

fn tool_definitions() -> Vec<Value> {
    vec![
        json!({
            "name": "search_code",
            "description": "Semantic code search over the active project (plus optional cross-repos). Use when: looking for code by meaning ('where is auth handled?', 'how does config load?'), looking up an identifier ('handle_session_start'), exploring an unfamiliar area, or checking whether logic ALREADY EXISTS before implementing something new. Prefer over Grep for anything that isn't a literal string or regex match. Args: query (required), optional top_k, language_filter, path_prefix, repos (absolute paths to additional indexed repos; the active project is always included and these are added to it). Returns: hits grouped by directory ordered by best-hit score, each with repo, file path, line range, kind, name, snippet, and score; cross-repo runs also include repo_errors for unindexed or model-mismatched repos.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Natural-language description or identifier name. Multiple words work best." },
                    "top_k": { "type": "integer", "minimum": 1, "description": "Maximum results to return (default: from config, usually 5-10)" },
                    "language_filter": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Restrict to specific languages, e.g. [\"rust\"], [\"python\", \"javascript\"]"
                    },
                    "path_prefix": { "type": "string", "description": "Restrict to files under this project-relative path prefix, e.g. \"src/hooks\"" },
                    "repos": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Absolute paths to other already-indexed repos to search read-only. The active project is always included; these are added to it. Repos that are unindexed or use a different model surface in repo_errors."
                    }
                },
                "required": ["query"]
            }
        }),
        json!({
            "name": "get_index_status",
            "description": "Report current chunk count, file count, embedding model, and staleness for the active index. Use when: checking whether the index is fresh before relying on search results, diagnosing why search returns nothing, or confirming a reindex completed. Returns: file_count, chunk_count, model, stale (true means files changed since last index), and index_present.",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        }),
        json!({
            "name": "reindex",
            "description": "Rebuild the active project index by scanning all files and re-embedding changed chunks. Use when: get_index_status reports stale, after large file additions or deletions, or when search results look wrong. Pass force: true to wipe the existing index first; this is required after changing the embedding model. Don't use for: a single changed file (use reindex_file instead). Returns: file_count and chunk_count.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "force": { "type": "boolean", "description": "When true, wipes the existing index before rebuilding (required after embedding model change)" }
                }
            }
        }),
        json!({
            "name": "clear_index",
            "description": "Delete all stored chunks and the manifest for the active project. Use when: the index is corrupted, switching embedding models (clear then reindex with force: true via reindex), or resetting a test environment. Does not delete source files. After clearing, call reindex to rebuild.",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        }),
        json!({
            "name": "reindex_file",
            "description": "Re-embed one file in the active project without touching other chunks. Use when: you just edited a file and want search to reflect the change immediately, without waiting for a full reindex. Don't use for: bulk updates (use reindex) or files outside the active project root. Args: path (project-relative or absolute path inside the project root). Returns: chunks written for that file.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Project-relative or absolute path within the project root" }
                },
                "required": ["path"]
            }
        }),
        json!({
            "name": "overview",
            "description": "Map of the indexed repo: for each directory, file count, chunk count, languages, and the top identifiers by frequency. Use when: orienting in an unfamiliar codebase, deciding where to start a task, getting a structural sense of what lives where before diving into search_code. Optional path_prefix narrows the map to a subtree. Returns: per-directory rollups (directory path, file_count, chunk_count, languages, top_identifiers) sorted by path, plus repo-wide totals.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path_prefix": { "type": "string", "description": "Restrict output to files under this project-relative path prefix, e.g. \"src/hooks\"" }
                }
            }
        }),
        json!({
            "name": "find_duplicates",
            "description": "Find near-identical code chunks across files using stored embeddings. Use when: BEFORE adding new logic, checking whether equivalent code already exists; auditing for copy-paste within this repo or across an explicit list of already-indexed repos. Args: optional min_similarity (0-1, default 0.85; raise toward 1.0 for exact copies, lower for looser matches), limit (default 50), repos (absolute paths; when set, ONLY these repos are scanned and the active project is NOT auto-added, unlike search_code which always includes it). Returns: pairs sorted by similarity descending, each with repo, file path, line range, and name for both sides; plus repo_errors for any listed repo that is unindexed or model-mismatched.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "min_similarity": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Cosine similarity floor (default 0.85). Higher = stricter / fewer pairs."
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Maximum number of pairs to return (default 50)"
                    },
                    "repos": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Absolute paths to other already-indexed repos to include. When specified, ONLY these paths are scanned; the active project is not auto-added."
                    }
                }
            }
        }),
    ]
}

fn success_response(id: Option<Value>, result: Value) -> Value {
    json!({
        "jsonrpc": JSONRPC_VERSION,
        "id": id.unwrap_or(Value::Null),
        "result": result,
    })
}

fn error_response(id: Option<Value>, code: i64, message: String, data: Option<Value>) -> Value {
    let mut error = Map::new();
    error.insert("code".to_owned(), Value::Number(code.into()));
    error.insert("message".to_owned(), Value::String(message));

    if let Some(data) = data {
        error.insert("data".to_owned(), data);
    }

    Value::Object(Map::from_iter([
        (
            "jsonrpc".to_owned(),
            Value::String(JSONRPC_VERSION.to_owned()),
        ),
        ("id".to_owned(), id.unwrap_or(Value::Null)),
        ("error".to_owned(), Value::Object(error)),
    ]))
}

fn tool_success_result(payload: Value) -> Result<Value> {
    let text = serde_json::to_string(&payload).map_err(ClaudixError::from)?;

    Ok(json!({
        "content": [TextContent { kind: "text", text }],
        "structuredContent": payload,
    }))
}

fn tool_error_result(error: ClaudixError) -> Value {
    let message = error.to_string();
    let recovery = error.recovery_hint();
    let text = match recovery {
        Some(recovery) => format!("{message}. Recovery: {recovery}"),
        None => message.clone(),
    };

    let mut structured = Map::new();
    structured.insert("error".to_owned(), Value::String(message));
    if let Some(recovery) = recovery {
        structured.insert("recovery".to_owned(), Value::String(recovery.to_owned()));
    }

    json!({
        "content": [TextContent { kind: "text", text }],
        "structuredContent": Value::Object(structured),
        "isError": true,
    })
}

async fn write_message(writer: &mut io::Stdout, response: &Value) -> Result<()> {
    let encoded = serde_json::to_vec(response).map_err(ClaudixError::from)?;
    writer.write_all(&encoded).await?;
    writer.write_all(b"\n").await?;
    writer.flush().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initialize_result_advertises_server_info() {
        let result = initialize_result(&Value::Null);

        assert_eq!(result["protocolVersion"], PROTOCOL_VERSION);
        assert_eq!(result["serverInfo"]["name"], env!("CARGO_PKG_NAME"));
        assert_eq!(result["serverInfo"]["version"], env!("CARGO_PKG_VERSION"));
        assert!(result["capabilities"]["tools"].is_object());
    }

    #[test]
    fn tools_list_returns_documented_tool_names() {
        let tools = tools_list_result()["tools"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        let names = tools
            .iter()
            .filter_map(|tool| tool["name"].as_str())
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
    fn tool_error_result_includes_recovery_hint() {
        let result = tool_error_result(ClaudixError::PathTraversal {
            path: "../escape.rs".into(),
            recovery: RecoveryHint("Use a path inside $CLAUDE_PROJECT_DIR"),
        });

        assert_eq!(result["isError"], Value::Bool(true));
        assert_eq!(
            result["structuredContent"]["recovery"],
            Value::String("Use a path inside $CLAUDE_PROJECT_DIR".to_owned())
        );
    }

    #[tokio::test]
    async fn reindex_file_rejects_empty_path() {
        let response = handle_line(
            Path::new("."),
            r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"reindex_file","arguments":{"path":"   "}}}"#,
        )
        .await;

        assert!(response.is_ok());
        let response = response.ok().flatten().unwrap_or(Value::Null);
        assert_eq!(response["result"]["isError"], Value::Bool(true));
        assert_eq!(
            response["result"]["structuredContent"]["recovery"],
            Value::String("Pass a non-empty path to reindex_file".to_owned())
        );
    }

    #[tokio::test]
    async fn notifications_initialized_produces_no_response() {
        let response = handle_line(
            Path::new("."),
            r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
        )
        .await;

        assert!(response.is_ok());
        assert!(response.ok().flatten().is_none());
    }

    #[tokio::test]
    async fn unknown_tool_returns_protocol_error() {
        let response = handle_line(
            Path::new("."),
            r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"nope","arguments":{}}}"#,
        )
        .await;

        assert!(response.is_ok());
        let response = response.ok().flatten().unwrap_or(Value::Null);
        assert_eq!(response["error"]["code"], Value::Number((-32602).into()));
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
