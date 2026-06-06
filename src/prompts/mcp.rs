//! MCP tool definitions: descriptions and input schemas served on `tools/list`.

use serde_json::{Value, json};

pub fn tool_definitions() -> Vec<Value> {
    vec![
        json!({
            "name": "search_code",
            "description": "Semantic code search over the active project (plus optional cross-repos). Use when: looking for code by meaning ('where is auth handled?', 'how does config load?'), looking up an identifier ('handle_session_start'), exploring an unfamiliar area, or checking whether logic ALREADY EXISTS before implementing something new. Prefer over Grep for anything that isn't a literal string or regex match. Args: query (required), optional top_k, language_filter, path_prefix, repos (absolute paths to additional indexed repos; the active project is always included and these are added to it). Returns: hits grouped by directory ordered by best-hit score, each with repo, file path, line range, kind, name, snippet, and score; cross-repo runs also include repo_errors for unindexed or model-mismatched repos. Hits flagged stale changed on disk after indexing: treat line numbers as approximate and Read the file to confirm.",
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
            "description": "Report current chunk count, file count, embedding model, and staleness for the active index. Use when: checking whether the index is fresh before relying on search results, diagnosing why search returns nothing, or confirming a reindex completed. Returns: file_count, chunk_count, model, stale (true means files changed since last index; call reindex to refresh), and index_present.",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        }),
        json!({
            "name": "reindex",
            "description": "Rebuild the active project index by scanning all files and re-embedding changed chunks. Use when: get_index_status reports stale, after large file additions or deletions, or when search results look wrong. Pass force: true to wipe and rebuild from scratch in one call; required after changing the embedding model or on a schema mismatch. Runs synchronously inside this tool call; large repos take a while. Don't use for: a single changed file (use reindex_file instead). Returns: file_count and chunk_count.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "force": { "type": "boolean", "description": "When true, wipes the existing index before rebuilding (required after embedding model change)" }
                }
            }
        }),
        json!({
            "name": "clear_index",
            "description": "Delete all stored chunks and the manifest for the active project. Use when: the index is corrupted or you're resetting a test environment. For embedding model switches call reindex with force: true instead (clears and rebuilds in one call). Does not delete source files. After clearing, call reindex to rebuild.",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        }),
        json!({
            "name": "reindex_file",
            "description": "Re-embed one file in the active project without touching other chunks. Use when: a file changed outside Edit/Write (Bash, git checkout, codegen) and search should reflect it immediately. Edits made with Edit/Write are reindexed automatically by the hook, so you rarely need this for your own edits. Don't use for: bulk updates (use reindex) or files outside the active project root. Args: path (project-relative or absolute path inside the project root). Returns: chunks written for that file.",
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
