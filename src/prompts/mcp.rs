//! MCP tool definitions: descriptions and input schemas served on `tools/list`.
//!
//! Descriptions carry purpose, triggers, and semantics the schema cannot express
//! — never an argument list or a return-field enumeration. The client already
//! renders `inputSchema`, and one call reveals the payload; restating either is
//! duplicate tokens in every session and rots silently when a shape changes.
//! Conditional guidance rides the response instead (see [`STALE_HITS_NOTE`]).

use serde_json::{Value, json};

/// Carried on a `search_code` response only when a hit is stale, rather than in
/// the always-billed tool description.
pub const STALE_HITS_NOTE: &str = "A stale hit's file changed on disk after indexing: treat its line numbers as approximate and Read the file to confirm.";

/// Carried on a `search_code` response only when the embedding endpoint was
/// unreachable and the ranking fell back to lexical-only. Surfaced at most once
/// per MCP session so the notice does not bill every degraded search.
pub const ENDPOINT_DOWN_NOTE: &str = "The embedding endpoint is unreachable, so these results are lexical-only (keyword and identifier matching, no semantic ranking). Restart the endpoint for full semantic search.";

/// The served catalog. `repos` is always advertised on the tools that accept
/// it, whether or not cross-repos are configured: hiding it gated the argument
/// on config and left a subagent without cross-repos unable to select a target
/// repo from the schema (settled 2026-08-24: advertise always, no exclude
/// option).
pub fn tool_definitions() -> Vec<Value> {
    vec![search_code(), reindex(), find_duplicates()]
}

fn search_code() -> Value {
    let mut properties = json!({
        "query": { "type": "string", "description": "Natural-language description or identifier name. Multiple words work best." },
        "top_k": { "type": "integer", "minimum": 1, "description": "Maximum results to return (default 10)" },
        "language_filter": {
            "type": "array",
            "items": { "type": "string" },
            "description": "Restrict to specific languages, e.g. [\"rust\"], [\"python\", \"javascript\"]"
        },
        "path_prefix": { "type": "string", "description": "Restrict to files under this project-relative path prefix, e.g. \"src/hooks\"" }
    });
    insert_repos_property(
        &mut properties,
        "Absolute paths to other indexed repos to search read-only. The active project is always searched.",
    );
    json!({
        "name": "search_code",
        "description": "Semantic code search over the indexed project. Use when: finding code by meaning ('where is auth handled'), looking up an identifier, exploring an unfamiliar area, or checking whether logic already exists before writing it. Prefer over Grep unless you need an exact literal or regex. Returns hits grouped by directory, best score first.",
        "inputSchema": {
            "type": "object",
            "properties": properties,
            "required": ["query"]
        }
    })
}

fn reindex() -> Value {
    json!({
        "name": "reindex",
        "description": "Re-embed the project index. Omit path to sweep the whole project; pass path for a single file changed outside Edit/Write, such as by Bash, git checkout, or codegen — Edit and Write reindex themselves via the hook. Runs synchronously; a full sweep of a large repo takes a while.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Reindex only this file (project-relative or absolute inside the project root)" },
                "force": { "type": "boolean", "description": "Wipe the index before rebuilding. Ignored when path is set." }
            }
        }
    })
}

fn find_duplicates() -> Value {
    let mut properties = json!({
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
        }
    });
    insert_repos_property(
        &mut properties,
        "Absolute paths to other indexed repos to scan read-only. The active project is always scanned.",
    );
    json!({
        "name": "find_duplicates",
        "description": "Near-identical code chunks, found by comparing stored embeddings. Use when: checking whether equivalent code already exists before adding logic, or auditing copy-paste. Returns pairs by similarity, highest first.",
        "inputSchema": {
            "type": "object",
            "properties": properties
        }
    })
}

/// Both cross-repo-capable tools take the same `repos` shape; only the verb in
/// the description differs.
fn insert_repos_property(properties: &mut Value, description: &str) {
    if let Some(map) = properties.as_object_mut() {
        map.insert(
            "repos".to_owned(),
            json!({
                "type": "array",
                "items": { "type": "string" },
                "description": description
            }),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names() -> Vec<String> {
        tool_definitions()
            .iter()
            .filter_map(|tool| tool.get("name")?.as_str().map(str::to_owned))
            .collect()
    }

    fn has_repos_param(tool: &Value) -> bool {
        tool.pointer("/inputSchema/properties/repos").is_some()
    }

    fn repos_description(tool: &Value) -> Option<&str> {
        tool.pointer("/inputSchema/properties/repos/description")?
            .as_str()
    }

    #[test]
    fn catalog_serves_the_three_tools() {
        let expected = vec!["search_code", "reindex", "find_duplicates"];
        assert_eq!(names(), expected);
    }

    /// A subagent with no cross-repos configured still gets `repos`: it is the
    /// only way the served schema lets it select a target repo, so the argument
    /// must appear whether or not any config names repos.
    #[test]
    fn repos_param_is_always_advertised_on_cross_repo_tools() {
        let with_repos: Vec<String> = tool_definitions()
            .iter()
            .filter(|tool| has_repos_param(tool))
            .filter_map(|tool| tool.get("name")?.as_str().map(str::to_owned))
            .collect();
        assert_eq!(with_repos, vec!["search_code", "find_duplicates"]);
    }

    #[test]
    fn repos_description_states_the_active_project_is_always_searched_or_scanned() {
        let with_repos: Vec<Value> = tool_definitions()
            .into_iter()
            .filter(has_repos_param)
            .collect();
        assert_eq!(
            with_repos.len(),
            2,
            "both cross-repo tools must advertise repos"
        );

        for tool in with_repos {
            let description = repos_description(&tool).unwrap_or_default();
            let verb = if tool["name"] == "search_code" {
                "searched"
            } else {
                "scanned"
            };
            assert!(
                description.contains(&format!("The active project is always {verb}")),
                "{} repos description must state the active project is always {verb}: {description:?}",
                tool["name"]
            );
        }
    }

    /// The description restating the schema is exactly the duplication this
    /// module exists to avoid, and it silently rots when an argument changes.
    #[test]
    fn descriptions_never_restate_args_or_returns() {
        for tool in tool_definitions() {
            let description = tool["description"].as_str();
            assert!(
                description.is_some_and(|text| text.len() > 40),
                "{} has no usable description — the checks below would pass vacuously",
                tool["name"]
            );
            let description = description.unwrap_or_default();
            for banned in ["Args:", "Returns:", "Arguments:"] {
                assert!(
                    !description.contains(banned),
                    "{} description restates {banned}",
                    tool["name"]
                );
            }
            assert!(
                !description.contains('\n'),
                "{} description has a literal newline",
                tool["name"]
            );
        }
    }

    /// A `required` key with no matching property is a schema the client will
    /// reject or, worse, one the model can never satisfy.
    #[test]
    fn every_required_field_exists_in_properties() {
        let mut checked = 0;
        for tool in tool_definitions() {
            let schema = &tool["inputSchema"];
            assert!(
                schema["properties"].is_object(),
                "{} has no properties object",
                tool["name"]
            );
            for required in schema["required"].as_array().unwrap_or(&vec![]) {
                let key = required.as_str().unwrap_or_default();
                assert!(
                    schema.pointer(&format!("/properties/{key}")).is_some(),
                    "{} requires {key} but never declares it",
                    tool["name"]
                );
                checked += 1;
            }
        }
        // Only `search_code` has a required field; if that stops being true the
        // loop above silently stops checking anything.
        assert_eq!(checked, 1, "expected exactly one required field in total");
    }
}
