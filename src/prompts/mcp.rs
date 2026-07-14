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

/// The served catalog. `cross_repo` advertises the `repos` parameter on the
/// tools that accept it; the handlers honor `repos` either way, so a session
/// that gains cross-repos mid-flight degrades to a working-but-unadvertised
/// parameter rather than a broken call.
pub fn tool_definitions(cross_repo: bool) -> Vec<Value> {
    vec![
        search_code(cross_repo),
        reindex(),
        overview(),
        find_duplicates(cross_repo),
    ]
}

fn search_code(cross_repo: bool) -> Value {
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
    if cross_repo {
        insert_repos_property(
            &mut properties,
            "Absolute paths to other indexed repos to search read-only, added to the active project.",
        );
    }
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

fn overview() -> Value {
    json!({
        "name": "overview",
        "description": "Per-directory map of the indexed repo: file and chunk counts, languages, and the most frequent identifiers. Use when: orienting in an unfamiliar codebase or deciding where to start, before reaching for search_code.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path_prefix": { "type": "string", "description": "Restrict output to files under this project-relative path prefix, e.g. \"src/hooks\"" }
            }
        }
    })
}

fn find_duplicates(cross_repo: bool) -> Value {
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
    if cross_repo {
        insert_repos_property(
            &mut properties,
            "Absolute paths to other indexed repos to scan read-only, added to the active project.",
        );
    }
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

    fn names(cross_repo: bool) -> Vec<String> {
        tool_definitions(cross_repo)
            .iter()
            .filter_map(|tool| tool.get("name")?.as_str().map(str::to_owned))
            .collect()
    }

    fn has_repos_param(tool: &Value) -> bool {
        tool.pointer("/inputSchema/properties/repos").is_some()
    }

    #[test]
    fn catalog_is_the_same_four_tools_regardless_of_cross_repo() {
        let expected = vec!["search_code", "reindex", "overview", "find_duplicates"];
        assert_eq!(names(false), expected);
        assert_eq!(names(true), expected);
    }

    #[test]
    fn repos_param_is_advertised_only_when_cross_repo_is_configured() {
        for tool in tool_definitions(false) {
            assert!(
                !has_repos_param(&tool),
                "{} advertised repos without cross-repos configured",
                tool["name"]
            );
        }

        let with_repos: Vec<String> = tool_definitions(true)
            .iter()
            .filter(|tool| has_repos_param(tool))
            .filter_map(|tool| tool.get("name")?.as_str().map(str::to_owned))
            .collect();
        assert_eq!(with_repos, vec!["search_code", "find_duplicates"]);
    }

    /// The description restating the schema is exactly the duplication this
    /// module exists to avoid, and it silently rots when an argument changes.
    #[test]
    fn descriptions_never_restate_args_or_returns() {
        for tool in tool_definitions(true) {
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
        for tool in tool_definitions(true) {
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
