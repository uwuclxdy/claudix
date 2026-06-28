use tree_sitter::Node;

use crate::error::Result;
use crate::types::{Chunk, ChunkKind, FileHash, Language, RelativePath};

use super::chunk_with_grammar_named;

pub(super) fn chunk(path: &RelativePath, file_hash: FileHash, content: &str) -> Result<Vec<Chunk>> {
    chunk_with_grammar_named(
        tree_sitter_sequel::LANGUAGE.into(),
        "SQL",
        path,
        Language::Sql,
        file_hash,
        content,
        sql_chunk_kind,
        sql_name,
    )
}

fn sql_chunk_kind(node: Node<'_>) -> Option<ChunkKind> {
    match node.kind() {
        "create_table" => Some(ChunkKind::Table),
        "create_view" | "create_materialized_view" => Some(ChunkKind::View),
        "create_function" => Some(ChunkKind::Function),
        "create_trigger" => Some(ChunkKind::Trigger),
        _ => None,
    }
}

/// SQL `CREATE` nodes don't expose a `name` field. For table/view/function/
/// trigger the object name is the first `object_reference` child (kept whole so
/// a qualified `schema.table` survives); for a trigger that's its own name, not
/// the table it fires on (which is a later `object_reference`).
fn sql_name(node: Node<'_>, content: &str) -> Option<String> {
    let mut cursor = node.walk();
    let name_node = node
        .named_children(&mut cursor)
        .find(|child| child.kind() == "object_reference")?;

    name_node
        .utf8_text(content.as_bytes())
        .ok()
        .map(|text| text.trim().to_owned())
}
