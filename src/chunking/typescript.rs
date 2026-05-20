use tree_sitter::Node;

use crate::error::Result;
use crate::types::{Chunk, ChunkKind, FileHash, Language, RelativePath};

use super::chunk_with_grammar;

pub(super) fn chunk(
    path: &RelativePath,
    language: Language,
    file_hash: FileHash,
    content: &str,
) -> Result<Vec<Chunk>> {
    chunk_with_grammar(
        tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
        "TypeScript",
        path,
        language,
        file_hash,
        content,
        typescript_chunk_kind,
    )
}

fn typescript_chunk_kind(node: Node<'_>) -> Option<ChunkKind> {
    match node.kind() {
        "function_declaration" | "function_expression" => Some(ChunkKind::Function),
        "arrow_function" => {
            // Emit only when directly inside a variable_declarator so we don't
            // produce a chunk for every inline arrow.
            let parent_is_declarator = node
                .parent()
                .is_some_and(|p| p.kind() == "variable_declarator");
            parent_is_declarator.then_some(ChunkKind::Function)
        }
        "method_definition" => Some(ChunkKind::Method),
        "class_declaration" | "class_expression" => Some(ChunkKind::Class),
        "interface_declaration" => Some(ChunkKind::Interface),
        _ => None,
    }
}
