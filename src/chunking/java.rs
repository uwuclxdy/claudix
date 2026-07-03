use tree_sitter::Node;

use crate::error::Result;
use crate::types::{Chunk, ChunkKind, FileHash, Language, RelativePath};

use super::chunk_with_grammar;

pub(super) fn chunk(path: &RelativePath, file_hash: FileHash, content: &str) -> Result<Vec<Chunk>> {
    chunk_with_grammar(
        tree_sitter_java::LANGUAGE.into(),
        "Java",
        path,
        Language::Java,
        file_hash,
        content,
        java_chunk_kind,
    )
}

fn java_chunk_kind(node: Node<'_>) -> Option<ChunkKind> {
    match node.kind() {
        // Nested types reuse the same node kinds inside a class body, so the
        // recursive walk chunks inner classes/enums alongside their outer type.
        // A record is a class-like reference type; group them under Class.
        "class_declaration" | "record_declaration" => Some(ChunkKind::Class),
        // Annotation types (`@interface`) are interface-like.
        "interface_declaration" | "annotation_type_declaration" => Some(ChunkKind::Interface),
        "enum_declaration" => Some(ChunkKind::Enum),
        "method_declaration" | "constructor_declaration" => Some(ChunkKind::Method),
        _ => None,
    }
}
