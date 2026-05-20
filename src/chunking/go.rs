use tree_sitter::Node;

use crate::error::Result;
use crate::types::{Chunk, ChunkKind, FileHash, Language, RelativePath};

use super::chunk_with_grammar;

pub(super) fn chunk(path: &RelativePath, file_hash: FileHash, content: &str) -> Result<Vec<Chunk>> {
    chunk_with_grammar(
        tree_sitter_go::LANGUAGE.into(),
        "Go",
        path,
        Language::Go,
        file_hash,
        content,
        go_chunk_kind,
    )
}

fn go_chunk_kind(node: Node<'_>) -> Option<ChunkKind> {
    match node.kind() {
        "function_declaration" => Some(ChunkKind::Function),
        "method_declaration" => Some(ChunkKind::Method),
        "type_declaration" => {
            let has_struct = node.named_children(&mut node.walk()).any(|child| {
                child.kind() == "type_spec" && type_spec_contains(child, "struct_type")
            });
            let has_interface = node.named_children(&mut node.walk()).any(|child| {
                child.kind() == "type_spec" && type_spec_contains(child, "interface_type")
            });

            if has_struct {
                Some(ChunkKind::Struct)
            } else if has_interface {
                Some(ChunkKind::Interface)
            } else {
                None
            }
        }
        _ => None,
    }
}

fn type_spec_contains(type_spec: Node<'_>, target_kind: &str) -> bool {
    type_spec
        .named_children(&mut type_spec.walk())
        .any(|child| child.kind() == target_kind)
}
