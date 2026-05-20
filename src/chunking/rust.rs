use tree_sitter::Node;

use crate::error::Result;
use crate::types::{Chunk, ChunkKind, FileHash, Language, RelativePath};

use super::chunk_with_grammar;

pub(super) fn chunk(path: &RelativePath, file_hash: FileHash, content: &str) -> Result<Vec<Chunk>> {
    chunk_with_grammar(
        tree_sitter_rust::LANGUAGE.into(),
        "Rust",
        path,
        Language::Rust,
        file_hash,
        content,
        rust_chunk_kind,
    )
}

fn rust_chunk_kind(node: Node<'_>) -> Option<ChunkKind> {
    match node.kind() {
        "function_item" => Some(function_kind(node)),
        "function_signature_item" => Some(ChunkKind::Method),
        "struct_item" => Some(ChunkKind::Struct),
        "enum_item" => Some(ChunkKind::Enum),
        "trait_item" => Some(ChunkKind::Trait),
        "impl_item" => Some(ChunkKind::Impl),
        "mod_item" => {
            // `mod foo;` has no body — only index inline `mod foo { ... }` blocks.
            let has_body = node
                .named_children(&mut node.walk())
                .any(|child| child.kind() == "declaration_list");
            has_body.then_some(ChunkKind::Module)
        }
        "macro_definition" => Some(ChunkKind::Macro),
        _ => None,
    }
}

fn function_kind(node: Node<'_>) -> ChunkKind {
    let mut current = node.parent();
    while let Some(parent) = current {
        if parent.kind() == "impl_item" {
            return ChunkKind::Method;
        }
        current = parent.parent();
    }

    ChunkKind::Function
}

/// Extend a node's start byte upward across `///` or `//!` doc-comment lines
/// so the embedded chunk includes the function/struct documentation. Called
/// from `build_chunk` only for `Language::Rust`.
pub(super) fn extend_start_for_rust_docs(content: &str, node_start: usize) -> usize {
    let mut start = node_start;

    loop {
        let current_line_start = super::line_start(content, start);
        if current_line_start == 0 {
            return start;
        }

        let previous_line_end = current_line_start.saturating_sub(1);
        let previous_line_start = super::line_start(content, previous_line_end);
        let previous_line =
            &content[previous_line_start..super::line_end(content, previous_line_start)];

        if is_rust_doc_comment(previous_line) {
            start = previous_line_start;
            continue;
        }

        return start;
    }
}

fn is_rust_doc_comment(line: &str) -> bool {
    let trimmed = line.trim_start();
    trimmed.starts_with("///") || trimmed.starts_with("//!")
}
