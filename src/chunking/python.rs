use tree_sitter::Node;

use crate::error::Result;
use crate::types::{Chunk, ChunkKind, FileHash, Language, RelativePath};

use super::chunk_with_grammar;

pub(super) fn chunk(path: &RelativePath, file_hash: FileHash, content: &str) -> Result<Vec<Chunk>> {
    chunk_with_grammar(
        tree_sitter_python::LANGUAGE.into(),
        "Python",
        path,
        Language::Python,
        file_hash,
        content,
        python_chunk_kind,
    )
}

fn python_chunk_kind(node: Node<'_>) -> Option<ChunkKind> {
    match node.kind() {
        "function_definition" => Some(ChunkKind::Function),
        "decorated_definition" => {
            // Only emit for decorated functions; decorated classes are handled
            // when recursion reaches the inner `class_definition`.
            let has_function = node
                .named_children(&mut node.walk())
                .any(|child| child.kind() == "function_definition");
            has_function.then_some(ChunkKind::Function)
        }
        "class_definition" => {
            // Skip if the parent is already a decorated_definition we emitted.
            let parent_is_decorated = node
                .parent()
                .is_some_and(|p| p.kind() == "decorated_definition");
            (!parent_is_decorated).then_some(ChunkKind::Class)
        }
        _ => None,
    }
}
