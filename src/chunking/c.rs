use tree_sitter::Node;

use crate::error::Result;
use crate::types::{Chunk, ChunkKind, FileHash, Language, RelativePath};

use super::{chunk_with_grammar_named, default_name};

pub(super) fn chunk(path: &RelativePath, file_hash: FileHash, content: &str) -> Result<Vec<Chunk>> {
    chunk_with_grammar_named(
        tree_sitter_c::LANGUAGE.into(),
        "C",
        path,
        Language::C,
        file_hash,
        content,
        c_chunk_kind,
        c_name,
    )
}

fn c_chunk_kind(node: Node<'_>) -> Option<ChunkKind> {
    match node.kind() {
        "function_definition" => Some(ChunkKind::Function),
        // A `body` marks a real definition; without one this is a `struct Foo x;`
        // type reference that would only yield a useless one-line chunk.
        "struct_specifier" | "union_specifier" => has_body(node).then_some(ChunkKind::Struct),
        "enum_specifier" => has_body(node).then_some(ChunkKind::Enum),
        // Object- and function-like macros are first-class, searchable C symbols;
        // preprocessor-heavy headers are mostly these.
        "preproc_def" | "preproc_function_def" => Some(ChunkKind::Macro),
        _ => None,
    }
}

fn has_body(node: Node<'_>) -> bool {
    node.child_by_field_name("body").is_some()
}

/// C buries a function's name in a nested `declarator` chain rather than a
/// `name` field. Aggregates and macros do expose `name`; an anonymous aggregate
/// in `typedef struct { … } Foo;` borrows the alias name from its parent.
fn c_name(node: Node<'_>, content: &str) -> Option<String> {
    if node.kind() == "function_definition" {
        return node
            .child_by_field_name("declarator")
            .and_then(|declarator| declarator_name(declarator, content));
    }

    if let Some(name) = default_name(node, content) {
        return Some(name);
    }

    node.parent()
        .filter(|parent| parent.kind() == "type_definition")
        .and_then(|parent| parent.child_by_field_name("declarator"))
        .and_then(|declarator| declarator_name(declarator, content))
}

/// Walk the `declarator` field chain (pointer/array/function wrappers) down to
/// the innermost identifier that names the symbol.
fn declarator_name(node: Node<'_>, content: &str) -> Option<String> {
    match node.kind() {
        "identifier" | "type_identifier" | "field_identifier" => {
            node.utf8_text(content.as_bytes()).ok().map(str::to_owned)
        }
        _ => node
            .child_by_field_name("declarator")
            .and_then(|child| declarator_name(child, content)),
    }
}
