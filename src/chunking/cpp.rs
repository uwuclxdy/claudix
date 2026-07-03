use tree_sitter::Node;

use crate::error::Result;
use crate::types::{Chunk, ChunkKind, FileHash, Language, RelativePath};

use super::{chunk_with_grammar_named, default_name};

pub(super) fn chunk(path: &RelativePath, file_hash: FileHash, content: &str) -> Result<Vec<Chunk>> {
    chunk_with_grammar_named(
        tree_sitter_cpp::LANGUAGE.into(),
        "C++",
        path,
        Language::Cpp,
        file_hash,
        content,
        cpp_chunk_kind,
        cpp_name,
    )
}

fn cpp_chunk_kind(node: Node<'_>) -> Option<ChunkKind> {
    match node.kind() {
        // A function defined inside a class/struct body is a method; a free or
        // out-of-line definition is a function.
        "function_definition" => Some(if inside_type_body(node) {
            ChunkKind::Method
        } else {
            ChunkKind::Function
        }),
        "class_specifier" => has_body(node).then_some(ChunkKind::Class),
        "struct_specifier" | "union_specifier" => has_body(node).then_some(ChunkKind::Struct),
        "enum_specifier" => has_body(node).then_some(ChunkKind::Enum),
        "namespace_definition" => Some(ChunkKind::Module),
        "preproc_def" | "preproc_function_def" => Some(ChunkKind::Macro),
        _ => None,
    }
}

fn has_body(node: Node<'_>) -> bool {
    node.child_by_field_name("body").is_some()
}

/// A member function's `function_definition` sits inside the class body
/// (`field_declaration_list`); a free function's does not. Stop the ancestor
/// walk at the enclosing definition boundary so a namespace-scoped free
/// function isn't mistaken for a method.
fn inside_type_body(node: Node<'_>) -> bool {
    let mut current = node.parent();
    while let Some(parent) = current {
        match parent.kind() {
            "field_declaration_list" => return true,
            "translation_unit" | "namespace_definition" => return false,
            _ => current = parent.parent(),
        }
    }
    false
}

/// C++ hides function/method names in a nested `declarator` chain that may end
/// in a qualified `Foo::bar`, a `~Dtor`, or an `operator==`. Types, namespaces
/// and macros expose a `name` field.
fn cpp_name(node: Node<'_>, content: &str) -> Option<String> {
    if node.kind() == "function_definition" {
        return node
            .child_by_field_name("declarator")
            .and_then(|declarator| declarator_name(declarator, content));
    }

    default_name(node, content)
}

fn declarator_name(node: Node<'_>, content: &str) -> Option<String> {
    match node.kind() {
        "identifier"
        | "field_identifier"
        | "type_identifier"
        | "qualified_identifier"
        | "destructor_name"
        | "operator_name"
        | "template_function" => node.utf8_text(content.as_bytes()).ok().map(str::to_owned),
        _ => node
            .child_by_field_name("declarator")
            .and_then(|child| declarator_name(child, content)),
    }
}
