use tree_sitter::Node;

use crate::error::Result;
use crate::types::{Chunk, ChunkKind, FileHash, Language, RelativePath};

use super::chunk_with_grammar;

pub(super) fn chunk(path: &RelativePath, file_hash: FileHash, content: &str) -> Result<Vec<Chunk>> {
    chunk_with_grammar(
        tree_sitter_c_sharp::LANGUAGE.into(),
        "C#",
        path,
        Language::CSharp,
        file_hash,
        content,
        csharp_chunk_kind,
    )
}

fn csharp_chunk_kind(node: Node<'_>) -> Option<ChunkKind> {
    match node.kind() {
        // Block `namespace App { ... }` wraps the file's members as its body, so
        // it's a real container chunk. A file-scoped `namespace App;` is a
        // distinct grammar node (`file_scoped_namespace_declaration`) not matched
        // here, so it never becomes a useless one-line chunk.
        "namespace_declaration" => Some(ChunkKind::Module),
        // A record is a reference type like a class; group them under Class.
        "class_declaration" | "record_declaration" => Some(ChunkKind::Class),
        "interface_declaration" => Some(ChunkKind::Interface),
        "struct_declaration" => Some(ChunkKind::Struct),
        "enum_declaration" => Some(ChunkKind::Enum),
        // Properties can carry getter/setter bodies worth searching, so they're
        // indexed as members alongside methods and constructors.
        "method_declaration" | "constructor_declaration" | "property_declaration" => {
            Some(ChunkKind::Method)
        }
        _ => None,
    }
}
