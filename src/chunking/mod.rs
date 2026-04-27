use tree_sitter::{Node, Parser};

use crate::error::{ClaudixError, Result};
use crate::types::{
    ByteRange, Chunk, ChunkId, ChunkKind, FileHash, Language, LineRange, RelativePath,
};

pub trait Chunker {
    fn chunk(
        &self,
        path: &RelativePath,
        language: Language,
        file_hash: FileHash,
        content: &str,
    ) -> Result<Vec<Chunk>>;
}

#[derive(Debug, Default)]
pub struct MultiLanguageChunker;

impl MultiLanguageChunker {
    pub fn new() -> Self {
        Self
    }
}

impl Chunker for MultiLanguageChunker {
    fn chunk(
        &self,
        path: &RelativePath,
        language: Language,
        file_hash: FileHash,
        content: &str,
    ) -> Result<Vec<Chunk>> {
        match language {
            Language::Rust => chunk_rust(path, file_hash, content),
            _ => Ok(Vec::new()),
        }
    }
}

fn chunk_rust(path: &RelativePath, file_hash: FileHash, content: &str) -> Result<Vec<Chunk>> {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_rust::LANGUAGE.into())
        .map_err(|error| ClaudixError::TreeSitter(error.to_string()))?;

    let tree = parser
        .parse(content, None)
        .ok_or_else(|| ClaudixError::TreeSitter("failed to parse Rust source".to_owned()))?;

    let mut chunks = Vec::new();
    collect_rust_chunks(tree.root_node(), path, file_hash, content, &mut chunks)?;
    chunks.sort_by_key(|chunk| (chunk.byte_range.start, chunk.byte_range.end));
    Ok(chunks)
}

fn collect_rust_chunks(
    node: Node<'_>,
    path: &RelativePath,
    file_hash: FileHash,
    content: &str,
    chunks: &mut Vec<Chunk>,
) -> Result<()> {
    if let Some(kind) = rust_chunk_kind(node) {
        chunks.push(build_chunk(path, file_hash, content, node, kind)?);
    }

    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        collect_rust_chunks(child, path, file_hash, content, chunks)?;
    }

    Ok(())
}

fn rust_chunk_kind(node: Node<'_>) -> Option<ChunkKind> {
    match node.kind() {
        "function_item" => Some(function_kind(node)),
        "function_signature_item" => Some(ChunkKind::Method),
        "struct_item" => Some(ChunkKind::Struct),
        "enum_item" => Some(ChunkKind::Enum),
        "trait_item" => Some(ChunkKind::Trait),
        "impl_item" => Some(ChunkKind::Impl),
        "mod_item" => Some(ChunkKind::Module),
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

fn build_chunk(
    path: &RelativePath,
    file_hash: FileHash,
    content: &str,
    node: Node<'_>,
    kind: ChunkKind,
) -> Result<Chunk> {
    let start = extend_start_for_rust_docs(content, node.start_byte());
    let end = node.end_byte();
    let byte_range = ByteRange {
        start: u32::try_from(start)
            .map_err(|_| ClaudixError::TreeSitter("chunk start overflowed u32".to_owned()))?,
        end: u32::try_from(end)
            .map_err(|_| ClaudixError::TreeSitter("chunk end overflowed u32".to_owned()))?,
    };

    let line_range = LineRange {
        start: line_number_for_byte(content, start),
        end: inclusive_end_line(content, start, end),
    };

    let name = node
        .child_by_field_name("name")
        .and_then(|child| child.utf8_text(content.as_bytes()).ok())
        .map(str::to_owned);

    let chunk_content = content
        .get(start..end)
        .ok_or_else(|| {
            ClaudixError::TreeSitter("chunk byte range was not utf-8 aligned".to_owned())
        })?
        .to_owned();

    Ok(Chunk {
        id: chunk_id(file_hash, byte_range),
        file_path: path.clone(),
        language: Language::Rust,
        kind,
        name,
        line_range,
        byte_range,
        file_hash,
        content: chunk_content,
    })
}

fn chunk_id(file_hash: FileHash, byte_range: ByteRange) -> ChunkId {
    let mut payload = [0_u8; 24];
    payload[..16].copy_from_slice(&file_hash.0);
    payload[16..20].copy_from_slice(&byte_range.start.to_be_bytes());
    payload[20..24].copy_from_slice(&byte_range.end.to_be_bytes());

    ChunkId(xxhash_rust::xxh3::xxh3_128(&payload) as u64)
}

fn extend_start_for_rust_docs(content: &str, node_start: usize) -> usize {
    let mut start = node_start;

    loop {
        let current_line_start = line_start(content, start);
        if current_line_start == 0 {
            return start;
        }

        let previous_line_end = current_line_start.saturating_sub(1);
        let previous_line_start = line_start(content, previous_line_end);
        let previous_line = &content[previous_line_start..line_end(content, previous_line_start)];

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

fn line_start(content: &str, byte_index: usize) -> usize {
    content[..byte_index]
        .rfind('\n')
        .map(|index| index + 1)
        .unwrap_or(0)
}

fn line_end(content: &str, line_start: usize) -> usize {
    content[line_start..]
        .find('\n')
        .map(|offset| line_start + offset)
        .unwrap_or(content.len())
}

fn line_number_for_byte(content: &str, byte_index: usize) -> u32 {
    let line_count = content[..byte_index]
        .bytes()
        .filter(|byte| *byte == b'\n')
        .count();

    u32::try_from(line_count + 1).unwrap_or(u32::MAX)
}

fn inclusive_end_line(content: &str, start: usize, end: usize) -> u32 {
    if end <= start {
        return line_number_for_byte(content, start);
    }

    line_number_for_byte(content, end.saturating_sub(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hash_for(content: &str) -> FileHash {
        FileHash(xxhash_rust::xxh3::xxh3_128(content.as_bytes()).to_be_bytes())
    }

    #[test]
    fn rust_chunker_extracts_named_top_level_items() {
        let source = "/// Greets a user.\npub fn greet(name: &str) -> String {\n    format!(\"hello {name}\")\n}\n\npub struct Greeter;\n";
        let chunker = MultiLanguageChunker::new();

        let chunks = chunker.chunk(
            &RelativePath::new("src/lib.rs"),
            Language::Rust,
            hash_for(source),
            source,
        );
        assert!(chunks.is_ok());
        let chunks = chunks.ok().unwrap_or_else(|| unreachable!());

        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].kind, ChunkKind::Function);
        assert_eq!(chunks[0].name.as_deref(), Some("greet"));
        assert_eq!(chunks[0].line_range.start, 1);
        assert!(chunks[0].content.starts_with("/// Greets a user."));
        assert_eq!(chunks[1].kind, ChunkKind::Struct);
        assert_eq!(chunks[1].name.as_deref(), Some("Greeter"));
    }

    #[test]
    fn rust_chunker_marks_impl_functions_as_methods() {
        let source = "pub struct Counter;\n\nimpl Counter {\n    pub fn new() -> Self {\n        Self\n    }\n}\n";
        let chunker = MultiLanguageChunker::new();

        let chunks = chunker.chunk(
            &RelativePath::new("src/counter.rs"),
            Language::Rust,
            hash_for(source),
            source,
        );
        assert!(chunks.is_ok());
        let chunks = chunks.ok().unwrap_or_else(|| unreachable!());

        let method = chunks
            .iter()
            .find(|chunk| chunk.name.as_deref() == Some("new"));
        assert!(method.is_some());
        let method = method.unwrap_or_else(|| unreachable!());
        assert_eq!(method.kind, ChunkKind::Method);

        let has_impl_chunk = chunks.iter().any(|chunk| chunk.kind == ChunkKind::Impl);
        assert!(has_impl_chunk);
    }

    #[test]
    fn chunk_ids_are_deterministic_for_same_input() {
        let source = "pub fn greet() {}\n";
        let chunker = MultiLanguageChunker::new();

        let first = chunker.chunk(
            &RelativePath::new("src/lib.rs"),
            Language::Rust,
            hash_for(source),
            source,
        );
        assert!(first.is_ok());
        let first = first.ok().unwrap_or_else(|| unreachable!());

        let second = chunker.chunk(
            &RelativePath::new("src/lib.rs"),
            Language::Rust,
            hash_for(source),
            source,
        );
        assert!(second.is_ok());
        let second = second.ok().unwrap_or_else(|| unreachable!());

        assert_eq!(first.len(), 1);
        assert_eq!(first[0].id, second[0].id);
        assert_eq!(first[0].byte_range, second[0].byte_range);
    }

    #[test]
    fn non_rust_languages_return_no_chunks_yet() {
        let chunker = MultiLanguageChunker::new();
        let source = "def greet(name):\n    return f'hello {name}'\n";

        let chunks = chunker.chunk(
            &RelativePath::new("src/app.py"),
            Language::Python,
            hash_for(source),
            source,
        );
        assert!(chunks.is_ok());
        let chunks = chunks.ok().unwrap_or_else(|| unreachable!());

        assert!(chunks.is_empty());
    }
}
