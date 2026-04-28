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

const DEFAULT_CHUNK_LINES: usize = 60;
const DEFAULT_OVERLAP_LINES: usize = 5;

#[derive(Debug)]
pub struct MultiLanguageChunker {
    fallback_chunk_lines: usize,
    fallback_overlap_lines: usize,
}

impl Default for MultiLanguageChunker {
    fn default() -> Self {
        Self {
            fallback_chunk_lines: DEFAULT_CHUNK_LINES,
            fallback_overlap_lines: DEFAULT_OVERLAP_LINES,
        }
    }
}

impl MultiLanguageChunker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_fallback_params(chunk_lines: usize, overlap_lines: usize) -> Self {
        Self {
            fallback_chunk_lines: chunk_lines,
            fallback_overlap_lines: overlap_lines,
        }
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
            Language::Python => chunk_python(path, file_hash, content),
            Language::TypeScript | Language::JavaScript => {
                chunk_typescript(path, language, file_hash, content)
            }
            Language::Go => chunk_go(path, file_hash, content),
            Language::Java | Language::C | Language::Cpp => chunk_fallback(
                path,
                language,
                file_hash,
                content,
                self.fallback_chunk_lines,
                self.fallback_overlap_lines,
            ),
            Language::Unknown => Ok(Vec::new()),
        }
    }
}

// ---------------------------------------------------------------------------
// Rust chunker
// ---------------------------------------------------------------------------

fn chunk_rust(path: &RelativePath, file_hash: FileHash, content: &str) -> Result<Vec<Chunk>> {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_rust::LANGUAGE.into())
        .map_err(|error| ClaudixError::TreeSitter(error.to_string()))?;

    let tree = parser
        .parse(content, None)
        .ok_or_else(|| ClaudixError::TreeSitter("failed to parse Rust source".to_owned()))?;

    let mut chunks = Vec::new();
    collect_chunks(
        tree.root_node(),
        path,
        Language::Rust,
        file_hash,
        content,
        &mut chunks,
        rust_chunk_kind,
    )?;
    chunks.sort_by_key(|chunk| (chunk.byte_range.start, chunk.byte_range.end));
    Ok(chunks)
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

// ---------------------------------------------------------------------------
// Python chunker
// ---------------------------------------------------------------------------

fn chunk_python(path: &RelativePath, file_hash: FileHash, content: &str) -> Result<Vec<Chunk>> {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_python::LANGUAGE.into())
        .map_err(|error| ClaudixError::TreeSitter(error.to_string()))?;

    let tree = parser
        .parse(content, None)
        .ok_or_else(|| ClaudixError::TreeSitter("failed to parse Python source".to_owned()))?;

    let mut chunks = Vec::new();
    collect_chunks(
        tree.root_node(),
        path,
        Language::Python,
        file_hash,
        content,
        &mut chunks,
        python_chunk_kind,
    )?;
    chunks.sort_by_key(|chunk| (chunk.byte_range.start, chunk.byte_range.end));
    Ok(chunks)
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
            if has_function {
                Some(ChunkKind::Function)
            } else {
                None
            }
        }
        "class_definition" => {
            // Skip if the parent is already a decorated_definition we emitted.
            let parent_is_decorated = node
                .parent()
                .map(|p| p.kind() == "decorated_definition")
                .unwrap_or(false);
            if parent_is_decorated {
                None
            } else {
                Some(ChunkKind::Class)
            }
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// TypeScript / JavaScript chunker
// ---------------------------------------------------------------------------

fn chunk_typescript(
    path: &RelativePath,
    language: Language,
    file_hash: FileHash,
    content: &str,
) -> Result<Vec<Chunk>> {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into())
        .map_err(|error| ClaudixError::TreeSitter(error.to_string()))?;

    let tree = parser
        .parse(content, None)
        .ok_or_else(|| ClaudixError::TreeSitter("failed to parse TypeScript source".to_owned()))?;

    let mut chunks = Vec::new();
    collect_chunks(
        tree.root_node(),
        path,
        language,
        file_hash,
        content,
        &mut chunks,
        typescript_chunk_kind,
    )?;
    chunks.sort_by_key(|chunk| (chunk.byte_range.start, chunk.byte_range.end));
    Ok(chunks)
}

fn typescript_chunk_kind(node: Node<'_>) -> Option<ChunkKind> {
    match node.kind() {
        "function_declaration" | "function_expression" => Some(ChunkKind::Function),
        "arrow_function" => {
            // Emit only when directly inside a variable_declarator so we don't
            // produce a chunk for every inline arrow.
            let parent_is_declarator = node
                .parent()
                .map(|p| p.kind() == "variable_declarator")
                .unwrap_or(false);
            if parent_is_declarator {
                Some(ChunkKind::Function)
            } else {
                None
            }
        }
        "method_definition" => Some(ChunkKind::Method),
        "class_declaration" | "class_expression" => Some(ChunkKind::Class),
        "interface_declaration" => Some(ChunkKind::Interface),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Go chunker
// ---------------------------------------------------------------------------

fn chunk_go(path: &RelativePath, file_hash: FileHash, content: &str) -> Result<Vec<Chunk>> {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_go::LANGUAGE.into())
        .map_err(|error| ClaudixError::TreeSitter(error.to_string()))?;

    let tree = parser
        .parse(content, None)
        .ok_or_else(|| ClaudixError::TreeSitter("failed to parse Go source".to_owned()))?;

    let mut chunks = Vec::new();
    collect_chunks(
        tree.root_node(),
        path,
        Language::Go,
        file_hash,
        content,
        &mut chunks,
        go_chunk_kind,
    )?;
    chunks.sort_by_key(|chunk| (chunk.byte_range.start, chunk.byte_range.end));
    Ok(chunks)
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

// ---------------------------------------------------------------------------
// Fallback sliding-window chunker (unknown languages)
// ---------------------------------------------------------------------------

/// Split `content` into overlapping line-based chunks.
///
/// `chunk_size` — number of lines per chunk (default 50).
/// `overlap`    — lines shared between adjacent chunks (default 0).
pub fn chunk_fallback(
    path: &RelativePath,
    language: Language,
    file_hash: FileHash,
    content: &str,
    chunk_size: usize,
    overlap: usize,
) -> Result<Vec<Chunk>> {
    if content.is_empty() {
        return Ok(Vec::new());
    }

    let chunk_size = chunk_size.max(1);
    let step = if overlap < chunk_size {
        chunk_size - overlap
    } else {
        1
    };

    // Collect byte offsets of the start of every line.
    let mut line_starts: Vec<usize> = vec![0];
    for (offset, byte) in content.bytes().enumerate() {
        if byte == b'\n' && offset + 1 < content.len() {
            line_starts.push(offset + 1);
        }
    }
    let total_lines = line_starts.len();

    let mut chunks = Vec::new();
    let mut window_start = 0_usize;

    while window_start < total_lines {
        let window_end = (window_start + chunk_size).min(total_lines);
        let byte_start = line_starts[window_start];
        let byte_end = if window_end < total_lines {
            line_starts[window_end]
        } else {
            content.len()
        };

        let chunk_content = content
            .get(byte_start..byte_end)
            .ok_or_else(|| {
                ClaudixError::TreeSitter("fallback chunk byte range not utf-8 aligned".to_owned())
            })?
            .to_owned();

        let start_line = u32::try_from(window_start + 1)
            .map_err(|_| ClaudixError::TreeSitter("line number overflowed u32".to_owned()))?;
        let end_line = u32::try_from(window_end)
            .map_err(|_| ClaudixError::TreeSitter("line number overflowed u32".to_owned()))?;

        let byte_range = ByteRange {
            start: u32::try_from(byte_start)
                .map_err(|_| ClaudixError::TreeSitter("chunk start overflowed u32".to_owned()))?,
            end: u32::try_from(byte_end)
                .map_err(|_| ClaudixError::TreeSitter("chunk end overflowed u32".to_owned()))?,
        };

        chunks.push(Chunk {
            id: chunk_id(file_hash, byte_range),
            file_path: path.clone(),
            language,
            kind: ChunkKind::Other,
            name: None,
            line_range: LineRange {
                start: start_line,
                end: end_line,
            },
            byte_range,
            file_hash,
            content: chunk_content,
        });

        if window_end == total_lines {
            break;
        }
        window_start += step;
    }

    Ok(chunks)
}

// ---------------------------------------------------------------------------
// Shared tree-sitter helpers
// ---------------------------------------------------------------------------

fn collect_chunks(
    node: Node<'_>,
    path: &RelativePath,
    language: Language,
    file_hash: FileHash,
    content: &str,
    chunks: &mut Vec<Chunk>,
    kind_fn: fn(Node<'_>) -> Option<ChunkKind>,
) -> Result<()> {
    if let Some(kind) = kind_fn(node) {
        chunks.push(build_chunk(path, language, file_hash, content, node, kind)?);
    }

    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        collect_chunks(child, path, language, file_hash, content, chunks, kind_fn)?;
    }

    Ok(())
}

fn build_chunk(
    path: &RelativePath,
    language: Language,
    file_hash: FileHash,
    content: &str,
    node: Node<'_>,
    kind: ChunkKind,
) -> Result<Chunk> {
    let start = if language == Language::Rust {
        extend_start_for_rust_docs(content, node.start_byte())
    } else {
        node.start_byte()
    };
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
        language,
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

    // -----------------------------------------------------------------------
    // Rust
    // -----------------------------------------------------------------------

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

    // -----------------------------------------------------------------------
    // Python
    // -----------------------------------------------------------------------

    #[test]
    fn python_chunker_empty_returns_empty() {
        let chunker = MultiLanguageChunker::new();
        let chunks = chunker
            .chunk(
                &RelativePath::new("app.py"),
                Language::Python,
                hash_for(""),
                "",
            )
            .ok()
            .unwrap_or_else(|| unreachable!());
        assert!(chunks.is_empty());
    }

    #[test]
    fn python_chunker_extracts_function() {
        let source = "def greet(name):\n    return f'hello {name}'\n";
        let chunker = MultiLanguageChunker::new();

        let chunks = chunker
            .chunk(
                &RelativePath::new("app.py"),
                Language::Python,
                hash_for(source),
                source,
            )
            .ok()
            .unwrap_or_else(|| unreachable!());

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].kind, ChunkKind::Function);
        assert_eq!(chunks[0].name.as_deref(), Some("greet"));
    }

    #[test]
    fn python_chunker_extracts_class() {
        let source = "class Dog:\n    def bark(self):\n        print('woof')\n";
        let chunker = MultiLanguageChunker::new();

        let chunks = chunker
            .chunk(
                &RelativePath::new("dog.py"),
                Language::Python,
                hash_for(source),
                source,
            )
            .ok()
            .unwrap_or_else(|| unreachable!());

        let class_chunk = chunks.iter().find(|c| c.kind == ChunkKind::Class);
        assert!(class_chunk.is_some());
        let class_chunk = class_chunk.unwrap_or_else(|| unreachable!());
        assert_eq!(class_chunk.name.as_deref(), Some("Dog"));
    }

    #[test]
    fn python_chunker_extracts_decorated_function() {
        let source = "@staticmethod\ndef helper():\n    pass\n";
        let chunker = MultiLanguageChunker::new();

        let chunks = chunker
            .chunk(
                &RelativePath::new("util.py"),
                Language::Python,
                hash_for(source),
                source,
            )
            .ok()
            .unwrap_or_else(|| unreachable!());

        assert!(!chunks.is_empty());
        let func = chunks.iter().find(|c| c.kind == ChunkKind::Function);
        assert!(func.is_some());
    }

    // -----------------------------------------------------------------------
    // TypeScript
    // -----------------------------------------------------------------------

    #[test]
    fn typescript_chunker_empty_returns_empty() {
        let chunker = MultiLanguageChunker::new();
        let chunks = chunker
            .chunk(
                &RelativePath::new("app.ts"),
                Language::TypeScript,
                hash_for(""),
                "",
            )
            .ok()
            .unwrap_or_else(|| unreachable!());
        assert!(chunks.is_empty());
    }

    #[test]
    fn typescript_chunker_extracts_function_declaration() {
        let source = "function greet(name: string): string {\n  return `hello ${name}`;\n}\n";
        let chunker = MultiLanguageChunker::new();

        let chunks = chunker
            .chunk(
                &RelativePath::new("greet.ts"),
                Language::TypeScript,
                hash_for(source),
                source,
            )
            .ok()
            .unwrap_or_else(|| unreachable!());

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].kind, ChunkKind::Function);
        assert_eq!(chunks[0].name.as_deref(), Some("greet"));
    }

    #[test]
    fn typescript_chunker_extracts_class() {
        let source = "class Animal {\n  name: string;\n  constructor(name: string) { this.name = name; }\n}\n";
        let chunker = MultiLanguageChunker::new();

        let chunks = chunker
            .chunk(
                &RelativePath::new("animal.ts"),
                Language::TypeScript,
                hash_for(source),
                source,
            )
            .ok()
            .unwrap_or_else(|| unreachable!());

        let class_chunk = chunks.iter().find(|c| c.kind == ChunkKind::Class);
        assert!(class_chunk.is_some());
        let class_chunk = class_chunk.unwrap_or_else(|| unreachable!());
        assert_eq!(class_chunk.name.as_deref(), Some("Animal"));
    }

    #[test]
    fn typescript_chunker_extracts_interface() {
        let source = "interface Shape {\n  area(): number;\n}\n";
        let chunker = MultiLanguageChunker::new();

        let chunks = chunker
            .chunk(
                &RelativePath::new("shape.ts"),
                Language::TypeScript,
                hash_for(source),
                source,
            )
            .ok()
            .unwrap_or_else(|| unreachable!());

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].kind, ChunkKind::Interface);
        assert_eq!(chunks[0].name.as_deref(), Some("Shape"));
    }

    // -----------------------------------------------------------------------
    // Go
    // -----------------------------------------------------------------------

    #[test]
    fn go_chunker_empty_returns_empty() {
        let chunker = MultiLanguageChunker::new();
        let chunks = chunker
            .chunk(
                &RelativePath::new("main.go"),
                Language::Go,
                hash_for(""),
                "",
            )
            .ok()
            .unwrap_or_else(|| unreachable!());
        assert!(chunks.is_empty());
    }

    #[test]
    fn go_chunker_extracts_function() {
        let source =
            "package main\n\nfunc Greet(name string) string {\n\treturn \"hello \" + name\n}\n";
        let chunker = MultiLanguageChunker::new();

        let chunks = chunker
            .chunk(
                &RelativePath::new("main.go"),
                Language::Go,
                hash_for(source),
                source,
            )
            .ok()
            .unwrap_or_else(|| unreachable!());

        let func = chunks.iter().find(|c| c.kind == ChunkKind::Function);
        assert!(func.is_some());
        let func = func.unwrap_or_else(|| unreachable!());
        assert_eq!(func.name.as_deref(), Some("Greet"));
    }

    #[test]
    fn go_chunker_extracts_struct_type() {
        let source = "package main\n\ntype Dog struct {\n\tName string\n}\n";
        let chunker = MultiLanguageChunker::new();

        let chunks = chunker
            .chunk(
                &RelativePath::new("dog.go"),
                Language::Go,
                hash_for(source),
                source,
            )
            .ok()
            .unwrap_or_else(|| unreachable!());

        let struct_chunk = chunks.iter().find(|c| c.kind == ChunkKind::Struct);
        assert!(struct_chunk.is_some());
    }

    #[test]
    fn go_chunker_extracts_interface_type() {
        let source = "package main\n\ntype Animal interface {\n\tSpeak() string\n}\n";
        let chunker = MultiLanguageChunker::new();

        let chunks = chunker
            .chunk(
                &RelativePath::new("animal.go"),
                Language::Go,
                hash_for(source),
                source,
            )
            .ok()
            .unwrap_or_else(|| unreachable!());

        let iface = chunks.iter().find(|c| c.kind == ChunkKind::Interface);
        assert!(iface.is_some());
    }

    // -----------------------------------------------------------------------
    // Fallback sliding-window
    // -----------------------------------------------------------------------

    #[test]
    fn fallback_empty_returns_empty() {
        let result = chunk_fallback(
            &RelativePath::new("file.txt"),
            Language::Unknown,
            hash_for(""),
            "",
            50,
            0,
        );
        assert!(result.is_ok());
        assert!(result.ok().unwrap_or_else(|| unreachable!()).is_empty());
    }

    #[test]
    fn fallback_file_smaller_than_chunk_size_returns_single_chunk() {
        let source = "line1\nline2\nline3\n";
        let result = chunk_fallback(
            &RelativePath::new("file.txt"),
            Language::Unknown,
            hash_for(source),
            source,
            50,
            0,
        );
        assert!(result.is_ok());
        let chunks = result.ok().unwrap_or_else(|| unreachable!());
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].kind, ChunkKind::Other);
        assert!(chunks[0].name.is_none());
    }

    #[test]
    fn fallback_10_lines_chunk_size_5_no_overlap_returns_2_chunks() {
        let source = (1..=10)
            .map(|i| format!("line{i}"))
            .collect::<Vec<_>>()
            .join("\n")
            + "\n";
        let result = chunk_fallback(
            &RelativePath::new("file.txt"),
            Language::Unknown,
            hash_for(&source),
            &source,
            5,
            0,
        );
        assert!(result.is_ok());
        let chunks = result.ok().unwrap_or_else(|| unreachable!());
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].line_range.start, 1);
        assert_eq!(chunks[0].line_range.end, 5);
        assert_eq!(chunks[1].line_range.start, 6);
        assert_eq!(chunks[1].line_range.end, 10);
    }

    #[test]
    fn fallback_overlap_produces_overlapping_chunks() {
        // 10 lines, chunk_size=6, overlap=2 → step=4 → windows at 0,4
        let source = (1..=10)
            .map(|i| format!("line{i}"))
            .collect::<Vec<_>>()
            .join("\n")
            + "\n";
        let result = chunk_fallback(
            &RelativePath::new("file.txt"),
            Language::Unknown,
            hash_for(&source),
            &source,
            6,
            2,
        );
        assert!(result.is_ok());
        let chunks = result.ok().unwrap_or_else(|| unreachable!());
        // window 0..6 and 4..10
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].line_range.start, 1);
        assert_eq!(chunks[0].line_range.end, 6);
        assert_eq!(chunks[1].line_range.start, 5);
    }

    #[test]
    fn python_chunker_returns_function_chunk() {
        let chunker = MultiLanguageChunker::new();
        let source = "def greet(name):\n    return f'hello {name}'\n";

        let chunks = chunker
            .chunk(
                &RelativePath::new("src/app.py"),
                Language::Python,
                hash_for(source),
                source,
            )
            .ok()
            .unwrap_or_else(|| unreachable!());

        assert!(!chunks.is_empty());
        assert_eq!(chunks[0].kind, ChunkKind::Function);
    }

    #[test]
    fn unknown_language_returns_empty() {
        let chunker = MultiLanguageChunker::new();
        let source = "some text\n";

        let chunks = chunker
            .chunk(
                &RelativePath::new("data.txt"),
                Language::Unknown,
                hash_for(source),
                source,
            )
            .ok()
            .unwrap_or_else(|| unreachable!());

        assert!(
            chunks.is_empty(),
            "Unknown language should return empty (config files etc.)"
        );
    }

    #[test]
    fn java_language_falls_back_to_sliding_window() {
        let chunker = MultiLanguageChunker::new();
        let source = "public class Hello {\n    public static void main(String[] args) {\n        System.out.println(\"Hello\");\n    }\n}\n";

        let chunks = chunker
            .chunk(
                &RelativePath::new("Hello.java"),
                Language::Java,
                hash_for(source),
                source,
            )
            .ok()
            .unwrap_or_else(|| unreachable!());

        assert!(
            !chunks.is_empty(),
            "Java should fall back to sliding-window chunker"
        );
        assert_eq!(chunks[0].kind, ChunkKind::Other);
    }

    #[test]
    fn c_language_falls_back_to_sliding_window() {
        let chunker = MultiLanguageChunker::new();
        let source = "int add(int a, int b) { return a + b; }\n";

        let chunks = chunker
            .chunk(
                &RelativePath::new("math.c"),
                Language::C,
                hash_for(source),
                source,
            )
            .ok()
            .unwrap_or_else(|| unreachable!());

        assert!(
            !chunks.is_empty(),
            "C should fall back to sliding-window chunker"
        );
    }
}
