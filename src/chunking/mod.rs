mod c;
mod cpp;
mod csharp;
mod go;
mod java;
mod python;
mod rust;
mod sql;
mod typescript;

use tree_sitter::{Node, Parser};

use crate::error::{ClaudixError, Result};
use crate::types::{
    ByteRange, Chunk, ChunkId, ChunkKind, FileHash, Language, LineRange, RelativePath,
};

pub(crate) const DEFAULT_CHUNK_LINES: usize = 60;
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

    pub fn chunk_as_text(
        &self,
        path: &RelativePath,
        language: Language,
        file_hash: FileHash,
        content: &str,
    ) -> Result<Vec<Chunk>> {
        chunk_fallback(
            path,
            language,
            file_hash,
            content,
            self.fallback_chunk_lines,
            self.fallback_overlap_lines,
        )
    }

    pub fn chunk(
        &self,
        path: &RelativePath,
        language: Language,
        file_hash: FileHash,
        content: &str,
    ) -> Result<Vec<Chunk>> {
        match language {
            Language::Rust => rust::chunk(path, file_hash, content),
            Language::Python => python::chunk(path, file_hash, content),
            Language::TypeScript | Language::JavaScript => {
                typescript::chunk(path, language, file_hash, content)
            }
            Language::Go => go::chunk(path, file_hash, content),
            Language::CSharp => csharp::chunk(path, file_hash, content),
            Language::Sql => sql::chunk(path, file_hash, content),
            Language::Java => java::chunk(path, file_hash, content),
            Language::C => c::chunk(path, file_hash, content),
            Language::Cpp => cpp::chunk(path, file_hash, content),
            Language::Unknown => Ok(Vec::new()),
        }
    }
}

/// The grammar backing a language's first-class chunker, for callers that need
/// to re-parse chunk text rather than produce chunks from it. Kept here so the
/// language-to-grammar mapping has one home; a copy in the measurement harness
/// would silently go stale the next time a language is added.
///
/// Test-only: production code reaches grammars through the per-language
/// `chunk` entry points, which carry the kind and name resolvers too.
#[cfg(test)]
pub(crate) fn grammar_for(language: Language) -> Option<tree_sitter::Language> {
    Some(match language {
        Language::Rust => tree_sitter_rust::LANGUAGE.into(),
        Language::Python => tree_sitter_python::LANGUAGE.into(),
        Language::TypeScript | Language::JavaScript => {
            tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()
        }
        Language::Go => tree_sitter_go::LANGUAGE.into(),
        Language::CSharp => tree_sitter_c_sharp::LANGUAGE.into(),
        Language::Sql => tree_sitter_sequel::LANGUAGE.into(),
        Language::Java => tree_sitter_java::LANGUAGE.into(),
        Language::C => tree_sitter_c::LANGUAGE.into(),
        Language::Cpp => tree_sitter_cpp::LANGUAGE.into(),
        Language::Unknown => return None,
    })
}

/// Resolves a chunk node's symbol name from the source. Most grammars expose a
/// `name` field (see `default_name`); SQL doesn't, so it supplies its own.
type NameFn = fn(Node<'_>, &str) -> Option<String>;

/// Generic tree-sitter chunker shell. Per-language modules call this with
/// their grammar and kind classifier; the parse-and-walk machinery is
/// identical across grammars so it lives here in one place. Name resolution
/// defaults to the `name` field; use `chunk_with_grammar_named` to override it.
pub(super) fn chunk_with_grammar(
    grammar: tree_sitter::Language,
    grammar_name: &'static str,
    path: &RelativePath,
    language: Language,
    file_hash: FileHash,
    content: &str,
    kind_fn: fn(Node<'_>) -> Option<ChunkKind>,
) -> Result<Vec<Chunk>> {
    chunk_with_grammar_named(
        grammar,
        grammar_name,
        path,
        language,
        file_hash,
        content,
        kind_fn,
        default_name,
    )
}

/// Like `chunk_with_grammar`, but with a custom name resolver for grammars
/// whose declaration nodes don't carry a `name` field.
#[allow(clippy::too_many_arguments)]
pub(super) fn chunk_with_grammar_named(
    grammar: tree_sitter::Language,
    grammar_name: &'static str,
    path: &RelativePath,
    language: Language,
    file_hash: FileHash,
    content: &str,
    kind_fn: fn(Node<'_>) -> Option<ChunkKind>,
    name_fn: NameFn,
) -> Result<Vec<Chunk>> {
    let mut parser = Parser::new();
    parser
        .set_language(&grammar)
        .map_err(|error| ClaudixError::TreeSitter(error.to_string()))?;

    let tree = parser.parse(content, None).ok_or_else(|| {
        ClaudixError::TreeSitter(format!("failed to parse {grammar_name} source"))
    })?;

    let mut chunks = Vec::new();
    collect_chunks(
        tree.root_node(),
        path,
        language,
        file_hash,
        content,
        &mut chunks,
        kind_fn,
        name_fn,
    )?;
    chunks.sort_by_key(|chunk| (chunk.byte_range.start, chunk.byte_range.end));
    Ok(chunks)
}

/// Default name resolver: the node's `name` field, read as UTF-8. Grammars that
/// place the symbol name elsewhere (SQL) pass their own `NameFn`.
fn default_name(node: Node<'_>, content: &str) -> Option<String> {
    node.child_by_field_name("name")
        .and_then(|child| child.utf8_text(content.as_bytes()).ok())
        .map(str::to_owned)
}

/// Split `content` into line-based chunks on content-defined boundaries. Used
/// for languages without a tree-sitter grammar and for `chunk_as_text`
/// (force-included files).
///
/// A segment ends where a line's content hash lands on the target residue, so a
/// boundary depends on what a line says rather than where it sits. That is the
/// whole point: an edit above a boundary leaves the boundary's line untouched,
/// so it stays a boundary and every chunk below the edit keeps its exact bytes.
/// A fixed line stride instead shifts every later window, re-hashing the file
/// and defeating the change-neighbor snapshot's per-chunk narrowing.
///
/// `chunk_size` — target segment length in lines (average, not a hard period).
/// `overlap`    — lines each chunk shares with the segment before it, pulled in
///                as a leading prefix. The shared lines come from a
///                content-defined boundary, so overlap does not reintroduce the
///                line-shift fragility. It changes chunk boundaries, not count.
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

    // Collect byte offsets of the start of every line. A trailing newline does
    // not open an empty final line.
    let mut line_starts: Vec<usize> = vec![0];
    for (offset, byte) in content.bytes().enumerate() {
        if byte == b'\n' && offset + 1 < content.len() {
            line_starts.push(offset + 1);
        }
    }
    let total_lines = line_starts.len();

    // Bounds that keep a pathological file (thousands of identical lines) from
    // collapsing to one chunk or shattering into thousands: below `min_lines`
    // no boundary is accepted, at `max_lines` one is forced. Expected segment
    // length is `min_lines + divisor`, so the divisor is sized to land the
    // average on `chunk_size`. Ceiling: a `chunk_size` of a handful of lines
    // drives `divisor` toward 1, degrading to near-per-line chunking; production
    // always passes `DEFAULT_CHUNK_LINES` (60), well clear of that.
    let target = chunk_size.max(1);
    let min_lines = (target / 4).max(1);
    let max_lines = target.saturating_mul(2).max(min_lines + 1);
    let divisor = u64::try_from(target.saturating_sub(min_lines).max(1)).unwrap_or(u64::MAX);

    // Content-defined segmentation over line indices: (start, end) inclusive.
    let mut segments: Vec<(usize, usize)> = Vec::new();
    let mut seg_start = 0_usize;
    for line in 0..total_lines {
        let len = line - seg_start + 1;
        let forced = len >= max_lines;
        let boundary =
            len >= min_lines && line_is_boundary(content, &line_starts, total_lines, line, divisor);
        if forced || boundary {
            segments.push((seg_start, line));
            seg_start = line + 1;
        }
    }
    if seg_start < total_lines {
        segments.push((seg_start, total_lines - 1));
    }

    let mut chunks = Vec::with_capacity(segments.len());
    for (seg_start, seg_end) in segments {
        // Honor `overlap` as a shared prefix pulled from the previous segment's
        // (content-defined, stable) tail.
        let start_line = seg_start.saturating_sub(overlap);
        let byte_start = line_starts[start_line];
        let byte_end = if seg_end + 1 < total_lines {
            line_starts[seg_end + 1]
        } else {
            content.len()
        };

        let chunk_content = content
            .get(byte_start..byte_end)
            .ok_or_else(|| {
                ClaudixError::TreeSitter("fallback chunk byte range not utf-8 aligned".to_owned())
            })?
            .to_owned();

        let start_line_no = u32::try_from(start_line + 1)
            .map_err(|_| ClaudixError::TreeSitter("line number overflowed u32".to_owned()))?;
        let end_line_no = u32::try_from(seg_end + 1)
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
                start: start_line_no,
                end: end_line_no,
            },
            byte_range,
            file_hash,
            content: chunk_content,
        });
    }

    Ok(chunks)
}

/// Whether `line` ends a content-defined segment: its content hash (newline
/// included) lands on the target residue. Per-line hashing is enough for
/// shift-tolerance because a boundary line keeps its bytes across an unrelated
/// edit; a windowed rolling hash would only add robustness to single-line
/// duplication, which `min`/`max` already bound.
fn line_is_boundary(
    content: &str,
    line_starts: &[usize],
    total_lines: usize,
    line: usize,
    divisor: u64,
) -> bool {
    let start = line_starts[line];
    let end = if line + 1 < total_lines {
        line_starts[line + 1]
    } else {
        content.len()
    };
    xxhash_rust::xxh3::xxh3_64(&content.as_bytes()[start..end]) % divisor == divisor - 1
}

#[allow(clippy::too_many_arguments)]
fn collect_chunks(
    node: Node<'_>,
    path: &RelativePath,
    language: Language,
    file_hash: FileHash,
    content: &str,
    chunks: &mut Vec<Chunk>,
    kind_fn: fn(Node<'_>) -> Option<ChunkKind>,
    name_fn: NameFn,
) -> Result<()> {
    if let Some(kind) = kind_fn(node) {
        chunks.push(build_chunk(
            path, language, file_hash, content, node, kind, name_fn,
        )?);
    }

    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        collect_chunks(
            child, path, language, file_hash, content, chunks, kind_fn, name_fn,
        )?;
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn build_chunk(
    path: &RelativePath,
    language: Language,
    file_hash: FileHash,
    content: &str,
    node: Node<'_>,
    kind: ChunkKind,
    name_fn: NameFn,
) -> Result<Chunk> {
    // Rust doc-comments above a function/struct aren't inside the node, so
    // extend the start byte upward to capture them. Other grammars place
    // docstrings inside the body — no extension needed.
    let start = if language == Language::Rust {
        rust::extend_start_for_rust_docs(content, node.start_byte())
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

    let name = name_fn(node, content);

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

// Content-addressed: hashes only `(file_hash, byte_range)`, NOT `file_path`.
// See `ChunkId` docs — storage keys by `(file_path, byte_start, chunk_id)`, so
// cross-file collisions are disambiguated there, not here.
fn chunk_id(file_hash: FileHash, byte_range: ByteRange) -> ChunkId {
    let mut payload = [0_u8; 24];
    payload[..16].copy_from_slice(&file_hash.0);
    payload[16..20].copy_from_slice(&byte_range.start.to_be_bytes());
    payload[20..24].copy_from_slice(&byte_range.end.to_be_bytes());

    ChunkId(xxhash_rust::xxh3::xxh3_64(&payload))
}

pub(super) fn line_start(content: &str, byte_index: usize) -> usize {
    content.as_bytes()[..byte_index]
        .iter()
        .rposition(|&b| b == b'\n')
        .map(|i| i + 1)
        .unwrap_or(0)
}

pub(super) fn line_end(content: &str, line_start: usize) -> usize {
    content.as_bytes()[line_start..]
        .iter()
        .position(|&b| b == b'\n')
        .map(|offset| line_start + offset)
        .unwrap_or(content.len())
}

fn line_number_for_byte(content: &str, byte_index: usize) -> u32 {
    let line_count = content.as_bytes()[..byte_index]
        .iter()
        .filter(|&&b| b == b'\n')
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
    fn chunk_overlap_lines_only_affects_the_fallback_chunker() {
        // Tree-sitter languages chunk semantically and ignore the overlap param;
        // only the sliding-window fallback (force-indexed files via
        // `chunk_as_text`, unknown types) honors it. Guards the README's
        // "fallback chunks only" wording.
        let no_overlap = MultiLanguageChunker::with_fallback_params(10, 0);
        let with_overlap = MultiLanguageChunker::with_fallback_params(10, 4);

        let rs = "fn a() {}\nfn b() {}\nfn c() {}\n";
        let rs_no = no_overlap
            .chunk(&RelativePath::new("t.rs"), Language::Rust, hash_for(rs), rs)
            .ok()
            .unwrap_or_else(|| unreachable!());
        let rs_ov = with_overlap
            .chunk(&RelativePath::new("t.rs"), Language::Rust, hash_for(rs), rs)
            .ok()
            .unwrap_or_else(|| unreachable!());
        assert_eq!(
            rs_no.len(),
            rs_ov.len(),
            "tree-sitter chunk count must not change with overlap"
        );

        // Force-indexed content routes through the content-defined fallback,
        // where overlap shares a leading prefix between adjacent chunks: same
        // chunk count, more total bytes once the shared lines are duplicated.
        let text: String = (0..40)
            .map(|i| format!("distinct fallback line {i}\n"))
            .collect();
        let text_no = no_overlap
            .chunk_as_text(
                &RelativePath::new("notes.txt"),
                Language::Unknown,
                hash_for(&text),
                &text,
            )
            .ok()
            .unwrap_or_else(|| unreachable!());
        let text_ov = with_overlap
            .chunk_as_text(
                &RelativePath::new("notes.txt"),
                Language::Unknown,
                hash_for(&text),
                &text,
            )
            .ok()
            .unwrap_or_else(|| unreachable!());
        assert_eq!(
            text_no.len(),
            text_ov.len(),
            "overlap must not change fallback chunk count"
        );
        let bytes = |cs: &[Chunk]| cs.iter().map(|c| c.content.len()).sum::<usize>();
        assert!(
            bytes(&text_ov) > bytes(&text_no),
            "fallback overlap must duplicate boundary lines: no={}, ov={}",
            bytes(&text_no),
            bytes(&text_ov)
        );
    }

    // -----------------------------------------------------------------------
    // Rust
    // -----------------------------------------------------------------------

    #[test]
    fn rust_chunker_skips_mod_pointer_declarations() {
        // `mod error;` is a pointer declaration with no body — indexing it
        // produces useless 1-token chunks that outscore the real content.
        let source = "mod error;\nmod tests;\n\npub mod inline {\n    pub fn helper() {}\n}\n";
        let chunker = MultiLanguageChunker::default();

        let chunks = chunker
            .chunk(
                &RelativePath::new("src/main.rs"),
                Language::Rust,
                hash_for(source),
                source,
            )
            .ok()
            .unwrap_or_else(|| unreachable!());

        let mod_chunks: Vec<_> = chunks
            .iter()
            .filter(|c| c.kind == ChunkKind::Module)
            .collect();
        assert_eq!(
            mod_chunks.len(),
            1,
            "only the inline mod block should be indexed"
        );
        assert_eq!(mod_chunks[0].name.as_deref(), Some("inline"));
    }

    #[test]
    fn rust_chunker_extracts_named_top_level_items() {
        let source = "/// Greets a user.\npub fn greet(name: &str) -> String {\n    format!(\"hello {name}\")\n}\n\npub struct Greeter;\n";
        let chunker = MultiLanguageChunker::default();

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
        let source = "pub struct Counter;\n\nimpl Counter {\n    pub fn new() -> Self {\n        Self\n    }\n\n    pub fn increment(&mut self) {}\n}\n\nimpl Default for Counter {\n    fn default() -> Self {\n        Self::new()\n    }\n}\n";
        let chunker = MultiLanguageChunker::default();

        let chunks = chunker.chunk(
            &RelativePath::new("src/counter.rs"),
            Language::Rust,
            hash_for(source),
            source,
        );
        assert!(chunks.is_ok());
        let chunks = chunks.ok().unwrap_or_else(|| unreachable!());

        let method_names = chunks
            .iter()
            .filter(|chunk| chunk.kind == ChunkKind::Method)
            .map(|chunk| chunk.name.as_deref())
            .collect::<Vec<_>>();
        assert_eq!(
            method_names,
            vec![Some("new"), Some("increment"), Some("default")]
        );

        let impl_chunks = chunks
            .iter()
            .filter(|chunk| chunk.kind == ChunkKind::Impl)
            .count();
        assert_eq!(impl_chunks, 2);
    }

    #[test]
    fn rust_chunker_handles_multibyte_chars_without_panic() {
        // é is 0xC3 0xA9 — a 2-byte UTF-8 sequence. The node's exclusive end_byte
        // sits after the last byte (0xA9), and end_byte - 1 = 0xA9 which is a
        // continuation byte. line_number_for_byte must not slice the str there.
        let source = "pub fn café() -> &'static str {\n    \"espresso\"\n}\n";
        let chunker = MultiLanguageChunker::default();
        let result = chunker.chunk(
            &RelativePath::new("src/lib.rs"),
            Language::Rust,
            hash_for(source),
            source,
        );
        assert!(
            result.is_ok(),
            "chunking with multi-byte ident must not panic: {result:?}"
        );
        let chunks = result.ok().unwrap_or_else(|| unreachable!());
        assert!(!chunks.is_empty());
        assert_eq!(chunks[0].name.as_deref(), Some("café"));
    }

    #[test]
    fn chunk_ids_are_deterministic_for_same_input() {
        let source = "pub fn greet() {}\n";
        let chunker = MultiLanguageChunker::default();

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
        let chunker = MultiLanguageChunker::default();
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
        let chunker = MultiLanguageChunker::default();

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
        let chunker = MultiLanguageChunker::default();

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
        let chunker = MultiLanguageChunker::default();

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
        let chunker = MultiLanguageChunker::default();
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
        let chunker = MultiLanguageChunker::default();

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
        let chunker = MultiLanguageChunker::default();

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
        let chunker = MultiLanguageChunker::default();

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
        let chunker = MultiLanguageChunker::default();
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
        let chunker = MultiLanguageChunker::default();

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
        let chunker = MultiLanguageChunker::default();

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
        let chunker = MultiLanguageChunker::default();

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
    // C#
    // -----------------------------------------------------------------------

    fn csharp_chunks(source: &str) -> Vec<Chunk> {
        MultiLanguageChunker::default()
            .chunk(
                &RelativePath::new("App.cs"),
                Language::CSharp,
                hash_for(source),
                source,
            )
            .ok()
            .unwrap_or_else(|| unreachable!())
    }

    #[test]
    fn csharp_chunker_empty_returns_empty() {
        assert!(csharp_chunks("").is_empty());
    }

    #[test]
    fn csharp_chunker_extracts_class_and_method() {
        let source = "public class User {\n    public void Rename(string name) {\n        Name = name;\n    }\n}\n";
        let chunks = csharp_chunks(source);

        let class_chunk = chunks
            .iter()
            .find(|c| c.kind == ChunkKind::Class)
            .unwrap_or_else(|| unreachable!());
        assert_eq!(class_chunk.name.as_deref(), Some("User"));

        let method = chunks
            .iter()
            .find(|c| c.kind == ChunkKind::Method)
            .unwrap_or_else(|| unreachable!());
        assert_eq!(method.name.as_deref(), Some("Rename"));
    }

    #[test]
    fn csharp_chunker_extracts_interface_struct_enum() {
        let source = "interface IRepo {\n    void Save();\n}\nstruct Point {\n    public int X;\n}\nenum Color { Red, Green }\n";
        let chunks = csharp_chunks(source);

        assert_eq!(
            chunks
                .iter()
                .find(|c| c.kind == ChunkKind::Interface)
                .and_then(|c| c.name.as_deref()),
            Some("IRepo")
        );
        assert_eq!(
            chunks
                .iter()
                .find(|c| c.kind == ChunkKind::Struct)
                .and_then(|c| c.name.as_deref()),
            Some("Point")
        );
        assert_eq!(
            chunks
                .iter()
                .find(|c| c.kind == ChunkKind::Enum)
                .and_then(|c| c.name.as_deref()),
            Some("Color")
        );
    }

    #[test]
    fn csharp_chunker_block_namespace_is_module_file_scoped_is_skipped() {
        let block = "namespace App.Services {\n    public class Svc {}\n}\n";
        let module = csharp_chunks(block)
            .into_iter()
            .find(|c| c.kind == ChunkKind::Module)
            .unwrap_or_else(|| unreachable!());
        assert_eq!(module.name.as_deref(), Some("App.Services"));

        // File-scoped namespace has no body; it must not produce a chunk.
        let file_scoped = "namespace App;\npublic class Svc {}\n";
        assert!(
            csharp_chunks(file_scoped)
                .iter()
                .all(|c| c.kind != ChunkKind::Module),
            "file-scoped namespace must not be chunked"
        );
    }

    #[test]
    fn csharp_chunker_record_is_class_and_property_is_method() {
        let source = "public record Money(decimal Amount);\npublic class Account {\n    public string Owner { get; set; }\n}\n";
        let chunks = csharp_chunks(source);

        assert!(
            chunks
                .iter()
                .any(|c| c.kind == ChunkKind::Class && c.name.as_deref() == Some("Money")),
            "record should chunk as Class"
        );
        assert!(
            chunks
                .iter()
                .any(|c| c.kind == ChunkKind::Method && c.name.as_deref() == Some("Owner")),
            "property should chunk as Method"
        );
    }

    // -----------------------------------------------------------------------
    // SQL
    // -----------------------------------------------------------------------

    fn sql_chunks(source: &str) -> Vec<Chunk> {
        MultiLanguageChunker::default()
            .chunk(
                &RelativePath::new("schema.sql"),
                Language::Sql,
                hash_for(source),
                source,
            )
            .ok()
            .unwrap_or_else(|| unreachable!())
    }

    #[test]
    fn sql_chunker_empty_returns_empty() {
        assert!(sql_chunks("").is_empty());
    }

    #[test]
    fn sql_chunker_extracts_table_with_qualified_name() {
        let source =
            "CREATE TABLE public.users (\n    id INT PRIMARY KEY,\n    email TEXT NOT NULL\n);\n";
        let table = sql_chunks(source)
            .into_iter()
            .find(|c| c.kind == ChunkKind::Table)
            .unwrap_or_else(|| unreachable!());
        assert_eq!(table.name.as_deref(), Some("public.users"));
    }

    #[test]
    fn sql_chunker_extracts_view_and_function() {
        // The function's qualified return type `app.amount` is a second
        // `object_reference`; the name resolver must still pick the first (`add`).
        let source = "CREATE VIEW active_users AS SELECT id FROM users WHERE active;\nCREATE FUNCTION add(a INT, b INT) RETURNS app.amount AS $$ SELECT a + b $$ LANGUAGE sql;\n";
        let chunks = sql_chunks(source);

        assert_eq!(
            chunks
                .iter()
                .find(|c| c.kind == ChunkKind::View)
                .and_then(|c| c.name.as_deref()),
            Some("active_users")
        );
        assert_eq!(
            chunks
                .iter()
                .find(|c| c.kind == ChunkKind::Function)
                .and_then(|c| c.name.as_deref()),
            Some("add")
        );
    }

    #[test]
    fn sql_chunker_extracts_trigger_name_not_table() {
        let source = "CREATE TRIGGER audit_ins AFTER INSERT ON accounts FOR EACH ROW EXECUTE FUNCTION log_change();\n";
        let trigger = sql_chunks(source)
            .into_iter()
            .find(|c| c.kind == ChunkKind::Trigger)
            .unwrap_or_else(|| unreachable!());
        assert_eq!(trigger.name.as_deref(), Some("audit_ins"));
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
    fn fallback_covers_every_line_and_partitions_without_overlap() {
        // Content-defined boundaries land where the content puts them, so assert
        // coverage and ordering rather than exact windows: with no overlap the
        // segments partition the file — first chunk at line 1, last at the final
        // line, each chunk resuming exactly where the previous ended, none over
        // the max bound.
        let source: String = (0..200)
            .map(|i| format!("distinct fallback line number {i}\n"))
            .collect();
        let chunks = chunk_fallback(
            &RelativePath::new("file.txt"),
            Language::Unknown,
            hash_for(&source),
            &source,
            30,
            0,
        )
        .ok()
        .unwrap_or_else(|| unreachable!());

        assert!(
            chunks.len() >= 2,
            "fixture must split, got {}",
            chunks.len()
        );
        assert_eq!(chunks[0].line_range.start, 1);
        assert_eq!(chunks[chunks.len() - 1].line_range.end, 200);
        for pair in chunks.windows(2) {
            assert_eq!(
                pair[1].line_range.start,
                pair[0].line_range.end + 1,
                "no-overlap segments must partition without gaps or overlap"
            );
            let span = pair[0].line_range.end - pair[0].line_range.start + 1;
            assert!(
                span <= 60,
                "segment of {span} lines exceeds the 2×target max"
            );
        }
    }

    #[test]
    fn fallback_overlap_shares_a_prefix_with_the_previous_chunk() {
        // Overlap pulls the previous segment's last `overlap` lines into each
        // chunk as a leading prefix, so adjacent chunks share exactly that many
        // lines. Segmentation (and thus chunk count) is unaffected by overlap.
        let source: String = (0..200)
            .map(|i| format!("distinct fallback line number {i}\n"))
            .collect();
        let overlap = 4_u32;
        let chunks = chunk_fallback(
            &RelativePath::new("file.txt"),
            Language::Unknown,
            hash_for(&source),
            &source,
            30,
            overlap as usize,
        )
        .ok()
        .unwrap_or_else(|| unreachable!());

        assert!(
            chunks.len() >= 2,
            "fixture must split, got {}",
            chunks.len()
        );
        for pair in chunks.windows(2) {
            let shared = pair[0].line_range.end - pair[1].line_range.start + 1;
            assert_eq!(
                shared, overlap,
                "each chunk must share exactly `overlap` lines with the prior"
            );
        }
    }

    #[test]
    fn fallback_insert_at_top_perturbs_a_bounded_chunk_set() {
        // Content-defined boundaries track what a line says, not where it sits,
        // so a one-line insert at the top of a fallback-chunked file leaves
        // every chunk below the insert byte-identical. Only the chunk holding
        // the insert re-hashes, which is what lets the change-neighbor snapshot
        // narrow to the edit. The old line-index splitter shifted every window
        // and re-hashed the whole file; this test reds against it.
        let base: String = (0..300)
            .map(|i| format!("unique fallback content line number {i} lorem ipsum dolor\n"))
            .collect();
        let inserted = format!("a freshly inserted top line that did not exist before\n{base}");

        let content_hashes = |src: &str| -> std::collections::HashSet<u128> {
            chunk_fallback(
                &RelativePath::new("notes.md"),
                Language::Unknown,
                hash_for(src),
                src,
                DEFAULT_CHUNK_LINES,
                DEFAULT_OVERLAP_LINES,
            )
            .ok()
            .unwrap_or_else(|| unreachable!())
            .iter()
            .map(|c| xxhash_rust::xxh3::xxh3_128(c.content.as_bytes()))
            .collect()
        };

        let before = content_hashes(&base);
        let after = content_hashes(&inserted);

        // A single-chunk file would pass the bound trivially, so require the
        // fixture to have split into several chunks first.
        assert!(
            before.len() >= 4,
            "fixture must produce several fallback chunks, got {}",
            before.len()
        );

        // Chunks whose exact content vanished after the insert. Content-defined
        // boundaries hold this to the touched chunk (and at most its overlap
        // neighbor) in the common case; the line-index splitter drops nearly all
        // of them. This fixture exercises that common case — a top insert has a
        // ~1/divisor chance of shifting the first segment's boundary and dropping
        // more (the residual documented in `subsystems/chunking.md`), so the
        // bound is typical behavior, not a universal proof.
        let dropped = before.difference(&after).count();
        assert!(
            dropped <= 2,
            "a top insert must perturb a bounded chunk set, dropped {dropped} of {}",
            before.len()
        );
    }

    #[test]
    fn python_chunker_returns_function_chunk() {
        let chunker = MultiLanguageChunker::default();
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
        let chunker = MultiLanguageChunker::default();
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

    // -----------------------------------------------------------------------
    // Java
    // -----------------------------------------------------------------------

    fn java_chunks(source: &str) -> Vec<Chunk> {
        MultiLanguageChunker::default()
            .chunk(
                &RelativePath::new("App.java"),
                Language::Java,
                hash_for(source),
                source,
            )
            .ok()
            .unwrap_or_else(|| unreachable!())
    }

    fn find_named(chunks: &[Chunk], kind: ChunkKind, name: &str) -> bool {
        chunks
            .iter()
            .any(|c| c.kind == kind && c.name.as_deref() == Some(name))
    }

    #[test]
    fn java_chunker_empty_returns_empty() {
        assert!(java_chunks("").is_empty());
    }

    #[test]
    fn java_chunker_extracts_class_and_method() {
        let source = "public class Hello {\n    public static void main(String[] args) {\n        System.out.println(\"Hello\");\n    }\n}\n";
        let chunks = java_chunks(source);

        assert!(
            find_named(&chunks, ChunkKind::Class, "Hello"),
            "class should be symbol-anchored, not a window"
        );
        assert!(find_named(&chunks, ChunkKind::Method, "main"));
        assert!(
            chunks.iter().all(|c| c.kind != ChunkKind::Other),
            "no sliding-window fallback chunks expected"
        );
    }

    #[test]
    fn java_chunker_interface_enum_and_record() {
        let source = "interface Greeter {\n    String greet();\n}\nenum Color { RED, GREEN }\npublic record Point(int x, int y) {}\n";
        let chunks = java_chunks(source);

        assert!(find_named(&chunks, ChunkKind::Interface, "Greeter"));
        assert!(find_named(&chunks, ChunkKind::Enum, "Color"));
        // A record is a class-like reference type.
        assert!(find_named(&chunks, ChunkKind::Class, "Point"));
    }

    #[test]
    fn java_chunker_extracts_nested_classes() {
        // Inner types are the same node kinds inside a class body; the recursive
        // walk must chunk both the outer and inner class plus their methods.
        let source = "public class Outer {\n    private int value;\n    public class Inner {\n        public int get() { return value; }\n    }\n    public void run() {}\n}\n";
        let chunks = java_chunks(source);

        assert!(find_named(&chunks, ChunkKind::Class, "Outer"));
        assert!(find_named(&chunks, ChunkKind::Class, "Inner"));
        assert!(find_named(&chunks, ChunkKind::Method, "get"));
        assert!(find_named(&chunks, ChunkKind::Method, "run"));
    }

    // -----------------------------------------------------------------------
    // C
    // -----------------------------------------------------------------------

    fn c_chunks(source: &str) -> Vec<Chunk> {
        MultiLanguageChunker::default()
            .chunk(
                &RelativePath::new("app.c"),
                Language::C,
                hash_for(source),
                source,
            )
            .ok()
            .unwrap_or_else(|| unreachable!())
    }

    #[test]
    fn c_chunker_empty_returns_empty() {
        assert!(c_chunks("").is_empty());
    }

    #[test]
    fn c_chunker_extracts_function_and_struct() {
        let source = "struct Point {\n    int x;\n    int y;\n};\nint add(int a, int b) {\n    return a + b;\n}\n";
        let chunks = c_chunks(source);

        assert!(
            find_named(&chunks, ChunkKind::Function, "add"),
            "function name lives in a nested declarator, must still resolve"
        );
        assert!(find_named(&chunks, ChunkKind::Struct, "Point"));
        assert!(chunks.iter().all(|c| c.kind != ChunkKind::Other));
    }

    #[test]
    fn c_chunker_pointer_return_function_name_resolves() {
        // The name sits under a `pointer_declarator`, one level deeper.
        let source = "char *dup(const char *s) {\n    return 0;\n}\n";
        assert!(find_named(&c_chunks(source), ChunkKind::Function, "dup"));
    }

    #[test]
    fn c_chunker_handles_preprocessor_heavy_source() {
        // Macros, includes and #ifdef guards must not derail symbol anchoring:
        // the guarded struct and the function still chunk, and macros chunk too.
        let source = "#include <stdio.h>\n#define MAX 100\n#define SQUARE(x) ((x) * (x))\n\n#ifdef FEATURE\nstruct Config {\n    int level;\n};\n#endif\n\nint compute(int n) {\n    return SQUARE(n) + MAX;\n}\n";
        let chunks = c_chunks(source);

        assert!(find_named(&chunks, ChunkKind::Function, "compute"));
        assert!(
            find_named(&chunks, ChunkKind::Struct, "Config"),
            "struct inside #ifdef must still be chunked"
        );
        assert!(find_named(&chunks, ChunkKind::Macro, "SQUARE"));
        assert!(find_named(&chunks, ChunkKind::Macro, "MAX"));
    }

    // -----------------------------------------------------------------------
    // C++
    // -----------------------------------------------------------------------

    fn cpp_chunks(source: &str) -> Vec<Chunk> {
        MultiLanguageChunker::default()
            .chunk(
                &RelativePath::new("app.cpp"),
                Language::Cpp,
                hash_for(source),
                source,
            )
            .ok()
            .unwrap_or_else(|| unreachable!())
    }

    #[test]
    fn cpp_chunker_empty_returns_empty() {
        assert!(cpp_chunks("").is_empty());
    }

    #[test]
    fn cpp_chunker_class_with_inline_method() {
        let source = "class Widget {\npublic:\n    void draw() {}\n};\nvoid render() {}\n";
        let chunks = cpp_chunks(source);

        assert!(find_named(&chunks, ChunkKind::Class, "Widget"));
        // Inline member function is a method; the free function is a function.
        assert!(find_named(&chunks, ChunkKind::Method, "draw"));
        assert!(find_named(&chunks, ChunkKind::Function, "render"));
        assert!(chunks.iter().all(|c| c.kind != ChunkKind::Other));
    }

    #[test]
    fn cpp_chunker_handles_templates_and_namespaces() {
        // Templates wrap the class/function node and namespaces nest them; the
        // walk must still reach every symbol with its name and kind.
        let source = "namespace geo {\ntemplate <typename T>\nclass Point {\npublic:\n    T norm() const { return T(); }\n};\n\ntemplate <typename T>\nT dot(Point<T> a, Point<T> b) { return T(); }\n}\n";
        let chunks = cpp_chunks(source);

        assert!(find_named(&chunks, ChunkKind::Module, "geo"));
        assert!(find_named(&chunks, ChunkKind::Class, "Point"));
        assert!(
            find_named(&chunks, ChunkKind::Method, "norm"),
            "templated member function must chunk as a method"
        );
        assert!(
            find_named(&chunks, ChunkKind::Function, "dot"),
            "templated free function must chunk as a function"
        );
    }
}
