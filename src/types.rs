use std::fmt;
use std::path::{Path, PathBuf};

/// Deterministic chunk identifier derived from (file_hash, byte_range).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ChunkId(pub u64);

impl fmt::Display for ChunkId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// 16-byte xxh3 file hash.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct FileHash(pub [u8; 16]);

impl fmt::Display for FileHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for b in &self.0 {
            write!(f, "{b:02x}")?;
        }
        Ok(())
    }
}

/// Embedding vector dimensionality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Dimension(pub u16);

/// Repo-relative path, always forward-slash normalized.
#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct RelativePath(String);

pub(crate) fn path_prefix_matches(path: &str, prefix: &str) -> bool {
    let Some(rest) = path.strip_prefix(prefix) else {
        return false;
    };
    rest.is_empty() || rest.starts_with('/') || rest.starts_with('.')
}

impl RelativePath {
    pub fn new(s: impl Into<String>) -> Self {
        let raw = s.into();
        let normalized = raw.replace('\\', "/");
        Self(normalized)
    }

    pub fn from_path(path: &Path) -> Self {
        Self::new(path.to_string_lossy().as_ref())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn starts_with(&self, prefix: &RelativePath) -> bool {
        path_prefix_matches(&self.0, prefix.as_str())
    }

    pub fn to_path_buf(&self) -> PathBuf {
        PathBuf::from(&self.0)
    }
}

impl fmt::Display for RelativePath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Byte range within a file (0-indexed, half-open).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ByteRange {
    pub start: u32,
    pub end: u32,
}

/// Line range within a file (1-indexed, inclusive).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LineRange {
    pub start: u32,
    pub end: u32,
}

/// Source language.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Language {
    Rust,
    Python,
    JavaScript,
    TypeScript,
    Go,
    Java,
    C,
    Cpp,
    Unknown,
}

impl Language {
    pub fn from_extension(ext: &str) -> Self {
        match ext {
            "rs" => Self::Rust,
            "py" => Self::Python,
            "js" | "mjs" | "cjs" => Self::JavaScript,
            "ts" | "tsx" => Self::TypeScript,
            "go" => Self::Go,
            "java" => Self::Java,
            "c" | "h" => Self::C,
            "cpp" | "cc" | "cxx" | "hpp" | "hxx" => Self::Cpp,
            _ => Self::Unknown,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Rust => "rust",
            Self::Python => "python",
            Self::JavaScript => "javascript",
            Self::TypeScript => "typescript",
            Self::Go => "go",
            Self::Java => "java",
            Self::C => "c",
            Self::Cpp => "cpp",
            Self::Unknown => "unknown",
        }
    }
}

impl fmt::Display for Language {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Semantic kind of a code chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChunkKind {
    Function,
    Method,
    Struct,
    Class,
    Enum,
    Trait,
    Interface,
    Module,
    Impl,
    Macro,
    Other,
}

impl ChunkKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Function => "function",
            Self::Method => "method",
            Self::Struct => "struct",
            Self::Class => "class",
            Self::Enum => "enum",
            Self::Trait => "trait",
            Self::Interface => "interface",
            Self::Module => "module",
            Self::Impl => "impl",
            Self::Macro => "macro",
            Self::Other => "other",
        }
    }
}

impl fmt::Display for ChunkKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Fundamental indexable unit of source code.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Chunk {
    pub id: ChunkId,
    pub file_path: RelativePath,
    pub language: Language,
    pub kind: ChunkKind,
    pub name: Option<String>,
    pub line_range: LineRange,
    pub byte_range: ByteRange,
    pub file_hash: FileHash,
    pub content: String,
}

/// Chunk with its embedding vector.
#[derive(Debug, Clone)]
pub struct EmbeddedChunk {
    pub chunk: Chunk,
    pub vector: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relative_path_normalizes_windows_separators() {
        let path = RelativePath::new(r"src\nested\file.rs");

        assert_eq!(path.as_str(), "src/nested/file.rs");
        assert_eq!(path.to_string(), "src/nested/file.rs");
    }

    #[test]
    fn relative_path_prefix_match_uses_normalized_form() {
        let path = RelativePath::new("src/nested/file.rs");
        let prefix = RelativePath::new(r"src\nested");

        assert!(path.starts_with(&prefix));
    }

    #[test]
    fn file_hash_displays_as_lowercase_hex() {
        let hash = FileHash([0xAB; 16]);

        assert_eq!(hash.to_string(), "abababababababababababababababab");
    }

    #[test]
    fn chunk_id_display_is_decimal() {
        assert_eq!(ChunkId(42).to_string(), "42");
    }

    #[test]
    fn language_detects_known_extensions() {
        assert_eq!(Language::from_extension("rs"), Language::Rust);
        assert_eq!(Language::from_extension("tsx"), Language::TypeScript);
        assert_eq!(Language::from_extension("hpp"), Language::Cpp);
        assert_eq!(Language::from_extension("unknown"), Language::Unknown);
    }

    #[test]
    fn language_display_matches_serialized_name() {
        assert_eq!(Language::JavaScript.to_string(), "javascript");
        assert_eq!(Language::Unknown.to_string(), "unknown");
    }

    #[test]
    fn chunk_kind_display_matches_kind_name() {
        assert_eq!(ChunkKind::Function.to_string(), "function");
        assert_eq!(ChunkKind::Macro.to_string(), "macro");
        assert_eq!(ChunkKind::Other.to_string(), "other");
    }

    #[test]
    fn relative_path_starts_with_respects_segment_boundary() {
        let prefix = RelativePath::new("src/math");
        assert!(RelativePath::new("src/math.rs").starts_with(&prefix));
        assert!(RelativePath::new("src/math/util.rs").starts_with(&prefix));
        assert!(RelativePath::new("src/math").starts_with(&prefix));
        assert!(!RelativePath::new("src/mathematics.rs").starts_with(&prefix));
        assert!(!RelativePath::new("src/mathx").starts_with(&prefix));
    }
}
