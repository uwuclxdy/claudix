use std::fmt;
use std::path::{Component, Path, PathBuf};

use crate::error::{ClaudixError, RecoveryHint, Result};

/// Content-addressed chunk identifier: xxh3 of `(file_hash, byte_range)`.
///
/// Deliberately NOT row-unique. `FileHash` is xxh3 of file content, so two
/// different files with byte-identical content and matching byte ranges hash to
/// the same `ChunkId`. Storage keys rows by `(file_path, byte_start, chunk_id)`
/// — `file_path` disambiguates such collisions — so this is correct for the
/// current store. Do NOT promote `ChunkId` to a standalone primary or dedup key
/// without folding `file_path` into the hash first, or cross-file duplicates
/// will silently merge.
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
///
/// INVARIANT (advisory): `new`/`from_path` only normalize separators — they do
/// NOT enforce relativity. An absolute (`/etc/passwd`) or escaping (`../..`,
/// `C:\…`) string is accepted as-is. Any caller that joins this against a root
/// or crosses a trust boundary (hook payloads, stored rows, tool input) MUST
/// call [`RelativePath::reject_escape`] first to keep operations inside
/// `$CLAUDE_PROJECT_DIR`.
#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct RelativePath(String);

pub(crate) fn path_prefix_matches(path: &str, prefix: &str) -> bool {
    let prefix = prefix.trim_end_matches('/');
    let Some(rest) = path.strip_prefix(prefix) else {
        return false;
    };
    rest.is_empty()
        || rest.starts_with('/')
        || rest
            .strip_prefix('.')
            .is_some_and(|extension| !extension.contains(['/', '.']))
}

/// Reject absolute paths and parent-dir / root-dir / drive-prefix components.
///
/// Used wherever a project-relative path crosses a trust boundary (config
/// keys, hook payloads, search results) to prevent operations outside
/// `$CLAUDE_PROJECT_DIR`. `recovery` is the hint surfaced to the user when
/// rejection fires.
pub(crate) fn reject_path_escape(path: &Path, recovery: &'static str) -> Result<()> {
    if path.is_absolute() {
        return Err(ClaudixError::PathTraversal {
            path: path.to_path_buf(),
            recovery: RecoveryHint(recovery),
        });
    }
    for component in path.components() {
        if matches!(
            component,
            Component::ParentDir | Component::RootDir | Component::Prefix(_)
        ) {
            return Err(ClaudixError::PathTraversal {
                path: path.to_path_buf(),
                recovery: RecoveryHint(recovery),
            });
        }
    }
    Ok(())
}

impl RelativePath {
    pub(crate) fn reject_escape(&self, recovery: &'static str) -> Result<()> {
        reject_path_escape(&self.to_path_buf(), recovery)
    }

    /// Normalize separators only. Does NOT enforce relativity — see the type
    /// docs; trust-boundary callers must follow with [`Self::reject_escape`].
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

    /// Parse the canonical form used in storage / serialized chunks. Unknown
    /// strings round-trip to `Unknown` so corrupt data doesn't propagate.
    pub fn from_storage(value: &str) -> Self {
        match value {
            "rust" => Self::Rust,
            "python" => Self::Python,
            "javascript" => Self::JavaScript,
            "typescript" => Self::TypeScript,
            "go" => Self::Go,
            "java" => Self::Java,
            "c" => Self::C,
            "cpp" => Self::Cpp,
            _ => Self::Unknown,
        }
    }

    /// Parse a user-supplied language filter (CLI / MCP input). Accepts
    /// short aliases the storage form doesn't carry.
    pub fn from_filter_input(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "rust" => Some(Self::Rust),
            "python" => Some(Self::Python),
            "javascript" | "js" => Some(Self::JavaScript),
            "typescript" | "ts" => Some(Self::TypeScript),
            "go" => Some(Self::Go),
            "java" => Some(Self::Java),
            "c" => Some(Self::C),
            "cpp" | "c++" => Some(Self::Cpp),
            "unknown" => Some(Self::Unknown),
            _ => None,
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

    /// Parse the canonical form used in storage. Unknown strings map to
    /// `Other` so legacy or corrupt rows don't break read paths.
    pub fn from_storage(value: &str) -> Self {
        match value {
            "function" => Self::Function,
            "method" => Self::Method,
            "struct" => Self::Struct,
            "class" => Self::Class,
            "enum" => Self::Enum,
            "trait" => Self::Trait,
            "interface" => Self::Interface,
            "module" => Self::Module,
            "impl" => Self::Impl,
            "macro" => Self::Macro,
            _ => Self::Other,
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
    fn path_prefix_matches_exact_directory_and_extension_boundaries() {
        assert!(path_prefix_matches("src/math", "src/math"));
        assert!(path_prefix_matches("src/math/add.rs", "src/math"));
        assert!(path_prefix_matches("src/math.rs", "src/math"));
        assert!(!path_prefix_matches("src/math_extra.rs", "src/math"));
        assert!(!path_prefix_matches("src/math.rs.bak", "src/math"));
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

        let prefix_slash = RelativePath::new("src/math/");
        assert!(RelativePath::new("src/math/util.rs").starts_with(&prefix_slash));
        assert!(!RelativePath::new("src/mathematics.rs").starts_with(&prefix_slash));
    }
}
