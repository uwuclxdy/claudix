use std::path::PathBuf;
use thiserror::Error;

use crate::prompts::hints;

#[derive(Debug, Clone, Copy)]
pub struct RecoveryHint(pub &'static str);

impl std::fmt::Display for RecoveryHint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}

#[derive(Debug, Error)]
pub enum ClaudixError {
    #[error("config invalid: {message}")]
    ConfigInvalid {
        message: String,
        recovery: RecoveryHint,
    },

    #[error("embedding endpoint unreachable: {endpoint}")]
    EmbeddingUnreachable {
        endpoint: String,
        #[source]
        source: reqwest::Error,
        recovery: RecoveryHint,
    },

    #[error("embedding endpoint timed out after {timeout_ms}ms: {endpoint}")]
    EmbeddingTimedOut {
        endpoint: String,
        timeout_ms: u64,
        recovery: RecoveryHint,
    },

    #[error("embedding endpoint requires auth (HTTP {status}): {endpoint}")]
    EmbeddingAuthRejected {
        endpoint: String,
        status: u16,
        recovery: RecoveryHint,
    },

    #[error("embedding endpoint returned HTTP {status}: {endpoint}")]
    EmbeddingHttpStatus {
        endpoint: String,
        status: u16,
        recovery: RecoveryHint,
    },

    // 200 with a body the provider cannot decode, or a decode that violates the
    // OpenAI embeddings shape (count mismatch, invalid index, non-finite
    // values). Treated as endpoint-unavailable so FallbackProvider switches to
    // bundled instead of aborting the call: LM Studio returns 200 + HTML during
    // warm-up, and a model-loading server can silently emit truncated batches.
    #[error("embedding endpoint returned a bad payload: {endpoint}")]
    EmbeddingEndpointBadPayload {
        endpoint: String,
        recovery: RecoveryHint,
    },

    #[error("schema version mismatch: store={store}, binary={binary}")]
    SchemaMismatch {
        store: u32,
        binary: u32,
        recovery: RecoveryHint,
    },

    #[error("dimension mismatch: store={store_dim}, model={model_dim}")]
    DimensionMismatch {
        store_dim: u16,
        model_dim: u16,
        recovery: RecoveryHint,
    },

    #[error("embedding model mismatch: store={store_model}, active={active_model}")]
    EmbeddingModelMismatch {
        store_model: String,
        active_model: String,
        recovery: RecoveryHint,
    },

    #[error("path traversal: {path:?} is outside project root")]
    PathTraversal {
        path: PathBuf,
        recovery: RecoveryHint,
    },

    #[error("bundled assets missing for model {model_id}")]
    BundledAssetsMissing {
        model_id: String,
        recovery: RecoveryHint,
    },

    #[error(
        "bundled asset from {url} failed verification: expected sha256 {expected}, got {actual}"
    )]
    BundledAssetCorrupt {
        url: String,
        expected: String,
        actual: String,
        recovery: RecoveryHint,
    },

    #[error("embedding error: {0}")]
    Embedding(String),

    #[error("tree-sitter error: {0}")]
    TreeSitter(String),

    #[error("store error: {0}")]
    Store(String),

    #[error("mcp server error: {0}")]
    Mcp(String),

    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("lance error: {0}")]
    Lance(#[from] lancedb::Error),

    #[error("not a git repository: {path:?}")]
    NotAGitRepository {
        path: PathBuf,
        recovery: RecoveryHint,
    },

    #[error("git enumeration error: {0}")]
    Git(String),

    #[error("ignore pattern error: {0}")]
    Ignore(#[from] ignore::Error),

    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),
}

impl ClaudixError {
    pub fn recovery_hint(&self) -> Option<&'static str> {
        match self {
            Self::ConfigInvalid { recovery, .. } => Some(recovery.0),
            Self::EmbeddingUnreachable { recovery, .. } => Some(recovery.0),
            Self::EmbeddingTimedOut { recovery, .. } => Some(recovery.0),
            Self::EmbeddingAuthRejected { recovery, .. } => Some(recovery.0),
            Self::EmbeddingHttpStatus { recovery, .. } => Some(recovery.0),
            Self::EmbeddingEndpointBadPayload { recovery, .. } => Some(recovery.0),
            Self::SchemaMismatch { recovery, .. } => Some(recovery.0),
            Self::DimensionMismatch { recovery, .. } => Some(recovery.0),
            Self::EmbeddingModelMismatch { recovery, .. } => Some(recovery.0),
            Self::NotAGitRepository { recovery, .. } => Some(recovery.0),
            Self::PathTraversal { recovery, .. } => Some(recovery.0),
            Self::BundledAssetsMissing { recovery, .. } => Some(recovery.0),
            Self::BundledAssetCorrupt { recovery, .. } => Some(recovery.0),
            // High-frequency runtime variants: return actionable hints.
            Self::Io(_) => Some(hints::IO_CHECK_DISK),
            Self::Store(_) => Some(hints::STORE_DOCTOR),
            Self::Lance(_) => Some(hints::LANCE_DOCTOR),
            Self::Git(_) => Some(hints::GIT_ENUM_DOCTOR),
            Self::Ignore(_) => Some(hints::IGNORE_PATTERN),
            Self::Http(_) => Some(hints::HTTP_DOCTOR),
            Self::Embedding(_) => Some(hints::EMBEDDING_GENERIC),
            Self::TreeSitter(_) => Some(hints::TREE_SITTER_REINDEX),
            // JSON serialization errors are not user-actionable; no hint.
            Self::Serialization(_) => None,
            // A failed MCP serve/transport is environmental (stdio closed, init
            // refused), not fixable via a recovery hint.
            Self::Mcp(_) => None,
        }
    }

    /// True when the HTTP embedding endpoint is unusable this session — offline,
    /// too slow, rejecting auth, or erroring — so a fallback provider should take
    /// over rather than aborting.
    pub fn is_endpoint_unavailable(&self) -> bool {
        matches!(
            self,
            Self::EmbeddingUnreachable { .. }
                | Self::EmbeddingTimedOut { .. }
                | Self::EmbeddingAuthRejected { .. }
                | Self::EmbeddingHttpStatus { .. }
                | Self::EmbeddingEndpointBadPayload { .. }
        )
    }
}

pub type Result<T> = std::result::Result<T, ClaudixError>;
