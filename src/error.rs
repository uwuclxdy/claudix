use std::path::PathBuf;
use thiserror::Error;

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

    #[error("bundled download confirmation required for model {model_id}")]
    BundledDownloadConfirmationRequired {
        model_id: String,
        recovery: RecoveryHint,
    },

    #[error("sha256 mismatch for bundled asset {asset}")]
    BundledAssetChecksumMismatch {
        asset: String,
        expected_sha256: String,
        actual_sha256: String,
        recovery: RecoveryHint,
    },

    #[error("embedding error: {0}")]
    Embedding(String),

    #[error("tree-sitter error: {0}")]
    TreeSitter(String),

    #[error("store error: {0}")]
    Store(String),

    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("lance error: {0}")]
    Lance(#[from] lancedb::Error),

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
            Self::SchemaMismatch { recovery, .. } => Some(recovery.0),
            Self::DimensionMismatch { recovery, .. } => Some(recovery.0),
            Self::EmbeddingModelMismatch { recovery, .. } => Some(recovery.0),
            Self::PathTraversal { recovery, .. } => Some(recovery.0),
            Self::BundledAssetsMissing { recovery, .. } => Some(recovery.0),
            Self::BundledDownloadConfirmationRequired { recovery, .. } => Some(recovery.0),
            Self::BundledAssetChecksumMismatch { recovery, .. } => Some(recovery.0),
            _ => None,
        }
    }
}

pub type Result<T> = std::result::Result<T, ClaudixError>;
