pub mod chunking;
pub mod config;
pub mod embedding;
pub mod enumeration;
pub mod error;
pub mod hooks;
pub mod mcp;
pub mod search;
pub mod store;
pub mod types;

pub use error::{ClaudixError, Result};
pub use types::{
    ByteRange, Chunk, ChunkId, ChunkKind, Dimension, EmbeddedChunk, FileHash, Language, LineRange,
    RelativePath,
};

use std::path::PathBuf;
use std::sync::Arc;

/// Root composition object — owns all domain components.
pub struct Claudix {
    pub config: Arc<config::Config>,
    pub project_root: PathBuf,
}

impl Claudix {
    pub async fn new(project_root: PathBuf, config: Arc<config::Config>) -> Result<Self> {
        Ok(Self {
            config,
            project_root,
        })
    }
}
