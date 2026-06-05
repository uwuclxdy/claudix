use crate::error::{ClaudixError, RecoveryHint, Result};
use crate::types::{Dimension, EmbeddedChunk};

/// Lightweight projection of a stored chunk that omits the embedding vector.
///
/// Used by read paths that need only scalar metadata — overview, incremental
/// hash comparison, etc. — so the large float arrays are never loaded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkMetadata {
    pub file_path: String,
    pub file_hash: [u8; 16],
    pub language: String,
    pub name: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StoredChunk {
    pub chunk_id: u64,
    pub file_path: String,
    pub language: String,
    pub kind: String,
    pub name: Option<String>,
    pub line_start: u32,
    pub line_end: u32,
    pub byte_start: u32,
    pub byte_end: u32,
    pub file_hash: [u8; 16],
    pub content: String,
    pub vector: Vec<f32>,
}

impl StoredChunk {
    pub(super) fn from_embedded_chunk(chunk: &EmbeddedChunk, dimension: Dimension) -> Result<Self> {
        validate_vector(&chunk.vector, dimension)?;

        Ok(Self {
            chunk_id: chunk.chunk.id.0,
            file_path: chunk.chunk.file_path.as_str().to_owned(),
            language: chunk.chunk.language.to_string(),
            kind: chunk.chunk.kind.to_string(),
            name: chunk.chunk.name.clone(),
            line_start: chunk.chunk.line_range.start,
            line_end: chunk.chunk.line_range.end,
            byte_start: chunk.chunk.byte_range.start,
            byte_end: chunk.chunk.byte_range.end,
            file_hash: chunk.chunk.file_hash.0,
            content: chunk.chunk.content.clone(),
            vector: chunk.vector.clone(),
        })
    }
}

pub(crate) fn stored_chunks_from_embedded(
    chunks: &[EmbeddedChunk],
    dimension: Dimension,
) -> Result<Vec<StoredChunk>> {
    chunks
        .iter()
        .map(|chunk| StoredChunk::from_embedded_chunk(chunk, dimension))
        .collect()
}

pub(super) fn validate_vector(vector: &[f32], dimension: Dimension) -> Result<()> {
    if vector.len() != usize::from(dimension.0) {
        return Err(ClaudixError::DimensionMismatch {
            store_dim: dimension.0,
            model_dim: u16::try_from(vector.len()).unwrap_or(u16::MAX),
            recovery: RecoveryHint(
                "Reindex the project after aligning embedding dimensions with the active model",
            ),
        });
    }

    if vector.iter().any(|value| !value.is_finite()) {
        return Err(ClaudixError::Store(
            "embedding vector contains non-finite values".to_owned(),
        ));
    }

    Ok(())
}
