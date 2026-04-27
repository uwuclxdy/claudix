pub const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Manifest {
    pub schema_version: u32,
    pub embedding_model: String,
    pub dimensions: u16,
    pub last_full_index_at: Option<String>,
    pub last_incremental_at: Option<String>,
    pub chunk_count: u64,
    pub file_count: u64,
}
