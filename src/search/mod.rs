use crate::types::{Chunk, Language, RelativePath};

#[derive(Debug, Clone)]
pub struct SearchQuery {
    pub query: String,
    pub top_k: usize,
    pub language_filter: Option<Vec<Language>>,
    pub path_prefix: Option<RelativePath>,
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub chunk: Chunk,
    pub score: f32,
}
