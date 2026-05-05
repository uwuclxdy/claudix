use serde::Deserialize;

use super::EmbeddingProvider;

#[derive(Debug, Clone, Default, Deserialize)]
pub struct PartialConfig {
    #[serde(default)]
    pub embedding: PartialEmbeddingConfig,
    #[serde(default)]
    pub indexing: PartialIndexingConfig,
    #[serde(default)]
    pub search: PartialSearchConfig,
    #[serde(default)]
    pub hooks: PartialHooksConfig,
    #[serde(default)]
    pub paths: PartialPathsConfig,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct PartialEmbeddingConfig {
    #[serde(default)]
    pub provider: Option<EmbeddingProvider>,
    #[serde(default)]
    pub endpoint: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub dimensions: Option<u16>,
    #[serde(default)]
    pub batch_size: Option<usize>,
    #[serde(default)]
    pub timeout_ms: Option<u64>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct PartialIndexingConfig {
    #[serde(default)]
    pub respect_gitignore: Option<bool>,
    #[serde(default)]
    pub follow_symlinks: Option<bool>,
    #[serde(default)]
    pub max_file_size_kb: Option<u64>,
    #[serde(default)]
    pub chunk_overlap_lines: Option<usize>,
    #[serde(default)]
    pub reindex_after_hours: Option<u64>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct PartialSearchConfig {
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub hybrid_weights: PartialHybridWeights,
    #[serde(default)]
    pub identifier_boost: Option<f32>,
    #[serde(default)]
    pub similarity_threshold: Option<f32>,
    #[serde(default)]
    pub min_score: Option<f32>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct PartialHybridWeights {
    #[serde(default)]
    pub dense: Option<f32>,
    #[serde(default)]
    pub bm25: Option<f32>,
    #[serde(default)]
    pub rrf: Option<f32>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct PartialHooksConfig {
    #[serde(default)]
    pub intercept_grep: Option<bool>,
    #[serde(default)]
    pub auto_reembed_on_edit: Option<bool>,
    #[serde(default)]
    pub auto_index_on_session_start: Option<bool>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct PartialPathsConfig {
    #[serde(default)]
    pub index_dir: Option<String>,
    #[serde(default)]
    pub log_dir: Option<String>,
}

impl PartialConfig {
    pub fn merge(self, other: Self) -> Self {
        Self {
            embedding: self.embedding.merge(other.embedding),
            indexing: self.indexing.merge(other.indexing),
            search: self.search.merge(other.search),
            hooks: self.hooks.merge(other.hooks),
            paths: self.paths.merge(other.paths),
        }
    }
}

impl PartialEmbeddingConfig {
    fn merge(self, other: Self) -> Self {
        Self {
            provider: other.provider.or(self.provider),
            endpoint: other.endpoint.or(self.endpoint),
            model: other.model.or(self.model),
            dimensions: other.dimensions.or(self.dimensions),
            batch_size: other.batch_size.or(self.batch_size),
            timeout_ms: other.timeout_ms.or(self.timeout_ms),
        }
    }
}

impl PartialIndexingConfig {
    fn merge(self, other: Self) -> Self {
        Self {
            respect_gitignore: other.respect_gitignore.or(self.respect_gitignore),
            follow_symlinks: other.follow_symlinks.or(self.follow_symlinks),
            max_file_size_kb: other.max_file_size_kb.or(self.max_file_size_kb),
            chunk_overlap_lines: other.chunk_overlap_lines.or(self.chunk_overlap_lines),
            reindex_after_hours: other.reindex_after_hours.or(self.reindex_after_hours),
        }
    }
}

impl PartialSearchConfig {
    fn merge(self, other: Self) -> Self {
        Self {
            top_k: other.top_k.or(self.top_k),
            hybrid_weights: self.hybrid_weights.merge(other.hybrid_weights),
            identifier_boost: other.identifier_boost.or(self.identifier_boost),
            similarity_threshold: other.similarity_threshold.or(self.similarity_threshold),
            min_score: other.min_score.or(self.min_score),
        }
    }
}

impl PartialHybridWeights {
    fn merge(self, other: Self) -> Self {
        Self {
            dense: other.dense.or(self.dense),
            bm25: other.bm25.or(self.bm25),
            rrf: other.rrf.or(self.rrf),
        }
    }
}

impl PartialHooksConfig {
    fn merge(self, other: Self) -> Self {
        Self {
            intercept_grep: other.intercept_grep.or(self.intercept_grep),
            auto_reembed_on_edit: other.auto_reembed_on_edit.or(self.auto_reembed_on_edit),
            auto_index_on_session_start: other
                .auto_index_on_session_start
                .or(self.auto_index_on_session_start),
        }
    }
}

impl PartialPathsConfig {
    fn merge(self, other: Self) -> Self {
        Self {
            index_dir: other.index_dir.or(self.index_dir),
            log_dir: other.log_dir.or(self.log_dir),
        }
    }
}
