use crate::chunking::DEFAULT_CHUNK_LINES;
use crate::error::{ClaudixError, RecoveryHint, Result};
use crate::prompts::hints;

use super::{Config, EmbeddingProvider, validate_project_relative_path};

pub fn validate(config: &Config) -> Result<()> {
    validate_project_relative_path(&config.paths.index_dir, "paths.index_dir")?;
    validate_project_relative_path(&config.paths.log_dir, "paths.log_dir")?;

    let allow_stub_embedding_model = allow_stub_embedding_model(config);

    if matches!(config.embedding.provider, EmbeddingProvider::Http)
        && config.embedding.endpoint.trim().is_empty()
    {
        return Err(ClaudixError::ConfigInvalid {
            message: "embedding.endpoint is required when embedding.provider = \"http\"".into(),
            recovery: RecoveryHint(hints::SET_ENDPOINT),
        });
    }

    if matches!(config.embedding.provider, EmbeddingProvider::Bundled)
        && !allow_stub_embedding_model
        && config.embedding.model != crate::embedding::bundled::BUNDLED_MODEL_ID
    {
        return Err(ClaudixError::ConfigInvalid {
            message: format!(
                "embedding.model must be {} when embedding.provider = \"bundled\"",
                crate::embedding::bundled::BUNDLED_MODEL_ID
            ),
            recovery: RecoveryHint(hints::SET_MODEL_BUNDLED),
        });
    }

    if matches!(config.embedding.provider, EmbeddingProvider::Bundled)
        && !allow_stub_embedding_model
        && config.embedding.dimensions != crate::embedding::bundled::BUNDLED_DIMENSIONS.0
    {
        return Err(ClaudixError::ConfigInvalid {
            message: format!(
                "embedding.dimensions must be {} when embedding.provider = \"bundled\"",
                crate::embedding::bundled::BUNDLED_DIMENSIONS.0
            ),
            recovery: RecoveryHint(hints::BUNDLED_DIMENSIONS_384),
        });
    }

    if config.embedding.dimensions == 0 {
        return Err(ClaudixError::ConfigInvalid {
            message: "embedding.dimensions must be > 0".into(),
            recovery: RecoveryHint(hints::SET_DIMENSIONS_POSITIVE),
        });
    }

    if config.embedding.batch_size == 0 {
        return Err(ClaudixError::ConfigInvalid {
            message: "embedding.batch_size must be > 0".into(),
            recovery: RecoveryHint(hints::SET_BATCH_SIZE),
        });
    }

    if config.embedding.timeout_ms == 0 {
        return Err(ClaudixError::ConfigInvalid {
            message: "embedding.timeout_ms must be > 0".into(),
            recovery: RecoveryHint(hints::SET_TIMEOUT_MS),
        });
    }

    if config.indexing.max_file_size_kb == 0 {
        return Err(ClaudixError::ConfigInvalid {
            message: "indexing.max_file_size_kb must be > 0".into(),
            recovery: RecoveryHint(hints::SET_MAX_FILE_SIZE_KB),
        });
    }

    if config.indexing.chunk_overlap_lines >= DEFAULT_CHUNK_LINES {
        return Err(ClaudixError::ConfigInvalid {
            message: format!(
                "indexing.chunk_overlap_lines must be less than {DEFAULT_CHUNK_LINES}"
            ),
            recovery: RecoveryHint(hints::SET_CHUNK_OVERLAP),
        });
    }

    if config.indexing.reindex_after_hours == 0 {
        return Err(ClaudixError::ConfigInvalid {
            message: "indexing.reindex_after_hours must be > 0".into(),
            recovery: RecoveryHint(hints::SET_REINDEX_AFTER_HOURS),
        });
    }

    if config.hooks.related_top_k == 0 {
        return Err(ClaudixError::ConfigInvalid {
            message: "hooks.related_top_k must be > 0".into(),
            recovery: RecoveryHint(hints::SET_RELATED_TOP_K),
        });
    }

    if !(0.0..=1.0).contains(&config.hooks.related_min_similarity) {
        return Err(ClaudixError::ConfigInvalid {
            message: "hooks.related_min_similarity must be between 0 and 1".into(),
            recovery: RecoveryHint(hints::SET_RELATED_MIN_SIMILARITY),
        });
    }

    if config.search.top_k == 0 {
        return Err(ClaudixError::ConfigInvalid {
            message: "search.top_k must be > 0".into(),
            recovery: RecoveryHint(hints::SET_SEARCH_TOP_K),
        });
    }

    if !(0.0..=1.0).contains(&config.search.similarity_threshold) {
        return Err(ClaudixError::ConfigInvalid {
            message: "search.similarity_threshold must be between 0 and 1".into(),
            recovery: RecoveryHint(hints::SET_SIMILARITY_THRESHOLD),
        });
    }

    if !(0.0..=1.0).contains(&config.search.min_score) {
        return Err(ClaudixError::ConfigInvalid {
            message: "search.min_score must be between 0 and 1".into(),
            recovery: RecoveryHint(hints::SET_MIN_SCORE),
        });
    }

    if config
        .search
        .cross_repos
        .iter()
        .any(|repo| repo.trim().is_empty())
    {
        return Err(ClaudixError::ConfigInvalid {
            message: "search.cross_repos entries must be non-empty paths".into(),
            recovery: RecoveryHint(hints::REMOVE_EMPTY_CROSS_REPOS),
        });
    }

    if !config.search.identifier_boost.is_finite() || config.search.identifier_boost <= 0.0 {
        return Err(ClaudixError::ConfigInvalid {
            message: "search.identifier_boost must be a finite positive number".into(),
            recovery: RecoveryHint(hints::SET_IDENTIFIER_BOOST),
        });
    }

    let weights = &config.search.hybrid_weights;
    if !weights.dense.is_finite() || !weights.bm25.is_finite() || !weights.rrf.is_finite() {
        return Err(ClaudixError::ConfigInvalid {
            message: "search.hybrid_weights components must be finite numbers".into(),
            recovery: RecoveryHint(hints::SET_HYBRID_WEIGHTS_FINITE),
        });
    }
    if weights.dense < 0.0 || weights.bm25 < 0.0 || weights.rrf < 0.0 {
        return Err(ClaudixError::ConfigInvalid {
            message: "search.hybrid_weights components must be >= 0".into(),
            recovery: RecoveryHint(hints::SET_HYBRID_WEIGHTS_NON_NEGATIVE),
        });
    }
    if weights.dense == 0.0 && weights.bm25 == 0.0 && weights.rrf == 0.0 {
        return Err(ClaudixError::ConfigInvalid {
            message: "search.hybrid_weights must not all be zero".into(),
            recovery: RecoveryHint(hints::SET_HYBRID_WEIGHTS_NONZERO),
        });
    }

    Ok(())
}

#[cfg(any(test, feature = "test-stub"))]
fn allow_stub_embedding_model(config: &Config) -> bool {
    config.embedding.model.starts_with("stub")
}

#[cfg(not(any(test, feature = "test-stub")))]
fn allow_stub_embedding_model(_config: &Config) -> bool {
    false
}
