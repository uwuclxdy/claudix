use crate::error::{ClaudixError, RecoveryHint, Result};

use super::{Config, EmbeddingProvider, validate_project_relative_path};

pub fn validate(config: &Config) -> Result<()> {
    validate_project_relative_path(&config.paths.index_dir, "paths.index_dir")?;
    validate_project_relative_path(&config.paths.log_dir, "paths.log_dir")?;

    if allow_stub_embedding_model(config) {
        return Ok(());
    }

    if matches!(config.embedding.provider, EmbeddingProvider::Http)
        && config.embedding.endpoint.trim().is_empty()
    {
        return Err(ClaudixError::ConfigInvalid {
            message: "embedding.endpoint is required when embedding.provider = \"http\"".into(),
            recovery: RecoveryHint("Set [embedding].endpoint or switch provider to bundled"),
        });
    }

    #[cfg(feature = "bundled-embedder")]
    if matches!(config.embedding.provider, EmbeddingProvider::Bundled)
        && config.embedding.model != crate::embedding::bundled::BUNDLED_MODEL_ID
    {
        return Err(ClaudixError::ConfigInvalid {
            message: format!(
                "embedding.model must be {} when embedding.provider = \"bundled\"",
                crate::embedding::bundled::BUNDLED_MODEL_ID
            ),
            recovery: RecoveryHint(
                "Set [embedding].model = \"bge-small-en-v1.5\" for the bundled provider",
            ),
        });
    }

    #[cfg(feature = "bundled-embedder")]
    if matches!(config.embedding.provider, EmbeddingProvider::Bundled)
        && config.embedding.dimensions != crate::embedding::bundled::BUNDLED_DIMENSIONS.0
    {
        return Err(ClaudixError::ConfigInvalid {
            message: format!(
                "embedding.dimensions must be {} when embedding.provider = \"bundled\"",
                crate::embedding::bundled::BUNDLED_DIMENSIONS.0
            ),
            recovery: RecoveryHint("Set [embedding].dimensions = 384 for the bundled provider"),
        });
    }

    if config.embedding.dimensions == 0 {
        return Err(ClaudixError::ConfigInvalid {
            message: "embedding.dimensions must be > 0".into(),
            recovery: RecoveryHint("Set [embedding].dimensions to a positive integer"),
        });
    }

    if config.embedding.batch_size == 0 {
        return Err(ClaudixError::ConfigInvalid {
            message: "embedding.batch_size must be > 0".into(),
            recovery: RecoveryHint("Set [embedding].batch_size to a positive integer"),
        });
    }

    if config.search.top_k == 0 {
        return Err(ClaudixError::ConfigInvalid {
            message: "search.top_k must be > 0".into(),
            recovery: RecoveryHint("Set [search].top_k to a positive integer"),
        });
    }

    if !(0.0..=1.0).contains(&config.search.similarity_threshold) {
        return Err(ClaudixError::ConfigInvalid {
            message: "search.similarity_threshold must be between 0 and 1".into(),
            recovery: RecoveryHint("Set [search].similarity_threshold to a value in [0, 1]"),
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
