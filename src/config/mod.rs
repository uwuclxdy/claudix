mod merge;
mod validate;

pub use merge::PartialConfig;
pub use validate::validate;

use serde::{Deserialize, Serialize};
use std::path::{Component, Path, PathBuf};

use crate::error::{ClaudixError, RecoveryHint, Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    pub provider: EmbeddingProvider,
    pub endpoint: String,
    pub model: String,
    pub dimensions: u16,
    pub batch_size: usize,
    pub timeout_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum EmbeddingProvider {
    Bundled,
    Http,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexingConfig {
    pub respect_gitignore: bool,
    pub follow_symlinks: bool,
    pub max_file_size_kb: u64,
    pub chunk_overlap_lines: usize,
    pub reindex_after_hours: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchConfig {
    pub top_k: usize,
    pub hybrid_weights: HybridWeights,
    pub identifier_boost: f32,
    pub similarity_threshold: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridWeights {
    pub dense: f32,
    pub bm25: f32,
    pub rrf: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HooksConfig {
    pub intercept_grep: bool,
    pub auto_reembed_on_edit: bool,
    pub auto_index_on_session_start: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathsConfig {
    pub index_dir: PathBuf,
    pub log_dir: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub embedding: EmbeddingConfig,
    pub indexing: IndexingConfig,
    pub search: SearchConfig,
    pub hooks: HooksConfig,
    pub paths: PathsConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            embedding: EmbeddingConfig {
                provider: EmbeddingProvider::Bundled,
                endpoint: String::new(),
                model: "bge-small-en-v1.5".into(),
                dimensions: 384,
                batch_size: 32,
                timeout_ms: 30_000,
            },
            indexing: IndexingConfig {
                respect_gitignore: true,
                follow_symlinks: false,
                max_file_size_kb: 512,
                chunk_overlap_lines: 5,
                reindex_after_hours: 24,
            },
            search: SearchConfig {
                top_k: 10,
                hybrid_weights: HybridWeights {
                    dense: 0.55,
                    bm25: 0.30,
                    rrf: 0.15,
                },
                identifier_boost: 1.4,
                similarity_threshold: 0.30,
            },
            hooks: HooksConfig {
                intercept_grep: true,
                auto_reembed_on_edit: true,
                auto_index_on_session_start: true,
            },
            paths: PathsConfig {
                index_dir: PathBuf::from(".claudix/index"),
                log_dir: PathBuf::from(".claudix/logs"),
            },
        }
    }
}

/// Load config from the standard two-file stack and merge.
pub fn load(project_root: &Path) -> Result<Config> {
    let global_path = dirs_global();
    let project_path = project_root.join(".claude").join("claudix.toml");
    let test_override = cirrus_config_path();

    load_from_paths(
        global_path.as_deref(),
        &project_path,
        test_override.as_deref(),
    )
}

fn load_from_paths(
    global_path: Option<&Path>,
    project_path: &Path,
    test_override: Option<&Path>,
) -> Result<Config> {
    let mut partial = PartialConfig::default();

    if let Some(path) = global_path
        && path.exists()
    {
        partial = partial.merge(read_partial_config(
            path,
            "global config",
            "Fix ~/.claude/claudix.toml",
        )?);
    }

    if project_path.exists() {
        partial = partial.merge(read_partial_config(
            project_path,
            "project config",
            "Fix .claude/claudix.toml",
        )?);
    }

    if let Some(path) = test_override
        && path.exists()
    {
        partial = partial.merge(read_partial_config(
            path,
            "CIRRUS_CONFIG",
            "Fix the file at CIRRUS_CONFIG path",
        )?);
    }

    let config = Config::from_partial(partial);
    validate(&config)?;
    Ok(config)
}

fn read_partial_config(
    path: &Path,
    label: &str,
    recovery_hint: &'static str,
) -> Result<PartialConfig> {
    let text = std::fs::read_to_string(path)?;

    toml::from_str(&text).map_err(|e| ClaudixError::ConfigInvalid {
        message: format!("{label}: {e}"),
        recovery: RecoveryHint(recovery_hint),
    })
}

fn path_from_partial(value: Option<String>, default: PathBuf) -> PathBuf {
    value.map(PathBuf::from).unwrap_or(default)
}

pub(super) fn is_relative_path(path: &Path) -> bool {
    !path.is_absolute()
        && path.components().all(|component| {
            !matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
}

pub(crate) fn validate_project_relative_path(path: &Path, field_name: &'static str) -> Result<()> {
    if is_relative_path(path) {
        return Ok(());
    }

    Err(ClaudixError::ConfigInvalid {
        message: format!("{field_name} must be a relative path inside the project root"),
        recovery: RecoveryHint("Set the path to a project-relative value such as .claudix/index"),
    })
}

fn dirs_global() -> Option<PathBuf> {
    dirs::home_dir().map(|h| h.join(".claude").join("claudix.toml"))
}

fn cirrus_config_path() -> Option<PathBuf> {
    #[cfg(feature = "test-stub")]
    {
        std::env::var("CIRRUS_CONFIG").ok().map(PathBuf::from)
    }

    #[cfg(not(feature = "test-stub"))]
    {
        None
    }
}

impl Config {
    fn from_partial(partial: PartialConfig) -> Self {
        let defaults = Self::default();

        Self {
            embedding: EmbeddingConfig {
                provider: partial
                    .embedding
                    .provider
                    .unwrap_or(defaults.embedding.provider),
                endpoint: partial
                    .embedding
                    .endpoint
                    .unwrap_or(defaults.embedding.endpoint),
                model: partial.embedding.model.unwrap_or(defaults.embedding.model),
                dimensions: partial
                    .embedding
                    .dimensions
                    .unwrap_or(defaults.embedding.dimensions),
                batch_size: partial
                    .embedding
                    .batch_size
                    .unwrap_or(defaults.embedding.batch_size),
                timeout_ms: partial
                    .embedding
                    .timeout_ms
                    .unwrap_or(defaults.embedding.timeout_ms),
            },
            indexing: IndexingConfig {
                respect_gitignore: partial
                    .indexing
                    .respect_gitignore
                    .unwrap_or(defaults.indexing.respect_gitignore),
                follow_symlinks: partial
                    .indexing
                    .follow_symlinks
                    .unwrap_or(defaults.indexing.follow_symlinks),
                max_file_size_kb: partial
                    .indexing
                    .max_file_size_kb
                    .unwrap_or(defaults.indexing.max_file_size_kb),
                chunk_overlap_lines: partial
                    .indexing
                    .chunk_overlap_lines
                    .unwrap_or(defaults.indexing.chunk_overlap_lines),
                reindex_after_hours: partial
                    .indexing
                    .reindex_after_hours
                    .unwrap_or(defaults.indexing.reindex_after_hours),
            },
            search: SearchConfig {
                top_k: partial.search.top_k.unwrap_or(defaults.search.top_k),
                hybrid_weights: HybridWeights {
                    dense: partial
                        .search
                        .hybrid_weights
                        .dense
                        .unwrap_or(defaults.search.hybrid_weights.dense),
                    bm25: partial
                        .search
                        .hybrid_weights
                        .bm25
                        .unwrap_or(defaults.search.hybrid_weights.bm25),
                    rrf: partial
                        .search
                        .hybrid_weights
                        .rrf
                        .unwrap_or(defaults.search.hybrid_weights.rrf),
                },
                identifier_boost: partial
                    .search
                    .identifier_boost
                    .unwrap_or(defaults.search.identifier_boost),
                similarity_threshold: partial
                    .search
                    .similarity_threshold
                    .unwrap_or(defaults.search.similarity_threshold),
            },
            hooks: HooksConfig {
                intercept_grep: partial
                    .hooks
                    .intercept_grep
                    .unwrap_or(defaults.hooks.intercept_grep),
                auto_reembed_on_edit: partial
                    .hooks
                    .auto_reembed_on_edit
                    .unwrap_or(defaults.hooks.auto_reembed_on_edit),
                auto_index_on_session_start: partial
                    .hooks
                    .auto_index_on_session_start
                    .unwrap_or(defaults.hooks.auto_index_on_session_start),
            },
            paths: PathsConfig {
                index_dir: path_from_partial(partial.paths.index_dir, defaults.paths.index_dir),
                log_dir: path_from_partial(partial.paths.log_dir, defaults.paths.log_dir),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn default_config_is_valid() {
        let config = Config::default();
        assert!(validate(&config).is_ok());
    }

    #[test]
    fn nested_toml_parses_into_partial_config() {
        let text = r#"
[embedding]
provider = "http"
endpoint = "http://localhost:1234"

[search]
top_k = 15
hybrid_weights = { dense = 0.6, bm25 = 0.25, rrf = 0.15 }
"#;

        let parsed: std::result::Result<PartialConfig, toml::de::Error> = toml::from_str(text);
        assert!(parsed.is_ok());

        let parsed = parsed.ok();
        assert_eq!(
            parsed
                .as_ref()
                .and_then(|cfg| cfg.embedding.provider.as_ref()),
            Some(&EmbeddingProvider::Http)
        );
        assert_eq!(
            parsed
                .as_ref()
                .and_then(|cfg| cfg.embedding.endpoint.as_deref()),
            Some("http://localhost:1234")
        );
        assert_eq!(parsed.as_ref().and_then(|cfg| cfg.search.top_k), Some(15));
        assert_eq!(
            parsed
                .as_ref()
                .and_then(|cfg| cfg.search.hybrid_weights.dense),
            Some(0.6)
        );
        assert_eq!(
            parsed
                .as_ref()
                .and_then(|cfg| cfg.search.hybrid_weights.bm25),
            Some(0.25)
        );
        assert_eq!(
            parsed
                .as_ref()
                .and_then(|cfg| cfg.search.hybrid_weights.rrf),
            Some(0.15)
        );
    }

    #[test]
    fn merge_project_over_global() {
        let global = PartialConfig {
            search: merge::PartialSearchConfig {
                top_k: Some(5),
                ..Default::default()
            },
            ..Default::default()
        };
        let project = PartialConfig {
            search: merge::PartialSearchConfig {
                top_k: Some(20),
                ..Default::default()
            },
            ..Default::default()
        };

        let merged = global.merge(project);
        assert_eq!(merged.search.top_k, Some(20));
    }

    #[test]
    fn load_merges_global_and_project_files() {
        let temp = tempdir();
        assert!(temp.is_ok());
        let temp = temp.ok();
        assert!(temp.is_some());
        let temp = temp.unwrap_or_else(|| unreachable!());

        let global_path = temp.path().join("global.toml");
        let project_root = temp.path().join("project");
        let project_config_dir = project_root.join(".claude");
        let project_config_path = project_config_dir.join("claudix.toml");

        assert!(fs::create_dir_all(&project_config_dir).is_ok());
        assert!(
            fs::write(
                &global_path,
                r#"
[embedding]
provider = "http"
endpoint = "http://global.example"

[search]
top_k = 5
hybrid_weights = { dense = 0.7, bm25 = 0.2, rrf = 0.1 }
"#,
            )
            .is_ok()
        );
        assert!(
            fs::write(
                &project_config_path,
                r#"
[search]
top_k = 20

[paths]
index_dir = ".claudix/custom-index"
"#,
            )
            .is_ok()
        );

        let loaded = load_from_paths(Some(&global_path), &project_config_path, None);
        assert!(loaded.is_ok());
        let loaded = loaded.ok();

        assert_eq!(
            loaded.as_ref().map(|cfg| cfg.embedding.endpoint.as_str()),
            Some("http://global.example")
        );
        assert_eq!(loaded.as_ref().map(|cfg| cfg.search.top_k), Some(20));
        assert_eq!(
            loaded.as_ref().map(|cfg| cfg.search.hybrid_weights.dense),
            Some(0.7)
        );
        assert_eq!(
            loaded
                .as_ref()
                .map(|cfg| cfg.paths.index_dir.to_string_lossy().into_owned()),
            Some(".claudix/custom-index".to_owned())
        );
    }

    #[test]
    fn http_provider_requires_endpoint() {
        let mut config = Config::default();
        config.embedding.provider = EmbeddingProvider::Http;
        config.embedding.endpoint = String::new();
        assert!(validate(&config).is_err());
    }

    #[test]
    fn stub_model_still_validates_numeric_fields() {
        let mut config = Config::default();
        config.embedding.model = "stub-model".to_owned();
        config.search.top_k = 0;

        let error = validate(&config);
        assert!(matches!(error, Err(ClaudixError::ConfigInvalid { .. })));
    }

    #[test]
    fn reject_zero_max_file_size() {
        let mut config = Config::default();
        config.indexing.max_file_size_kb = 0;

        let error = validate(&config);
        assert!(matches!(error, Err(ClaudixError::ConfigInvalid { .. })));
    }

    #[test]
    fn reject_chunk_overlap_at_or_above_chunk_size() {
        let mut config = Config::default();
        config.indexing.chunk_overlap_lines = 60;

        let error = validate(&config);
        assert!(matches!(error, Err(ClaudixError::ConfigInvalid { .. })));
    }

    #[test]
    fn accept_chunk_overlap_below_chunk_size() {
        let mut config = Config::default();
        config.indexing.chunk_overlap_lines = 59;

        assert!(validate(&config).is_ok());
    }

    #[test]
    fn reject_zero_reindex_after_hours() {
        let mut config = Config::default();
        config.indexing.reindex_after_hours = 0;

        let error = validate(&config);
        assert!(matches!(error, Err(ClaudixError::ConfigInvalid { .. })));
    }

    #[test]
    fn reject_absolute_paths_outside_project() {
        let mut config = Config::default();
        config.paths.log_dir = PathBuf::from("/tmp/claudix-logs");

        let error = validate(&config);
        assert!(matches!(error, Err(ClaudixError::ConfigInvalid { .. })));
    }

    #[test]
    fn reject_parent_directory_path_segments() {
        let mut config = Config::default();
        config.paths.index_dir = PathBuf::from("../outside");

        let error = validate(&config);
        assert!(matches!(error, Err(ClaudixError::ConfigInvalid { .. })));
    }

    #[test]
    fn reject_all_zero_hybrid_weights() {
        let mut config = Config::default();
        config.search.hybrid_weights = HybridWeights { dense: 0.0, bm25: 0.0, rrf: 0.0 };

        let error = validate(&config);
        assert!(matches!(error, Err(ClaudixError::ConfigInvalid { .. })));
    }

    #[test]
    fn reject_zero_embedding_timeout() {
        let mut config = Config::default();
        config.embedding.timeout_ms = 0;

        let error = validate(&config);
        assert!(matches!(error, Err(ClaudixError::ConfigInvalid { .. })));
    }

    #[test]
    fn reject_invalid_identifier_boost() {
        for identifier_boost in [0.0, -0.1, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut config = Config::default();
            config.search.identifier_boost = identifier_boost;

            let error = validate(&config);
            assert!(matches!(error, Err(ClaudixError::ConfigInvalid { .. })));
        }
    }

    #[test]
    fn reject_negative_hybrid_weights() {
        let mut config = Config::default();
        config.search.hybrid_weights = HybridWeights { dense: -0.1, bm25: 0.5, rrf: 0.5 };

        let error = validate(&config);
        assert!(matches!(error, Err(ClaudixError::ConfigInvalid { .. })));
    }

    #[test]
    fn reject_non_finite_hybrid_weights() {
        for dense in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut config = Config::default();
            config.search.hybrid_weights = HybridWeights { dense, bm25: 0.5, rrf: 0.5 };

            let error = validate(&config);
            assert!(matches!(error, Err(ClaudixError::ConfigInvalid { .. })));
        }
    }

    #[test]
    fn accept_partial_nonzero_hybrid_weights() {
        let mut config = Config::default();
        config.search.hybrid_weights = HybridWeights { dense: 1.0, bm25: 0.0, rrf: 0.0 };

        assert!(validate(&config).is_ok());
    }
}
