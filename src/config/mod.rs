mod merge;
mod validate;

pub use merge::PartialConfig;
pub use validate::validate;

use serde::{Deserialize, Serialize};
use std::path::{Component, Path, PathBuf};

use crate::error::{ClaudixError, RecoveryHint, Result};
use crate::prompts::hints;

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
    pub min_score: f32,
    /// Additional already-indexed repos to query alongside the active project,
    /// read-only. Paths are not required to exist at load time; a missing or
    /// unindexed one surfaces as a `RepoError` at search time.
    pub cross_repos: Vec<String>,
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
    /// Surface semantically related code after an edit. Set false to disable.
    pub surface_related_on_edit: bool,
    /// Surface semantically related code after a ranged read. Opt-in, default false.
    pub surface_related_on_read: bool,
    /// Maximum number of related-code hits surfaced per edit or read.
    pub related_top_k: usize,
    /// Cosine similarity floor for related-code hits (0.0–1.0).
    pub related_min_similarity: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathsConfig {
    pub index_dir: PathBuf,
    pub log_dir: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub watch: bool,
    /// Dev-only: resolve the binary from `cargo install` (`~/.cargo/bin/claudix`)
    /// instead of the downloaded release. Read by `scripts/ensure-binary.sh`;
    /// the running binary cannot change which binary launched it, so this field
    /// exists to be a recognized, doctor-visible key, not to drive resolution.
    pub development_mode: bool,
    pub embedding: EmbeddingConfig,
    pub indexing: IndexingConfig,
    pub search: SearchConfig,
    pub hooks: HooksConfig,
    pub paths: PathsConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            watch: false,
            development_mode: false,
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
                min_score: 0.05,
                cross_repos: Vec::new(),
            },
            hooks: HooksConfig {
                intercept_grep: true,
                auto_reembed_on_edit: true,
                auto_index_on_session_start: true,
                surface_related_on_edit: true,
                surface_related_on_read: true,
                related_top_k: 5,
                related_min_similarity: 0.65,
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
            hints::FIX_GLOBAL_CONFIG,
        )?);
    }

    if project_path.exists() {
        partial = partial.merge(read_partial_config(
            project_path,
            "project config",
            hints::FIX_PROJECT_CONFIG,
        )?);
    }

    if let Some(path) = test_override
        && path.exists()
    {
        partial = partial.merge(read_partial_config(
            path,
            "CIRRUS_CONFIG",
            hints::FIX_CIRRUS_CONFIG,
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
        recovery: RecoveryHint(hints::PROJECT_RELATIVE_PATH),
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
            watch: partial.watch.unwrap_or(defaults.watch),
            development_mode: partial
                .development_mode
                .unwrap_or(defaults.development_mode),
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
                min_score: partial
                    .search
                    .min_score
                    .unwrap_or(defaults.search.min_score),
                cross_repos: partial
                    .search
                    .cross_repos
                    .unwrap_or(defaults.search.cross_repos),
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
                surface_related_on_edit: partial
                    .hooks
                    .surface_related_on_edit
                    .unwrap_or(defaults.hooks.surface_related_on_edit),
                surface_related_on_read: partial
                    .hooks
                    .surface_related_on_read
                    .unwrap_or(defaults.hooks.surface_related_on_read),
                related_top_k: partial
                    .hooks
                    .related_top_k
                    .unwrap_or(defaults.hooks.related_top_k),
                related_min_similarity: partial
                    .hooks
                    .related_min_similarity
                    .unwrap_or(defaults.hooks.related_min_similarity),
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
    fn development_mode_defaults_to_false() {
        assert!(!Config::default().development_mode);
        assert!(validate(&Config::default()).is_ok());
    }

    #[test]
    fn development_mode_parses_from_toml() {
        let parsed: std::result::Result<PartialConfig, toml::de::Error> =
            toml::from_str("development_mode = true");
        assert_eq!(parsed.ok().and_then(|cfg| cfg.development_mode), Some(true));
    }

    #[test]
    fn development_mode_project_overrides_global() {
        let global = PartialConfig {
            development_mode: Some(false),
            ..Default::default()
        };
        let project = PartialConfig {
            development_mode: Some(true),
            ..Default::default()
        };
        assert_eq!(global.merge(project).development_mode, Some(true));
    }

    #[test]
    fn from_partial_applies_development_mode() {
        let partial = PartialConfig {
            development_mode: Some(true),
            ..Default::default()
        };
        assert!(Config::from_partial(partial).development_mode);
    }

    #[test]
    fn nested_toml_parses_into_partial_config() {
        let text = r#"
watch = true

[embedding]
provider = "http"
endpoint = "http://localhost:1234"

[search]
top_k = 15
hybrid_weights = { dense = 0.6, bm25 = 0.25, rrf = 0.15 }
min_score = 0.45
"#;

        let parsed: std::result::Result<PartialConfig, toml::de::Error> = toml::from_str(text);
        assert!(parsed.is_ok());

        let parsed = parsed.ok();
        assert_eq!(parsed.as_ref().and_then(|cfg| cfg.watch), Some(true));
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
        assert_eq!(
            parsed.as_ref().and_then(|cfg| cfg.search.min_score),
            Some(0.45)
        );
    }

    #[test]
    fn merge_project_over_global() {
        let global = PartialConfig {
            watch: Some(false),
            search: merge::PartialSearchConfig {
                top_k: Some(5),
                ..Default::default()
            },
            ..Default::default()
        };
        let project = PartialConfig {
            watch: Some(true),
            search: merge::PartialSearchConfig {
                top_k: Some(20),
                ..Default::default()
            },
            ..Default::default()
        };

        let merged = global.merge(project);
        assert_eq!(merged.watch, Some(true));
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
        config.search.hybrid_weights = HybridWeights {
            dense: 0.0,
            bm25: 0.0,
            rrf: 0.0,
        };

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
    fn reject_invalid_similarity_threshold() {
        for threshold in [-0.1, 1.1, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut config = Config::default();
            config.search.similarity_threshold = threshold;

            let error = validate(&config);
            assert!(
                matches!(error, Err(ClaudixError::ConfigInvalid { .. })),
                "expected rejection for similarity_threshold = {threshold}"
            );
        }
    }

    #[test]
    fn reject_negative_hybrid_weights() {
        let mut config = Config::default();
        config.search.hybrid_weights = HybridWeights {
            dense: -0.1,
            bm25: 0.5,
            rrf: 0.5,
        };

        let error = validate(&config);
        assert!(matches!(error, Err(ClaudixError::ConfigInvalid { .. })));
    }

    #[test]
    fn reject_non_finite_hybrid_weights() {
        for dense in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut config = Config::default();
            config.search.hybrid_weights = HybridWeights {
                dense,
                bm25: 0.5,
                rrf: 0.5,
            };

            let error = validate(&config);
            assert!(matches!(error, Err(ClaudixError::ConfigInvalid { .. })));
        }
    }

    #[test]
    fn accept_partial_nonzero_hybrid_weights() {
        let mut config = Config::default();
        config.search.hybrid_weights = HybridWeights {
            dense: 1.0,
            bm25: 0.0,
            rrf: 0.0,
        };

        assert!(validate(&config).is_ok());
    }

    #[test]
    fn reject_empty_string_entry_in_cross_repos() {
        let mut config = Config::default();
        config.search.cross_repos = vec!["/ok/path".to_owned(), "   ".to_owned()];

        let error = validate(&config);
        assert!(matches!(error, Err(ClaudixError::ConfigInvalid { .. })));
    }

    #[test]
    fn accept_cross_repos_with_paths_that_do_not_exist() {
        // Validation does NOT require paths to exist; missing/unindexed repos
        // surface as RepoError at search time, not config-load time.
        let mut config = Config::default();
        config.search.cross_repos = vec![
            "/definitely/does/not/exist".to_owned(),
            "/another/missing".to_owned(),
        ];

        assert!(validate(&config).is_ok());
    }

    #[test]
    fn cross_repos_parses_from_search_section_toml() {
        let text = r#"
[search]
cross_repos = ["/path/one", "/path/two"]
"#;
        let parsed: std::result::Result<PartialConfig, toml::de::Error> = toml::from_str(text);
        assert!(parsed.is_ok());
        let parsed = parsed.ok().unwrap_or_else(|| unreachable!());
        assert_eq!(
            parsed.search.cross_repos.as_deref(),
            Some(["/path/one".to_owned(), "/path/two".to_owned()].as_slice())
        );
    }
}
