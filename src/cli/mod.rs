mod input;
mod install;
mod watch;

use std::collections::HashMap;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::Serialize;

use crate::config;
use crate::error::{ClaudixError, RecoveryHint, Result};
use crate::hooks::HookEvent;
use crate::search::SearchQuery;
use crate::store::{IndexLockGuard, Store};
use crate::types::{RelativePath, path_prefix_matches};
use crate::{Claudix, IndexFileStatus, IndexProgress};

use input::{
    parse_language_filter, parse_path_prefix, validate_search_query, validate_search_top_k,
};

/// Maximum number of top identifiers surfaced per directory in `OverviewOutput`.
const TOP_IDENTIFIERS_CAP: usize = 8;

pub use install::{run_install, setup_state};
pub use watch::run_watch;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SearchHit {
    pub file_path: String,
    pub language: String,
    pub kind: String,
    pub name: Option<String>,
    pub line_start: u32,
    pub line_end: u32,
    pub score: f32,
    pub stale: bool,
    pub snippet: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DirectoryGroup {
    pub directory: String,
    pub hits: Vec<SearchHit>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SearchOutput {
    pub groups: Vec<DirectoryGroup>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct IndexOutput {
    pub file_count: usize,
    pub chunk_count: usize,
}

pub struct StderrIndexProgress;

impl IndexProgress for StderrIndexProgress {
    fn file(&mut self, path: &RelativePath, status: IndexFileStatus) -> Result<()> {
        let mut stderr = io::stderr().lock();
        match status {
            IndexFileStatus::Indexed => writeln!(stderr, "indexed {}", path.as_str())?,
            IndexFileStatus::Verified => writeln!(stderr, "verified {}", path.as_str())?,
            IndexFileStatus::Skipped(reason) => {
                writeln!(stderr, "skipped {}: {reason}", path.as_str())?
            }
        }
        stderr.flush()?;
        Ok(())
    }
}

struct FileIndexProgress {
    writer: io::BufWriter<fs::File>,
}

impl FileIndexProgress {
    fn try_open(log_dir: &Path) -> Option<Self> {
        fs::create_dir_all(log_dir).ok()?;
        fs::File::create(log_dir.join("index.log"))
            .ok()
            .map(|f| Self {
                writer: io::BufWriter::new(f),
            })
    }
}

impl IndexProgress for FileIndexProgress {
    fn file(&mut self, path: &RelativePath, status: IndexFileStatus) -> Result<()> {
        match status {
            IndexFileStatus::Indexed => writeln!(self.writer, "indexed {}", path.as_str())?,
            IndexFileStatus::Verified => writeln!(self.writer, "verified {}", path.as_str())?,
            IndexFileStatus::Skipped(reason) => {
                writeln!(self.writer, "skipped {}: {reason}", path.as_str())?
            }
        }
        self.writer.flush()?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ClearOutput {
    pub cleared: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct StatusOutput {
    pub chunk_count: usize,
    pub file_count: usize,
    pub model: Option<String>,
    pub dimensions: Option<u16>,
    pub last_full_index_at: Option<String>,
    pub last_incremental_at: Option<String>,
    /// True when the index is missing or older than `reindex_after_hours`.
    pub stale: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DoctorOutput {
    pub project_root: String,
    pub index_present: bool,
    pub chunk_count: usize,
    pub file_count: usize,
    pub model: Option<String>,
    pub dimensions: Option<u16>,
    pub embedding_provider: String,
    pub embedding_healthy: bool,
    /// Why the embedding check failed, when `embedding_healthy` is false and the
    /// cause is not a model mismatch — auth, timeout, unreachable, HTTP status.
    pub embedding_error: Option<String>,
    /// True when the stored index model differs from the active config model.
    /// Distinct from `embedding_healthy = false` caused by the server being unreachable.
    pub embedding_model_mismatch: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct InstallOutput {
    pub plugin_root: String,
    pub binary_path: String,
    pub config_path: String,
    pub wrote_config: bool,
    pub embedding_healthy: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SetupState {
    Ready,
    Missing(Vec<&'static str>),
}

/// Per-language chunk count within a directory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LanguageCount {
    pub language: String,
    pub chunk_count: usize,
}

/// Aggregated stats for one immediate parent directory in the index.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DirectoryRollup {
    /// Repo-relative directory path, or `"."` for root-level files.
    pub path: String,
    pub file_count: usize,
    pub chunk_count: usize,
    /// Languages present in this directory, sorted by chunk count desc then name asc.
    pub languages: Vec<LanguageCount>,
    /// Most frequent non-empty chunk names, capped at [`TOP_IDENTIFIERS_CAP`], sorted by
    /// frequency desc then name asc.
    pub top_identifiers: Vec<String>,
}

/// Structural map of the indexed repo, grouped by immediate parent directory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct OverviewOutput {
    /// One entry per directory, sorted by path ascending.
    pub directories: Vec<DirectoryRollup>,
    /// Distinct files in the filtered view.
    pub file_count: usize,
    /// Total chunks in the filtered view.
    pub chunk_count: usize,
}

pub async fn run_overview(
    project_root: impl AsRef<Path>,
    path_prefix: Option<String>,
) -> Result<OverviewOutput> {
    let project_root = canonical_project_root(project_root.as_ref())?;
    let config = config::load(&project_root)?;
    let store = Store::new(&project_root, &config)?;

    // Validate and normalize the path prefix the same way search does.
    let prefix: Option<RelativePath> = parse_path_prefix(path_prefix)?;

    let chunks = store.read_chunks().await?;

    // Group chunks by the immediate parent directory of their file_path.
    // Splitting on `/` is sound because file_path comes from RelativePath,
    // which forward-slash-normalizes on construction on every platform.
    // Aggregation is light counting over an already-loaded Vec — no spawn_blocking needed.
    let mut dir_files: HashMap<String, std::collections::HashSet<String>> = HashMap::new();
    let mut dir_chunks: HashMap<String, usize> = HashMap::new();
    let mut dir_languages: HashMap<String, HashMap<String, usize>> = HashMap::new();
    let mut dir_names: HashMap<String, HashMap<String, usize>> = HashMap::new();

    for chunk in &chunks {
        // Apply path-prefix filter, consistent with `apply_filters` in search.
        if let Some(ref prefix) = prefix
            && !path_prefix_matches(&chunk.file_path, prefix.as_str())
        {
            continue;
        }

        let dir = immediate_parent_dir(&chunk.file_path);
        dir_files
            .entry(dir.clone())
            .or_default()
            .insert(chunk.file_path.clone());
        *dir_chunks.entry(dir.clone()).or_insert(0) += 1;
        *dir_languages
            .entry(dir.clone())
            .or_default()
            .entry(chunk.language.clone())
            .or_insert(0) += 1;
        if let Some(ref name) = chunk.name
            && !name.is_empty()
        {
            *dir_names
                .entry(dir.clone())
                .or_default()
                .entry(name.clone())
                .or_insert(0) += 1;
        }
    }

    let mut directories: Vec<DirectoryRollup> = dir_files
        .keys()
        .map(|dir| {
            let file_count = dir_files[dir].len();
            let chunk_count = dir_chunks[dir];

            let mut languages: Vec<LanguageCount> = dir_languages
                .get(dir)
                .map(|lang_map| {
                    lang_map
                        .iter()
                        .map(|(lang, count)| LanguageCount {
                            language: lang.clone(),
                            chunk_count: *count,
                        })
                        .collect()
                })
                .unwrap_or_default();
            // Sort by chunk count desc, then language name asc for determinism.
            languages.sort_by(|a, b| {
                b.chunk_count
                    .cmp(&a.chunk_count)
                    .then_with(|| a.language.cmp(&b.language))
            });

            let top_identifiers: Vec<String> = dir_names
                .get(dir)
                .map(|name_map| {
                    let mut pairs: Vec<(&String, usize)> =
                        name_map.iter().map(|(n, c)| (n, *c)).collect();
                    // Frequency desc, then name asc for determinism, then cap.
                    pairs.sort_by(|(a_name, a_count), (b_name, b_count)| {
                        b_count.cmp(a_count).then_with(|| a_name.cmp(b_name))
                    });
                    pairs.truncate(TOP_IDENTIFIERS_CAP);
                    pairs.into_iter().map(|(name, _)| name.clone()).collect()
                })
                .unwrap_or_default();

            DirectoryRollup {
                path: dir.clone(),
                file_count,
                chunk_count,
                languages,
                top_identifiers,
            }
        })
        .collect();

    // Sort directories by path ascending for deterministic, navigable output.
    directories.sort_by(|a, b| a.path.cmp(&b.path));

    let file_count: usize = directories.iter().map(|d| d.file_count).sum();
    let chunk_count: usize = directories.iter().map(|d| d.chunk_count).sum();

    Ok(OverviewOutput {
        directories,
        file_count,
        chunk_count,
    })
}

/// Returns the immediate parent directory of a repo-relative file path,
/// normalized to forward slashes. Root-level files (no `/`) return `"."`.
///
/// Used by `run_overview` for directory grouping. Feature 5 can reuse this
/// for grouping search results.
pub(crate) fn immediate_parent_dir(file_path: &str) -> String {
    match file_path.rfind('/') {
        Some(pos) => file_path[..pos].to_owned(),
        None => ".".to_owned(),
    }
}

pub async fn run_search(
    project_root: impl AsRef<Path>,
    query: String,
    top_k: Option<usize>,
    language_filter: Option<Vec<String>>,
    path_prefix: Option<String>,
) -> Result<SearchOutput> {
    validate_search_query(&query)?;
    let project_root = canonical_project_root(project_root.as_ref())?;
    let config = config::load(&project_root)?;
    let top_k = top_k.unwrap_or(config.search.top_k);
    validate_search_top_k(top_k)?;
    let claudix = Claudix::new(project_root, Arc::new(config)).await?;

    run_search_with_claudix(&claudix, query, top_k, language_filter, path_prefix).await
}

pub async fn run_index(project_root: impl AsRef<Path>, progress: bool) -> Result<IndexOutput> {
    let session = IndexSession::new(project_root).await?;
    let log_dir = session
        .claudix
        .project_root()
        .join(&session.claudix.config().paths.log_dir);
    let mut stderr_progress = StderrIndexProgress;
    let mut file_progress = (!progress)
        .then(|| FileIndexProgress::try_open(&log_dir))
        .flatten();
    let progress: &mut dyn IndexProgress = if progress {
        &mut stderr_progress
    } else if let Some(ref mut fp) = file_progress {
        fp
    } else {
        &mut ()
    };
    let stats = session.claudix.index_full(progress).await?;

    Ok(IndexOutput {
        file_count: stats.file_count,
        chunk_count: stats.chunk_count,
    })
}

struct IndexSession {
    claudix: Claudix,
    _lock: IndexLockGuard,
}

impl IndexSession {
    async fn new(project_root: impl AsRef<Path>) -> Result<Self> {
        let project_root = canonical_project_root(project_root.as_ref())?;
        require_git_repo(&project_root)?;
        let config = config::load(&project_root)?;
        let store = Store::new(&project_root, &config)?;
        let lock = store
            .acquire_index_lock()
            .ok_or_else(|| crate::error::ClaudixError::Store("index already running".to_owned()))?;
        let claudix = match Claudix::new(project_root.clone(), Arc::new(config.clone())).await {
            Ok(claudix) => claudix,
            Err(error) if requires_clean_reindex(&error) => {
                store.clear_chunks(&config).await?;
                Claudix::new(project_root, Arc::new(config)).await?
            }
            Err(error) => return Err(error),
        };

        Ok(Self {
            claudix,
            _lock: lock,
        })
    }
}

pub async fn run_status(project_root: impl AsRef<Path>) -> Result<StatusOutput> {
    let project_root = canonical_project_root(project_root.as_ref())?;
    let config = config::load(&project_root)?;
    let store = Store::new(&project_root, &config)?;
    status_from_store(&store, &config).await
}

pub async fn run_reindex_file(
    project_root: impl AsRef<Path>,
    path: impl AsRef<Path>,
) -> Result<IndexOutput> {
    let project_root = canonical_project_root(project_root.as_ref())?;
    let config = config::load(&project_root)?;
    let store = Store::new(&project_root, &config)?;

    // Block on the shared chunk-writer lock instead of short-circuiting on a
    // running full index: bailing here loses the user's edit until they save
    // again, since the reindex-file child returns 0 with no retry path.
    let _reindex_lock = store.acquire_reindex_lock()?;
    let claudix = Claudix::new(project_root, Arc::new(config)).await?;
    let stats = claudix.reindex_file(path.as_ref()).await?;

    Ok(IndexOutput {
        file_count: stats.file_count,
        chunk_count: stats.chunk_count,
    })
}

pub async fn run_doctor(project_root: impl AsRef<Path>) -> Result<DoctorOutput> {
    let project_root = canonical_project_root(project_root.as_ref())?;
    let config = config::load(&project_root)?;
    let store = Store::new(&project_root, &config)?;
    let status = status_from_store(&store, &config).await?;

    let claudix = Claudix::new(project_root.clone(), Arc::new(config.clone())).await;
    let (embedding_healthy, embedding_model_mismatch, embedding_error) = match claudix {
        Ok(claudix) => match claudix.embedder_health_check().await {
            Ok(()) => (true, false, None),
            Err(error) => (false, false, Some(error.to_string())),
        },
        Err(ClaudixError::EmbeddingModelMismatch { .. }) => (false, true, None),
        Err(error) => (false, false, Some(error.to_string())),
    };

    Ok(DoctorOutput {
        project_root: project_root.display().to_string(),
        index_present: status.chunk_count > 0 || status.model.is_some(),
        chunk_count: status.chunk_count,
        file_count: status.file_count,
        model: status.model,
        dimensions: status.dimensions,
        embedding_provider: match config.embedding.provider {
            config::EmbeddingProvider::Bundled => "bundled".to_owned(),
            config::EmbeddingProvider::Http => "http".to_owned(),
        },
        embedding_healthy,
        embedding_error,
        embedding_model_mismatch,
    })
}

pub async fn run_clear_index(project_root: impl AsRef<Path>) -> Result<ClearOutput> {
    let project_root = canonical_project_root(project_root.as_ref())?;
    let config = config::load(&project_root)?;
    let store = Store::new(&project_root, &config)?;
    store.clear_chunks(&config).await?;

    Ok(ClearOutput { cleared: true })
}

fn requires_clean_reindex(error: &ClaudixError) -> bool {
    matches!(
        error,
        ClaudixError::SchemaMismatch { .. }
            | ClaudixError::EmbeddingModelMismatch { .. }
            | ClaudixError::DimensionMismatch { .. }
    )
}

pub fn parse_hook_event(value: &str) -> Result<HookEvent> {
    match value {
        "SessionStart" => Ok(HookEvent::SessionStart),
        "PostToolUse" => Ok(HookEvent::PostToolUse),
        "PreToolUse" => Ok(HookEvent::PreToolUse),
        "UserPromptSubmit" => Ok(HookEvent::UserPromptSubmit),
        _ => Err(ClaudixError::ConfigInvalid {
            message: format!("unknown hook event: {value}"),
            recovery: RecoveryHint(
                "Use one of: SessionStart, PostToolUse, PreToolUse, UserPromptSubmit",
            ),
        }),
    }
}

async fn run_search_with_claudix(
    claudix: &Claudix,
    query: String,
    top_k: usize,
    language_filter: Option<Vec<String>>,
    path_prefix: Option<String>,
) -> Result<SearchOutput> {
    let query = SearchQuery {
        query,
        top_k,
        language_filter: parse_language_filter(language_filter)?,
        path_prefix: parse_path_prefix(path_prefix)?,
    };
    let results = claudix.search(query).await?;

    // Walk hits in score order (results is already score-desc from search).
    // Bucket by directory preserving first-seen order so the first directory
    // encountered owns the top hit — groups naturally ordered by best score.
    let mut dir_index: Vec<String> = Vec::new();
    let mut dir_hits: HashMap<String, Vec<SearchHit>> = HashMap::new();

    for result in results {
        let hit = SearchHit {
            file_path: result.chunk.file_path.to_string(),
            language: result.chunk.language.to_string(),
            kind: result.chunk.kind.to_string(),
            name: result.chunk.name,
            line_start: result.chunk.line_range.start,
            line_end: result.chunk.line_range.end,
            score: result.score,
            stale: result.stale,
            snippet: result.chunk.content,
        };
        let dir = immediate_parent_dir(&hit.file_path);
        if !dir_hits.contains_key(&dir) {
            dir_index.push(dir.clone());
        }
        dir_hits.entry(dir).or_default().push(hit);
    }

    let groups = dir_index
        .into_iter()
        .filter_map(|dir| {
            let hits = dir_hits.remove(&dir)?;
            Some(DirectoryGroup {
                directory: dir,
                hits,
            })
        })
        .collect();

    Ok(SearchOutput { groups })
}

async fn status_from_store(store: &Store, config: &crate::config::Config) -> Result<StatusOutput> {
    let manifest = store.read_manifest()?;
    let chunk_count = manifest
        .as_ref()
        .map(|m| m.chunk_count as usize)
        .unwrap_or(0);
    let file_count = manifest
        .as_ref()
        .map(|m| m.file_count as usize)
        .unwrap_or(0);

    let stale = manifest
        .as_ref()
        .map(|m| m.is_stale(config))
        .unwrap_or(true);

    Ok(StatusOutput {
        chunk_count,
        file_count,
        model: manifest
            .as_ref()
            .map(|manifest| manifest.embedding_model.clone()),
        dimensions: manifest.as_ref().map(|manifest| manifest.dimensions),
        last_full_index_at: manifest
            .as_ref()
            .and_then(|manifest| manifest.last_full_index_at.clone()),
        last_incremental_at: manifest
            .as_ref()
            .and_then(|manifest| manifest.last_incremental_at.clone()),
        stale,
    })
}

fn require_git_repo(project_root: &Path) -> Result<()> {
    if !crate::enumeration::is_git_repo(project_root) {
        return Err(ClaudixError::NotAGitRepository {
            path: project_root.to_path_buf(),
            recovery: RecoveryHint("Run claudix index from inside a git repository"),
        });
    }
    Ok(())
}

pub(super) fn canonical_project_root(project_root: &Path) -> Result<PathBuf> {
    project_root.canonicalize().map_err(ClaudixError::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::embedding::{Provider, StubProvider};
    use crate::store::{Manifest, Store};
    use crate::types::Dimension;

    mod fixture {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/common/fixture.rs"
        ));
    }

    mod test_support {
        use crate as claudix;

        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/common/test_support.rs"
        ));
    }

    use fixture::TestFixture;
    use test_support::{index_fixture, stub_config};

    struct CliHarness {
        _fixture: TestFixture,
        claudix: Claudix,
        store: Store,
    }

    fn test_claudix(project_root: PathBuf, config: Config) -> Result<Claudix> {
        let store = Store::new(&project_root, &config)?;
        let config = Arc::new(config);
        let embedder: Arc<dyn Provider> = Arc::new(StubProvider::with_model_id(
            config.embedding.model.clone(),
            Dimension(config.embedding.dimensions),
        ));

        Ok(Claudix::from_parts(project_root, config, embedder, store))
    }

    async fn cli_harness() -> Result<CliHarness> {
        let fixture = TestFixture::new("small_rust")?;
        let config = stub_config();
        let claudix = test_claudix(fixture.root().to_path_buf(), config.clone())?;
        index_fixture(
            &claudix.store,
            claudix.embedder.as_ref(),
            claudix.project_root(),
            &config,
        )
        .await?;
        let store = Store::new(fixture.root(), &config)?;

        Ok(CliHarness {
            _fixture: fixture,
            claudix,
            store,
        })
    }

    #[test]
    fn parse_hook_event_accepts_known_values() {
        let event = parse_hook_event("SessionStart");
        assert!(matches!(event, Ok(HookEvent::SessionStart)));

        let event = parse_hook_event("PostToolUse");
        assert!(matches!(event, Ok(HookEvent::PostToolUse)));

        let event = parse_hook_event("PreToolUse");
        assert!(matches!(event, Ok(HookEvent::PreToolUse)));

        let event = parse_hook_event("UserPromptSubmit");
        assert!(matches!(event, Ok(HookEvent::UserPromptSubmit)));
    }

    #[test]
    fn parse_hook_event_rejects_unknown_values() {
        let result = parse_hook_event("Unknown");
        assert!(matches!(result, Err(ClaudixError::ConfigInvalid { .. })));
    }

    #[tokio::test]
    async fn run_search_returns_ranked_hits() {
        let harness = cli_harness().await;
        assert!(harness.is_ok());
        let harness = harness.ok().unwrap_or_else(|| unreachable!());

        let output =
            run_search_with_claudix(&harness.claudix, "add".to_owned(), 5, None, None).await;
        assert!(output.is_ok());
        let output = output.ok().unwrap_or_else(|| unreachable!());

        assert!(!output.groups.is_empty());
        let top_hit = &output.groups[0].hits[0];
        assert_eq!(top_hit.name.as_deref(), Some("add"));
        assert_eq!(top_hit.file_path, "src/math.rs");
    }

    #[tokio::test]
    async fn run_search_applies_filters() {
        let harness = cli_harness().await;
        assert!(harness.is_ok());
        let harness = harness.ok().unwrap_or_else(|| unreachable!());

        let output = run_search_with_claudix(
            &harness.claudix,
            "add".to_owned(),
            5,
            Some(vec!["rust".to_owned()]),
            Some(RelativePath::new("src/math").to_string()),
        )
        .await;
        assert!(output.is_ok());
        let output = output.ok().unwrap_or_else(|| unreachable!());

        assert_eq!(output.groups.len(), 1);
        assert_eq!(output.groups[0].hits.len(), 1);
        assert_eq!(output.groups[0].hits[0].file_path, "src/math.rs");
    }

    #[test]
    fn clean_reindex_required_for_manifest_compatibility_errors() {
        assert!(requires_clean_reindex(&ClaudixError::SchemaMismatch {
            store: 0,
            binary: 1,
            recovery: RecoveryHint("reindex"),
        }));
        assert!(requires_clean_reindex(
            &ClaudixError::EmbeddingModelMismatch {
                store_model: "old".to_owned(),
                active_model: "new".to_owned(),
                recovery: RecoveryHint("reindex"),
            }
        ));
        assert!(requires_clean_reindex(&ClaudixError::DimensionMismatch {
            store_dim: 384,
            model_dim: 768,
            recovery: RecoveryHint("reindex"),
        }));
        assert!(!requires_clean_reindex(&ClaudixError::Store(
            "index already running".to_owned()
        )));
    }

    #[tokio::test]
    async fn run_index_clears_model_mismatch_and_reindexes() {
        let fixture = TestFixture::new("small_rust");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());
        let config = stub_config();
        let claude_dir = fixture.root().join(".claude");
        assert!(std::fs::create_dir_all(&claude_dir).is_ok());
        let config_text = toml::to_string(&config);
        assert!(config_text.is_ok());
        assert!(
            std::fs::write(
                claude_dir.join("claudix.toml"),
                config_text.ok().unwrap_or_default(),
            )
            .is_ok()
        );

        let store = Store::new(fixture.root(), &config);
        assert!(store.is_ok());
        let store = store.ok().unwrap_or_else(|| unreachable!());
        let old_manifest = Manifest::new("old-model", config.embedding.dimensions);
        assert!(store.write_manifest(&old_manifest).is_ok());

        let output = run_index(fixture.root(), false).await;
        assert!(output.is_ok());
        let output = output.ok().unwrap_or_else(|| unreachable!());
        assert!(output.chunk_count > 0);

        let manifest = store.read_manifest();
        assert!(manifest.is_ok());
        let manifest = manifest.ok().unwrap_or_else(|| unreachable!());
        let manifest = manifest.unwrap_or_else(|| unreachable!());
        assert_eq!(manifest.embedding_model, config.embedding.model);
        assert_eq!(manifest.dimensions, config.embedding.dimensions);
        assert_eq!(manifest.chunk_count as usize, output.chunk_count);
    }

    #[tokio::test]
    async fn run_status_reports_manifest_and_counts() {
        let harness = cli_harness().await;
        assert!(harness.is_ok());
        let harness = harness.ok().unwrap_or_else(|| unreachable!());

        let config = stub_config();
        let status = status_from_store(&harness.store, &config).await;
        assert!(status.is_ok());
        let status = status.ok().unwrap_or_else(|| unreachable!());

        assert_eq!(status.chunk_count, 3);
        assert_eq!(status.file_count, 2);
        assert_eq!(status.model.as_deref(), Some("stub-v1"));
        assert_eq!(status.dimensions, Some(8));
        assert!(!status.stale, "freshly indexed should not be stale");
    }

    #[tokio::test]
    async fn run_doctor_reports_index_and_embedding_health() {
        let harness = cli_harness().await;
        assert!(harness.is_ok());
        let harness = harness.ok().unwrap_or_else(|| unreachable!());

        let config = stub_config();
        let claude_dir = harness.claudix.project_root().join(".claude");
        assert!(std::fs::create_dir_all(&claude_dir).is_ok());
        let config_text = toml::to_string(&config);
        assert!(config_text.is_ok());
        assert!(
            std::fs::write(
                claude_dir.join("claudix.toml"),
                config_text.ok().unwrap_or_default(),
            )
            .is_ok()
        );

        let output = run_doctor(harness.claudix.project_root()).await;
        assert!(output.is_ok());
        let output = output.ok().unwrap_or_else(|| unreachable!());

        assert!(output.index_present);
        assert_eq!(output.chunk_count, 3);
        assert_eq!(output.file_count, 2);
        assert_eq!(output.model.as_deref(), Some("stub-v1"));
        assert_eq!(output.embedding_provider, "bundled");
        assert!(output.embedding_healthy);
    }

    #[tokio::test]
    async fn run_overview_returns_src_directory_rollup() {
        let harness = cli_harness().await;
        assert!(harness.is_ok());
        let harness = harness.ok().unwrap_or_else(|| unreachable!());

        let output = run_overview(harness.claudix.project_root(), None).await;
        assert!(output.is_ok());
        let output = output.ok().unwrap_or_else(|| unreachable!());

        // The small_rust fixture has files under src/ only.
        let src = output.directories.iter().find(|d| d.path == "src");
        assert!(src.is_some(), "expected a 'src' directory in the rollup");
        let src = src.unwrap_or_else(|| unreachable!());

        // src/ has src/math.rs and src/lib.rs — 2 files, 3 chunks total.
        assert_eq!(src.file_count, 2);
        assert_eq!(src.chunk_count, 3);

        // The fixture is Rust source — rust must appear in languages.
        assert!(
            src.languages.iter().any(|l| l.language == "rust"),
            "expected 'rust' among languages in src/"
        );
    }

    #[tokio::test]
    async fn run_overview_top_identifiers_include_known_fixture_name() {
        let harness = cli_harness().await;
        assert!(harness.is_ok());
        let harness = harness.ok().unwrap_or_else(|| unreachable!());

        let output = run_overview(harness.claudix.project_root(), None).await;
        assert!(output.is_ok());
        let output = output.ok().unwrap_or_else(|| unreachable!());

        let src = output.directories.iter().find(|d| d.path == "src");
        assert!(src.is_some());
        let src = src.unwrap_or_else(|| unreachable!());

        // src/math.rs defines `add` — it must appear in top identifiers.
        assert!(
            src.top_identifiers.iter().any(|name| name == "add"),
            "expected 'add' in top_identifiers for src/; got: {:?}",
            src.top_identifiers,
        );
    }

    #[tokio::test]
    async fn run_overview_path_prefix_narrows_to_subtree() {
        let harness = cli_harness().await;
        assert!(harness.is_ok());
        let harness = harness.ok().unwrap_or_else(|| unreachable!());

        // Prefix "src/math" should match only src/math.rs.
        let output =
            run_overview(harness.claudix.project_root(), Some("src/math".to_owned())).await;
        assert!(output.is_ok());
        let output = output.ok().unwrap_or_else(|| unreachable!());

        // Only chunks from src/math.rs pass the filter; src/lib.rs is excluded.
        assert!(output.file_count < 2, "prefix should exclude src/lib.rs");
        assert!(
            output.chunk_count > 0,
            "at least one chunk from src/math.rs"
        );

        // Every directory in the result must have files that match the prefix.
        for dir in &output.directories {
            assert!(
                path_prefix_matches(&format!("{}/file.rs", dir.path), "src/math")
                    || dir.path == "src",
                "unexpected directory outside prefix: {}",
                dir.path,
            );
        }
    }

    #[tokio::test]
    async fn run_overview_directories_sorted_ascending() {
        let harness = cli_harness().await;
        assert!(harness.is_ok());
        let harness = harness.ok().unwrap_or_else(|| unreachable!());

        let output = run_overview(harness.claudix.project_root(), None).await;
        assert!(output.is_ok());
        let output = output.ok().unwrap_or_else(|| unreachable!());

        let paths: Vec<&str> = output.directories.iter().map(|d| d.path.as_str()).collect();
        let mut sorted = paths.clone();
        sorted.sort();
        assert_eq!(
            paths, sorted,
            "directories must be sorted ascending by path"
        );
    }

    #[test]
    fn immediate_parent_dir_returns_dot_for_root_level_file() {
        assert_eq!(immediate_parent_dir("Cargo.toml"), ".");
        assert_eq!(immediate_parent_dir("lib.rs"), ".");
    }

    #[test]
    fn immediate_parent_dir_extracts_parent_segment() {
        assert_eq!(immediate_parent_dir("src/math.rs"), "src");
        assert_eq!(immediate_parent_dir("src/hooks/mod.rs"), "src/hooks");
    }

    #[tokio::test]
    async fn run_doctor_surfaces_specific_embedding_failure() {
        let fixture = TestFixture::new("small_rust");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());

        // Point at a dead port with a non-bundled model so no fallback masks the
        // failure; doctor must report the actual reason, not a flat "unreachable".
        let mut config = stub_config();
        config.embedding.provider = config::EmbeddingProvider::Http;
        config.embedding.endpoint = "http://127.0.0.1:1".to_owned();
        config.embedding.model = "no-fallback-model".to_owned();
        config.embedding.timeout_ms = 100;

        let claude_dir = fixture.root().join(".claude");
        assert!(std::fs::create_dir_all(&claude_dir).is_ok());
        let config_text = toml::to_string(&config);
        assert!(config_text.is_ok());
        assert!(
            std::fs::write(
                claude_dir.join("claudix.toml"),
                config_text.ok().unwrap_or_default(),
            )
            .is_ok()
        );

        let output = run_doctor(fixture.root()).await;
        assert!(output.is_ok());
        let output = output.ok().unwrap_or_else(|| unreachable!());

        assert!(!output.embedding_healthy);
        assert!(!output.embedding_model_mismatch);
        assert!(
            output
                .embedding_error
                .is_some_and(|reason| reason.contains("http://127.0.0.1:1"))
        );
    }

    #[tokio::test]
    async fn search_groups_hits_by_directory_ordered_by_best_score() {
        let harness = cli_harness().await;
        assert!(harness.is_ok());
        let harness = harness.ok().unwrap_or_else(|| unreachable!());

        // "greet add" spans both src/lib.rs and src/math.rs — two directories
        // are both under "src", so we expect exactly one group named "src".
        // The small_rust fixture has all files under src/, so one group.
        let output =
            run_search_with_claudix(&harness.claudix, "add greet".to_owned(), 10, None, None).await;
        assert!(output.is_ok());
        let output = output.ok().unwrap_or_else(|| unreachable!());

        assert!(!output.groups.is_empty(), "expected at least one group");

        // Every group's directory must equal immediate_parent_dir of its hits.
        for group in &output.groups {
            for hit in &group.hits {
                assert_eq!(
                    immediate_parent_dir(&hit.file_path),
                    group.directory,
                    "hit {} belongs in wrong group",
                    hit.file_path,
                );
            }
        }

        // Hits within each group must be in score-descending order.
        for group in &output.groups {
            let scores: Vec<f32> = group.hits.iter().map(|h| h.score).collect();
            let mut sorted = scores.clone();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            assert_eq!(
                scores, sorted,
                "hits in group '{}' not score-desc",
                group.directory
            );
        }
    }

    #[tokio::test]
    async fn search_grouping_preserves_top_hit_ranking() {
        let harness = cli_harness().await;
        assert!(harness.is_ok());
        let harness = harness.ok().unwrap_or_else(|| unreachable!());

        let output =
            run_search_with_claudix(&harness.claudix, "add".to_owned(), 5, None, None).await;
        assert!(output.is_ok());
        let output = output.ok().unwrap_or_else(|| unreachable!());

        assert!(!output.groups.is_empty());

        // The globally top-ranked hit must be groups[0].hits[0] — grouping must
        // not reorder the flat ranking, only reshape its presentation.
        let top = &output.groups[0].hits[0];
        let all_scores: Vec<f32> = output
            .groups
            .iter()
            .flat_map(|g| g.hits.iter().map(|h| h.score))
            .collect();
        let global_max = all_scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert_eq!(
            top.score, global_max,
            "top hit in groups[0] must have the globally highest score"
        );
    }
}
