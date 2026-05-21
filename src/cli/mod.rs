mod input;
mod install;
mod watch;

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
use crate::types::RelativePath;
use crate::{Claudix, IndexFileStatus, IndexProgress};

use input::{
    parse_language_filter, parse_path_prefix, validate_search_query, validate_search_top_k,
};

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
pub struct SearchOutput {
    pub hits: Vec<SearchHit>,
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
            .map(|f| Self { writer: io::BufWriter::new(f) })
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
    let (embedding_healthy, embedding_model_mismatch) = match claudix {
        Ok(claudix) => (claudix.embedder_health_check().await.is_ok(), false),
        Err(ClaudixError::EmbeddingModelMismatch { .. }) => (false, true),
        Err(_) => (false, false),
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

    Ok(SearchOutput {
        hits: results
            .into_iter()
            .map(|result| SearchHit {
                file_path: result.chunk.file_path.to_string(),
                language: result.chunk.language.to_string(),
                kind: result.chunk.kind.to_string(),
                name: result.chunk.name,
                line_start: result.chunk.line_range.start,
                line_end: result.chunk.line_range.end,
                score: result.score,
                stale: result.stale,
                snippet: result.chunk.content,
            })
            .collect(),
    })
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

        assert!(!output.hits.is_empty());
        assert_eq!(output.hits[0].name.as_deref(), Some("add"));
        assert_eq!(output.hits[0].file_path, "src/math.rs");
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

        assert_eq!(output.hits.len(), 1);
        assert_eq!(output.hits[0].file_path, "src/math.rs");
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
}
