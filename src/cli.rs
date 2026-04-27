use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::Serialize;

use crate::Claudix;
use crate::config;
use crate::error::{ClaudixError, RecoveryHint, Result};
use crate::hooks::HookEvent;
use crate::search::SearchQuery;
use crate::store::Store;
use crate::types::{Language, RelativePath};

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SearchHit {
    pub file_path: String,
    pub language: String,
    pub kind: String,
    pub name: Option<String>,
    pub line_start: u32,
    pub line_end: u32,
    pub score: String,
    pub snippet: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SearchOutput {
    pub hits: Vec<SearchHit>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct IndexOutput {
    pub file_count: usize,
    pub chunk_count: usize,
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
}

pub async fn run_search(
    project_root: impl AsRef<Path>,
    query: String,
    top_k: Option<usize>,
    language_filter: Option<Vec<String>>,
    path_prefix: Option<String>,
) -> Result<SearchOutput> {
    let project_root = canonical_project_root(project_root.as_ref())?;
    let config = config::load(&project_root)?;
    let top_k = top_k.unwrap_or(config.search.top_k);
    let claudix = Claudix::new(project_root, Arc::new(config)).await?;

    run_search_with_claudix(&claudix, query, top_k, language_filter, path_prefix).await
}

pub async fn run_index(project_root: impl AsRef<Path>) -> Result<IndexOutput> {
    let project_root = canonical_project_root(project_root.as_ref())?;
    let config = config::load(&project_root)?;
    let claudix = Claudix::new(project_root, Arc::new(config)).await?;
    let stats = claudix.index_full().await?;

    Ok(IndexOutput {
        file_count: stats.file_count,
        chunk_count: stats.chunk_count,
    })
}

pub async fn run_status(project_root: impl AsRef<Path>) -> Result<StatusOutput> {
    let project_root = canonical_project_root(project_root.as_ref())?;
    let config = config::load(&project_root)?;
    let store = Store::new(&project_root, &config)?;
    status_from_store(&store).await
}

pub async fn run_reindex_file(
    project_root: impl AsRef<Path>,
    path: impl AsRef<Path>,
) -> Result<IndexOutput> {
    let project_root = canonical_project_root(project_root.as_ref())?;
    let config = config::load(&project_root)?;
    let claudix = Claudix::new(project_root, Arc::new(config)).await?;
    let stats = claudix.reindex_file(path.as_ref()).await?;

    Ok(IndexOutput {
        file_count: stats.file_count,
        chunk_count: stats.chunk_count,
    })
}

pub async fn run_clear_index(project_root: impl AsRef<Path>) -> Result<ClearOutput> {
    let project_root = canonical_project_root(project_root.as_ref())?;
    let config = config::load(&project_root)?;
    let store = Store::new(&project_root, &config)?;
    store.clear_chunks(&config).await?;

    Ok(ClearOutput { cleared: true })
}

pub fn parse_hook_event(value: &str) -> Result<HookEvent> {
    match value {
        "SessionStart" => Ok(HookEvent::SessionStart),
        "PostToolUse" => Ok(HookEvent::PostToolUse),
        "PreToolUse" => Ok(HookEvent::PreToolUse),
        _ => Err(ClaudixError::ConfigInvalid {
            message: format!("unknown hook event: {value}"),
            recovery: RecoveryHint("Use one of: SessionStart, PostToolUse, PreToolUse"),
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
        path_prefix: path_prefix.map(RelativePath::new),
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
                score: format!("{:.3}", result.score),
                snippet: result.chunk.content,
            })
            .collect(),
    })
}

async fn status_from_store(store: &Store) -> Result<StatusOutput> {
    let manifest = store.read_manifest()?;
    let stats = store.chunk_stats().await?;

    Ok(StatusOutput {
        chunk_count: stats.chunk_count,
        file_count: stats.file_count,
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
    })
}

fn canonical_project_root(project_root: &Path) -> Result<PathBuf> {
    project_root.canonicalize().map_err(ClaudixError::from)
}

fn parse_language_filter(language_filter: Option<Vec<String>>) -> Result<Option<Vec<Language>>> {
    let Some(language_filter) = language_filter else {
        return Ok(None);
    };

    let mut parsed = Vec::with_capacity(language_filter.len());
    for language in language_filter {
        parsed.push(parse_language(&language)?);
    }

    Ok(Some(parsed))
}

fn parse_language(value: &str) -> Result<Language> {
    match value.to_ascii_lowercase().as_str() {
        "rust" => Ok(Language::Rust),
        "python" => Ok(Language::Python),
        "javascript" | "js" => Ok(Language::JavaScript),
        "typescript" | "ts" => Ok(Language::TypeScript),
        "go" => Ok(Language::Go),
        "java" => Ok(Language::Java),
        "c" => Ok(Language::C),
        "cpp" | "c++" => Ok(Language::Cpp),
        "unknown" => Ok(Language::Unknown),
        _ => Err(ClaudixError::ConfigInvalid {
            message: format!("unsupported language filter: {value}"),
            recovery: RecoveryHint(
                "Use one of: rust, python, javascript, typescript, go, java, c, cpp, unknown",
            ),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunking::{Chunker, MultiLanguageChunker};
    use crate::config::Config;
    use crate::embedding::{Provider, StubProvider};
    use crate::enumeration::FileEnumerator;
    use crate::store::Store;
    use crate::types::{Dimension, EmbeddedChunk};
    use tokio::{fs, task};

    mod fixture {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/common/fixture.rs"
        ));
    }

    use fixture::TestFixture;

    struct CliHarness {
        _fixture: TestFixture,
        claudix: Claudix,
        store: Store,
    }

    fn test_config() -> Config {
        let mut config = Config::default();
        config.embedding.model = "stub-v1".to_owned();
        config.embedding.dimensions = 8;
        config
    }

    fn test_claudix(project_root: PathBuf, config: Config) -> Result<Claudix> {
        let store = Store::new(&project_root, &config)?;
        let embedder: Arc<dyn Provider> = Arc::new(StubProvider::with_model_id(
            config.embedding.model.clone(),
            Dimension(config.embedding.dimensions),
        ));

        Ok(Claudix {
            config: Arc::new(config),
            project_root,
            embedder,
            store,
        })
    }

    async fn cli_harness() -> Result<CliHarness> {
        let fixture = TestFixture::new("small_rust")?;
        let config = test_config();
        let claudix = test_claudix(fixture.root().to_path_buf(), config.clone())?;
        index_fixture(&claudix, &config).await?;
        let store = Store::new(fixture.root(), &config)?;

        Ok(CliHarness {
            _fixture: fixture,
            claudix,
            store,
        })
    }

    async fn index_fixture(claudix: &Claudix, config: &Config) -> Result<()> {
        let enumerator = FileEnumerator::new(claudix.project_root().to_path_buf(), config.clone())?;
        let files = enumerator.enumerate()?;
        let mut chunks = Vec::new();

        for file in files {
            let content = fs::read_to_string(&file.absolute_path).await?;
            let path = file.relative_path.clone();
            let language = file.language;
            let file_hash = file.file_hash;

            let file_chunks = task::spawn_blocking(move || {
                MultiLanguageChunker::new().chunk(&path, language, file_hash, &content)
            })
            .await
            .map_err(|error| ClaudixError::TreeSitter(error.to_string()))??;
            chunks.extend(file_chunks);
        }

        let inputs = chunks
            .iter()
            .map(|chunk| chunk.content.as_str())
            .collect::<Vec<_>>();
        let vectors = claudix.embedder.embed(&inputs).await?;
        let embedded_chunks = chunks
            .into_iter()
            .zip(vectors)
            .map(|(chunk, vector)| EmbeddedChunk { chunk, vector })
            .collect::<Vec<_>>();

        claudix
            .store
            .replace_chunks(&embedded_chunks, claudix.config())
            .await?;
        Ok(())
    }

    #[test]
    fn parse_hook_event_accepts_known_values() {
        let event = parse_hook_event("SessionStart");
        assert!(matches!(event, Ok(HookEvent::SessionStart)));

        let event = parse_hook_event("PostToolUse");
        assert!(matches!(event, Ok(HookEvent::PostToolUse)));

        let event = parse_hook_event("PreToolUse");
        assert!(matches!(event, Ok(HookEvent::PreToolUse)));
    }

    #[test]
    fn parse_hook_event_rejects_unknown_values() {
        let result = parse_hook_event("Unknown");
        assert!(matches!(result, Err(ClaudixError::ConfigInvalid { .. })));
    }

    #[test]
    fn parse_language_filter_accepts_aliases() {
        let parsed = parse_language_filter(Some(vec!["rs".to_owned(), "ts".to_owned()]));
        assert!(parsed.is_err());

        let parsed = parse_language_filter(Some(vec!["rust".to_owned(), "ts".to_owned()]));
        assert!(matches!(
            parsed,
            Ok(Some(ref languages)) if languages == &vec![Language::Rust, Language::TypeScript]
        ));
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

    #[tokio::test]
    async fn run_status_reports_manifest_and_counts() {
        let harness = cli_harness().await;
        assert!(harness.is_ok());
        let harness = harness.ok().unwrap_or_else(|| unreachable!());

        let status = status_from_store(&harness.store).await;
        assert!(status.is_ok());
        let status = status.ok().unwrap_or_else(|| unreachable!());

        assert_eq!(status.chunk_count, 3);
        assert_eq!(status.file_count, 2);
        assert_eq!(status.model.as_deref(), Some("stub-v1"));
        assert_eq!(status.dimensions, Some(8));
    }
}
