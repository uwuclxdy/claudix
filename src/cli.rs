use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::Serialize;
use tokio::fs;

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
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct InstallOutput {
    pub plugin_root: String,
    pub binary_path: String,
    pub config_path: String,
    pub wrote_config: bool,
    pub next_step: Option<String>,
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

pub async fn run_doctor(project_root: impl AsRef<Path>) -> Result<DoctorOutput> {
    let project_root = canonical_project_root(project_root.as_ref())?;
    let config = config::load(&project_root)?;
    let store = Store::new(&project_root, &config)?;
    let status = status_from_store(&store).await?;
    let claudix = Claudix::new(project_root.clone(), Arc::new(config.clone())).await;
    let embedding_healthy = match claudix {
        Ok(claudix) => claudix.embedder_health_check().await.is_ok(),
        Err(_) => false,
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
    })
}

pub async fn run_clear_index(project_root: impl AsRef<Path>) -> Result<ClearOutput> {
    let project_root = canonical_project_root(project_root.as_ref())?;
    let config = config::load(&project_root)?;
    let store = Store::new(&project_root, &config)?;
    store.clear_chunks(&config).await?;

    Ok(ClearOutput { cleared: true })
}

pub async fn run_install(project_root: impl AsRef<Path>) -> Result<InstallOutput> {
    let project_root = canonical_project_root(project_root.as_ref())?;
    let plugin_root = plugin_root(&project_root)?;
    let binary_path = plugin_root.join("bin").join(binary_name());
    let config_path = global_config_path()?;

    install_plugin_assets(&project_root, &plugin_root).await?;

    let wrote_config = ensure_global_config(&config_path).await?;

    Ok(InstallOutput {
        plugin_root: plugin_root.display().to_string(),
        binary_path: binary_path.display().to_string(),
        config_path: config_path.display().to_string(),
        wrote_config,
        next_step: install_next_step(&plugin_root),
    })
}

pub async fn run_auto_install(project_root: impl AsRef<Path>) -> Option<String> {
    let project_root = project_root.as_ref();
    let source_root = plugin_asset_source(project_root).ok()?;
    let plugin_root =
        plugin_root_from_env(project_root, std::env::var_os("CLAUDE_PLUGIN_ROOT")).ok()?;
    let config_path = global_config_path().ok()?;

    let installed_assets = install_plugin_assets(&source_root, &plugin_root)
        .await
        .ok()?;
    let wrote_config = ensure_global_config(&config_path).await.ok()?;

    if installed_assets || wrote_config {
        Some(format!(
            "claudix: setup complete — config at {}. Restart Claude Code to load MCP, then run /claudix:index.",
            config_path.display()
        ))
    } else {
        None
    }
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

async fn install_plugin_assets(project_root: &Path, plugin_root: &Path) -> Result<bool> {
    let mut changed = false;
    changed |= copy_plugin_asset(
        project_root,
        ".claude-plugin/plugin.json",
        plugin_root.join(".claude-plugin").join("plugin.json"),
    )
    .await?;
    changed |= copy_plugin_asset(
        project_root,
        "hooks/hooks.json",
        plugin_root.join("hooks").join("hooks.json"),
    )
    .await?;
    changed |= copy_plugin_asset(
        project_root,
        "bin/claudix",
        plugin_root.join("bin").join(binary_name()),
    )
    .await?;
    make_executable(&plugin_root.join("bin").join(binary_name())).await?;
    changed |=
        copy_plugin_directory(project_root, "commands", plugin_root.join("commands")).await?;
    changed |= copy_plugin_directory(project_root, "scripts", plugin_root.join("scripts")).await?;
    Ok(changed)
}

async fn copy_plugin_asset(
    project_root: &Path,
    source_relative: &str,
    destination: PathBuf,
) -> Result<bool> {
    let source = required_plugin_asset(project_root, source_relative).await?;

    if source == destination || files_match(&source, &destination).await? {
        return Ok(false);
    }

    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent).await?;
    }
    fs::copy(source, &destination).await?;
    Ok(true)
}

async fn copy_plugin_directory(
    project_root: &Path,
    source_relative: &str,
    destination: PathBuf,
) -> Result<bool> {
    let source = required_plugin_asset(project_root, source_relative).await?;
    if source == destination || directories_match(&source, &destination).await? {
        return Ok(false);
    }

    if fs::try_exists(&destination).await? {
        fs::remove_dir_all(&destination).await?;
    }
    fs::create_dir_all(&destination).await?;

    let mut entries = fs::read_dir(source).await?;
    while let Some(entry) = entries.next_entry().await? {
        let file_type = entry.file_type().await?;
        if file_type.is_file() {
            let destination_file = destination.join(entry.file_name());
            fs::copy(entry.path(), &destination_file).await?;
            if destination_file
                .extension()
                .is_some_and(|extension| extension == "sh")
            {
                make_executable(&destination_file).await?;
            }
        }
    }

    Ok(true)
}

async fn files_match(left: &Path, right: &Path) -> Result<bool> {
    if !fs::try_exists(right).await? {
        return Ok(false);
    }

    let left_metadata = fs::metadata(left).await?;
    let right_metadata = fs::metadata(right).await?;
    if left_metadata.len() != right_metadata.len() {
        return Ok(false);
    }

    Ok(fs::read(left).await? == fs::read(right).await?)
}

async fn directories_match(left: &Path, right: &Path) -> Result<bool> {
    if !fs::try_exists(right).await? {
        return Ok(false);
    }

    let mut left_entries = directory_file_names(left).await?;
    let mut right_entries = directory_file_names(right).await?;
    left_entries.sort();
    right_entries.sort();
    if left_entries != right_entries {
        return Ok(false);
    }

    for entry in left_entries {
        if !files_match(&left.join(&entry), &right.join(&entry)).await? {
            return Ok(false);
        }
    }

    Ok(true)
}

async fn directory_file_names(path: &Path) -> Result<Vec<std::ffi::OsString>> {
    let mut file_names = Vec::new();
    let mut entries = fs::read_dir(path).await?;
    while let Some(entry) = entries.next_entry().await? {
        if entry.file_type().await?.is_file() {
            file_names.push(entry.file_name());
        }
    }
    Ok(file_names)
}

async fn required_plugin_asset(project_root: &Path, source_relative: &str) -> Result<PathBuf> {
    let source = project_root.join(source_relative);
    if fs::try_exists(&source).await? {
        return Ok(source);
    }

    Err(ClaudixError::ConfigInvalid {
        message: format!("required plugin asset missing: {}", source.display()),
        recovery: RecoveryHint("Restore the plugin metadata files before running claudix install"),
    })
}

fn install_next_step(plugin_root: &Path) -> Option<String> {
    if !plugin_root.starts_with(local_plugin_root()) {
        return None;
    }

    Some(format!(
        "Run `claude --plugin-dir {}` or install this path with `/plugin install {}`.",
        plugin_root.display(),
        plugin_root.display()
    ))
}

fn local_plugin_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("claudix-plugin")
}

async fn make_executable(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        let mut permissions = fs::metadata(path).await?.permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(path, permissions).await?;
    }

    Ok(())
}

async fn ensure_global_config(config_path: &Path) -> Result<bool> {
    if fs::try_exists(config_path).await? {
        return Ok(false);
    }

    if let Some(parent) = config_path.parent() {
        fs::create_dir_all(parent).await?;
    }

    fs::write(config_path, default_global_config()).await?;
    Ok(true)
}

fn default_global_config() -> &'static str {
    "# Global claudix configuration\n# Uncomment and edit values as needed.\n\n[embedding]\n# provider = \"bundled\"\n# model = \"bge-small-en-v1.5\"\n# dimensions = 384\n# endpoint = \"http://localhost:11434\"\n\n[indexing]\n# reindex_after_hours = 24\n"
}

fn plugin_root(project_root: &Path) -> Result<PathBuf> {
    plugin_root_from_env(project_root, std::env::var_os("CLAUDE_PLUGIN_ROOT"))
}

fn plugin_asset_source(project_root: &Path) -> Result<PathBuf> {
    if is_claudix_plugin_root(project_root) {
        return Ok(project_root.to_path_buf());
    }

    let source_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if is_claudix_plugin_root(&source_root) {
        return Ok(source_root);
    }

    plugin_root(project_root)
}

fn plugin_root_from_env(
    project_root: &Path,
    plugin_root_env: Option<std::ffi::OsString>,
) -> Result<PathBuf> {
    if is_claudix_plugin_root(project_root) {
        return Ok(local_plugin_root());
    }

    if let Some(path) = plugin_root_env {
        let plugin_root = PathBuf::from(path);
        if is_claudix_plugin_root(&plugin_root) {
            return Ok(plugin_root);
        }
    }

    Err(ClaudixError::ConfigInvalid {
        message: "CLAUDE_PLUGIN_ROOT is not set".into(),
        recovery: RecoveryHint(
            "Run claudix install from the plugin directory or plugin environment",
        ),
    })
}

fn is_claudix_plugin_root(path: &Path) -> bool {
    let manifest_path = path.join(".claude-plugin").join("plugin.json");
    let Ok(manifest) = std::fs::read_to_string(manifest_path) else {
        return false;
    };

    manifest.contains("\"name\": \"claudix\"")
}

fn global_config_path() -> Result<PathBuf> {
    dirs::home_dir()
        .map(|home| home.join(".claude").join("claudix.toml"))
        .ok_or_else(|| ClaudixError::ConfigInvalid {
            message: "home directory is not available".into(),
            recovery: RecoveryHint("Set HOME before running claudix install"),
        })
}

fn binary_name() -> &'static str {
    if cfg!(windows) {
        "claudix.exe"
    } else {
        "claudix"
    }
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
    use tempfile::tempdir;
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
        let config = Arc::new(config);
        let embedder: Arc<dyn Provider> = Arc::new(StubProvider::with_model_id(
            config.embedding.model.clone(),
            Dimension(config.embedding.dimensions),
        ));

        Ok(Claudix::from_parts(project_root, config, embedder, store))
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

    #[tokio::test]
    async fn run_doctor_reports_index_and_embedding_health() {
        let harness = cli_harness().await;
        assert!(harness.is_ok());
        let harness = harness.ok().unwrap_or_else(|| unreachable!());

        let mut config = test_config();
        config.hooks.session_start_warmup = false;
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
    async fn ensure_global_config_writes_default_once() {
        let temp = tempdir();
        assert!(temp.is_ok());
        let temp = temp.ok().unwrap_or_else(|| unreachable!());
        let config_path = temp.path().join(".claude").join("claudix.toml");

        let wrote_config = ensure_global_config(&config_path).await;
        assert!(wrote_config.is_ok());
        assert!(wrote_config.ok().unwrap_or(false));

        let contents = fs::read_to_string(&config_path).await;
        assert!(contents.is_ok());
        assert!(contents.ok().unwrap_or_default().contains("[embedding]"));

        let wrote_config = ensure_global_config(&config_path).await;
        assert!(wrote_config.is_ok());
        assert!(!wrote_config.ok().unwrap_or(true));
    }

    #[tokio::test]
    async fn install_copies_plugin_assets_into_plugin_root() {
        let fixture = TestFixture::new("small_rust");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());
        let plugin_root = fixture.root().join("plugin-root");

        let result =
            install_plugin_assets(Path::new(env!("CARGO_MANIFEST_DIR")), &plugin_root).await;
        assert!(result.is_ok());
        assert!(result.ok().unwrap_or(false));

        let second_result =
            install_plugin_assets(Path::new(env!("CARGO_MANIFEST_DIR")), &plugin_root).await;
        assert!(second_result.is_ok());
        assert!(!second_result.ok().unwrap_or(true));

        let plugin_manifest =
            fs::read_to_string(plugin_root.join(".claude-plugin").join("plugin.json")).await;
        assert!(plugin_manifest.is_ok());
        let plugin_manifest = plugin_manifest.ok().unwrap_or_default();
        assert!(plugin_manifest.contains("\"name\": \"claudix\""));
        assert!(plugin_manifest.contains("\"mcpServers\""));
        assert!(plugin_manifest.contains("\"args\": [\"mcp\"]"));

        let hooks_manifest = fs::read_to_string(plugin_root.join("hooks").join("hooks.json")).await;
        assert!(hooks_manifest.is_ok());
        assert!(
            hooks_manifest
                .ok()
                .unwrap_or_default()
                .contains("scripts/session-start.sh")
        );

        let wrapper = fs::read_to_string(plugin_root.join("bin").join("claudix")).await;
        assert!(wrapper.is_ok());
        assert!(wrapper.ok().unwrap_or_default().contains("CARGO_BIN"));

        let search_command =
            fs::read_to_string(plugin_root.join("commands").join("search.md")).await;
        assert!(search_command.is_ok());
        let search_command = search_command.ok().unwrap_or_default();
        assert!(search_command.contains("!`claudix search"));
        assert!(!search_command.contains("CLAUDE_PLUGIN_ROOT"));

        let updater = fs::read_to_string(plugin_root.join("scripts").join("check-update.sh")).await;
        assert!(updater.is_ok());
        assert!(updater.ok().unwrap_or_default().contains("github.com"));
    }

    #[test]
    fn plugin_root_uses_claudix_environment_value_outside_local_checkout() {
        let fixture = TestFixture::new("small_rust");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());
        let env_root = fixture.root().join("env-plugin-root");
        assert!(std::fs::create_dir_all(env_root.join(".claude-plugin")).is_ok());
        assert!(
            std::fs::write(
                env_root.join(".claude-plugin").join("plugin.json"),
                "{\"name\": \"claudix\"}",
            )
            .is_ok()
        );
        let project_root = fixture.root().join("project");
        assert!(std::fs::create_dir_all(&project_root).is_ok());

        let result = plugin_root_from_env(&project_root, Some(env_root.clone().into_os_string()));
        assert!(result.is_ok());
        assert_eq!(result.ok().unwrap_or_default(), env_root);
    }

    #[test]
    fn plugin_root_ignores_foreign_environment_value() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let fixture = TestFixture::new("small_rust");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());

        let result = plugin_root_from_env(root, Some(fixture.root().as_os_str().to_os_string()));
        assert!(result.is_ok());
        assert_eq!(
            result.ok().unwrap_or_default(),
            root.join("target").join("claudix-plugin")
        );
    }

    #[test]
    fn plugin_root_falls_back_to_local_manifest() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));

        let result = plugin_root_from_env(root, None);
        assert!(result.is_ok());
        assert_eq!(
            result.ok().unwrap_or_default(),
            root.join("target").join("claudix-plugin")
        );
    }

    #[test]
    fn plugin_root_requires_environment_or_local_manifest() {
        let fixture = TestFixture::new("small_rust");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());

        let result = plugin_root_from_env(fixture.root(), None);
        assert!(result.is_err());
    }

    #[test]
    fn install_next_step_is_present_for_local_bundle() {
        let plugin_root = local_plugin_root();

        let next_step = install_next_step(&plugin_root);
        assert!(next_step.is_some());
        assert!(next_step.unwrap_or_default().contains("/plugin install"));
    }

    #[test]
    fn install_next_step_is_absent_for_enabled_plugin_root() {
        let fixture = TestFixture::new("small_rust");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());

        let next_step = install_next_step(fixture.root());
        assert!(next_step.is_none());
    }

    #[test]
    fn default_global_config_includes_commented_defaults() {
        let config = default_global_config();
        assert!(config.contains("[embedding]"));
        assert!(config.contains("provider = \"bundled\""));
        assert!(config.contains("reindex_after_hours = 24"));
    }

    #[test]
    fn binary_name_matches_platform() {
        if cfg!(windows) {
            assert_eq!(binary_name(), "claudix.exe");
        } else {
            assert_eq!(binary_name(), "claudix");
        }
    }
}
