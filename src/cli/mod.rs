mod drain;
mod input;
mod install;
mod watch;

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{self, Write};
use std::path::{Component, Path, PathBuf, Prefix};
use std::sync::Arc;

use serde::Serialize;

use crate::config;
use crate::error::{ClaudixError, RecoveryHint, Result};
use crate::hooks::HookEvent;
use crate::prompts::hints;
use crate::search::SearchQuery;
use crate::search::duplicates::{self, LabeledChunk};
pub use crate::search::duplicates::{DuplicateChunk, DuplicatePair};
use crate::store::{IndexLockGuard, Store};
use crate::types::{RelativePath, path_prefix_matches};
use crate::{Claudix, IndexFileStatus, IndexProgress};

use input::{
    parse_language_filter, parse_path_prefix, validate_search_query, validate_search_top_k,
};

/// Default cosine-similarity floor for [`run_find_duplicates`]. Higher = stricter / fewer pairs.
pub(crate) const DEFAULT_MIN_SIMILARITY: f32 = 0.85;
/// Default maximum number of duplicate pairs returned by [`run_find_duplicates`].
pub(crate) const DEFAULT_DUPLICATE_LIMIT: usize = 50;
/// Hard ceiling on the combined chunk count fed to the O(n²) duplicate scan.
/// `limit` caps only the output heap, not the input, so a caller pointing at
/// many large repos could drive an unbounded pairwise scan (≈ n²/2 comparisons).
/// Beyond this the scan is skipped and a notice is surfaced instead.
pub(crate) const MAX_DUPLICATE_CORPUS_CHUNKS: usize = 50_000;

pub use drain::run_drain_reindex_queue;
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
    /// Serialized only when true: a `false` on every hit is dead weight in the
    /// agent's context, and `SearchOutput::stale_hint` explains the flag when
    /// any hit actually carries it. The repo a hit came from is not repeated
    /// here — it is the owning `DirectoryGroup::repo`.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub stale: bool,
    /// Capped at [`prompts::SNIPPET_MAX_LINES`]; an uncapped tree-sitter chunk
    /// can be an entire impl block.
    pub snippet: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DirectoryGroup {
    /// Canonical repo path the directory belongs to. Same-named directories in
    /// different repos do not merge — the group key is `(repo, directory)`.
    pub repo: String,
    pub directory: String,
    pub hits: Vec<SearchHit>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SearchOutput {
    pub groups: Vec<DirectoryGroup>,
    /// Repos that could not be searched (unindexed, mismatched, missing).
    /// Empty for single-repo searches.
    pub repo_errors: Vec<RepoError>,
    /// What a `stale` hit means, carried only when at least one hit is stale.
    /// Conditional guidance rides the response instead of the tool description,
    /// so a session that never sees a stale hit never pays for the explanation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stale_hint: Option<&'static str>,
    /// Set when results were ranked lexical-only: the endpoint-down notice
    /// when the embedding endpoint was unreachable, or the reindex hint when
    /// the stored index's embedding model/dimensions differ from the
    /// configured provider. The MCP layer throttles this to once per session;
    /// the non-cached CLI path (a fresh process) carries it every time.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub degraded_hint: Option<&'static str>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct IndexOutput {
    pub file_count: usize,
    pub chunk_count: usize,
    /// Set when the full pass enumerated files but stored no chunks; `main.rs`
    /// prints it on stderr after the summary. Skipped from serialized output
    /// unless present (the MCP reindex sweep then surfaces it to the agent).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub empty_index_warning: Option<String>,
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
    /// Dev-only flag from config: when on, `bin/claudix-bootstrap.js` runs the
    /// `cargo install` binary instead of the downloaded release.
    pub development_mode: bool,
    /// Absolute path of the binary serving this command, so it is unambiguous
    /// which build is running (cargo vs. cached release) during development.
    pub binary_path: String,
    /// Last `error:` line the node bootstrap recorded in install.log when a binary
    /// download/verify failed, so a permanently-dead MCP has a visible cause instead
    /// of failing silently across sessions. None when no install.log or no error.
    pub install_error: Option<String>,
    /// Absolute path to the bootstrap's install.log, if resolvable, for the agent
    /// to open and inspect. None when `CLAUDE_PLUGIN_DATA` is unset (manual shell run).
    pub install_log_path: Option<String>,
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

/// A repo that could not contribute to the duplicate scan, with a reason.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RepoError {
    pub repo: String,
    pub error: String,
}

/// Result of [`run_find_duplicates`].
///
/// Partial success is intentional: indexed repos produce pairs even when
/// some listed repos errored.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DuplicatesOutput {
    pub pairs: Vec<DuplicatePair>,
    pub repo_errors: Vec<RepoError>,
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

    // Metadata projection — skips embedding vectors; only file_path, file_hash,
    // language, and name are needed for the directory rollup.
    let metadata = store.read_chunk_metadata().await?;

    // Group chunks by the immediate parent directory of their file_path.
    // Splitting on `/` is sound because file_path comes from RelativePath,
    // which forward-slash-normalizes on construction on every platform.
    // Aggregation is light counting over an already-loaded Vec — no spawn_blocking needed.
    //
    // One aggregate per directory keeps the key clone count to at most one per
    // chunk (on the `entry().or_insert_with` path for new dirs, zero for existing).
    struct DirAggregate {
        files: std::collections::HashSet<String>,
        chunk_count: usize,
        languages: HashMap<String, usize>,
    }

    let mut dir_map: HashMap<String, DirAggregate> = HashMap::new();

    for chunk in &metadata {
        // Apply path-prefix filter, consistent with `apply_filters` in search.
        if let Some(ref prefix) = prefix
            && !path_prefix_matches(&chunk.file_path, prefix.as_str())
        {
            continue;
        }

        let dir = immediate_parent_dir(&chunk.file_path);
        let agg = dir_map.entry(dir).or_insert_with(|| DirAggregate {
            files: std::collections::HashSet::new(),
            chunk_count: 0,
            languages: HashMap::new(),
        });
        agg.files.insert(chunk.file_path.clone());
        agg.chunk_count += 1;
        *agg.languages.entry(chunk.language.clone()).or_insert(0) += 1;
    }

    let mut directories: Vec<DirectoryRollup> = dir_map
        .into_iter()
        .map(|(dir, agg)| {
            let mut languages: Vec<LanguageCount> = agg
                .languages
                .into_iter()
                .map(|(language, chunk_count)| LanguageCount {
                    language,
                    chunk_count,
                })
                .collect();
            // Sort by chunk count desc, then language name asc for determinism.
            languages.sort_by(|a, b| {
                b.chunk_count
                    .cmp(&a.chunk_count)
                    .then_with(|| a.language.cmp(&b.language))
            });

            DirectoryRollup {
                path: dir,
                file_count: agg.files.len(),
                chunk_count: agg.chunk_count,
                languages,
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

/// Read the manifest JSON sidecar directly from a repo path without opening a
/// Store (no LanceDB connection machinery). Derives the manifest path from the
/// repo's config the same way `Store::new` would, but avoids a second Store
/// construction in the `load_repo_chunks_readonly` call that always follows.
///
/// The manifest file name and location logic must stay in sync with
/// `Store::new` + `store::manifest::MANIFEST_FILE_NAME`.
fn read_manifest_at_repo(
    repo_path: &str,
) -> std::result::Result<Option<crate::store::Manifest>, RepoError> {
    let config = config::load(std::path::Path::new(repo_path)).map_err(|e| RepoError {
        repo: repo_path.to_owned(),
        error: e.to_string(),
    })?;

    // Replicate `Store::new` path derivation: canonicalize root, join the
    // config-relative index_dir (validated at config-load time to not escape),
    // take its parent as state_dir, append the manifest file name.
    let root = std::path::Path::new(repo_path)
        .canonicalize()
        .map_err(|e| RepoError {
            repo: repo_path.to_owned(),
            error: e.to_string(),
        })?;

    let index_dir = root.join(&config.paths.index_dir);

    let state_dir = index_dir.parent().ok_or_else(|| RepoError {
        repo: repo_path.to_owned(),
        error: "index path has no parent directory".to_owned(),
    })?;

    let manifest_path = state_dir.join(crate::store::manifest::MANIFEST_FILE_NAME);

    if !manifest_path.exists() {
        return Ok(None);
    }

    let text = fs::read_to_string(&manifest_path).map_err(|e| RepoError {
        repo: repo_path.to_owned(),
        error: e.to_string(),
    })?;

    let manifest: crate::store::Manifest = serde_json::from_str(&text).map_err(|e| RepoError {
        repo: repo_path.to_owned(),
        error: e.to_string(),
    })?;

    Ok(Some(manifest))
}

/// Read the embedding identity (model, dimensions) from a repo's manifest without
/// validating against any reference. Used to bootstrap the reference for the first
/// repo in a multi-repo scan.
fn peek_manifest_identity(repo_path: &str) -> std::result::Result<(String, u16), RepoError> {
    match read_manifest_at_repo(repo_path)? {
        Some(m) if m.chunk_count > 0 => Ok((m.embedding_model, m.dimensions)),
        _ => Err(RepoError {
            repo: repo_path.to_owned(),
            error: "not indexed".to_owned(),
        }),
    }
}

/// Open a repo read-only, confirm it is indexed and dimension-compatible,
/// and return its chunks labeled with the canonical repo path.
///
/// This helper is intentionally read-only: it calls only `Store::new`,
/// `read_manifest`, and `read_chunks` — never `ensure_layout` or `write_manifest`.
/// Feature 6 (cross-repo search) reuses this to load remote repos without writing.
///
/// Returns `Err(RepoError)` when the repo path is invalid, the index is absent,
/// the chunk count is zero, or the embedding identity (model + dimensions) does
/// not match `ref_model`/`ref_dims`.
pub(crate) async fn load_repo_chunks_readonly(
    repo_path: &str,
    ref_model: &str,
    ref_dims: u16,
) -> std::result::Result<(String, Vec<crate::store::StoredChunk>), RepoError> {
    let config = config::load(std::path::Path::new(repo_path)).map_err(|e| RepoError {
        repo: repo_path.to_owned(),
        error: e.to_string(),
    })?;

    let store = Store::new(repo_path, &config).map_err(|e| RepoError {
        repo: repo_path.to_owned(),
        error: e.to_string(),
    })?;

    // Use the canonical path the store resolved to as the stable repo key.
    let canonical_repo = store.project_root().display().to_string();

    let manifest = store
        .validate_manifest_compatibility(ref_model, ref_dims)
        .map_err(|e| RepoError {
            repo: canonical_repo.clone(),
            error: e.to_string(),
        })?;

    let Some(manifest) = manifest else {
        return Err(RepoError {
            repo: canonical_repo,
            error: "not indexed".to_owned(),
        });
    };

    if manifest.chunk_count == 0 {
        return Err(RepoError {
            repo: canonical_repo,
            error: "not indexed".to_owned(),
        });
    }

    let chunks = store.read_chunks().await.map_err(|e| RepoError {
        repo: canonical_repo.clone(),
        error: e.to_string(),
    })?;

    if chunks.is_empty() {
        return Err(RepoError {
            repo: canonical_repo,
            error: "index chunks missing".to_owned(),
        });
    }

    Ok((canonical_repo, chunks))
}

/// Best-effort canonical dedup key for a caller-supplied repo path: the same
/// `canonicalize` the store applies, falling back to the raw spelling when the
/// path doesn't resolve. Keyed before loading so the error branches dedup too —
/// an errored repo never reaches the store's canonical label, so two spellings
/// of one broken repo would otherwise each earn a `RepoError`.
pub(crate) fn repo_dedup_key(path: &str) -> String {
    let normalized = degrade_verbatim_prefix(path);
    Path::new(&normalized)
        .canonicalize()
        .map_or_else(|_| path.to_owned(), |c| c.display().to_string())
}

/// Rewrite a Windows verbatim (`\\?\`) disk or UNC prefix to its plain
/// equivalent (`C:`, `\\server\share`); a no-op elsewhere. Verbatim paths
/// bypass Win32's own path normalization, so `/` stops acting as a separator
/// and `.` components stop collapsing there. A path derived by string-editing
/// an already-canonical verbatim path (e.g. appending `/.`) then can't
/// canonicalize back to the same identity, breaking dedup. `canonicalize`
/// re-derives the correct verbatim form for the real filesystem path anyway,
/// so degrading the prefix first only restores normal parsing for the rest.
fn degrade_verbatim_prefix(path: &str) -> String {
    let Some(Component::Prefix(prefix)) = Path::new(path).components().next() else {
        return path.to_owned();
    };
    let plain_prefix = match prefix.kind() {
        Prefix::VerbatimDisk(drive) => format!("{}:", drive as char),
        Prefix::VerbatimUNC(server, share) => {
            format!(
                "\\\\{}\\{}",
                server.to_string_lossy(),
                share.to_string_lossy()
            )
        }
        _ => return path.to_owned(),
    };
    match path.get(prefix.as_os_str().len()..) {
        Some(rest) => format!("{plain_prefix}{rest}"),
        None => path.to_owned(),
    }
}

/// Find near-duplicate code chunks in the active repo, plus any repos in `repos`.
///
/// `project_root` is always scanned; `repos` adds to it, matching `search_code`.
/// Repos are deduped by the canonical path their store resolves to, so naming the
/// active project explicitly is a no-op however it is spelled.
///
/// The reference embedding identity (model + dimensions) is established from the
/// manifest of the first repo that loads successfully. All subsequent repos must
/// match that identity or they become `RepoError`s.
///
/// The O(n²) detection scan runs inside `tokio::task::spawn_blocking`.
pub async fn run_find_duplicates(
    project_root: impl AsRef<Path>,
    min_similarity: Option<f32>,
    limit: Option<usize>,
    repos: Option<Vec<String>>,
) -> Result<DuplicatesOutput> {
    let project_root = canonical_project_root(project_root.as_ref())?;
    let min_similarity = min_similarity.unwrap_or(DEFAULT_MIN_SIMILARITY);
    if !min_similarity.is_finite() || !(0.0..=1.0).contains(&min_similarity) {
        return Err(ClaudixError::ConfigInvalid {
            message: "min_similarity must be between 0 and 1".to_owned(),
            recovery: RecoveryHint(hints::FINITE_MIN_SIMILARITY),
        });
    }
    let limit = limit.unwrap_or(DEFAULT_DUPLICATE_LIMIT);
    if limit == 0 {
        return Err(ClaudixError::ConfigInvalid {
            message: "limit must be at least 1".to_owned(),
            recovery: RecoveryHint(hints::POSITIVE_LIMIT),
        });
    }

    // The active project always participates and `repos` adds to it, matching
    // `search_code`.
    let repo_paths: Vec<String> = std::iter::once(project_root.display().to_string())
        .chain(repos.into_iter().flatten())
        .collect();

    let mut all_chunks: Vec<crate::store::StoredChunk> = Vec::new();
    let mut repo_labels: Vec<Arc<str>> = Vec::new();
    let mut repo_errors: Vec<RepoError> = Vec::new();

    // The reference embedding identity is taken from the first repo whose manifest
    // loads successfully. Subsequent repos must match or they become RepoErrors.
    let mut ref_identity: Option<(String, u16)> = None;

    // Dedup on the canonical path, never the caller's spelling: `--repo .`, a
    // trailing slash, a symlink, macOS `/var` vs `/private/var`, and Windows
    // `\\?\C:\r` vs `C:\r` all name one repo. Loading one twice emits every
    // cross-file pair four times, each burning a `limit` slot, and doubles the
    // O(n²) scan; erroring one twice repeats its `RepoError` per spelling.
    // `search` dedups the same way.
    let mut seen: HashSet<String> = HashSet::new();

    for path in &repo_paths {
        if !seen.insert(repo_dedup_key(path)) {
            continue;
        }
        // Resolve reference identity lazily from the first successful manifest.
        if ref_identity.is_none() {
            match peek_manifest_identity(path) {
                Ok(identity) => ref_identity = Some(identity),
                Err(err) => {
                    repo_errors.push(err);
                    continue;
                }
            }
        }

        let Some((ref_model, ref_dims)) = ref_identity.as_ref() else {
            continue;
        };
        match load_repo_chunks_readonly(path, ref_model, *ref_dims).await {
            Ok((canonical, chunks)) => {
                let label: Arc<str> = Arc::from(canonical.as_str());
                for _ in &chunks {
                    repo_labels.push(Arc::clone(&label));
                }
                all_chunks.extend(chunks);
            }
            Err(err) => {
                repo_errors.push(err);
            }
        }
    }

    if all_chunks.is_empty() {
        return Ok(DuplicatesOutput {
            pairs: Vec::new(),
            repo_errors,
        });
    }

    // Cap the combined corpus before the O(n²) scan: `limit` bounds only the
    // output, so a large multi-repo input is a CPU/memory amplifier. Skip the
    // scan and surface a notice rather than churning through millions of pairs.
    if all_chunks.len() > MAX_DUPLICATE_CORPUS_CHUNKS {
        repo_errors.push(RepoError {
            repo: project_root.display().to_string(),
            error: format!(
                "duplicate scan skipped: {} chunks exceeds the {MAX_DUPLICATE_CORPUS_CHUNKS} cap; \
                 narrow the repo list",
                all_chunks.len()
            ),
        });
        return Ok(DuplicatesOutput {
            pairs: Vec::new(),
            repo_errors,
        });
    }

    // Build the labeled slice for the detection scan.
    // `spawn_blocking` keeps the O(n²) work off the async executor.
    let pairs = tokio::task::spawn_blocking(move || {
        let labeled: Vec<LabeledChunk<'_>> = all_chunks
            .iter()
            .zip(repo_labels.iter())
            .map(|(chunk, repo)| LabeledChunk {
                repo: repo.as_ref(),
                chunk,
            })
            .collect();
        duplicates::find_duplicates(&labeled, min_similarity, limit)
    })
    .await
    .map_err(|e| ClaudixError::Store(format!("duplicate scan task failed: {e}")))?;

    Ok(DuplicatesOutput { pairs, repo_errors })
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
    repos: Option<Vec<String>>,
) -> Result<SearchOutput> {
    run_search_impl(
        project_root,
        None,
        query,
        top_k,
        language_filter,
        path_prefix,
        repos,
    )
    .await
}

/// `run_search` that draws the provider from a session-lifetime cache; the MCP
/// server uses this so only the first search of a session pays the provider
/// build (the bundled ONNX session load dominates otherwise).
pub async fn run_search_cached(
    project_root: impl AsRef<Path>,
    provider_cache: &crate::embedding::ProviderCache,
    query: String,
    top_k: Option<usize>,
    language_filter: Option<Vec<String>>,
    path_prefix: Option<String>,
    repos: Option<Vec<String>>,
) -> Result<SearchOutput> {
    run_search_impl(
        project_root,
        Some(provider_cache),
        query,
        top_k,
        language_filter,
        path_prefix,
        repos,
    )
    .await
}

async fn run_search_impl(
    project_root: impl AsRef<Path>,
    provider_cache: Option<&crate::embedding::ProviderCache>,
    query: String,
    top_k: Option<usize>,
    language_filter: Option<Vec<String>>,
    path_prefix: Option<String>,
    repos: Option<Vec<String>>,
) -> Result<SearchOutput> {
    validate_search_query(&query)?;
    let project_root = canonical_project_root(project_root.as_ref())?;
    let config = Arc::new(config::load(&project_root)?);
    let top_k = top_k.unwrap_or(config.search.top_k);
    validate_search_top_k(top_k)?;
    // Active project is always in scope; union the config cross_repos with the
    // per-call repos, deduped at search time by canonical path.
    let repos = effective_cross_repos(&config.search.cross_repos, repos);
    let provider = match provider_cache {
        Some(cache) => cache.get_or_build(&config).await?,
        None => crate::build_provider(&config).await?,
    };
    // A stored index whose embedding model/dimensions differ from the
    // configured provider degrades to lexical ranking with the reindex hint
    // instead of erroring: `Searcher::search_all` re-derives the mismatch
    // against the manifest. Any other construction error still fails the call,
    // and every write path keeps the strict [`Claudix::with_embedder`] check.
    let claudix = match Claudix::with_embedder(
        project_root.clone(),
        Arc::clone(&config),
        Arc::clone(&provider),
    ) {
        Ok(claudix) => claudix,
        Err(ClaudixError::EmbeddingModelMismatch { .. })
        | Err(ClaudixError::DimensionMismatch { .. }) => {
            Claudix::with_embedder_unvalidated(project_root, config, provider)?
        }
        Err(error) => return Err(error),
    };

    run_search_with_claudix(&claudix, query, top_k, language_filter, path_prefix, repos).await
}

/// Union the configured `cross_repos` with the per-call `repos`, preserving
/// order and dropping exact-string duplicates. Canonical-path dedup happens
/// later in the searcher (it needs to resolve each path through the store).
fn effective_cross_repos(cross_repos: &[String], repos: Option<Vec<String>>) -> Vec<String> {
    let mut seen = HashSet::new();
    cross_repos
        .iter()
        .cloned()
        .chain(repos.into_iter().flatten())
        .filter(|repo| seen.insert(repo.clone()))
        .collect()
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
    let stats = match session.claudix.index_full(progress).await {
        Ok(stats) => stats,
        Err(error) => {
            // The background index runs with stderr → null, so its terminal
            // error would otherwise vanish. Leave a breadcrumb in index.log so
            // the failure notice can surface the cause.
            append_index_log_error(&log_dir, &error);
            return Err(error);
        }
    };

    Ok(IndexOutput {
        file_count: stats.file_count,
        chunk_count: stats.chunk_count,
        empty_index_warning: stats.empty_index_warning,
    })
}

/// Append a terminal error line to `index.log` so a failed background index
/// leaves a debuggable trace. Best-effort: a logging failure must never mask
/// the real indexing error.
fn append_index_log_error(log_dir: &Path, error: &ClaudixError) {
    if fs::create_dir_all(log_dir).is_err() {
        return;
    }
    let Ok(mut file) = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_dir.join("index.log"))
    else {
        return;
    };
    let _ = writeln!(file, "error: {error}");
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
    require_git_repo(&project_root)?;
    let config = config::load(&project_root)?;
    let store = Store::new(&project_root, &config)?;

    // Block on the shared chunk-writer lock instead of short-circuiting on a
    // running full index: bailing here loses the user's edit until they save
    // again, since the reindex-file child returns 0 with no retry path.
    let _reindex_lock = store.acquire_reindex_lock()?;
    let claudix = Claudix::new(project_root, Arc::new(config)).await?;
    // No session identity: a manual reindex cannot be attributed, so it never
    // writes a change-neighbors marker (the ack would drop it unread anyway).
    let stats = claudix.reindex_file(path.as_ref(), None).await?;

    Ok(IndexOutput {
        file_count: stats.file_count,
        chunk_count: stats.chunk_count,
        empty_index_warning: stats.empty_index_warning,
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

    let install_log = install_data_dir().map(|dir| dir.join("install.log"));
    let install_error = install_log
        .as_ref()
        .and_then(|path| crate::util::last_error_line(path));

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
        development_mode: config.development_mode,
        binary_path: std::env::current_exe()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|_| "<unknown>".to_owned()),
        install_error,
        install_log_path: install_log.map(|path| path.display().to_string()),
    })
}

/// Resolve the claudix binary cache dir from the environment, mirroring
/// `bin/claudix-bootstrap.js`'s resolution so `/doctor` reads the same
/// `install.log` the bootstrap writes. `None` when no env hints a cache dir
/// (e.g. `claudix doctor` run manually from a shell).
fn install_data_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("CLAUDE_PLUGIN_DATA").map(PathBuf::from) {
        return Some(dir);
    }
    if let Some(dir) = std::env::var_os("CLAUDIX_HOME").map(PathBuf::from) {
        return Some(dir);
    }
    let base = std::env::var_os("XDG_DATA_HOME").map(PathBuf::from);
    let base = match base {
        Some(base) => base,
        None => dirs::home_dir()?.join(".local").join("share"),
    };
    Some(base.join("claudix"))
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
            recovery: RecoveryHint(hints::VALID_HOOK_EVENTS),
        }),
    }
}

async fn run_search_with_claudix(
    claudix: &Claudix,
    query: String,
    top_k: usize,
    language_filter: Option<Vec<String>>,
    path_prefix: Option<String>,
    repos: Vec<String>,
) -> Result<SearchOutput> {
    let query = SearchQuery {
        query,
        top_k,
        language_filter: parse_language_filter(language_filter)?,
        path_prefix: parse_path_prefix(path_prefix)?,
        repos,
    };
    let found = claudix.search(query).await?;

    // Walk hits in score order (results is already score-desc from search).
    // Bucket by (repo, directory) preserving first-seen order so the first key
    // encountered owns the top hit — groups naturally ordered by best score.
    // Keying on repo too keeps same-named directories in different repos apart.
    let mut group_index: Vec<(String, String)> = Vec::new();
    let mut grouped: HashMap<(String, String), Vec<SearchHit>> = HashMap::new();

    let mut any_stale = false;
    for result in found.results {
        let dir = immediate_parent_dir(result.chunk.file_path.as_str());
        let key = (result.repo, dir);
        any_stale |= result.stale;
        let hit = SearchHit {
            file_path: result.chunk.file_path.to_string(),
            language: result.chunk.language.to_string(),
            kind: result.chunk.kind.to_string(),
            name: result.chunk.name,
            line_start: result.chunk.line_range.start,
            line_end: result.chunk.line_range.end,
            score: result.score,
            stale: result.stale,
            snippet: crate::prompts::truncate_snippet(
                &result.chunk.content,
                crate::prompts::SNIPPET_MAX_LINES,
            ),
        };
        if !grouped.contains_key(&key) {
            group_index.push(key.clone());
        }
        grouped.entry(key).or_default().push(hit);
    }

    let groups = group_index
        .into_iter()
        .filter_map(|key| {
            let hits = grouped.remove(&key)?;
            let (repo, directory) = key;
            Some(DirectoryGroup {
                repo,
                directory,
                hits,
            })
        })
        .collect();

    Ok(SearchOutput {
        groups,
        repo_errors: found.repo_errors,
        stale_hint: any_stale.then_some(crate::prompts::mcp::STALE_HITS_NOTE),
        degraded_hint: found.degraded_hint,
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
            recovery: RecoveryHint(hints::GIT_REPO_REQUIRED),
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
    use test_support::{index_fixture, stub_config, write_fixture_config};

    struct CliHarness {
        // Some when the harness owns its fixture (private harness); None when
        // the fixture is shared and owned by SHARED_FIXTURE_DIR.
        _fixture: Option<TestFixture>,
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

    // Shared indexed fixture directory — built once per test process.
    // The TempDir is stored here to keep the path alive for the process lifetime.
    static SHARED_FIXTURE_DIR: std::sync::OnceLock<(tempfile::TempDir, PathBuf)> =
        std::sync::OnceLock::new();

    fn shared_fixture_dir() -> &'static (tempfile::TempDir, PathBuf) {
        SHARED_FIXTURE_DIR.get_or_init(|| {
            // Spawn a fresh OS thread so `block_on` isn't called from within
            // an existing tokio runtime (which `#[tokio::test]` provides).
            // `TestFixture::new` initialises a real git repo so `FileEnumerator`
            // (called inside `index_fixture`) can enumerate files via git ls-files.
            // The git cost is paid once here for the entire test process.
            std::thread::spawn(|| {
                let fixture = TestFixture::new("small_rust").expect("shared fixture copy failed");
                let config = stub_config();
                let root = fixture.root().to_path_buf();
                let claudix =
                    test_claudix(root.clone(), config.clone()).expect("shared claudix init failed");

                tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("shared fixture tokio runtime")
                    .block_on(async {
                        index_fixture(
                            &claudix.store,
                            claudix.embedder.as_ref(),
                            claudix.project_root(),
                            &config,
                        )
                        .await
                        .expect("shared fixture indexing failed");
                    });

                fixture.into_parts()
            })
            .join()
            .expect("shared fixture init thread panicked")
        })
    }

    /// Harness for read-only tests: opens a fresh Store/Claudix against the
    /// shared pre-indexed fixture — no copy, no git, no indexing per test.
    async fn cli_harness() -> Result<CliHarness> {
        let (_, root) = shared_fixture_dir();
        let config = stub_config();
        let claudix = test_claudix(root.clone(), config.clone())?;
        let store = Store::new(root, &config)?;
        Ok(CliHarness {
            _fixture: None,
            claudix,
            store,
        })
    }

    /// Harness for tests that mutate fixture state (write files, delete the
    /// index dir, etc.) — each call gets its own isolated copy.
    async fn cli_harness_private() -> Result<CliHarness> {
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
            _fixture: Some(fixture),
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

        let output = run_search_with_claudix(
            &harness.claudix,
            "add".to_owned(),
            5,
            None,
            None,
            Vec::new(),
        )
        .await;
        assert!(output.is_ok());
        let output = output.ok().unwrap_or_else(|| unreachable!());

        assert!(!output.groups.is_empty());
        let top_hit = &output.groups[0].hits[0];
        assert_eq!(top_hit.name.as_deref(), Some("add"));
        assert_eq!(top_hit.file_path, "src/math.rs");
    }

    /// A healthy search carries no degradation notice; an endpoint-down search
    /// over the same indexed corpus returns lexical hits plus the notice.
    #[tokio::test]
    async fn search_output_degraded_hint_tracks_endpoint_availability() {
        let healthy = cli_harness().await;
        assert!(healthy.is_ok());
        let healthy = healthy.ok().unwrap_or_else(|| unreachable!());
        let output = run_search_with_claudix(
            &healthy.claudix,
            "add".to_owned(),
            5,
            None,
            None,
            Vec::new(),
        )
        .await;
        assert!(output.is_ok());
        let output = output.ok().unwrap_or_else(|| unreachable!());
        assert!(
            output.degraded_hint.is_none(),
            "a healthy search must carry no degradation notice"
        );

        struct EndpointDownProvider;

        #[async_trait::async_trait]
        impl Provider for EndpointDownProvider {
            fn name(&self) -> &str {
                "endpoint-down"
            }

            fn dimensions(&self) -> Dimension {
                Dimension(8)
            }

            fn model_id(&self) -> &str {
                "stub-v1"
            }

            async fn embed(&self, _batch: &[&str]) -> Result<Vec<Vec<f32>>> {
                Err(ClaudixError::EmbeddingTimedOut {
                    endpoint: "http://127.0.0.1:1234".to_owned(),
                    timeout_ms: 50,
                    recovery: RecoveryHint(hints::EMBEDDING_GENERIC),
                })
            }

            async fn health_check(&self) -> Result<()> {
                Ok(())
            }
        }

        let (_, root) = shared_fixture_dir();
        let config = Arc::new(stub_config());
        let store = Store::new(root, config.as_ref());
        assert!(store.is_ok());
        let store = store.ok().unwrap_or_else(|| unreachable!());
        let embedder: Arc<dyn Provider> = Arc::new(EndpointDownProvider);
        let degraded = Claudix::from_parts(root.clone(), config, embedder, store);

        let output =
            run_search_with_claudix(&degraded, "add".to_owned(), 5, None, None, Vec::new()).await;
        assert!(
            output.is_ok(),
            "endpoint-down search must degrade, not error: {output:?}"
        );
        let output = output.ok().unwrap_or_else(|| unreachable!());
        assert!(
            !output.groups.is_empty(),
            "lexical fallback must still return hits"
        );
        assert_eq!(
            output.degraded_hint,
            Some(crate::prompts::mcp::ENDPOINT_DOWN_NOTE),
            "an endpoint-down search must carry the degradation notice"
        );
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
            Vec::new(),
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
            recovery: RecoveryHint(hints::RUN_REINDEX),
        }));
        assert!(requires_clean_reindex(
            &ClaudixError::EmbeddingModelMismatch {
                store_model: "old".to_owned(),
                active_model: "new".to_owned(),
                recovery: RecoveryHint(hints::RUN_REINDEX),
            }
        ));
        assert!(requires_clean_reindex(&ClaudixError::DimensionMismatch {
            store_dim: 384,
            model_dim: 768,
            recovery: RecoveryHint(hints::RUN_REINDEX),
        }));
        assert!(!requires_clean_reindex(&ClaudixError::Store(
            "index already running".to_owned()
        )));
    }

    #[test]
    fn append_index_log_error_writes_error_line() {
        let dir = tempfile::tempdir().unwrap_or_else(|_| unreachable!());
        let log_dir = dir.path().join("logs");
        append_index_log_error(&log_dir, &ClaudixError::Store("boom".to_owned()));
        let text =
            std::fs::read_to_string(log_dir.join("index.log")).unwrap_or_else(|_| unreachable!());
        assert!(
            text.starts_with("error:") && text.contains("boom"),
            "log must capture the terminal error so the failure notice can quote it, got: {text}"
        );
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
    async fn run_index_clears_dimension_mismatch_and_reindexes() {
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
        // Same model, wrong dimensions: only the dimension check fires.
        let stale_manifest =
            Manifest::new(&config.embedding.model, config.embedding.dimensions * 2);
        assert!(store.write_manifest(&stale_manifest).is_ok());

        let output = run_index(fixture.root(), false).await;
        assert!(
            output.is_ok(),
            "run_index must auto-clear and reindex on dimension mismatch: {:?}",
            output.err()
        );
        let output = output.ok().unwrap_or_else(|| unreachable!());
        assert!(output.chunk_count > 0);

        let manifest = store.read_manifest();
        assert!(manifest.is_ok());
        let manifest = manifest.ok().unwrap_or_else(|| unreachable!());
        let manifest = manifest.unwrap_or_else(|| unreachable!());
        assert_eq!(
            manifest.dimensions, config.embedding.dimensions,
            "dimensions must match config after the clean reindex"
        );
    }

    #[tokio::test]
    async fn run_index_warns_when_files_enumerate_but_nothing_chunks() {
        // Every file in this fixture is unknown-language, so the pass stores
        // zero chunks while the enumeration is non-empty — the silent-empty
        // shape that must print a warning. Run three passes: pass 1 chunks for
        // real while the store's state files are still enumerable; pass 2
        // re-chunks for real after the store's `.claudix/.gitignore` shrinks
        // the enumerated set and invalidates the recorded hashes; pass 3 takes
        // the manifest fast path. All three must warn.
        let fixture = TestFixture::new("unknown_only");
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

        for pass in 1..=3 {
            let output = run_index(fixture.root(), false).await;
            assert!(output.is_ok(), "run_index (pass {pass}) failed");
            let output = output.ok().unwrap_or_else(|| unreachable!());
            assert_eq!(
                (output.file_count, output.chunk_count),
                (0, 0),
                "pass {pass}: the unknown-only corpus must store no chunks"
            );
            let warning = output
                .empty_index_warning
                .unwrap_or_else(|| unreachable!("pass {pass} must warn"));
            assert!(
                warning.contains("files enumerated but 0 chunks indexed"),
                "pass {pass}: the warning must name the cause, got: {warning}"
            );
            assert!(
                warning.contains(
                    "hint: add a root .indexinclude with a `*` rule to text-index these files"
                ),
                "pass {pass}: the warning must name the fix, got: {warning}"
            );
        }
    }

    #[tokio::test]
    async fn run_index_stays_silent_when_files_chunk() {
        // small_rust chunks 3 chunks from 2 files: a corpus that stores chunks
        // must never warn, whatever individual files were skipped.
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

        let output = run_index(fixture.root(), false).await;
        assert!(output.is_ok());
        let output = output.ok().unwrap_or_else(|| unreachable!());
        assert!(output.chunk_count > 0);
        assert!(
            output.empty_index_warning.is_none(),
            "a pass that stored chunks must not warn, got: {:?}",
            output.empty_index_warning
        );
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
        let output = run_search_with_claudix(
            &harness.claudix,
            "add greet".to_owned(),
            10,
            None,
            None,
            Vec::new(),
        )
        .await;
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

        let output = run_search_with_claudix(
            &harness.claudix,
            "add".to_owned(),
            5,
            None,
            None,
            Vec::new(),
        )
        .await;
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

    // ── find_duplicates tests ────────────────────────────────────────────────

    /// Build a temporary fixture with two Rust source files having identical content,
    /// index them with the stub provider, and return (fixture, store) so tests can
    /// call `run_find_duplicates`.
    async fn dup_harness() -> Result<(TestFixture, Store)> {
        let fixture = TestFixture::new("small_rust")?;
        // Write a second file whose content is byte-identical to src/math.rs so
        // the StubProvider (content-hash seed) emits the same vector → cosine = 1.0.
        let dup_content = std::fs::read_to_string(fixture.root().join("src").join("math.rs"))
            .map_err(ClaudixError::from)?;
        std::fs::write(fixture.root().join("src").join("math_copy.rs"), dup_content)
            .map_err(ClaudixError::from)?;

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
        Ok((fixture, store))
    }

    #[tokio::test]
    async fn find_duplicates_returns_pair_for_identical_content() {
        let result = dup_harness().await;
        assert!(result.is_ok());
        let (fixture, _store) = result.ok().unwrap_or_else(|| unreachable!());

        let output = run_find_duplicates(fixture.root(), None, None, None).await;
        assert!(output.is_ok());
        let output = output.ok().unwrap_or_else(|| unreachable!());

        // math.rs and math_copy.rs share identical content → identical vectors → cosine 1.0.
        assert!(
            !output.pairs.is_empty(),
            "expected at least one pair from identical file content"
        );
        // Both chunks name the right files.
        let pair = &output.pairs[0];
        let paths = [pair.a.file_path.as_str(), pair.b.file_path.as_str()];
        assert!(
            paths.iter().any(|p| p.contains("math.rs")),
            "expected math.rs in the pair; got {:?}",
            paths
        );
        assert!(
            paths.iter().any(|p| p.contains("math_copy.rs")),
            "expected math_copy.rs in the pair; got {:?}",
            paths
        );
        assert!(
            pair.similarity > 0.99,
            "similarity should be ~1.0 for identical content"
        );
    }

    #[tokio::test]
    async fn find_duplicates_returns_empty_for_unique_repo() {
        // small_rust has lib.rs and math.rs with distinct content → no duplicates.
        let harness = cli_harness().await;
        assert!(harness.is_ok());
        let harness = harness.ok().unwrap_or_else(|| unreachable!());

        let output = run_find_duplicates(harness.claudix.project_root(), None, None, None).await;
        assert!(output.is_ok());
        let output = output.ok().unwrap_or_else(|| unreachable!());

        assert!(
            output.pairs.is_empty(),
            "distinct-content repo should have no duplicates"
        );
    }

    #[tokio::test]
    async fn find_duplicates_threshold_sensitivity() {
        let result = dup_harness().await;
        assert!(result.is_ok());
        let (fixture, _store) = result.ok().unwrap_or_else(|| unreachable!());

        // At threshold 0.99 the identical pair still shows.
        let high = run_find_duplicates(fixture.root(), Some(0.99), None, None).await;
        assert!(high.is_ok());
        let high = high.ok().unwrap_or_else(|| unreachable!());
        assert!(
            !high.pairs.is_empty(),
            "threshold 0.99 should still find identical pair"
        );

        let ceiling = run_find_duplicates(fixture.root(), Some(1.01), None, None).await;
        assert!(
            ceiling.is_err(),
            "threshold > 1.0 must be rejected before scanning"
        );
    }

    #[tokio::test]
    async fn find_duplicates_rejects_invalid_threshold_and_limit() {
        let harness = cli_harness().await;
        assert!(harness.is_ok());
        let harness = harness.ok().unwrap_or_else(|| unreachable!());

        let nan =
            run_find_duplicates(harness.claudix.project_root(), Some(f32::NAN), None, None).await;
        assert!(nan.is_err(), "NaN threshold must be rejected");

        let negative =
            run_find_duplicates(harness.claudix.project_root(), Some(-0.1), None, None).await;
        assert!(negative.is_err(), "negative threshold must be rejected");

        let zero_limit =
            run_find_duplicates(harness.claudix.project_root(), None, Some(0), None).await;
        assert!(zero_limit.is_err(), "zero limit must be rejected");
    }

    #[tokio::test]
    async fn find_duplicates_same_file_not_reported() {
        // Even when identical chunks come from the same file, only cross-file pairs
        // should appear. The dup_harness fixture has math.rs AND math_copy.rs (different
        // files), so the pair is expected — but confirm no entry has a.file_path == b.file_path.
        let result = dup_harness().await;
        assert!(result.is_ok());
        let (fixture, _store) = result.ok().unwrap_or_else(|| unreachable!());

        let output = run_find_duplicates(fixture.root(), Some(0.0), None, None).await;
        assert!(output.is_ok());
        let output = output.ok().unwrap_or_else(|| unreachable!());

        for pair in &output.pairs {
            assert!(
                pair.a.file_path != pair.b.file_path || pair.a.repo != pair.b.repo,
                "same-file pair must not be reported: {} == {}",
                pair.a.file_path,
                pair.b.file_path,
            );
        }
    }

    #[tokio::test]
    async fn find_duplicates_partial_success_on_unindexed_path() {
        let result = dup_harness().await;
        assert!(result.is_ok());
        let (fixture, _store) = result.ok().unwrap_or_else(|| unreachable!());

        let indexed = fixture.root().display().to_string();
        let unindexed = "/tmp/nonexistent-claudix-test-repo-12345".to_owned();

        let output = run_find_duplicates(
            fixture.root(),
            None,
            None,
            Some(vec![indexed, unindexed.clone()]),
        )
        .await;
        assert!(output.is_ok());
        let output = output.ok().unwrap_or_else(|| unreachable!());

        // The unindexed path must produce a RepoError.
        assert!(
            output.repo_errors.iter().any(|e| e.repo == unindexed
                || e.error.contains("not indexed")
                || e.error.contains("No such")),
            "expected a repo_error for the unindexed path; got: {:?}",
            output.repo_errors,
        );
        // Pairs from the indexed repo are still returned.
        assert!(
            !output.pairs.is_empty(),
            "indexed repo should still produce pairs despite the error"
        );
    }

    #[tokio::test]
    async fn load_repo_chunks_readonly_rejects_dimension_mismatch() {
        let harness = cli_harness().await;
        assert!(harness.is_ok());
        let harness = harness.ok().unwrap_or_else(|| unreachable!());

        let root = harness.claudix.project_root().display().to_string();
        // Manifest has dimensions=8 (stub_config). Asking with wrong dims triggers mismatch.
        let result = load_repo_chunks_readonly(&root, "stub-v1", 999).await;
        assert!(
            result.is_err(),
            "mismatched dimensions must produce a RepoError"
        );
        let err = result.err().unwrap_or_else(|| unreachable!());
        assert!(
            err.error.contains("mismatch"),
            "error should mention mismatch; got: {}",
            err.error,
        );
    }

    #[tokio::test]
    async fn load_repo_chunks_readonly_rejects_model_mismatch() {
        let harness = cli_harness().await;
        assert!(harness.is_ok());
        let harness = harness.ok().unwrap_or_else(|| unreachable!());

        let root = harness.claudix.project_root().display().to_string();
        let result = load_repo_chunks_readonly(&root, "different-model", 8).await;
        assert!(result.is_err(), "mismatched model must produce a RepoError");
        let err = result.err().unwrap_or_else(|| unreachable!());
        assert!(
            err.error.contains("mismatch"),
            "error should mention mismatch; got: {}",
            err.error,
        );
    }

    #[tokio::test]
    async fn load_repo_chunks_readonly_rejects_missing_chunk_table() {
        let harness = cli_harness_private().await;
        assert!(harness.is_ok());
        let harness = harness.ok().unwrap_or_else(|| unreachable!());

        let index_dir = harness.claudix.store.state_dir_path().join("index");
        let remove = std::fs::remove_dir_all(&index_dir);
        assert!(remove.is_ok());

        let root = harness.claudix.project_root().display().to_string();
        let result = load_repo_chunks_readonly(&root, "stub-v1", 8).await;
        assert!(
            result.is_err(),
            "manifest claiming chunks without a chunks table must produce a RepoError"
        );
        let err = result.err().unwrap_or_else(|| unreachable!());
        assert!(
            err.error.contains("chunks missing"),
            "error should mention missing chunks; got: {}",
            err.error,
        );
    }

    #[tokio::test]
    async fn find_duplicates_multi_repo_detects_cross_repo_pair() {
        // Build two separate repos with an identical file, index each, then scan both.
        let fixture_a = TestFixture::new("small_rust");
        assert!(fixture_a.is_ok());
        let fixture_a = fixture_a.ok().unwrap_or_else(|| unreachable!());

        let fixture_b = TestFixture::new("small_rust");
        assert!(fixture_b.is_ok());
        let fixture_b = fixture_b.ok().unwrap_or_else(|| unreachable!());

        let config = stub_config();

        // Index repo A.
        let claudix_a = test_claudix(fixture_a.root().to_path_buf(), config.clone());
        assert!(claudix_a.is_ok());
        let claudix_a = claudix_a.ok().unwrap_or_else(|| unreachable!());
        let index_a = index_fixture(
            &claudix_a.store,
            claudix_a.embedder.as_ref(),
            claudix_a.project_root(),
            &config,
        )
        .await;
        assert!(index_a.is_ok());

        // Index repo B.
        let claudix_b = test_claudix(fixture_b.root().to_path_buf(), config.clone());
        assert!(claudix_b.is_ok());
        let claudix_b = claudix_b.ok().unwrap_or_else(|| unreachable!());
        let index_b = index_fixture(
            &claudix_b.store,
            claudix_b.embedder.as_ref(),
            claudix_b.project_root(),
            &config,
        )
        .await;
        assert!(index_b.is_ok());

        let repo_a = fixture_a.root().display().to_string();
        let repo_b = fixture_b.root().display().to_string();

        // Listing the active project explicitly is a no-op: it is always scanned,
        // and the exact-string dedup keeps it from being paired with itself.
        let output = run_find_duplicates(
            fixture_a.root(),
            Some(0.99),
            None,
            Some(vec![repo_a.clone(), repo_b.clone()]),
        )
        .await;
        assert!(output.is_ok());
        let output = output.ok().unwrap_or_else(|| unreachable!());

        assert!(
            output.repo_errors.is_empty(),
            "both repos are indexed; no errors expected"
        );

        // Both repos contain the same fixture content → at least one cross-repo pair.
        assert!(
            !output.pairs.is_empty(),
            "identical fixture content across repos must produce cross-repo pairs"
        );

        // At least one pair must have a.repo != b.repo.
        let has_cross_repo = output.pairs.iter().any(|p| p.a.repo != p.b.repo);
        assert!(
            has_cross_repo,
            "at least one pair must span two different repos"
        );
    }

    /// Naming the active project in a spelling that isn't its canonical path must
    /// still dedup. An exact-string filter passes the sibling test (fixtures are
    /// pre-canonicalized) but fails here, and on Windows it fails always: the
    /// repo loads twice under one label and every cross-file pair is emitted four
    /// times, each burning a `limit` slot.
    #[tokio::test]
    async fn find_duplicates_dedups_a_non_canonical_spelling_of_the_active_project() {
        // dup_harness, not a bare fixture: this needs a repo that actually
        // produces pairs, or every assertion below holds on an empty vec.
        let result = dup_harness().await;
        assert!(result.is_ok());
        let (fixture, _store) = result.ok().unwrap_or_else(|| unreachable!());
        assert!(write_fixture_config(fixture.root(), &stub_config()).is_ok());

        let baseline = run_find_duplicates(fixture.root(), None, None, None).await;
        assert!(baseline.is_ok());
        let baseline = baseline.ok().unwrap_or_else(|| unreachable!());
        assert!(
            !baseline.pairs.is_empty(),
            "harness must produce pairs, else this test proves nothing"
        );

        // Same repo, spelled so the store canonicalizes it back to the root.
        let uncanonical = format!("{}/.", fixture.root().display());
        let output = run_find_duplicates(fixture.root(), None, None, Some(vec![uncanonical])).await;
        assert!(output.is_ok(), "duplicate scan failed: {output:?}");
        let output = output.ok().unwrap_or_else(|| unreachable!());

        // A second load of the same repo emits every pair 4x under one label.
        assert_eq!(
            output.pairs.len(),
            baseline.pairs.len(),
            "naming the active project a second way changed the pair count — it was scanned twice"
        );
        let mut keys: Vec<String> = output
            .pairs
            .iter()
            .map(|pair| {
                format!(
                    "{}|{}:{}-{}|{}:{}-{}",
                    pair.a.repo,
                    pair.a.file_path,
                    pair.a.line_start,
                    pair.a.line_end,
                    pair.b.file_path,
                    pair.b.line_start,
                    pair.b.line_end
                )
            })
            .collect();
        let total = keys.len();
        keys.sort();
        keys.dedup();
        assert_eq!(
            keys.len(),
            total,
            "identical pairs repeated — the repo was loaded more than once"
        );
    }

    /// Two spellings of one broken repo must collapse to one `RepoError`, the
    /// same way two spellings of a loadable repo collapse to one corpus entry.
    /// The Ok-branch label dedup can't cover it: an errored repo never reaches
    /// the store's canonical label. Reachable since the active project is
    /// auto-added — an unindexed project plus any respelling of it in `repos`.
    #[tokio::test]
    async fn find_duplicates_reports_one_error_for_two_spellings_of_a_broken_repo() {
        // A bare fixture, never indexed: the active project itself is broken.
        let fixture = TestFixture::new("small_rust");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());

        let respelled = format!("{}/.", fixture.root().display());
        let output = run_find_duplicates(fixture.root(), None, None, Some(vec![respelled])).await;
        assert!(output.is_ok(), "duplicate scan failed: {output:?}");
        let output = output.ok().unwrap_or_else(|| unreachable!());

        assert!(
            output.pairs.is_empty(),
            "unindexed repos cannot produce pairs"
        );
        assert_eq!(
            output.repo_errors.len(),
            1,
            "one broken repo spelled two ways must yield one error; got: {:?}",
            output.repo_errors,
        );
        assert!(
            output.repo_errors[0].error.contains("not indexed"),
            "expected a 'not indexed' error; got: {:?}",
            output.repo_errors,
        );
    }

    /// `repos` adds to the active project rather than replacing it, matching
    /// `search_code`. Naming only the other repo must still scan this one, or a
    /// caller silently audits everything except the code they are working in.
    #[tokio::test]
    async fn find_duplicates_scans_active_project_when_repos_names_only_another() {
        let setup = dual_repo_harness().await;
        assert!(setup.is_ok());
        let (fixture_a, _fixture_b, repo_a, repo_b) = setup.ok().unwrap_or_else(|| unreachable!());

        // repo_b only — repo_a is the active project and must be scanned anyway.
        let output =
            run_find_duplicates(fixture_a.root(), Some(0.99), None, Some(vec![repo_b])).await;
        assert!(output.is_ok(), "duplicate scan failed: {output:?}");
        let output = output.ok().unwrap_or_else(|| unreachable!());

        assert!(output.repo_errors.is_empty(), "both repos are indexed");
        let touches_active = output
            .pairs
            .iter()
            .any(|pair| pair.a.repo == repo_a || pair.b.repo == repo_a);
        assert!(
            touches_active,
            "the unlisted active project {repo_a} must still be scanned; pairs: {:?}",
            output.pairs
        );
    }

    // ── cross-repo search tests (Feature 6) ─────────────────────────────────

    /// Set up two repos backed by the small_rust fixture and indexed with the
    /// same stub model so their vectors are comparable. Returns the canonical
    /// path of each. Caller passes one as `project_root`, the other as `repos`.
    async fn dual_repo_harness() -> Result<(TestFixture, TestFixture, String, String)> {
        let fixture_a = TestFixture::new("small_rust")?;
        let fixture_b = TestFixture::new("small_rust")?;
        let config = stub_config();

        let claudix_a = test_claudix(fixture_a.root().to_path_buf(), config.clone())?;
        index_fixture(
            &claudix_a.store,
            claudix_a.embedder.as_ref(),
            claudix_a.project_root(),
            &config,
        )
        .await?;
        write_fixture_config(fixture_a.root(), &config)?;

        let claudix_b = test_claudix(fixture_b.root().to_path_buf(), config.clone())?;
        index_fixture(
            &claudix_b.store,
            claudix_b.embedder.as_ref(),
            claudix_b.project_root(),
            &config,
        )
        .await?;
        write_fixture_config(fixture_b.root(), &config)?;

        let repo_a = fixture_a
            .root()
            .canonicalize()
            .map_err(ClaudixError::from)?
            .display()
            .to_string();
        let repo_b = fixture_b
            .root()
            .canonicalize()
            .map_err(ClaudixError::from)?
            .display()
            .to_string();
        Ok((fixture_a, fixture_b, repo_a, repo_b))
    }

    /// `stale_hint` is the whole point of moving the staleness explanation out of
    /// the always-billed tool description: it must be absent on a fresh index and
    /// present the moment a hit goes stale. `stale: false` must also stay off the
    /// wire — that is per-hit dead weight for the common case.
    #[tokio::test]
    async fn stale_hint_and_flag_ride_the_payload_only_when_a_hit_is_stale() {
        let fixture = TestFixture::new("small_rust");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());
        let config = stub_config();
        let claudix = test_claudix(fixture.root().to_path_buf(), config.clone());
        assert!(claudix.is_ok());
        let claudix = claudix.ok().unwrap_or_else(|| unreachable!());
        assert!(
            index_fixture(
                &claudix.store,
                claudix.embedder.as_ref(),
                claudix.project_root(),
                &config,
            )
            .await
            .is_ok()
        );
        assert!(write_fixture_config(fixture.root(), &config).is_ok());

        let fresh = run_search(fixture.root(), "add".to_owned(), Some(10), None, None, None).await;
        assert!(fresh.is_ok(), "search failed: {fresh:?}");
        let fresh = fresh.ok().unwrap_or_else(|| unreachable!());
        assert!(!fresh.groups.is_empty(), "fixture must produce hits");
        assert_eq!(
            fresh.stale_hint, None,
            "a fresh index must not pay for the stale explanation"
        );
        // `stale: false` is skipped, so it never reaches the agent's context.
        let json = serde_json::to_string(&fresh);
        assert!(json.is_ok());
        let json = json.unwrap_or_default();
        assert!(!json.contains("\"stale\""), "stale: false was serialized");
        assert!(!json.contains("stale_hint"), "stale_hint was serialized");

        // Touch an indexed file so its stored hash no longer matches disk.
        let math = fixture.root().join("src/math.rs");
        let existing = std::fs::read_to_string(&math);
        assert!(existing.is_ok());
        assert!(
            std::fs::write(
                &math,
                format!("{}\n// drift\n", existing.unwrap_or_default())
            )
            .is_ok()
        );

        let drifted =
            run_search(fixture.root(), "add".to_owned(), Some(10), None, None, None).await;
        assert!(drifted.is_ok(), "search failed: {drifted:?}");
        let drifted = drifted.ok().unwrap_or_else(|| unreachable!());
        let any_stale = drifted
            .groups
            .iter()
            .flat_map(|g| g.hits.iter())
            .any(|hit| hit.stale);
        assert!(
            any_stale,
            "editing an indexed file must mark its hits stale"
        );
        assert_eq!(
            drifted.stale_hint,
            Some(crate::prompts::mcp::STALE_HITS_NOTE),
            "a stale hit must carry its explanation"
        );
    }

    /// The cap has to bite on the payload `run_search` actually builds, not just
    /// in `truncate_snippet` — a tested helper nobody calls caps nothing. The
    /// fixture's own chunks are a few lines each, so this writes a function long
    /// enough to prove the wiring rather than passing vacuously on short chunks.
    #[tokio::test]
    async fn search_caps_snippet_lines_on_an_oversized_chunk() {
        let fixture = TestFixture::new("small_rust");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());

        let body = (0..60)
            .map(|n| format!("    let value_{n} = {n};"))
            .collect::<Vec<_>>()
            .join("\n");
        let long_fn = format!("pub fn subtract_many_numbers() -> i64 {{\n{body}\n    0\n}}\n");
        assert!(std::fs::write(fixture.root().join("src/long.rs"), &long_fn).is_ok());

        let config = stub_config();
        let claudix = test_claudix(fixture.root().to_path_buf(), config.clone());
        assert!(claudix.is_ok());
        let claudix = claudix.ok().unwrap_or_else(|| unreachable!());
        assert!(
            index_fixture(
                &claudix.store,
                claudix.embedder.as_ref(),
                claudix.project_root(),
                &config,
            )
            .await
            .is_ok()
        );
        assert!(write_fixture_config(fixture.root(), &config).is_ok());

        let output = run_search(
            fixture.root(),
            "subtract many numbers".to_owned(),
            Some(10),
            None,
            None,
            None,
        )
        .await;
        assert!(output.is_ok(), "search failed: {output:?}");
        let output = output.ok().unwrap_or_else(|| unreachable!());

        let hits: Vec<&SearchHit> = output.groups.iter().flat_map(|g| g.hits.iter()).collect();
        assert!(!hits.is_empty(), "expected the long function to be indexed");
        let capped = hits
            .iter()
            .find(|hit| hit.name.as_deref() == Some("subtract_many_numbers"));
        assert!(
            capped.is_some(),
            "the 62-line function must surface, else this test proves nothing"
        );
        let capped = capped.unwrap_or_else(|| unreachable!());

        // 20 kept lines + the ellipsis marker.
        assert_eq!(
            capped.snippet.lines().count(),
            crate::prompts::SNIPPET_MAX_LINES + 1
        );
        assert!(capped.snippet.ends_with('…'));
        for hit in &hits {
            assert!(
                hit.snippet.lines().count() <= crate::prompts::SNIPPET_MAX_LINES + 1,
                "{} snippet exceeded the cap",
                hit.file_path
            );
        }
    }

    #[tokio::test]
    async fn search_spans_active_and_listed_repo_with_correct_labels() {
        let setup = dual_repo_harness().await;
        assert!(setup.is_ok());
        let (fixture_a, _fixture_b, repo_a, repo_b) = setup.ok().unwrap_or_else(|| unreachable!());

        let output = run_search(
            fixture_a.root(),
            "add".to_owned(),
            Some(10),
            None,
            None,
            Some(vec![repo_b.clone()]),
        )
        .await;
        assert!(output.is_ok(), "cross-repo search failed: {output:?}");
        let output = output.ok().unwrap_or_else(|| unreachable!());

        assert!(output.repo_errors.is_empty(), "no errors expected");

        // Hits surface from both repos, each correctly labeled. The group owns
        // the repo label; hits no longer repeat it.
        let labelled = |repo: &str| {
            output
                .groups
                .iter()
                .any(|g| g.repo == repo && !g.hits.is_empty())
        };
        assert!(
            labelled(&repo_a),
            "expected at least one hit from active repo {repo_a}"
        );
        assert!(
            labelled(&repo_b),
            "expected at least one hit from extra repo {repo_b}"
        );
    }

    #[tokio::test]
    async fn search_partial_success_on_unindexed_repo() {
        let harness = cli_harness_private().await;
        assert!(harness.is_ok());
        let harness = harness.ok().unwrap_or_else(|| unreachable!());
        let write = write_fixture_config(harness.claudix.project_root(), &stub_config());
        assert!(write.is_ok());

        let unindexed = "/tmp/nonexistent-claudix-cross-repo-search-9999".to_owned();
        let output = run_search(
            harness.claudix.project_root(),
            "add".to_owned(),
            Some(10),
            None,
            None,
            Some(vec![unindexed.clone()]),
        )
        .await;
        assert!(output.is_ok());
        let output = output.ok().unwrap_or_else(|| unreachable!());

        // Unindexed path surfaces as a RepoError; the active repo still returns hits.
        assert!(
            output.repo_errors.iter().any(|e| e.repo == unindexed
                || e.error.contains("not indexed")
                || e.error.contains("No such")),
            "expected RepoError for unindexed path; got: {:?}",
            output.repo_errors,
        );
        assert!(
            !output.groups.is_empty(),
            "active repo must still produce hits despite the error"
        );
    }

    /// The search twin of
    /// `find_duplicates_reports_one_error_for_two_spellings_of_a_broken_repo`:
    /// two spellings of one broken extra repo must collapse to one `RepoError`.
    #[tokio::test]
    async fn search_reports_one_error_for_two_spellings_of_a_broken_repo() {
        let harness = cli_harness_private().await;
        assert!(harness.is_ok());
        let harness = harness.ok().unwrap_or_else(|| unreachable!());
        let write = write_fixture_config(harness.claudix.project_root(), &stub_config());
        assert!(write.is_ok());

        // Exists as a repo but was never indexed: it errors while both of its
        // spellings still canonicalize to one path.
        let broken = TestFixture::new("small_rust");
        assert!(broken.is_ok());
        let broken = broken.ok().unwrap_or_else(|| unreachable!());
        let spelled = broken.root().display().to_string();
        let respelled = format!("{spelled}/.");

        let output = run_search(
            harness.claudix.project_root(),
            "add".to_owned(),
            Some(10),
            None,
            None,
            Some(vec![spelled, respelled]),
        )
        .await;
        assert!(output.is_ok());
        let output = output.ok().unwrap_or_else(|| unreachable!());

        assert_eq!(
            output.repo_errors.len(),
            1,
            "one broken repo spelled two ways must yield one error; got: {:?}",
            output.repo_errors,
        );
        assert!(
            !output.groups.is_empty(),
            "active repo hits must survive the broken repo"
        );
    }

    #[tokio::test]
    async fn search_partial_success_on_model_mismatch() {
        let fixture_a = TestFixture::new("small_rust");
        assert!(fixture_a.is_ok());
        let fixture_a = fixture_a.ok().unwrap_or_else(|| unreachable!());
        let fixture_b = TestFixture::new("small_rust");
        assert!(fixture_b.is_ok());
        let fixture_b = fixture_b.ok().unwrap_or_else(|| unreachable!());

        // Active repo uses stub-v1; extra repo uses stub-v2 → mismatch.
        let config_a = stub_config();
        let config_b = {
            let mut c = stub_config();
            c.embedding.model = "stub-v2".to_owned();
            c
        };

        let claudix_a = test_claudix(fixture_a.root().to_path_buf(), config_a.clone());
        assert!(claudix_a.is_ok());
        let claudix_a = claudix_a.ok().unwrap_or_else(|| unreachable!());
        let _ = index_fixture(
            &claudix_a.store,
            claudix_a.embedder.as_ref(),
            claudix_a.project_root(),
            &config_a,
        )
        .await;
        let write_a = write_fixture_config(fixture_a.root(), &config_a);
        assert!(write_a.is_ok());

        let claudix_b = test_claudix(fixture_b.root().to_path_buf(), config_b.clone());
        assert!(claudix_b.is_ok());
        let claudix_b = claudix_b.ok().unwrap_or_else(|| unreachable!());
        let _ = index_fixture(
            &claudix_b.store,
            claudix_b.embedder.as_ref(),
            claudix_b.project_root(),
            &config_b,
        )
        .await;
        let write_b = write_fixture_config(fixture_b.root(), &config_b);
        assert!(write_b.is_ok());

        let repo_b = fixture_b.root().display().to_string();
        let output = run_search(
            fixture_a.root(),
            "add".to_owned(),
            Some(10),
            None,
            None,
            Some(vec![repo_b.clone()]),
        )
        .await;
        assert!(output.is_ok());
        let output = output.ok().unwrap_or_else(|| unreachable!());

        assert!(
            output
                .repo_errors
                .iter()
                .any(|e| e.error.contains("mismatch")),
            "expected mismatch error for extra repo; got: {:?}",
            output.repo_errors,
        );
        // Active repo still returns hits.
        assert!(
            !output.groups.is_empty(),
            "active repo hits must survive a sibling repo's mismatch"
        );
    }

    #[tokio::test]
    async fn search_dedupes_active_when_listed_in_repos() {
        let harness = cli_harness_private().await;
        assert!(harness.is_ok());
        let harness = harness.ok().unwrap_or_else(|| unreachable!());
        let write = write_fixture_config(harness.claudix.project_root(), &stub_config());
        assert!(write.is_ok());

        // Baseline: no extra repos.
        let baseline = run_search(
            harness.claudix.project_root(),
            "add".to_owned(),
            Some(10),
            None,
            None,
            None,
        )
        .await;
        assert!(baseline.is_ok());
        let baseline = baseline.ok().unwrap_or_else(|| unreachable!());
        let baseline_hits: usize = baseline.groups.iter().map(|g| g.hits.len()).sum();

        // List the active repo path explicitly — must not double-count.
        let active_path = harness.claudix.project_root().display().to_string();
        let echoed = run_search(
            harness.claudix.project_root(),
            "add".to_owned(),
            Some(10),
            None,
            None,
            Some(vec![active_path]),
        )
        .await;
        assert!(echoed.is_ok());
        let echoed = echoed.ok().unwrap_or_else(|| unreachable!());
        let echoed_hits: usize = echoed.groups.iter().map(|g| g.hits.len()).sum();

        assert_eq!(
            baseline_hits, echoed_hits,
            "listing active path must not duplicate hits"
        );
        assert!(echoed.repo_errors.is_empty());
    }

    #[tokio::test]
    async fn search_groups_separate_same_named_dirs_per_repo() {
        let setup = dual_repo_harness().await;
        assert!(setup.is_ok());
        let (fixture_a, _fixture_b, _repo_a, repo_b) = setup.ok().unwrap_or_else(|| unreachable!());

        let output = run_search(
            fixture_a.root(),
            "add".to_owned(),
            Some(20),
            None,
            None,
            Some(vec![repo_b]),
        )
        .await;
        assert!(output.is_ok());
        let output = output.ok().unwrap_or_else(|| unreachable!());

        // Both repos have a "src" directory; the grouping must keep them apart.
        let src_groups: Vec<&DirectoryGroup> = output
            .groups
            .iter()
            .filter(|g| g.directory == "src")
            .collect();
        assert!(
            src_groups.len() >= 2,
            "expected two distinct 'src' groups (one per repo), got {}",
            src_groups.len(),
        );
        let repos: HashSet<&str> = src_groups.iter().map(|g| g.repo.as_str()).collect();
        assert!(
            repos.len() >= 2,
            "src groups must come from distinct repos, got: {:?}",
            repos,
        );
    }

    #[tokio::test]
    async fn search_does_not_write_into_extra_repo() {
        let setup = dual_repo_harness().await;
        assert!(setup.is_ok());
        let (fixture_a, fixture_b, _repo_a, repo_b) = setup.ok().unwrap_or_else(|| unreachable!());

        // Snapshot the extra repo's .claudix state-dir tree before the search.
        let state_dir = fixture_b.root().join(".claudix");
        let before = snapshot_paths(&state_dir);

        let output = run_search(
            fixture_a.root(),
            "add".to_owned(),
            Some(10),
            None,
            None,
            Some(vec![repo_b]),
        )
        .await;
        assert!(output.is_ok());

        let after = snapshot_paths(&state_dir);
        assert_eq!(
            before, after,
            "cross-repo search must not write into the extra repo's .claudix dir",
        );
    }

    /// Snapshot every file path and its byte length under `dir` (recursive).
    /// Used to assert read-only behavior: nothing in the snapshot changes.
    fn snapshot_paths(dir: &Path) -> std::collections::BTreeSet<(PathBuf, u64)> {
        fn walk(dir: &Path, into: &mut std::collections::BTreeSet<(PathBuf, u64)>) {
            let Ok(entries) = std::fs::read_dir(dir) else {
                return;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                let Ok(metadata) = entry.metadata() else {
                    continue;
                };
                if metadata.is_dir() {
                    walk(&path, into);
                } else {
                    into.insert((path, metadata.len()));
                }
            }
        }
        let mut set = std::collections::BTreeSet::new();
        walk(dir, &mut set);
        set
    }

    #[test]
    fn effective_cross_repos_orders_config_then_call_args() {
        let cfg = vec!["/cfg/a".to_owned(), "/cfg/b".to_owned()];
        let call = Some(vec!["/cfg/b".to_owned(), "/call/c".to_owned()]);
        let merged = effective_cross_repos(&cfg, call);
        // Config first, call second, exact-string duplicates dropped.
        assert_eq!(merged, vec!["/cfg/a", "/cfg/b", "/call/c"]);
    }

    #[test]
    fn effective_cross_repos_empty_when_both_empty() {
        let merged = effective_cross_repos(&[], None);
        assert!(merged.is_empty());
    }

    /// `run_doctor` must surface `development_mode` from config and `binary_path`
    /// from the running process. The default config has `development_mode = false`;
    /// a project config with `development_mode = true` must flip the field.
    #[tokio::test]
    async fn run_doctor_surfaces_development_mode_and_binary_path() {
        let fixture = TestFixture::new("small_rust");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());

        // `development_mode` is a top-level key — must precede any section header.
        let claude_dir = fixture.root().join(".claude");
        assert!(std::fs::create_dir_all(&claude_dir).is_ok());
        assert!(
            std::fs::write(
                claude_dir.join("claudix.toml"),
                "development_mode = true\n\n[embedding]\nmodel = \"stub-v1\"\ndimensions = 8\n",
            )
            .is_ok()
        );

        let output = run_doctor(fixture.root()).await;
        assert!(output.is_ok(), "run_doctor failed: {:?}", output.err());
        let output = output.ok().unwrap_or_else(|| unreachable!());

        assert!(
            output.development_mode,
            "development_mode must be true when set in project config"
        );
        assert!(
            !output.binary_path.is_empty(),
            "binary_path must be a non-empty string"
        );
        assert!(
            output.binary_path != "<unknown>" || std::env::current_exe().is_err(),
            "binary_path must resolve to the current exe path when available"
        );
    }

    /// `run_doctor` must surface `development_mode = false` when the project config
    /// explicitly sets it to false, overriding any global setting.
    #[tokio::test]
    async fn run_doctor_development_mode_false_when_project_config_disables_it() {
        let fixture = TestFixture::new("small_rust");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());

        // `development_mode` is a top-level key — must appear before any
        // section header so the TOML parser does not assign it to [embedding].
        // This ensures the project layer wins over any global config the test
        // host may have (e.g. ~/.claude/claudix.toml with development_mode = true).
        let claude_dir = fixture.root().join(".claude");
        assert!(std::fs::create_dir_all(&claude_dir).is_ok());
        assert!(
            std::fs::write(
                claude_dir.join("claudix.toml"),
                "development_mode = false\n\n[embedding]\nmodel = \"stub-v1\"\ndimensions = 8\n",
            )
            .is_ok()
        );

        let output = run_doctor(fixture.root()).await;
        assert!(output.is_ok(), "run_doctor failed: {:?}", output.err());
        let output = output.ok().unwrap_or_else(|| unreachable!());

        assert!(
            !output.development_mode,
            "development_mode must be false when project config explicitly disables it"
        );
    }

    /// `run_index` must clear the store and produce a working index when
    /// `validate_manifest_compatibility` returns `SchemaMismatch` (future schema
    /// bump). This exercises the `requires_clean_reindex` → `clear_chunks` →
    /// `Claudix::new` retry path inside `IndexSession::new`.
    #[tokio::test]
    async fn run_index_clears_schema_mismatch_and_reindexes() {
        let fixture = TestFixture::new("small_rust");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());

        // Write a project config so run_index can load it.
        let claude_dir = fixture.root().join(".claude");
        assert!(std::fs::create_dir_all(&claude_dir).is_ok());
        assert!(
            std::fs::write(
                claude_dir.join("claudix.toml"),
                "[embedding]\nmodel = \"stub-v1\"\ndimensions = 8\n",
            )
            .is_ok()
        );

        let config = stub_config();
        let store = Store::new(fixture.root(), &config);
        assert!(store.is_ok());
        let store = store.ok().unwrap_or_else(|| unreachable!());

        // Inject a manifest with a future schema_version to trigger SchemaMismatch.
        let mut stale_manifest =
            Manifest::new(&config.embedding.model, config.embedding.dimensions);
        stale_manifest.schema_version = crate::store::SCHEMA_VERSION + 1;
        assert!(store.write_manifest(&stale_manifest).is_ok());

        // run_index must detect SchemaMismatch, clear, and reindex successfully.
        let output = run_index(fixture.root(), false).await;
        assert!(
            output.is_ok(),
            "run_index must succeed after schema mismatch: {:?}",
            output.err()
        );
        let output = output.ok().unwrap_or_else(|| unreachable!());
        assert!(
            output.chunk_count > 0,
            "reindex after schema mismatch must produce chunks"
        );

        // The manifest written after the reindex must carry the current schema version.
        let manifest = store.read_manifest();
        assert!(manifest.is_ok());
        let manifest = manifest.ok().unwrap_or_else(|| unreachable!());
        assert!(manifest.is_some());
        let manifest = manifest.unwrap_or_else(|| unreachable!());
        assert_eq!(
            manifest.schema_version,
            crate::store::SCHEMA_VERSION,
            "schema_version must match binary after clean reindex"
        );
    }
}
