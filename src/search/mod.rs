pub mod duplicates;
pub mod neighbors;

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use futures::future::join_all;
use tokio::{fs, task};

use crate::cli::{RepoError, load_repo_chunks_readonly};
use crate::config::SearchConfig;
use crate::embedding::Provider;
use crate::enumeration::hash_bytes;
use crate::error::{ClaudixError, RecoveryHint, Result};
use crate::prompts::hints;
use crate::store::{Store, StoredChunk};
use crate::types::{
    ByteRange, Chunk, ChunkId, ChunkKind, Dimension, FileHash, Language, LineRange, RelativePath,
    path_prefix_matches,
};

#[derive(Debug, Clone)]
pub struct SearchQuery {
    pub query: String,
    pub top_k: usize,
    pub language_filter: Option<Vec<Language>>,
    pub path_prefix: Option<RelativePath>,
    /// Additional repos to search read-only alongside the active project.
    /// The active project is always included; this set is union'd with it and
    /// deduped by canonical path.
    pub repos: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    pub chunk: Chunk,
    pub score: f32,
    pub stale: bool,
    /// Canonical path of the repo this hit came from. Staleness resolves the
    /// chunk's file against this root.
    pub repo: String,
}

/// Results plus the repos that could not contribute, for partial-success
/// reporting. `repo_errors` is empty for a plain single-repo search.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResults {
    pub results: Vec<SearchResult>,
    pub repo_errors: Vec<RepoError>,
}

/// Warn when a full search (embed + rank + staleness) exceeds this. Fires at the
/// default `warn` level so a slow index read or oversized corpus surfaces.
const QUERY_BUDGET_WARN_MS: u64 = 200;

#[derive(Clone)]
pub struct Searcher {
    project_root: PathBuf,
    store: Store,
    embedder: Arc<dyn Provider>,
    config: SearchConfig,
}

impl Searcher {
    pub fn new(
        project_root: PathBuf,
        store: Store,
        embedder: Arc<dyn Provider>,
        config: SearchConfig,
    ) -> Self {
        Self {
            project_root,
            store,
            embedder,
            config,
        }
    }

    pub async fn search(&self, query: SearchQuery) -> Result<SearchResults> {
        let search_start = std::time::Instant::now();
        let limit = effective_top_k(query.top_k, self.config.top_k);
        if limit == 0 || query.query.trim().is_empty() {
            return Ok(SearchResults {
                results: Vec::new(),
                repo_errors: Vec::new(),
            });
        }

        let mut found = self.search_all(query).await?;
        found.results = deduplicate_by_file_path(found.results);
        found.results.truncate(limit);

        let elapsed = search_start.elapsed();
        if elapsed.as_millis() as u64 > QUERY_BUDGET_WARN_MS {
            tracing::warn!(
                elapsed_ms = elapsed.as_millis(),
                budget_ms = QUERY_BUDGET_WARN_MS,
                "search exceeded budget"
            );
        }

        Ok(found)
    }

    async fn search_all(&self, query: SearchQuery) -> Result<SearchResults> {
        if query.query.trim().is_empty() {
            return Ok(SearchResults {
                results: Vec::new(),
                repo_errors: Vec::new(),
            });
        }

        let (labeled_rows, repo_errors) = self.collect_labeled_rows(&query).await?;
        if labeled_rows.is_empty() {
            return Ok(SearchResults {
                results: Vec::new(),
                repo_errors,
            });
        }

        // Embed the query once with the active embedder. Its model/dimensions
        // are the reference identity every cross-repo's vectors must match.
        // When the active provider has bundled fallback enabled, this call falls
        // back before ranking the non-empty corpus.
        let vectors = self.embedder.embed(&[query.query.as_str()]).await?;
        if vectors.len() != 1 {
            return Err(ClaudixError::Embedding(format!(
                "provider returned {} vectors for 1 query",
                vectors.len()
            )));
        }
        let query_vector = vectors.into_iter().next().unwrap_or_default();
        validate_query_vector(&query_vector, self.embedder.dimensions())?;

        let config = self.config.clone();
        let mut results =
            task::spawn_blocking(move || rank_rows(query, labeled_rows, query_vector, config))
                .await
                .map_err(|error| ClaudixError::Store(format!("search task failed: {error}")))??;
        mark_stale_results(&mut results).await?;
        Ok(SearchResults {
            results,
            repo_errors,
        })
    }

    /// Build the union corpus: active project (always included) plus the
    /// resolved extra repos, each labeled with its canonical path. Errored
    /// repos are collected separately so the rest still search (partial success).
    async fn collect_labeled_rows(
        &self,
        query: &SearchQuery,
    ) -> Result<(Vec<(Arc<str>, StoredChunk)>, Vec<RepoError>)> {
        let active_repo: Arc<str> = Arc::from(self.project_root.display().to_string().as_str());
        let active_rows = self.store.read_chunks().await?;

        let mut labeled: Vec<(Arc<str>, StoredChunk)> = active_rows
            .into_iter()
            .map(|row| (Arc::clone(&active_repo), row))
            .collect();
        let mut repo_errors = Vec::new();

        if query.repos.is_empty() {
            return Ok((labeled, repo_errors));
        }

        // Reference identity comes from the active manifest so the single query
        // embedding is comparable to every repo's stored vectors.
        let ref_model = self.embedder.model_id();
        let ref_dims = self.embedder.dimensions().0;
        let mut seen: HashSet<Arc<str>> = HashSet::from([Arc::clone(&active_repo)]);

        for repo in &query.repos {
            match load_repo_chunks_readonly(repo, ref_model, ref_dims).await {
                Ok((canonical, rows)) => {
                    let canonical: Arc<str> = Arc::from(canonical.as_str());
                    if !seen.insert(Arc::clone(&canonical)) {
                        continue;
                    }
                    labeled.extend(rows.into_iter().map(|row| (Arc::clone(&canonical), row)));
                }
                Err(error) => repo_errors.push(error),
            }
        }

        Ok((labeled, repo_errors))
    }
}

#[derive(Debug)]
struct DocumentStats {
    term_frequencies: HashMap<String, usize>,
    length: usize,
}

impl DocumentStats {
    fn from_content(content: &str) -> Self {
        let tokens = tokenize(content);
        let length = tokens.len();
        let mut term_frequencies = HashMap::new();

        for token in tokens {
            *term_frequencies.entry(token).or_insert(0) += 1;
        }

        Self {
            term_frequencies,
            length,
        }
    }
}

/// Per-row intermediate scores used during hybrid ranking.
///
/// Replaces the nine parallel `Vec`s that the old code materialised: instead,
/// a single pass over `filtered_rows` fills one `RowScore` per row, which are
/// then sorted / normalised in-place before the final `filter_map` collect.
/// Rank computation still needs a sorted pass over scores, and normalisation
/// needs min/max — so two auxiliary passes remain — but no parallel allocations.
#[derive(Default)]
struct RowScore {
    dense: f32,
    bm25: f32,
    rrf: f32,
    /// Populated after `rank_positions` + `reciprocal_rank_fusion`.
    dense_norm: f32,
    bm25_norm: f32,
    rrf_norm: f32,
}

fn rank_rows(
    query: SearchQuery,
    rows: Vec<(Arc<str>, StoredChunk)>,
    query_vector: Vec<f32>,
    config: SearchConfig,
) -> Result<Vec<SearchResult>> {
    let query_tokens = tokenize(&query.query);
    let filtered_rows = apply_filters(rows, &query);
    if filtered_rows.is_empty() {
        return Ok(Vec::new());
    }

    // Build document stats for the whole filtered corpus once, so BM25 IDF is
    // computed against real document frequency rather than a degenerate n=1
    // corpus that treats every token as equally rare.
    let docs: Vec<DocumentStats> = filtered_rows
        .iter()
        .map(|(_, row)| DocumentStats::from_content(&row.content))
        .collect();
    let bm25_per_row = bm25_scores(&docs, &query_tokens);

    // Single pass: dense scores zipped with the pre-computed cross-corpus BM25.
    let mut scores: Vec<RowScore> = filtered_rows
        .iter()
        .zip(bm25_per_row)
        .map(|((_, row), bm25)| {
            let dense = cosine_similarity(&query_vector, &row.vector).max(0.0);
            RowScore {
                dense,
                bm25,
                ..Default::default()
            }
        })
        .collect();

    // Rank positions need sorted score order — one pass each.
    let dense_ranks = rank_positions_from(&scores, |s| s.dense);
    let bm25_ranks = rank_positions_from(&scores, |s| s.bm25);

    // Fill RRF and normalised columns in-place.
    const RRF_K: f32 = 60.0;
    for (i, s) in scores.iter_mut().enumerate() {
        s.rrf = dense_ranks[i].map_or(0.0, |r| 1.0 / (RRF_K + r as f32))
            + bm25_ranks[i].map_or(0.0, |r| 1.0 / (RRF_K + r as f32));
    }

    normalize_field(&mut scores, |s| s.dense, |s, v| s.dense_norm = v);
    normalize_field(&mut scores, |s| s.bm25, |s, v| s.bm25_norm = v);
    normalize_field(&mut scores, |s| s.rrf, |s, v| s.rrf_norm = v);

    let mut results = filtered_rows
        .into_iter()
        .zip(scores)
        .filter_map(|((repo, row), s)| {
            let identifier_hit = row
                .name
                .as_deref()
                .is_some_and(|name| name_contains_query_token(name, &query_tokens));
            let lexical_hit = s.bm25 > 0.0 || identifier_hit;
            let dense_hit = s.dense >= config.similarity_threshold;
            let combined_score = config.hybrid_weights.dense * s.dense_norm
                + config.hybrid_weights.bm25 * s.bm25_norm
                + config.hybrid_weights.rrf * s.rrf_norm;
            if !lexical_hit && !dense_hit {
                return None;
            }

            let boosted_score = if identifier_hit {
                (combined_score * config.identifier_boost).max(config.identifier_boost * 0.3)
            } else {
                combined_score
            };
            let boosted_score = boosted_score.clamp(0.0, 1.0);

            if boosted_score < config.min_score {
                return None;
            }

            Some(SearchResult {
                chunk: stored_chunk_to_chunk(row),
                score: boosted_score,
                stale: false,
                repo: repo.to_string(),
            })
        })
        .collect::<Vec<_>>();

    sort_results(&mut results);
    Ok(results)
}

/// `rank_positions` variant that reads scores from a `RowScore` slice via a
/// field accessor, avoiding a temporary score `Vec`.
fn rank_positions_from<F>(scores: &[RowScore], get: F) -> Vec<Option<usize>>
where
    F: Fn(&RowScore) -> f32,
{
    let mut indexed: Vec<(usize, f32)> = scores
        .iter()
        .enumerate()
        .filter_map(|(i, s)| {
            let v = get(s);
            if v > 0.0 { Some((i, v)) } else { None }
        })
        .collect();
    indexed.sort_by(|l, r| compare_scores_desc(l.1, r.1, l.0, r.0));

    let mut positions = vec![None; scores.len()];
    for (rank, (index, _)) in indexed.into_iter().enumerate() {
        positions[index] = Some(rank + 1);
    }
    positions
}

/// Normalise a score field across all rows in-place (min-max, floor 0).
fn normalize_field<Get, Set>(rows: &mut [RowScore], get: Get, set: Set)
where
    Get: Fn(&RowScore) -> f32,
    Set: Fn(&mut RowScore, f32),
{
    let max = rows.iter().map(&get).fold(0.0_f32, f32::max);
    if max <= 0.0 {
        for row in rows.iter_mut() {
            set(row, 0.0);
        }
        return;
    }
    for row in rows.iter_mut() {
        set(row, (get(row) / max).clamp(0.0, 1.0));
    }
}

fn effective_top_k(requested: usize, default_top_k: usize) -> usize {
    if requested == 0 {
        default_top_k
    } else {
        requested
    }
}

async fn mark_stale_results(results: &mut [SearchResult]) -> Result<()> {
    // Build one future per result and drive them concurrently. Each future
    // resolves to `(index, is_stale)` so we can write back without borrowing
    // the slice across the await.
    let checks = results
        .iter()
        .enumerate()
        .map(|(i, result)| {
            let repo_root = PathBuf::from(&result.repo);
            let chunk = result.chunk.clone();
            async move {
                let stale = result_is_stale(&repo_root, &chunk).await.unwrap_or(true);
                (i, stale)
            }
        })
        .collect::<Vec<_>>();

    for (i, stale) in join_all(checks).await {
        results[i].stale = stale;
    }

    Ok(())
}

/// Upper bound on a file we will read+hash for the staleness check, mirroring
/// the indexing default `[indexing].max_file_size_kb` (512 KB). A file larger
/// than this could not have been indexed under the default, so re-reading it on
/// every search returning its chunk is wasted I/O; treat oversized as stale.
const MAX_STALE_CHECK_BYTES: u64 = 512 * 1024;

/// Returns `true` when the file on disk differs from what was indexed.
///
/// `StoredChunk` carries no per-row index timestamp and `byte_range.end` is a
/// chunk boundary, not the total file size, so no cheap size shortcut is
/// available beyond the stat below. We `stat` first and cap the read at
/// [`MAX_STALE_CHECK_BYTES`]: an oversized file is reported stale without an
/// unbounded read. Otherwise read and re-hash the full file; missing /
/// unreadable → stale (fail-open).
async fn result_is_stale(repo_root: &Path, chunk: &Chunk) -> Result<bool> {
    let path = resolve_chunk_path(repo_root, &chunk.file_path)?;
    let Ok(metadata) = fs::metadata(&path).await else {
        return Ok(true);
    };
    if metadata.len() > MAX_STALE_CHECK_BYTES {
        return Ok(true);
    }
    let Ok(contents) = fs::read(path).await else {
        return Ok(true);
    };

    Ok(hash_bytes(&contents) != chunk.file_hash)
}

fn resolve_chunk_path(repo_root: &Path, relative_path: &RelativePath) -> Result<PathBuf> {
    relative_path.reject_escape("Only read search result files inside the declared repo roots")?;
    Ok(repo_root.join(relative_path.to_path_buf()))
}

fn validate_query_vector(vector: &[f32], dimensions: Dimension) -> Result<()> {
    if vector.len() != usize::from(dimensions.0) {
        return Err(ClaudixError::DimensionMismatch {
            store_dim: dimensions.0,
            model_dim: u16::try_from(vector.len()).unwrap_or(u16::MAX),
            recovery: RecoveryHint(hints::REBUILD_INDEX_DIMENSIONS),
        });
    }

    if vector.iter().any(|value| !value.is_finite()) {
        return Err(ClaudixError::Embedding(
            "provider returned non-finite query embedding values".to_owned(),
        ));
    }

    Ok(())
}

fn apply_filters(
    rows: Vec<(Arc<str>, StoredChunk)>,
    query: &SearchQuery,
) -> Vec<(Arc<str>, StoredChunk)> {
    let language_filter = query.language_filter.as_ref().map(|languages| {
        languages
            .iter()
            .map(|language| language.as_str())
            .collect::<HashSet<_>>()
    });
    let path_prefix = query.path_prefix.as_ref().map(RelativePath::as_str);

    rows.into_iter()
        .filter(|(_, row)| {
            if let Some(language_filter) = &language_filter
                && !language_filter.contains(row.language.as_str())
            {
                return false;
            }

            if let Some(path_prefix) = path_prefix
                && !path_prefix_matches(&row.file_path, path_prefix)
            {
                return false;
            }

            true
        })
        .collect()
}

fn bm25_scores(documents: &[DocumentStats], query_tokens: &[String]) -> Vec<f32> {
    const K1: f32 = 1.2;
    const B: f32 = 0.75;

    if documents.is_empty() {
        return Vec::new();
    }

    let unique_tokens = query_tokens.iter().cloned().collect::<HashSet<_>>();
    if unique_tokens.is_empty() {
        return vec![0.0; documents.len()];
    }

    let doc_count = documents.len() as f32;
    let average_length = documents.iter().map(|doc| doc.length).sum::<usize>() as f32 / doc_count;
    let mut document_frequency = HashMap::new();

    for token in &unique_tokens {
        let matches = documents
            .iter()
            .filter(|doc| doc.term_frequencies.contains_key(token))
            .count();
        document_frequency.insert(token.clone(), matches as f32);
    }

    documents
        .iter()
        .map(|doc| {
            unique_tokens
                .iter()
                .map(|token| {
                    let term_frequency = *doc.term_frequencies.get(token).unwrap_or(&0) as f32;
                    if term_frequency == 0.0 {
                        return 0.0;
                    }

                    let frequency = *document_frequency.get(token).unwrap_or(&0.0);
                    let idf = ((doc_count - frequency + 0.5) / (frequency + 0.5) + 1.0).ln();
                    let length = doc.length.max(1) as f32;
                    let numerator = term_frequency * (K1 + 1.0);
                    let denominator =
                        term_frequency + K1 * (1.0 - B + B * (length / average_length.max(1.0)));

                    idf * (numerator / denominator)
                })
                .sum()
        })
        .collect()
}

fn deduplicate_by_file_path(results: Vec<SearchResult>) -> Vec<SearchResult> {
    let mut seen_paths = HashSet::new();
    let mut deduplicated = Vec::new();

    for result in results {
        // Key on (repo, file) so a same-named file in two repos keeps both.
        if seen_paths.insert((result.repo.clone(), result.chunk.file_path.clone())) {
            deduplicated.push(result);
        }
    }

    deduplicated
}

fn sort_results(results: &mut [SearchResult]) {
    results.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.repo.cmp(&right.repo))
            .then_with(|| {
                left.chunk
                    .file_path
                    .as_str()
                    .cmp(right.chunk.file_path.as_str())
            })
            .then_with(|| {
                left.chunk
                    .line_range
                    .start
                    .cmp(&right.chunk.line_range.start)
            })
            .then_with(|| left.chunk.line_range.end.cmp(&right.chunk.line_range.end))
    });
}

fn compare_scores_desc(
    left_score: f32,
    right_score: f32,
    left_index: usize,
    right_index: usize,
) -> Ordering {
    right_score
        .partial_cmp(&left_score)
        .unwrap_or(Ordering::Equal)
        .then_with(|| left_index.cmp(&right_index))
}

pub(crate) fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    if left.len() != right.len() || left.is_empty() {
        return 0.0;
    }

    let mut dot = 0.0;
    let mut left_norm = 0.0;
    let mut right_norm = 0.0;

    for index in 0..left.len() {
        dot += left[index] * right[index];
        left_norm += left[index] * left[index];
        right_norm += right[index] * right[index];
    }

    if left_norm == 0.0 || right_norm == 0.0 {
        return 0.0;
    }

    dot / (left_norm.sqrt() * right_norm.sqrt())
}

fn tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() {
            current.push(ch.to_ascii_lowercase());
        } else if !current.is_empty() {
            tokens.push(std::mem::take(&mut current));
        }
    }

    if !current.is_empty() {
        tokens.push(current);
    }

    tokens
}

fn name_contains_query_token(name: &str, query_tokens: &[String]) -> bool {
    let lowercase_name = name.to_ascii_lowercase();
    query_tokens
        .iter()
        .any(|token| !token.is_empty() && lowercase_name.contains(token))
}

fn stored_chunk_to_chunk(row: StoredChunk) -> Chunk {
    let StoredChunk {
        chunk_id,
        file_path,
        language,
        kind,
        name,
        line_start,
        line_end,
        byte_start,
        byte_end,
        file_hash,
        content,
        vector: _,
    } = row;

    Chunk {
        id: ChunkId(chunk_id),
        file_path: RelativePath::new(file_path),
        language: Language::from_storage(&language),
        kind: ChunkKind::from_storage(&kind),
        name,
        line_range: LineRange {
            start: line_start,
            end: line_end,
        },
        byte_range: ByteRange {
            start: byte_start,
            end: byte_end,
        },
        file_hash: FileHash(file_hash),
        content,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;

    use crate::embedding::{Provider, StubProvider};
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

    struct FixedProvider {
        dimension: Dimension,
        vectors: Vec<Vec<f32>>,
    }

    #[async_trait]
    impl Provider for FixedProvider {
        fn name(&self) -> &str {
            "fixed"
        }

        fn dimensions(&self) -> Dimension {
            self.dimension
        }

        fn model_id(&self) -> &str {
            "fixed-model"
        }

        async fn embed(&self, batch: &[&str]) -> Result<Vec<Vec<f32>>> {
            Ok(self.vectors.iter().take(batch.len()).cloned().collect())
        }

        async fn health_check(&self) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn tokenize_splits_snake_case() {
        assert_eq!(
            tokenize("handle_session_start"),
            vec!["handle", "session", "start"]
        );
        assert_eq!(
            tokenize("fn full_index_running"),
            vec!["fn", "full", "index", "running"]
        );
    }

    #[test]
    fn cosine_similarity_returns_zero_for_dimension_mismatch() {
        assert_eq!(cosine_similarity(&[1.0, 0.0], &[1.0, 0.0, 0.0]), 0.0);
        assert_eq!(cosine_similarity(&[1.0, 0.0, 0.0], &[1.0, 0.0]), 0.0);
        assert_eq!(cosine_similarity(&[], &[1.0]), 0.0);
    }

    #[test]
    fn bm25_scores_match_snake_case_content() {
        let docs = vec![
            DocumentStats::from_content("async fn handle_session_start(project_root: &Path)"),
            DocumentStats::from_content("pub fn with_fallback_params(max: usize) -> Self"),
        ];
        let scores = bm25_scores(&docs, &tokenize("session start hook"));
        assert!(
            scores[0] > 0.0,
            "handle_session_start should match 'session start'"
        );
        assert_eq!(scores[1], 0.0, "with_fallback_params should not match");
    }

    #[test]
    fn bm25_rare_term_outranks_common_term() {
        // 9 docs contain only "foo" (common, df = 9/10); 1 contains only
        // "zygote" (rare, df = 1/10). For query "foo zygote" the rare-term doc
        // must win on IDF — which a hardcoded IDF = 1.0 could never produce.
        let mut docs: Vec<DocumentStats> = (0..9)
            .map(|_| DocumentStats::from_content("foo bar"))
            .collect();
        docs.push(DocumentStats::from_content("zygote baz"));

        let scores = bm25_scores(&docs, &tokenize("foo zygote"));
        let rare_score = scores[9];
        let common_score = scores[0];

        assert!(rare_score > 0.0, "rare-term doc must score its query token");
        assert!(
            common_score > 0.0,
            "common-term doc must score its query token"
        );
        assert!(
            rare_score > common_score,
            "rare term (high IDF) must outscore common term (low IDF): rare={rare_score}, common={common_score}"
        );
    }

    /// Wrap raw `StoredChunk`s with a fake repo label for `rank_rows` tests.
    fn label_rows(
        rows: Vec<crate::store::StoredChunk>,
    ) -> Vec<(Arc<str>, crate::store::StoredChunk)> {
        let label: Arc<str> = Arc::from("/test/repo");
        rows.into_iter().map(|r| (Arc::clone(&label), r)).collect()
    }

    #[test]
    fn rank_rows_handles_dominant_dense_with_bm25_hits() -> Result<()> {
        use crate::config::{HybridWeights, SearchConfig};
        use crate::store::StoredChunk;

        let config = SearchConfig {
            top_k: 10,
            hybrid_weights: HybridWeights {
                dense: 0.55,
                bm25: 0.30,
                rrf: 0.15,
            },
            identifier_boost: 1.4,
            similarity_threshold: 0.30,
            min_score: 0.0,
            cross_repos: Vec::new(),
        };

        let make_row = |name: &str, content: &str, vec: Vec<f32>| StoredChunk {
            chunk_id: 0,
            file_path: format!("src/{name}.rs"),
            language: "rust".into(),
            kind: "function".into(),
            name: Some(name.into()),
            line_start: 1,
            line_end: 5,
            byte_start: 0,
            byte_end: 100,
            file_hash: [0u8; 16],
            content: content.into(),
            vector: vec,
        };

        // Simulate: new() has high cosine (0.9 of max), but no BM25
        // handle_session_start has lower cosine (0.3 of max), but strong BM25
        // The BM25 + identifier_boost should push handle_session_start above new()
        let rows = vec![
            make_row(
                "new",
                "pub fn new(max: usize) -> Self { Self { max } }",
                vec![0.9, 0.1, 0.0, 0.0],
            ),
            make_row(
                "handle_session_start",
                "async fn handle_session_start(root: &Path) -> Result<Option<Value>> { let config = load(root); }",
                vec![0.3, 0.8, 0.0, 0.0],
            ),
        ];
        let query_vector = vec![1.0, 0.0, 0.0, 0.0];
        let query = SearchQuery {
            query: "session start hook".into(),
            top_k: 10,
            language_filter: None,
            path_prefix: None,
            repos: Vec::new(),
        };

        let results = rank_rows(query, label_rows(rows), query_vector, config)?;
        assert_eq!(results.len(), 2, "both chunks should pass the filter");
        assert_eq!(
            results[0].chunk.name.as_deref(),
            Some("handle_session_start"),
            "identifier_boost should lift handle_session_start above new()"
        );
        Ok(())
    }

    #[test]
    fn rank_rows_filters_results_below_min_score() -> Result<()> {
        use crate::config::{HybridWeights, SearchConfig};
        use crate::store::StoredChunk;

        let config = SearchConfig {
            top_k: 10,
            hybrid_weights: HybridWeights {
                dense: 0.55,
                bm25: 0.30,
                rrf: 0.15,
            },
            identifier_boost: 1.0,
            similarity_threshold: 0.0,
            min_score: 0.50,
            cross_repos: Vec::new(),
        };
        let row = |name: &str, vector: Vec<f32>| StoredChunk {
            chunk_id: 0,
            file_path: format!("src/{name}.rs"),
            language: "rust".into(),
            kind: "function".into(),
            name: Some(name.into()),
            line_start: 1,
            line_end: 5,
            byte_start: 0,
            byte_end: 100,
            file_hash: [0u8; 16],
            content: format!("pub fn {name}() {{}}"),
            vector,
        };
        let rows = vec![
            row("strong_match", vec![1.0, 0.0, 0.0, 0.0]),
            row("weak_candidate", vec![0.0, 1.0, 0.0, 0.0]),
        ];
        let query = SearchQuery {
            query: "strong".into(),
            top_k: 10,
            language_filter: None,
            path_prefix: None,
            repos: Vec::new(),
        };

        let results = rank_rows(query, label_rows(rows), vec![1.0, 0.0, 0.0, 0.0], config)?;
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk.name.as_deref(), Some("strong_match"));
        Ok(())
    }

    #[test]
    fn rank_rows_returns_multiple_results_for_code_query() -> Result<()> {
        use crate::config::{HybridWeights, SearchConfig};
        use crate::store::StoredChunk;

        let config = SearchConfig {
            top_k: 10,
            hybrid_weights: HybridWeights {
                dense: 0.55,
                bm25: 0.30,
                rrf: 0.15,
            },
            identifier_boost: 1.4,
            similarity_threshold: 0.30,
            min_score: 0.0,
            cross_repos: Vec::new(),
        };

        let make_row = |name: &str, content: &str, sim: f32| {
            let v: Vec<f32> = vec![sim, 0.0, 0.0, 0.0];
            StoredChunk {
                chunk_id: 0,
                file_path: format!("src/{name}.rs"),
                language: "rust".into(),
                kind: "function".into(),
                name: Some(name.into()),
                line_start: 1,
                line_end: 5,
                byte_start: 0,
                byte_end: 100,
                file_hash: [0u8; 16],
                content: content.into(),
                vector: v,
            }
        };

        let rows = vec![
            make_row(
                "new",
                "pub fn new(max: usize) -> Self { Self { max } }",
                0.8,
            ),
            make_row(
                "handle_session_start",
                "async fn handle_session_start(root: &Path) { let config = load(root); }",
                0.2,
            ),
            make_row(
                "full_index_running",
                "pub fn full_index_running(&self) -> bool { let lock = self.lock_path(); }",
                0.1,
            ),
        ];

        let query_vector = vec![1.0, 0.0, 0.0, 0.0];
        let query = SearchQuery {
            query: "session start hook message".into(),
            top_k: 10,
            language_filter: None,
            path_prefix: None,
            repos: Vec::new(),
        };

        let results = rank_rows(query, label_rows(rows), query_vector, config)?;
        assert!(
            results.len() >= 2,
            "BM25 should match handle_session_start and others: got {} results",
            results.len()
        );
        // handle_session_start should rank above new() because BM25 matches override low dense
        let top_name = results[0].chunk.name.as_deref().unwrap_or("");
        assert_ne!(
            top_name, "new",
            "trivial new() should not rank first when BM25 matches exist"
        );
        Ok(())
    }

    struct SearchHarness {
        _fixture: TestFixture,
        searcher: Searcher,
    }

    async fn search_harness() -> Result<SearchHarness> {
        let fixture = TestFixture::new("small_rust")?;
        let config = stub_config();
        let store = Store::new(fixture.root(), &config)?;
        let embedder: Arc<dyn Provider> = Arc::new(StubProvider::with_model_id(
            config.embedding.model.clone(),
            Dimension(config.embedding.dimensions),
        ));

        index_fixture(&store, embedder.as_ref(), fixture.root(), &config).await?;

        Ok(SearchHarness {
            searcher: Searcher::new(
                fixture.root().to_path_buf(),
                store,
                embedder,
                config.search.clone(),
            ),
            _fixture: fixture,
        })
    }

    #[tokio::test]
    async fn search_returns_empty_before_embedding_when_corpus_empty() -> Result<()> {
        let fixture = TestFixture::new("small_rust")?;
        let config = stub_config();
        let store = Store::new(fixture.root(), &config)?;
        let embedder: Arc<dyn Provider> = Arc::new(FixedProvider {
            dimension: Dimension(384),
            vectors: Vec::new(),
        });
        let searcher = Searcher::new(
            fixture.root().to_path_buf(),
            store,
            embedder,
            config.search.clone(),
        );

        let output = searcher
            .search_all(SearchQuery {
                query: "add".to_owned(),
                top_k: 10,
                language_filter: None,
                path_prefix: None,
                repos: Vec::new(),
            })
            .await?;

        assert!(output.results.is_empty());
        assert!(output.repo_errors.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn search_rejects_non_finite_query_embedding() {
        let harness = search_harness().await;
        assert!(harness.is_ok());
        let harness = harness.ok().unwrap_or_else(|| unreachable!());
        let embedder: Arc<dyn Provider> = Arc::new(FixedProvider {
            dimension: Dimension(384),
            vectors: vec![vec![f32::NAN; 384]],
        });
        let searcher = Searcher::new(
            harness.searcher.project_root,
            harness.searcher.store,
            embedder,
            harness.searcher.config,
        );

        let error = searcher
            .search_all(SearchQuery {
                query: "add".to_owned(),
                top_k: 10,
                language_filter: None,
                path_prefix: None,
                repos: Vec::new(),
            })
            .await;

        assert!(
            matches!(error, Err(ClaudixError::Embedding(message)) if message.contains("non-finite query"))
        );
    }

    #[tokio::test]
    async fn search_prefers_identifier_matches() {
        let harness = search_harness().await;
        assert!(harness.is_ok());
        let harness = harness.ok().unwrap_or_else(|| unreachable!());

        let results = harness
            .searcher
            .search(SearchQuery {
                query: "add".to_owned(),
                top_k: 10,
                language_filter: None,
                path_prefix: None,
                repos: Vec::new(),
            })
            .await;
        assert!(results.is_ok());
        let results = results.ok().unwrap_or_else(|| unreachable!()).results;

        assert!(!results.is_empty());
        assert_eq!(results[0].chunk.name.as_deref(), Some("add"));
        assert_eq!(results[0].chunk.file_path.as_str(), "src/math.rs");
        assert!(!results[0].stale);
    }

    #[tokio::test]
    async fn search_marks_modified_source_file_stale() {
        let harness = search_harness().await;
        assert!(harness.is_ok());
        let harness = harness.ok().unwrap_or_else(|| unreachable!());
        assert!(
            tokio::fs::write(
                harness.searcher.project_root.join("src/math.rs"),
                "pub fn subtract(left: i32, right: i32) -> i32 { left - right }\n",
            )
            .await
            .is_ok()
        );

        let results = harness
            .searcher
            .search(SearchQuery {
                query: "add".to_owned(),
                top_k: 10,
                language_filter: None,
                path_prefix: None,
                repos: Vec::new(),
            })
            .await;
        assert!(results.is_ok());
        let results = results.ok().unwrap_or_else(|| unreachable!()).results;

        assert!(!results.is_empty());
        assert!(results[0].stale);
    }

    #[tokio::test]
    async fn search_marks_missing_source_file_stale() {
        let harness = search_harness().await;
        assert!(harness.is_ok());
        let harness = harness.ok().unwrap_or_else(|| unreachable!());
        assert!(
            tokio::fs::remove_file(harness.searcher.project_root.join("src/math.rs"))
                .await
                .is_ok()
        );

        let results = harness
            .searcher
            .search(SearchQuery {
                query: "add".to_owned(),
                top_k: 10,
                language_filter: None,
                path_prefix: None,
                repos: Vec::new(),
            })
            .await;
        assert!(results.is_ok());
        let results = results.ok().unwrap_or_else(|| unreachable!()).results;

        assert!(!results.is_empty());
        assert!(results[0].stale);
    }

    #[tokio::test]
    async fn result_is_stale_treats_oversized_file_as_stale() {
        // A file larger than the staleness read cap must be reported stale
        // without an unbounded read, even when its content hash would match.
        let dir = tempfile::tempdir().expect("tempdir");
        let big = vec![b'x'; (MAX_STALE_CHECK_BYTES as usize) + 1];
        let matching_hash = hash_bytes(&big);
        tokio::fs::write(dir.path().join("big.rs"), &big)
            .await
            .expect("write oversized file");

        let row = StoredChunk {
            chunk_id: 0,
            file_path: "big.rs".to_owned(),
            language: "rust".into(),
            kind: "function".into(),
            name: None,
            line_start: 1,
            line_end: 1,
            byte_start: 0,
            byte_end: 1,
            file_hash: matching_hash.0,
            content: String::new(),
            vector: vec![0.0],
        };
        let chunk = stored_chunk_to_chunk(row);

        let stale = result_is_stale(dir.path(), &chunk).await;
        assert!(matches!(stale, Ok(true)), "oversized file must be stale");
    }

    #[tokio::test]
    async fn result_is_stale_matches_unchanged_in_cap_file() {
        // A file at or below the cap with a matching hash is NOT stale.
        let dir = tempfile::tempdir().expect("tempdir");
        let small = b"pub fn ok() {}\n";
        let matching_hash = hash_bytes(small);
        tokio::fs::write(dir.path().join("ok.rs"), small)
            .await
            .expect("write file");

        let row = StoredChunk {
            chunk_id: 0,
            file_path: "ok.rs".to_owned(),
            language: "rust".into(),
            kind: "function".into(),
            name: None,
            line_start: 1,
            line_end: 1,
            byte_start: 0,
            byte_end: small.len() as u32,
            file_hash: matching_hash.0,
            content: String::new(),
            vector: vec![0.0],
        };
        let chunk = stored_chunk_to_chunk(row);

        let stale = result_is_stale(dir.path(), &chunk).await;
        assert!(matches!(stale, Ok(false)), "unchanged in-cap file is fresh");
    }

    #[tokio::test]
    async fn search_applies_language_filters() {
        let harness = search_harness().await;
        assert!(harness.is_ok());
        let harness = harness.ok().unwrap_or_else(|| unreachable!());

        let results = harness
            .searcher
            .search(SearchQuery {
                query: "greet".to_owned(),
                top_k: 10,
                language_filter: Some(vec![Language::Python]),
                path_prefix: None,
                repos: Vec::new(),
            })
            .await;
        assert!(results.is_ok());
        let results = results.ok().unwrap_or_else(|| unreachable!()).results;

        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn search_applies_path_prefix_filters() {
        let harness = search_harness().await;
        assert!(harness.is_ok());
        let harness = harness.ok().unwrap_or_else(|| unreachable!());

        let results = harness
            .searcher
            .search(SearchQuery {
                query: "add".to_owned(),
                top_k: 10,
                language_filter: Some(vec![Language::Rust]),
                path_prefix: Some(RelativePath::new("src/math")),
                repos: Vec::new(),
            })
            .await;
        assert!(results.is_ok());
        let results = results.ok().unwrap_or_else(|| unreachable!()).results;

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk.file_path.as_str(), "src/math.rs");
    }

    #[tokio::test]
    async fn search_deduplicates_top_results_by_file_path() {
        let harness = search_harness().await;
        assert!(harness.is_ok());
        let harness = harness.ok().unwrap_or_else(|| unreachable!());

        let results = harness
            .searcher
            .search(SearchQuery {
                query: "pub".to_owned(),
                top_k: 10,
                language_filter: None,
                path_prefix: None,
                repos: Vec::new(),
            })
            .await;
        assert!(results.is_ok());
        let results = results.ok().unwrap_or_else(|| unreachable!()).results;

        let unique_paths = results
            .iter()
            .map(|result| result.chunk.file_path.as_str())
            .collect::<HashSet<_>>();
        assert_eq!(unique_paths.len(), results.len());
        assert_eq!(unique_paths.len(), 2);
    }

    #[tokio::test]
    async fn search_all_keeps_multiple_chunks_from_same_file() {
        let harness = search_harness().await;
        assert!(harness.is_ok());
        let harness = harness.ok().unwrap_or_else(|| unreachable!());

        let results = harness
            .searcher
            .search_all(SearchQuery {
                query: "pub".to_owned(),
                top_k: 10,
                language_filter: None,
                path_prefix: None,
                repos: Vec::new(),
            })
            .await;
        assert!(results.is_ok());
        let results = results.ok().unwrap_or_else(|| unreachable!()).results;

        let lib_matches = results
            .iter()
            .filter(|result| result.chunk.file_path.as_str() == "src/lib.rs")
            .count();
        assert!(lib_matches > 1);
    }
}
