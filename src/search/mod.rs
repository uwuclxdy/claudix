use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use tokio::task;

use crate::config::SearchConfig;
use crate::embedding::Provider;
use crate::error::{ClaudixError, Result};
use crate::store::{Store, StoredChunk};
use crate::types::{
    ByteRange, Chunk, ChunkId, ChunkKind, FileHash, Language, LineRange, RelativePath,
};

#[derive(Debug, Clone)]
pub struct SearchQuery {
    pub query: String,
    pub top_k: usize,
    pub language_filter: Option<Vec<Language>>,
    pub path_prefix: Option<RelativePath>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    pub chunk: Chunk,
    pub score: f32,
}

#[derive(Clone)]
pub struct Searcher {
    store: Store,
    embedder: Arc<dyn Provider>,
    config: SearchConfig,
}

impl Searcher {
    pub fn new(store: Store, embedder: Arc<dyn Provider>, config: SearchConfig) -> Self {
        Self {
            store,
            embedder,
            config,
        }
    }

    pub async fn search(&self, query: SearchQuery) -> Result<Vec<SearchResult>> {
        let limit = effective_top_k(query.top_k, self.config.top_k);
        if limit == 0 || query.query.trim().is_empty() {
            return Ok(Vec::new());
        }

        let mut results = self.search_all(query).await?;
        results = deduplicate_by_file_path(results);
        results.truncate(limit);
        Ok(results)
    }

    pub async fn search_all(&self, query: SearchQuery) -> Result<Vec<SearchResult>> {
        if query.query.trim().is_empty() {
            return Ok(Vec::new());
        }

        let rows = self.store.read_chunks().await?;
        if rows.is_empty() {
            return Ok(Vec::new());
        }

        let vectors = self.embedder.embed(&[query.query.as_str()]).await?;
        if vectors.len() != 1 {
            return Err(ClaudixError::Embedding(format!(
                "provider returned {} vectors for 1 query",
                vectors.len()
            )));
        }

        let query_vector = vectors.into_iter().next().unwrap_or_default();
        let config = self.config.clone();

        task::spawn_blocking(move || rank_rows(query, rows, query_vector, config))
            .await
            .map_err(|error| ClaudixError::Store(format!("search task failed: {error}")))?
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

fn rank_rows(
    query: SearchQuery,
    rows: Vec<StoredChunk>,
    query_vector: Vec<f32>,
    config: SearchConfig,
) -> Result<Vec<SearchResult>> {
    let query_tokens = tokenize(&query.query);
    let filtered_rows = apply_filters(rows, &query);
    if filtered_rows.is_empty() {
        return Ok(Vec::new());
    }

    let documents = filtered_rows
        .iter()
        .map(|row| DocumentStats::from_content(&row.content))
        .collect::<Vec<_>>();
    let dense_scores = filtered_rows
        .iter()
        .map(|row| cosine_similarity(&query_vector, &row.vector).max(0.0))
        .collect::<Vec<_>>();
    let bm25_scores = bm25_scores(&documents, &query_tokens);
    let dense_ranks = rank_positions(&dense_scores);
    let bm25_ranks = rank_positions(&bm25_scores);
    let rrf_scores = reciprocal_rank_fusion(&dense_ranks, &bm25_ranks);

    let dense_normalized = normalize_scores(&dense_scores);
    let bm25_normalized = normalize_scores(&bm25_scores);
    let rrf_normalized = normalize_scores(&rrf_scores);

    let mut results = filtered_rows
        .into_iter()
        .enumerate()
        .filter_map(|(index, row)| {
            let identifier_hit = row
                .name
                .as_deref()
                .is_some_and(|name| name_contains_query_token(name, &query_tokens));
            let lexical_hit = bm25_scores[index] > 0.0 || identifier_hit;
            let dense_hit = dense_scores[index] >= config.similarity_threshold;
            let combined_score = config.hybrid_weights.dense * dense_normalized[index]
                + config.hybrid_weights.bm25 * bm25_normalized[index]
                + config.hybrid_weights.rrf * rrf_normalized[index];
            let boosted_score = if identifier_hit {
                combined_score * config.identifier_boost
            } else {
                combined_score
            };

            if boosted_score <= 0.0 {
                return None;
            }

            if !lexical_hit && !dense_hit {
                return None;
            }

            Some(SearchResult {
                chunk: stored_chunk_to_chunk(row),
                score: boosted_score,
            })
        })
        .collect::<Vec<_>>();

    sort_results(&mut results);
    Ok(results)
}

fn effective_top_k(requested: usize, default_top_k: usize) -> usize {
    if requested == 0 {
        default_top_k
    } else {
        requested
    }
}

fn apply_filters(rows: Vec<StoredChunk>, query: &SearchQuery) -> Vec<StoredChunk> {
    let language_filter = query.language_filter.as_ref().map(|languages| {
        languages
            .iter()
            .map(|language| language.as_str())
            .collect::<HashSet<_>>()
    });
    let path_prefix = query.path_prefix.as_ref().map(RelativePath::as_str);

    rows.into_iter()
        .filter(|row| {
            if let Some(language_filter) = &language_filter
                && !language_filter.contains(row.language.as_str())
            {
                return false;
            }

            if let Some(path_prefix) = path_prefix
                && !row.file_path.starts_with(path_prefix)
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

fn rank_positions(scores: &[f32]) -> Vec<Option<usize>> {
    let mut indexed_scores = scores
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, score)| *score > 0.0)
        .collect::<Vec<_>>();
    indexed_scores.sort_by(|left, right| compare_scores_desc(left.1, right.1, left.0, right.0));

    let mut positions = vec![None; scores.len()];
    for (rank, (index, _)) in indexed_scores.into_iter().enumerate() {
        positions[index] = Some(rank + 1);
    }

    positions
}

fn reciprocal_rank_fusion(dense_ranks: &[Option<usize>], bm25_ranks: &[Option<usize>]) -> Vec<f32> {
    const RRF_K: f32 = 60.0;

    dense_ranks
        .iter()
        .zip(bm25_ranks)
        .map(|(dense_rank, bm25_rank)| {
            dense_rank
                .map(|rank| 1.0 / (RRF_K + rank as f32))
                .unwrap_or(0.0)
                + bm25_rank
                    .map(|rank| 1.0 / (RRF_K + rank as f32))
                    .unwrap_or(0.0)
        })
        .collect()
}

fn normalize_scores(scores: &[f32]) -> Vec<f32> {
    let max_score = scores.iter().copied().fold(0.0, f32::max);
    if max_score <= 0.0 {
        return vec![0.0; scores.len()];
    }

    scores
        .iter()
        .map(|score| (score / max_score).clamp(0.0, 1.0))
        .collect()
}

fn deduplicate_by_file_path(results: Vec<SearchResult>) -> Vec<SearchResult> {
    let mut seen_paths = HashSet::new();
    let mut deduplicated = Vec::new();

    for result in results {
        if seen_paths.insert(result.chunk.file_path.clone()) {
            deduplicated.push(result);
        }
    }

    deduplicated
}

fn sort_results(results: &mut [SearchResult]) {
    results.sort_by(|left, right| {
        compare_scores_desc(
            left.score,
            right.score,
            left.chunk.line_range.start as usize,
            right.chunk.line_range.start as usize,
        )
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

fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    let length = left.len().min(right.len());
    if length == 0 {
        return 0.0;
    }

    let mut dot = 0.0;
    let mut left_norm = 0.0;
    let mut right_norm = 0.0;

    for index in 0..length {
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
        if ch.is_ascii_alphanumeric() || ch == '_' {
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
        language: stored_language(&language),
        kind: stored_chunk_kind(&kind),
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

fn stored_language(language: &str) -> Language {
    match language {
        "rust" => Language::Rust,
        "python" => Language::Python,
        "javascript" => Language::JavaScript,
        "typescript" => Language::TypeScript,
        "go" => Language::Go,
        "java" => Language::Java,
        "c" => Language::C,
        "cpp" => Language::Cpp,
        _ => Language::Unknown,
    }
}

fn stored_chunk_kind(kind: &str) -> ChunkKind {
    match kind {
        "function" => ChunkKind::Function,
        "method" => ChunkKind::Method,
        "struct" => ChunkKind::Struct,
        "class" => ChunkKind::Class,
        "enum" => ChunkKind::Enum,
        "trait" => ChunkKind::Trait,
        "interface" => ChunkKind::Interface,
        "module" => ChunkKind::Module,
        "impl" => ChunkKind::Impl,
        "macro" => ChunkKind::Macro,
        _ => ChunkKind::Other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
            _fixture: fixture,
            searcher: Searcher::new(store, embedder, config.search.clone()),
        })
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
            })
            .await;
        assert!(results.is_ok());
        let results = results.ok().unwrap_or_else(|| unreachable!());

        assert!(!results.is_empty());
        assert_eq!(results[0].chunk.name.as_deref(), Some("add"));
        assert_eq!(results[0].chunk.file_path.as_str(), "src/math.rs");
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
            })
            .await;
        assert!(results.is_ok());
        let results = results.ok().unwrap_or_else(|| unreachable!());

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
            })
            .await;
        assert!(results.is_ok());
        let results = results.ok().unwrap_or_else(|| unreachable!());

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
            })
            .await;
        assert!(results.is_ok());
        let results = results.ok().unwrap_or_else(|| unreachable!());

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
            })
            .await;
        assert!(results.is_ok());
        let results = results.ok().unwrap_or_else(|| unreachable!());

        let lib_matches = results
            .iter()
            .filter(|result| result.chunk.file_path.as_str() == "src/lib.rs")
            .count();
        assert!(lib_matches > 1);
    }
}
