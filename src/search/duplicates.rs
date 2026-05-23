//! Near-duplicate detection over labeled chunks.
//!
//! [`find_duplicates`] does a pairwise upper-triangle scan and is intentionally
//! kept pure (no I/O, no async). The caller runs it inside
//! `tokio::task::spawn_blocking` because it is O(n²) in chunk count.
//! For an on-demand tool this is acceptable; the caller scopes the input via
//! the repo list.

use serde::Serialize;

use crate::store::StoredChunk;

use super::cosine_similarity;

/// A chunk annotated with the repo it came from.
///
/// `repo` is the canonical repo path string (matching `DuplicateChunk.repo`
/// in the output). For a single active-repo run it is the project root path.
#[derive(Debug, Clone)]
pub struct LabeledChunk<'a> {
    pub repo: &'a str,
    pub chunk: &'a StoredChunk,
}

/// A single member of a near-duplicate pair.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DuplicateChunk {
    /// Canonical repo path string. For a single active-repo run, this is the project root path.
    pub repo: String,
    pub file_path: String,
    pub line_start: u32,
    pub line_end: u32,
    pub name: Option<String>,
}

/// Two chunks whose cosine similarity meets or exceeds the threshold.
///
/// Only cross-location pairs are reported: `(repo_a, file_a) != (repo_b, file_b)`.
/// Intra-file repetition is excluded — cross-file and cross-repo pairs are the
/// actionable signal. `similarity: f32` precludes `Eq`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DuplicatePair {
    pub a: DuplicateChunk,
    pub b: DuplicateChunk,
    /// Cosine similarity in [0, 1]; higher is more similar.
    pub similarity: f32,
}

/// Scan `chunks` for near-duplicate pairs.
///
/// Uses an upper-triangle pairwise scan (i < j) with [`cosine_similarity`].
/// A pair qualifies when both:
/// - the two chunks are in DIFFERENT `(repo, file_path)` locations, and
/// - their cosine similarity is `>= min_similarity`.
///
/// Results are sorted by similarity descending and truncated to `limit` entries.
/// `limit` caps OUTPUT only; there is no input truncation.
///
/// # Complexity
///
/// O(n²) in `chunks.len()`. Run inside `tokio::task::spawn_blocking`; the
/// caller controls the input size via the repo list.
pub fn find_duplicates(
    chunks: &[LabeledChunk<'_>],
    min_similarity: f32,
    limit: usize,
) -> Vec<DuplicatePair> {
    if chunks.len() < 2 || limit == 0 {
        return Vec::new();
    }

    let mut pairs: Vec<DuplicatePair> = Vec::new();

    for i in 0..chunks.len() {
        for j in (i + 1)..chunks.len() {
            let a = &chunks[i];
            let b = &chunks[j];

            // Only cross-location pairs.
            if a.repo == b.repo && a.chunk.file_path == b.chunk.file_path {
                continue;
            }

            let sim = cosine_similarity(&a.chunk.vector, &b.chunk.vector);
            if sim < min_similarity {
                continue;
            }

            pairs.push(DuplicatePair {
                a: DuplicateChunk {
                    repo: a.repo.to_owned(),
                    file_path: a.chunk.file_path.clone(),
                    line_start: a.chunk.line_start,
                    line_end: a.chunk.line_end,
                    name: a.chunk.name.clone(),
                },
                b: DuplicateChunk {
                    repo: b.repo.to_owned(),
                    file_path: b.chunk.file_path.clone(),
                    line_start: b.chunk.line_start,
                    line_end: b.chunk.line_end,
                    name: b.chunk.name.clone(),
                },
                similarity: sim,
            });
        }
    }

    pairs.sort_by(|x, y| {
        y.similarity
            .partial_cmp(&x.similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    pairs.truncate(limit);
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::StoredChunk;

    fn stored(file_path: &str, name: &str, vector: Vec<f32>) -> StoredChunk {
        StoredChunk {
            chunk_id: 0,
            file_path: file_path.to_owned(),
            language: "rust".into(),
            kind: "function".into(),
            name: Some(name.to_owned()),
            line_start: 1,
            line_end: 10,
            byte_start: 0,
            byte_end: 100,
            file_hash: [0u8; 16],
            content: format!("pub fn {name}() {{}}"),
            vector,
        }
    }

    fn labeled<'a>(repo: &'a str, chunk: &'a StoredChunk) -> LabeledChunk<'a> {
        LabeledChunk { repo, chunk }
    }

    #[test]
    fn identical_vectors_produce_a_pair() {
        let v = vec![1.0_f32, 0.0, 0.0, 0.0];
        let a = stored("src/a.rs", "foo", v.clone());
        let b = stored("src/b.rs", "bar", v.clone());
        let chunks = [labeled("/repo", &a), labeled("/repo", &b)];

        let pairs = find_duplicates(&chunks, 0.9, 50);
        assert_eq!(pairs.len(), 1);
        assert!(pairs[0].similarity > 0.99);
        assert_eq!(pairs[0].a.file_path, "src/a.rs");
        assert_eq!(pairs[0].b.file_path, "src/b.rs");
    }

    #[test]
    fn no_pairs_when_below_threshold() {
        let a = stored("src/a.rs", "foo", vec![1.0, 0.0, 0.0, 0.0]);
        let b = stored("src/b.rs", "bar", vec![0.0, 1.0, 0.0, 0.0]);
        let chunks = [labeled("/repo", &a), labeled("/repo", &b)];

        let pairs = find_duplicates(&chunks, 0.85, 50);
        assert!(pairs.is_empty());
    }

    #[test]
    fn same_file_pairs_excluded() {
        let v = vec![1.0_f32, 0.0, 0.0, 0.0];
        let a = stored("src/a.rs", "foo", v.clone());
        let b = stored("src/a.rs", "bar", v.clone());
        let chunks = [labeled("/repo", &a), labeled("/repo", &b)];

        let pairs = find_duplicates(&chunks, 0.0, 50);
        assert!(pairs.is_empty(), "same-file pairs must not be reported");
    }

    #[test]
    fn limit_caps_output_not_input() {
        let v = vec![1.0_f32, 0.0, 0.0, 0.0];
        let a = stored("src/a.rs", "foo", v.clone());
        let b = stored("src/b.rs", "bar", v.clone());
        let c = stored("src/c.rs", "baz", v.clone());
        let chunks = [
            labeled("/repo", &a),
            labeled("/repo", &b),
            labeled("/repo", &c),
        ];

        let pairs = find_duplicates(&chunks, 0.0, 1);
        assert_eq!(pairs.len(), 1);
    }

    #[test]
    fn cross_repo_pair_when_repos_differ() {
        let v = vec![1.0_f32, 0.0, 0.0, 0.0];
        let a = stored("src/a.rs", "foo", v.clone());
        let b = stored("src/a.rs", "foo", v.clone());
        // Same file_path but different repos — should qualify.
        let chunks = [labeled("/repo-a", &a), labeled("/repo-b", &b)];

        let pairs = find_duplicates(&chunks, 0.9, 50);
        assert_eq!(pairs.len(), 1);
        assert_ne!(pairs[0].a.repo, pairs[0].b.repo);
    }

    #[test]
    fn sorted_by_similarity_descending() {
        let high = vec![1.0_f32, 0.0, 0.0, 0.0];
        let mid = vec![0.9_f32, 0.1_f32.sqrt(), 0.0, 0.0];
        let a = stored("src/a.rs", "foo", high.clone());
        let b = stored("src/b.rs", "bar", high.clone());
        let c = stored("src/c.rs", "baz", mid.clone());
        let d = stored("src/d.rs", "qux", high.clone());
        let chunks = [
            labeled("/repo", &a),
            labeled("/repo", &b),
            labeled("/repo", &c),
            labeled("/repo", &d),
        ];

        let pairs = find_duplicates(&chunks, 0.0, 50);
        for window in pairs.windows(2) {
            assert!(
                window[0].similarity >= window[1].similarity,
                "pairs not sorted descending"
            );
        }
    }

    #[test]
    fn empty_input_returns_empty() {
        let pairs = find_duplicates(&[], 0.0, 50);
        assert!(pairs.is_empty());
    }

    #[test]
    fn single_chunk_returns_empty() {
        let a = stored("src/a.rs", "foo", vec![1.0, 0.0]);
        let chunks = [labeled("/repo", &a)];
        let pairs = find_duplicates(&chunks, 0.0, 50);
        assert!(pairs.is_empty());
    }
}
