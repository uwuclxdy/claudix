//! Near-duplicate detection over labeled chunks.
//!
//! [`find_duplicates`] does a pairwise upper-triangle scan and is intentionally
//! kept pure (no I/O, no async). The caller runs it inside
//! `tokio::task::spawn_blocking` because it is O(n²) in chunk count.
//! For an on-demand tool this is acceptable; the caller scopes the input via
//! the repo list.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use serde::Serialize;

use crate::store::StoredChunk;

use super::cosine_similarity;

/// Wrapper that gives `DuplicatePair` a min-heap ordering by similarity so a
/// `BinaryHeap<HeapEntry>` is a bounded max-heap (pop the smallest when over
/// the cap, keep the largest). Tie-breaking mirrors the original sort:
/// `partial_cmp` with `unwrap_or(Equal)` and no secondary key.
struct HeapEntry(DuplicatePair);

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: the *smallest* similarity is the highest priority to pop.
        // Reversed so BinaryHeap::peek() returns the weakest pair.
        other
            .0
            .similarity
            .partial_cmp(&self.0.similarity)
            .unwrap_or(Ordering::Equal)
    }
}

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

    // Min-heap capped at `limit`: we keep the `limit` highest-similarity pairs
    // without ever storing the full O(n²) set. The minimum-similarity entry is
    // at the top; when the heap exceeds the cap we pop it (discard lowest).
    // `DuplicateChunk` string clones are deferred until a pair actually enters
    // the heap, so rejected pairs allocate nothing beyond a similarity `f32`.
    let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::with_capacity(limit + 1);

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

            // Skip if this pair is weaker than the current heap minimum and
            // the heap is already full — defer the string clones entirely.
            if heap.len() == limit {
                if let Some(min_entry) = heap.peek() {
                    let min_sim = min_entry.0.similarity;
                    // partial_cmp on f32: treat NaN as equal (same as original).
                    if sim.partial_cmp(&min_sim).unwrap_or(Ordering::Equal) != Ordering::Greater {
                        continue;
                    }
                }
                heap.pop();
            }

            heap.push(HeapEntry(DuplicatePair {
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
            }));
        }
    }

    // Drain heap into a Vec sorted descending by similarity (mirrors original).
    let mut pairs: Vec<DuplicatePair> = heap.into_iter().map(|e| e.0).collect();
    pairs.sort_by(|x, y| {
        y.similarity
            .partial_cmp(&x.similarity)
            .unwrap_or(Ordering::Equal)
    });
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

    /// When `limit` is smaller than the total qualifying pairs, the heap must
    /// keep the *highest*-similarity ones and discard the lowest, regardless of
    /// scan order. This guards the heap cap logic against regressions.
    #[test]
    fn heap_cap_keeps_highest_similarity_pairs() {
        // Three files: a↔b = 1.0, a↔c = 0.95, b↔c = 0.90.
        // With limit=2 the output must be {a↔b=1.0, a↔c=0.95}.
        let a = stored("src/a.rs", "foo", vec![1.0, 0.0, 0.0, 0.0]);
        let b = stored("src/b.rs", "bar", vec![1.0, 0.0, 0.0, 0.0]);
        // c has a small orthogonal component so a↔c ≈ 0.95 and b↔c ≈ 0.95.
        let c = stored(
            "src/c.rs",
            "baz",
            vec![0.95_f32, 0.31225_f32, 0.0, 0.0], // |v| ≈ 1
        );
        let chunks = [
            labeled("/repo", &a),
            labeled("/repo", &b),
            labeled("/repo", &c),
        ];

        let pairs = find_duplicates(&chunks, 0.0, 2);
        assert_eq!(pairs.len(), 2, "limit=2 must yield exactly 2 pairs");
        // Highest pair first.
        assert!(
            pairs[0].similarity >= pairs[1].similarity,
            "output must be sorted descending"
        );
        // The top pair must be a↔b (similarity ≈ 1.0).
        assert!(
            pairs[0].similarity > 0.99,
            "a↔b (identical vectors) must be the top pair"
        );
        // The discarded pair (lowest of the three) must not appear.
        let min_sim = pairs[1].similarity;
        assert!(
            min_sim > 0.89,
            "pair with similarity ≈ 0.90 should be dropped, kept pair sim={min_sim}"
        );
    }

    /// Equal-similarity pairs (NaN treated as Equal) must not cause panics and
    /// the output must still be sorted non-strictly descending.
    #[test]
    fn tie_similarity_output_is_sorted() {
        let v = vec![1.0_f32, 0.0, 0.0, 0.0];
        let a = stored("src/a.rs", "foo", v.clone());
        let b = stored("src/b.rs", "bar", v.clone());
        let c = stored("src/c.rs", "baz", v.clone());
        let d = stored("src/d.rs", "qux", v.clone());
        let chunks = [
            labeled("/repo", &a),
            labeled("/repo", &b),
            labeled("/repo", &c),
            labeled("/repo", &d),
        ];

        // All pairs have similarity 1.0 — every ordering is valid as long as
        // it is non-strictly descending.
        let pairs = find_duplicates(&chunks, 0.0, 3);
        assert_eq!(pairs.len(), 3);
        for window in pairs.windows(2) {
            assert!(
                window[0].similarity >= window[1].similarity,
                "pairs not sorted descending on ties"
            );
        }
    }
}
