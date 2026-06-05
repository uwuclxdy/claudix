//! Semantic-neighbor scan over already-embedded chunks.
//!
//! `neighbors` is a pure cosine pass — no embedding call required when the
//! query vectors are already in hand (e.g. freshly embedded file chunks).
//! It is designed to be reused by duplicate detection, read-time surfacing,
//! and cross-repo features; keep it general.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::store::StoredChunk;
use crate::types::RelativePath;

use super::cosine_similarity;

/// Min-heap wrapper for [`Neighbor`] so a `BinaryHeap<NeighborEntry>` acts as
/// a bounded max-heap: the weakest neighbor sits at the top and is popped when
/// the heap exceeds `top_k`. Tie-breaking mirrors the original sort:
/// `partial_cmp` with `unwrap_or(Equal)`, no secondary key.
struct NeighborEntry(Neighbor);

impl PartialEq for NeighborEntry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for NeighborEntry {}

impl PartialOrd for NeighborEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for NeighborEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: smallest score has highest BinaryHeap priority (gets popped first).
        // Reversed so BinaryHeap::peek() returns the weakest neighbor.
        other
            .0
            .score
            .partial_cmp(&self.0.score)
            .unwrap_or(Ordering::Equal)
    }
}

/// A single neighbor hit returned by [`neighbors`].
#[derive(Debug, Clone, PartialEq)]
pub struct Neighbor {
    /// Repo-relative path of the neighbor.
    pub file_path: String,
    pub line_start: u32,
    pub line_end: u32,
    /// Symbol name, if the chunk has one.
    pub name: Option<String>,
    /// Highest cosine similarity across all query vectors for this file.
    pub score: f32,
}

/// Find the most semantically similar chunks elsewhere in the repo.
///
/// Scans `rows` with each vector in `query_vectors`, keeps the **best score
/// per file** (not per chunk), excludes `exclude_path`, drops results below
/// `min_similarity`, and truncates to `top_k` ordered by score descending.
///
/// # Parameters
/// - `rows` — all stored chunks (from `store.read_chunks()`).
/// - `query_vectors` — embedding vectors of the chunks in the edited file.
///   Multiple vectors are averaged per file via a max-over-vectors strategy:
///   each candidate file retains the highest similarity achieved by any single
///   query vector, which means a file that is similar to *any* edited chunk
///   surfaces rather than requiring it to be similar to *all* of them.
/// - `exclude_path` — the edited file; never returned as a neighbor.
/// - `top_k` — maximum result count.
/// - `min_similarity` — cosine floor; results below this are discarded.
///
/// Returns an empty `Vec` when nothing clears the floor (no marker written).
pub fn neighbors(
    rows: &[StoredChunk],
    query_vectors: &[Vec<f32>],
    exclude_path: &RelativePath,
    top_k: usize,
    min_similarity: f32,
) -> Vec<Neighbor> {
    if rows.is_empty() || query_vectors.is_empty() || top_k == 0 {
        return Vec::new();
    }

    // Per-file best score and representative chunk (highest-scoring chunk).
    let mut best: std::collections::HashMap<&str, (f32, &StoredChunk)> =
        std::collections::HashMap::new();

    for row in rows {
        if row.file_path == exclude_path.as_str() {
            continue;
        }

        let row_score = query_vectors
            .iter()
            .map(|qv| cosine_similarity(qv, &row.vector).max(0.0))
            .fold(0.0_f32, f32::max);

        let entry = best.entry(row.file_path.as_str()).or_insert((0.0, row));
        if row_score > entry.0 {
            *entry = (row_score, row);
        }
    }

    // Build a min-heap capped at `top_k` so only qualifying candidates
    // allocate their output `Neighbor` struct. The heap minimum (weakest score)
    // is popped when the cap is exceeded.
    let mut heap: BinaryHeap<NeighborEntry> = BinaryHeap::with_capacity(top_k + 1);

    for (score, row) in best.into_values() {
        if score < min_similarity {
            continue;
        }

        if heap.len() == top_k {
            if let Some(min_entry) = heap.peek()
                && score
                    .partial_cmp(&min_entry.0.score)
                    .unwrap_or(Ordering::Equal)
                    != Ordering::Greater
            {
                continue;
            }
            heap.pop();
        }

        heap.push(NeighborEntry(Neighbor {
            file_path: row.file_path.clone(),
            line_start: row.line_start,
            line_end: row.line_end,
            name: row.name.clone(),
            score,
        }));
    }

    let mut candidates: Vec<Neighbor> = heap.into_iter().map(|e| e.0).collect();
    candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
    candidates
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::StoredChunk;

    fn make_row(file_path: &str, name: &str, vector: Vec<f32>) -> StoredChunk {
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

    #[test]
    fn neighbors_excludes_edited_file() {
        let rows = vec![
            make_row("src/a.rs", "foo", vec![1.0, 0.0, 0.0, 0.0]),
            make_row("src/b.rs", "bar", vec![0.9, 0.1, 0.0, 0.0]),
        ];
        let query = vec![vec![1.0, 0.0, 0.0, 0.0]];
        let exclude = RelativePath::new("src/a.rs");

        let hits = neighbors(&rows, &query, &exclude, 5, 0.0);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].file_path, "src/b.rs");
    }

    #[test]
    fn neighbors_respects_min_similarity_floor() {
        let rows = vec![
            make_row("src/close.rs", "similar", vec![1.0, 0.0, 0.0, 0.0]),
            make_row("src/distant.rs", "unrelated", vec![0.0, 1.0, 0.0, 0.0]),
        ];
        let query = vec![vec![1.0, 0.0, 0.0, 0.0]];
        let exclude = RelativePath::new("src/edited.rs");

        let hits = neighbors(&rows, &query, &exclude, 5, 0.65);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].file_path, "src/close.rs");
    }

    #[test]
    fn neighbors_deduplicates_to_one_hit_per_file() {
        // Two chunks from the same file — only the best-scoring one surfaces.
        let mut chunk1 = make_row("src/multi.rs", "alpha", vec![0.7, 0.3, 0.0, 0.0]);
        let mut chunk2 = make_row("src/multi.rs", "beta", vec![0.9, 0.1, 0.0, 0.0]);
        chunk1.line_start = 1;
        chunk1.line_end = 10;
        chunk2.line_start = 20;
        chunk2.line_end = 30;
        let rows = vec![chunk1, chunk2];
        let query = vec![vec![1.0, 0.0, 0.0, 0.0]];
        let exclude = RelativePath::new("src/edited.rs");

        let hits = neighbors(&rows, &query, &exclude, 5, 0.0);
        assert_eq!(
            hits.len(),
            1,
            "two chunks from same file must collapse to one hit"
        );
        assert_eq!(hits[0].file_path, "src/multi.rs");
        // The best-scoring chunk (beta, 0.9) should win.
        assert_eq!(hits[0].name.as_deref(), Some("beta"));
    }

    #[test]
    fn neighbors_respects_top_k_limit() {
        let rows = vec![
            make_row("src/a.rs", "a", vec![0.9, 0.1, 0.0, 0.0]),
            make_row("src/b.rs", "b", vec![0.8, 0.2, 0.0, 0.0]),
            make_row("src/c.rs", "c", vec![0.7, 0.3, 0.0, 0.0]),
        ];
        let query = vec![vec![1.0, 0.0, 0.0, 0.0]];
        let exclude = RelativePath::new("src/edited.rs");

        let hits = neighbors(&rows, &query, &exclude, 2, 0.0);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].file_path, "src/a.rs");
        assert_eq!(hits[1].file_path, "src/b.rs");
    }

    #[test]
    fn neighbors_returns_empty_when_nothing_above_floor() {
        let rows = vec![make_row("src/b.rs", "bar", vec![0.0, 1.0, 0.0, 0.0])];
        let query = vec![vec![1.0, 0.0, 0.0, 0.0]];
        let exclude = RelativePath::new("src/edited.rs");

        let hits = neighbors(&rows, &query, &exclude, 5, 0.65);
        assert!(hits.is_empty());
    }

    #[test]
    fn neighbors_uses_max_over_multiple_query_vectors() {
        // file "src/b.rs" is similar to query_vectors[1] but not [0]
        let rows = vec![make_row("src/b.rs", "bar", vec![0.0, 1.0, 0.0, 0.0])];
        let query = vec![
            vec![1.0, 0.0, 0.0, 0.0], // not similar to b.rs
            vec![0.0, 1.0, 0.0, 0.0], // similar to b.rs (cosine = 1.0)
        ];
        let exclude = RelativePath::new("src/edited.rs");

        let hits = neighbors(&rows, &query, &exclude, 5, 0.65);
        assert_eq!(
            hits.len(),
            1,
            "file similar to any query vector must surface"
        );
        assert!((hits[0].score - 1.0).abs() < 1e-6);
    }
}
