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

use super::{cosine_with_norms, l2_norm};

/// Total order over neighbors, best-first: higher score wins, ties broken
/// deterministically by `file_path` then `line_start`. Without the tiebreak,
/// when more files tie on score than `top_k`, *which* survive the heap depends
/// on `HashMap` drain order — so identical input could yield different sets.
fn neighbor_rank(a: &Neighbor, b: &Neighbor) -> Ordering {
    b.score
        .partial_cmp(&a.score)
        .unwrap_or(Ordering::Equal)
        .then_with(|| a.file_path.cmp(&b.file_path))
        .then_with(|| a.line_start.cmp(&b.line_start))
}

/// Min-heap wrapper for [`Neighbor`] so a `BinaryHeap<NeighborEntry>` acts as
/// a bounded max-heap: the weakest neighbor (worst under [`neighbor_rank`]) sits
/// at the top and is popped when the heap exceeds `top_k`.
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
        // BinaryHeap is a max-heap; we want its top (peek/pop) to be the *worst*
        // neighbor. `neighbor_rank` is best-first (better compares `Less`), so a
        // worse neighbor already compares `Greater` — use it directly so the
        // worst sits at the top.
        neighbor_rank(&self.0, &other.0)
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

    // Norms are invariant across the row loop (query) and across the inner
    // query loop (row) — precompute both so the hot loop is dot-product only.
    let query_norms: Vec<f32> = query_vectors.iter().map(|qv| l2_norm(qv)).collect();

    for row in rows {
        if row.file_path == exclude_path.as_str() {
            continue;
        }

        let row_norm = l2_norm(&row.vector);
        let row_score = query_vectors
            .iter()
            .zip(&query_norms)
            .map(|(qv, &qv_norm)| cosine_with_norms(qv, qv_norm, &row.vector, row_norm).max(0.0))
            .fold(0.0_f32, f32::max);

        let entry = best.entry(row.file_path.as_str()).or_insert((0.0, row));
        if row_score > entry.0 {
            *entry = (row_score, row);
        }
    }

    // Build a min-heap capped at `top_k` so only qualifying candidates
    // allocate their output `Neighbor` struct. The weakest neighbor under
    // `neighbor_rank` (lowest score, then highest path/line) is popped when the
    // cap is exceeded.
    let mut heap: BinaryHeap<NeighborEntry> = BinaryHeap::with_capacity(top_k + 1);

    for (score, row) in best.into_values() {
        if score < min_similarity {
            continue;
        }

        let candidate = Neighbor {
            file_path: row.file_path.clone(),
            line_start: row.line_start,
            line_end: row.line_end,
            name: row.name.clone(),
            score,
        };

        if heap.len() == top_k {
            // Drop the candidate when it does not rank strictly better than the
            // current weakest (heap min). `neighbor_rank` is best-first, so a
            // better candidate compares `Less`.
            if let Some(min_entry) = heap.peek()
                && neighbor_rank(&candidate, &min_entry.0) != Ordering::Less
            {
                continue;
            }
            heap.pop();
        }

        heap.push(NeighborEntry(candidate));
    }

    let mut candidates: Vec<Neighbor> = heap.into_iter().map(|e| e.0).collect();
    candidates.sort_by(neighbor_rank);
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
    fn neighbors_tied_scores_yield_a_stable_set() {
        // Five files all identical to the query (score 1.0); top_k=3. The cap
        // forces a choice among tied scores — the deterministic tiebreak
        // (file_path) must pick the same three on every run, in path order.
        let v = vec![1.0_f32, 0.0, 0.0, 0.0];
        let rows = vec![
            make_row("src/e.rs", "e", v.clone()),
            make_row("src/b.rs", "b", v.clone()),
            make_row("src/d.rs", "d", v.clone()),
            make_row("src/a.rs", "a", v.clone()),
            make_row("src/c.rs", "c", v.clone()),
        ];
        let query = vec![v];
        let exclude = RelativePath::new("src/edited.rs");

        let expected = ["src/a.rs", "src/b.rs", "src/c.rs"];
        for _ in 0..16 {
            let hits = neighbors(&rows, &query, &exclude, 3, 0.0);
            let paths: Vec<&str> = hits.iter().map(|h| h.file_path.as_str()).collect();
            assert_eq!(paths, expected, "tied-score survivors must be stable");
        }
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

    #[test]
    fn neighbors_output_is_invariant_to_chunk_content() {
        // The change-neighbor and read-surfacing hooks feed `neighbors` rows read
        // without chunk text (`read_chunks_without_content`). This pins the
        // contract that makes that safe: the scan scores vectors and reads only
        // file_path / line / name, never `content`. If a future change makes it
        // depend on content, this test reds instead of those consumers silently
        // degrading against content-blanked rows.
        let with_content = vec![
            make_row("src/a.rs", "foo", vec![0.9, 0.1, 0.0, 0.0]),
            make_row("src/b.rs", "bar", vec![0.8, 0.2, 0.0, 0.0]),
        ];
        let mut blanked = with_content.clone();
        for row in &mut blanked {
            row.content = String::new();
        }
        let query = vec![vec![1.0, 0.0, 0.0, 0.0]];
        let exclude = RelativePath::new("src/edited.rs");

        assert_eq!(
            neighbors(&with_content, &query, &exclude, 5, 0.0),
            neighbors(&blanked, &query, &exclude, 5, 0.0),
            "neighbor hits must not depend on chunk content"
        );
    }

    #[test]
    fn neighbors_pool_grows_with_added_seeds() {
        // The uncapped pool grows monotonically as seeds are added: `neighbors`
        // keeps the best score per file across query vectors, so an extra seed
        // can only raise a file's score, never drop it below the floor. That is
        // why the multi-seed harness pool supersets the single-seed pool — the
        // harness seeds a superset of the single-seed's chunks (`spread_indices`
        // always includes the median) and measures at an uncapped `top_k`. It
        // says nothing about production's *capped* hint set, where a rising
        // competitor can still evict a file whose own score is unchanged.
        let rows = vec![
            make_row("src/a.rs", "a", vec![1.0, 0.0, 0.0, 0.0]),
            make_row("src/b.rs", "b", vec![0.0, 1.0, 0.0, 0.0]),
            make_row("src/c.rs", "c", vec![0.0, 0.0, 1.0, 0.0]),
        ];
        let exclude = RelativePath::new("src/edited.rs");
        // top_k = rows.len() leaves the heap effectively unbounded, so each call
        // returns the full above-floor pool rather than a capped view.
        let single = vec![vec![1.0, 0.0, 0.0, 0.0]];
        let multi = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];

        let files = |hits: Vec<Neighbor>| -> std::collections::HashSet<String> {
            hits.into_iter().map(|n| n.file_path).collect()
        };
        let single_files = files(neighbors(&rows, &single, &exclude, rows.len(), 0.65));
        let multi_files = files(neighbors(&rows, &multi, &exclude, rows.len(), 0.65));

        assert!(
            single_files.is_subset(&multi_files),
            "multi-seed pool must contain the single-seed pool: {single_files:?} vs {multi_files:?}"
        );
        assert!(
            multi_files.len() > single_files.len(),
            "the added seed must admit at least one more file"
        );
    }
}
