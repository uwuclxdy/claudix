//! Corpus-relative similarity floor for related-code hints.
//!
//! A fixed absolute cosine floor describes a whole embedding pipeline, not a
//! model: correcting the bundled provider's pooling head moved the best-neighbor
//! median 0.835 -> 0.873 on an identical corpus and doubled hint volume at a
//! fixed cutoff, while retrieval quality stayed flat. So the cutoff is derived
//! here from the index's own best-neighbor distribution and stored per index,
//! keeping selectivity stable as the model, pooling, or provider changes.

use std::collections::BTreeMap;

use crate::store::StoredChunk;
use crate::types::{Language, RelativePath};

use super::neighbors::neighbors;

/// Percentile of the best-neighbor-per-file distribution that becomes the floor.
///
/// Measured on the live index across two models (qwen3-8b @ 4096, gte-modernbert
/// @ 768): p30 holds ~70% of edits surfacing at least one hint on either model
/// (the clear rate is model-independent by construction), restoring the hint
/// volume the pooling fix had doubled, while lifting the precision of the hints
/// admitted from ~0.43 floor-free to ~0.63-0.67. It sits at qwen3's precision
/// knee and captures the bulk of gte's gain. Re-run `tests/measure/
/// hint_distribution.rs`'s floor sweep before changing it.
pub const FLOOR_PERCENTILE: u8 = 30;

/// Below this file count the percentile is too noisy to be worth storing; the
/// consumer falls open to the configured floor instead.
const MIN_FILES_FOR_FLOOR: usize = 8;

/// Above this chunk count the O(seeds x rows) scan is too costly to run at index
/// time; skip it and fall open. Read-time surfacing caps its own scan at the
/// same order of magnitude.
const MAX_CHUNKS_FOR_FLOOR: usize = 100_000;

/// Cap on how many files seed the scan. Beyond this the seed set is strided so
/// the percentile stays a cheap estimate on a huge repo rather than an
/// O(files x rows) blow-up; the inner scan still sees every row.
const MAX_SEED_FILES: usize = 400;

/// The `percentile`-th percentile of the best-neighbor-per-file cosine
/// distribution, or `None` when the corpus is too small to estimate a stable
/// one or too large to scan at index time. Both cases fall open to the
/// configured floor at the consumer.
///
/// The distribution mirrors the shipped edit-path query and the measurement
/// harness: one representative seed per file (its median chunk, or every chunk
/// for a fallback-chunked file that the edit path also queries whole), scored
/// against the best matching chunk in any other file.
pub fn corpus_similarity_floor(rows: &[StoredChunk], percentile: u8) -> Option<f32> {
    if rows.len() > MAX_CHUNKS_FOR_FLOOR {
        return None;
    }

    let mut by_file: BTreeMap<&str, Vec<&StoredChunk>> = BTreeMap::new();
    for row in rows {
        by_file.entry(row.file_path.as_str()).or_default().push(row);
    }
    if by_file.len() < MIN_FILES_FOR_FLOOR {
        return None;
    }
    // Deterministic median needs a stable within-file order.
    for chunks in by_file.values_mut() {
        chunks.sort_by_key(|a| (a.line_start, a.byte_start));
    }

    // Stride the seed files, never the rows: the best-neighbor score depends on
    // scanning every candidate, but the percentile only needs enough sampled
    // seeds to be stable.
    let stride = by_file.len().div_ceil(MAX_SEED_FILES).max(1);

    let mut best_scores: Vec<f32> = Vec::new();
    for (path, chunks) in by_file.iter().step_by(stride) {
        let exclude = RelativePath::new(*path);
        let seeds = seed_vectors(chunks);
        if seeds.is_empty() {
            continue;
        }
        let score = neighbors(rows, &seeds, &exclude, 1, 0.0)
            .first()
            .map_or(0.0, |neighbor| neighbor.score);
        best_scores.push(score);
    }
    if best_scores.is_empty() {
        return None;
    }

    best_scores.sort_by(f32::total_cmp);
    Some(percentile_value(&best_scores, percentile))
}

/// The query seeds for one file's best-neighbor score. A file with a first-class
/// grammar seeds from its median chunk, the deterministic stand-in for a single
/// edited chunk. A fallback-chunked file (no grammar) has line-windowed chunks
/// that shift wholesale on any edit, so the edit path queries it whole; match
/// that here by seeding from every chunk.
fn seed_vectors(chunks: &[&StoredChunk]) -> Vec<Vec<f32>> {
    let fallback = chunks
        .first()
        .is_some_and(|chunk| chunk.language == Language::Unknown.as_str());
    if fallback {
        chunks.iter().map(|chunk| chunk.vector.clone()).collect()
    } else {
        chunks
            .get(chunks.len() / 2)
            .map(|chunk| vec![chunk.vector.clone()])
            .unwrap_or_default()
    }
}

/// Nearest-rank percentile over an ascending slice, matching the measurement
/// harness so a stored floor equals the value that sweep printed.
fn percentile_value(sorted: &[f32], percentile: u8) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = (sorted.len() * percentile as usize)
        .div_ceil(100)
        .saturating_sub(1);
    sorted[rank.min(sorted.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(file_path: &str, name: &str, vector: Vec<f32>) -> StoredChunk {
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
    fn percentile_value_is_nearest_rank() {
        let sorted = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        // rank = ceil(10 * 30 / 100) - 1 = 2 -> the 3rd element (0-indexed 2).
        assert_eq!(percentile_value(&sorted, 30), 0.3);
        assert_eq!(percentile_value(&sorted, 100), 1.0);
        assert_eq!(percentile_value(&[], 30), 0.0);
    }

    #[test]
    fn too_few_files_yields_no_floor() {
        // Seven distinct files sit under MIN_FILES_FOR_FLOOR.
        let rows: Vec<StoredChunk> = (0..7)
            .map(|i| row(&format!("f{i}.rs"), "sym", vec![1.0, 0.0]))
            .collect();
        assert_eq!(corpus_similarity_floor(&rows, 30), None);
    }

    /// Spread across 8 distinct files with identical vectors, so without the
    /// chunk ceiling the scan would return `Some(1.0)` (every file a twin). The
    /// ceiling must be what returns `None` here, not the min-files guard — which
    /// the single-file spelling would have satisfied on its own, letting a
    /// deleted ceiling guard survive.
    #[test]
    fn a_corpus_over_the_chunk_ceiling_yields_no_floor() {
        let rows: Vec<StoredChunk> = (0..=MAX_CHUNKS_FOR_FLOOR)
            .map(|i| row(&format!("f{}.rs", i % 8), "s", vec![1.0, 0.0]))
            .collect();
        assert_eq!(corpus_similarity_floor(&rows, 30), None);
    }

    /// Eight files: seven identical to each other (best neighbor 1.0) and one
    /// orthogonal to all of them (best neighbor 0.0). The best-neighbor
    /// distribution is `[0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]` sorted, so the
    /// p30 (rank ceil(8*30/100)-1 = 2) is 1.0 and the p10 (rank 0) is 0.0.
    #[test]
    fn floor_is_the_requested_percentile_of_best_neighbors() {
        let mut rows: Vec<StoredChunk> = (0..7)
            .map(|i| row(&format!("twin{i}.rs"), "sym", vec![1.0, 0.0]))
            .collect();
        rows.push(row("lonely.rs", "sym", vec![0.0, 1.0]));

        assert_eq!(corpus_similarity_floor(&rows, 30), Some(1.0));
        assert_eq!(corpus_similarity_floor(&rows, 10), Some(0.0));
    }
}
