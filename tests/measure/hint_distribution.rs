//! Measurement harness for related-code hint noise. Numbers it produced are
//! recorded in `docs/subsystems/hooks.md`; open questions in `docs/todo.md`.
//!
//! Reads the repo's own live index and replays the hook's neighbor pipeline
//! against it, so the numbers come from the shipped code path rather than from
//! a whole-corpus duplicate scan (which can only model the pre-fix query shape).
//!
//! Linked into the lib test target from `src/lib.rs` via `#[path]` so it can
//! reach `NEIGHBOR_CANDIDATE_DEPTH` and the store internals. It is `#[ignore]`d
//! and mutates nothing: run it by hand after a change to the query set, the
//! ranking, or the dedup key.
//!
//! ```text
//! cargo test --lib hint_distribution -- --ignored --nocapture
//! CLAUDIX_MEASURE_ROOT=~/repos/py/cloudysec cargo test --lib hint_distribution -- --ignored --nocapture
//! ```
//!
//! `CLAUDIX_MEASURE_ROOT` points it at any other indexed repo, which is how the
//! pool ceiling gets checked against a corpus larger than this one. Reads only:
//! `Store::new` + `read_chunks` is the same pair the cross-repo loader uses.
//!
//! Needs a populated `.claudix/index` in the target repo; it prints a skip
//! notice and returns when there is none.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::{BTreeMap, BTreeSet, HashSet};

use crate::NEIGHBOR_CANDIDATE_DEPTH;
use crate::config;
use crate::search::neighbors::{Neighbor, neighbors};
use crate::store::{Store, StoredChunk};
use crate::types::{Language, RelativePath};

/// Edits per simulated session, `CLAUDIX_MEASURE_SESSION` to override. The
/// default matches the shape of the pre-fix baseline in `docs/todo.md` (10
/// edits to 10 different files) so the two are comparable. Longer sessions are
/// what put the seen-filter under real pressure: the ledger only grows, so the
/// pool cap can only start costing hints once a session has consumed enough of
/// one edit's candidates.
fn session_edits() -> usize {
    std::env::var("CLAUDIX_MEASURE_SESSION")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n| n > 0)
        .unwrap_or(10)
}

/// What seeds the neighbor query for one edit.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// Pre-fix: every chunk of the edited file.
    WholeFile,
    /// Shipped: only the chunks whose content changed. A single-chunk edit is
    /// the common case and the one the containment filter reduces to, so one
    /// chunk per edit is the faithful model.
    ChangedChunk,
}

impl Mode {
    fn label(self) -> &'static str {
        match self {
            Self::WholeFile => "whole-file query (pre-fix)",
            Self::ChangedChunk => "changed-chunk query (shipped)",
        }
    }
}

/// The uncapped, above-floor neighbor pool for one simulated edit.
struct EditPool {
    edited_path: String,
    pool: Vec<Neighbor>,
}

/// One neighbor's dedup identity. Mirrors the scope of
/// `change_neighbors::seen_key`: neighbor file plus symbol, never the edited
/// path. Modelled as a tuple rather than the tab-joined string because the
/// harness measures suppression, not the on-disk encoding.
fn dedup_id(n: &Neighbor) -> (String, Option<String>) {
    (n.file_path.clone(), n.name.clone())
}

/// Render a hit the way the hint line identifies it, for equal-list detection.
fn hit_id(n: &Neighbor) -> String {
    format!(
        "{}:{}-{}:{}",
        n.file_path,
        n.line_start,
        n.line_end,
        n.name.as_deref().unwrap_or("")
    )
}

/// Replay the ack-time consumer over an edit's pool.
///
/// Order matters and mirrors `take_change_neighbors_context`: the marker holds
/// at most `top_k * NEIGHBOR_CANDIDATE_DEPTH` candidates, the seen-filter
/// subtracts from those, and only then is the `top_k` hint budget applied. The
/// on-disk existence check is skipped: every neighbor here comes from the live
/// index of the repo being measured, so it exists by construction.
fn consume(
    pool: &[Neighbor],
    top_k: usize,
    seen: Option<&mut HashSet<(String, Option<String>)>>,
) -> Vec<Neighbor> {
    let marker_cap = top_k.saturating_mul(NEIGHBOR_CANDIDATE_DEPTH);
    let candidates = pool.iter().take(marker_cap);

    match seen {
        None => candidates.take(top_k).cloned().collect(),
        Some(seen) => {
            let mut out = Vec::new();
            for n in candidates {
                if out.len() == top_k {
                    break;
                }
                let id = dedup_id(n);
                if seen.contains(&id) {
                    continue;
                }
                seen.insert(id);
                out.push(n.clone());
            }
            out
        }
    }
}

#[derive(Default)]
struct RunStats {
    edits: usize,
    emitted: usize,
    silent_edits: usize,
    distinct_neighbor_files: BTreeSet<String>,
    appearances: BTreeMap<String, usize>,
    repeated_lists: usize,
    starved_by_cap: usize,
}

impl RunStats {
    fn hub(&self) -> Option<(&str, usize)> {
        self.appearances
            .iter()
            .max_by_key(|(_, count)| **count)
            .map(|(file, count)| (file.as_str(), *count))
    }

    fn report(&self, title: &str) {
        println!("\n  {title}");
        println!("    edits simulated          {}", self.edits);
        println!(
            "    hints emitted            {} ({:.2} per edit)",
            self.emitted,
            self.emitted as f64 / self.edits.max(1) as f64
        );
        println!(
            "    edits with zero hints    {} ({:.0}%)",
            self.silent_edits,
            100.0 * self.silent_edits as f64 / self.edits.max(1) as f64
        );
        println!(
            "    distinct neighbor files  {}",
            self.distinct_neighbor_files.len()
        );
        match self.hub() {
            Some((file, count)) => println!(
                "    busiest neighbor         {file} in {count}/{} edits ({:.0}%)",
                self.edits,
                100.0 * count as f64 / self.edits.max(1) as f64
            ),
            None => println!("    busiest neighbor         none"),
        }
        println!(
            "    repeated hint lists      {} of {} edits emitting a list already seen this session",
            self.repeated_lists, self.edits
        );
        println!(
            "    silent from the pool cap {} (unseen candidates existed past the marker cap)",
            self.starved_by_cap
        );
    }
}

/// Replay a run of edits, partitioned into sessions of [`session_edits`]. The
/// dedup ledger resets per session, matching SessionStart deleting it.
///
/// `order` decides which edits share a session. Index order groups directory
/// siblings, which is the seen-filter's best case and inflates every dedup
/// number; the shuffled order is the one to read.
fn replay(pools: &[EditPool], top_k: usize, dedup: bool, order: &[usize]) -> RunStats {
    let mut stats = RunStats::default();

    for session in order.chunks(session_edits()) {
        let mut seen: HashSet<(String, Option<String>)> = HashSet::new();
        let mut lists_this_session: HashSet<Vec<String>> = HashSet::new();

        for edit in session.iter().filter_map(|&i| pools.get(i)) {
            let hits = consume(
                &edit.pool,
                top_k,
                if dedup { Some(&mut seen) } else { None },
            );

            stats.edits += 1;
            stats.emitted += hits.len();
            if hits.is_empty() {
                stats.silent_edits += 1;
                // Starved only when the marker cap hid candidates that the
                // seen-filter had not already eaten. A pool that fits under the
                // cap and comes back empty is the filter working as designed;
                // this counts the edits where the over-fetch depth is what cost
                // the hint. Measured at zero everywhere; see `hooks.md`.
                let cap = top_k.saturating_mul(NEIGHBOR_CANDIDATE_DEPTH);
                if edit.pool.len() > cap
                    && edit.pool[cap..]
                        .iter()
                        .any(|n| !seen.contains(&dedup_id(n)))
                {
                    stats.starved_by_cap += 1;
                }
                continue;
            }

            for hit in &hits {
                stats.distinct_neighbor_files.insert(hit.file_path.clone());
                *stats.appearances.entry(hit.file_path.clone()).or_insert(0) += 1;
            }

            let list: Vec<String> = hits.iter().map(hit_id).collect();
            if !lists_this_session.insert(list) {
                stats.repeated_lists += 1;
            }
        }
    }

    stats
}

/// Distinct qualifying neighbor files per edit, uncapped — the pool ceiling
/// `NEIGHBOR_CANDIDATE_DEPTH` has to cover. Single-seed, so it is a floor on the
/// true ceiling rather than the ceiling (`docs/todo.md` item 2).
fn report_pool_ceiling(label: &str, pools: &[EditPool], top_k: usize) {
    let marker_cap = top_k.saturating_mul(NEIGHBOR_CANDIDATE_DEPTH);
    let mut sizes: Vec<usize> = pools.iter().map(|p| p.pool.len()).collect();
    sizes.sort_unstable();

    let max = sizes.last().copied().unwrap_or(0);
    // Nearest-rank p90: the smallest value at or above the 90th percentile, so
    // ceil(0.9n) - 1 rather than the one-rank-high 9n/10.
    let p90 = sizes
        .get((sizes.len() * 9).div_ceil(10).saturating_sub(1))
        .copied()
        .unwrap_or(0);
    let mean = sizes.iter().sum::<usize>() as f64 / sizes.len().max(1) as f64;
    let truncated = sizes.iter().filter(|&&s| s > marker_cap).count();

    println!("\n  candidate-pool ceiling, {label} (uncapped, above floor)");
    println!("    max {max}   p90 {p90}   mean {mean:.1}");
    println!(
        "    over the {marker_cap}-entry marker cap: {truncated} of {} edits",
        sizes.len()
    );
    if let Some(worst) = pools.iter().max_by_key(|p| p.pool.len()) {
        println!(
            "    widest pool: {} ({} neighbor files)",
            worst.edited_path,
            worst.pool.len()
        );
    }
}

/// Nearest-rank percentile over an ascending slice.
fn percentile(sorted: &[f32], p: usize) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = (sorted.len() * p).div_ceil(100).saturating_sub(1);
    sorted[rank.min(sorted.len() - 1)]
}

/// Where the configured floor lands on this provider's score scale. The floor
/// is an absolute cosine and each embedding model spreads its similarities
/// differently, so a floor tuned against one is worth re-checking against
/// another. Run it on the same repo under two providers and compare. The
/// bge-small-vs-qwen3 comparison it answered is in `docs/subsystems/hooks.md`.
///
/// It scans at floor `0.0` itself rather than taking a pool, because a floored
/// pool would silently mix a no-neighbor sentinel into the percentiles and drag
/// every one of them down.
///
/// Compare only runs whose corpus matches: the chunk count is printed above,
/// and a corpus that differs by even a few chunks moves the medians by enough
/// to swamp the difference being measured.
///
/// Pointing it at a second provider needs `CIRRUS_CONFIG` (a `[paths]
/// index_dir` plus that provider's `[embedding]` block) **and**
/// `--features test-stub`, since `cirrus_config_path` is gated on that feature
/// alone, not on `cfg(test)`. Without the feature the override is silently
/// ignored and the run reads the default index while still printing the other
/// provider's manifest identity, which looks exactly like a real result.
///
/// Give the second provider a whole sibling tree (`index_dir =
/// "<other>/index"`), never a subdirectory of `.claudix`: the state dir is
/// `index_dir.parent()`, so `.claudix/index-other` shares one `manifest.json`
/// with the live index and overwrites it.
fn report_score_distribution(
    rows: &[StoredChunk],
    by_file: &BTreeMap<String, Vec<&StoredChunk>>,
    top_k: usize,
    floor: f32,
) {
    let pools = scan(rows, build_edits(by_file, Mode::ChangedChunk), 0.0);

    let mut best: Vec<f32> = Vec::with_capacity(pools.len());
    let mut kth: Vec<f32> = Vec::with_capacity(pools.len());
    for edit in &pools {
        best.push(edit.pool.first().map_or(0.0, |n| n.score));
        kth.push(
            edit.pool
                .get(top_k.saturating_sub(1))
                .map_or(0.0, |n| n.score),
        );
    }
    best.sort_by(f32::total_cmp);
    kth.sort_by(f32::total_cmp);

    let edits = pools.len().max(1);
    let any = best.iter().filter(|&&s| s >= floor).count();
    let full = kth.iter().filter(|&&s| s >= floor).count();

    println!("\n  neighbor score distribution (floor removed)");
    println!(
        "    best neighbor per edit    p10 {:.3}  p50 {:.3}  p90 {:.3}  max {:.3}",
        percentile(&best, 10),
        percentile(&best, 50),
        percentile(&best, 90),
        best.last().copied().unwrap_or(0.0)
    );
    println!(
        "    rank-{top_k} neighbor per edit   p10 {:.3}  p50 {:.3}  p90 {:.3}  max {:.3}",
        percentile(&kth, 10),
        percentile(&kth, 50),
        percentile(&kth, 90),
        kth.last().copied().unwrap_or(0.0)
    );
    println!(
        "    at floor {floor}: {any} of {} edits ({:.0}%) emit at least one hint, {full} ({:.0}%) emit a full {top_k}",
        pools.len(),
        100.0 * any as f64 / edits as f64,
        100.0 * full as f64 / edits as f64
    );
    // The quartiles the p10/p50/p90 line above does not carry, so a candidate
    // floor can be read straight off the empirical spread rather than guessed.
    println!(
        "    best-neighbor quartiles   p25 {:.3}  p75 {:.3}",
        percentile(&best, 25),
        percentile(&best, 75)
    );
}

/// A file with no first-class chunker falls back to line-index windowing, so a
/// line insert or delete rewrites every window after it and the whole file
/// still reads as changed. Production keeps the file-wide query for those, and
/// modelling them as single-chunk edits is what would understate the shipped
/// hint volume.
fn is_fallback_chunked(chunks: &[&StoredChunk]) -> bool {
    chunks
        .first()
        .is_some_and(|c| c.language == Language::Unknown.as_str())
}

/// Build one edit per indexed file. `WholeFile` seeds with every chunk;
/// `ChangedChunk` seeds with the file's median chunk by position, a fixed pick
/// so the run is deterministic — except for fallback-chunked files, which
/// production still queries file-wide.
fn build_edits(
    by_file: &BTreeMap<String, Vec<&StoredChunk>>,
    mode: Mode,
) -> Vec<(String, Vec<Vec<f32>>)> {
    by_file
        .iter()
        .map(|(path, chunks)| {
            let whole_file = mode == Mode::WholeFile || is_fallback_chunked(chunks);
            let seeds = if whole_file {
                chunks.iter().map(|c| c.vector.clone()).collect()
            } else {
                vec![chunks[chunks.len() / 2].vector.clone()]
            };
            (path.clone(), seeds)
        })
        .collect()
}

/// Deterministic permutation of `0..len`, so sessions group files that are not
/// directory siblings. A real session's edits are not path-contiguous, and
/// path-contiguous ones hand the seen-filter its best case.
fn shuffled_order(len: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..len).collect();
    // xorshift64* from a fixed seed: reproducible across runs and machines
    // without pulling in a rand dependency for one shuffle.
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    for i in (1..len).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        order.swap(i, (state % (i as u64 + 1)) as usize);
    }
    order
}

fn scan(rows: &[StoredChunk], edits: Vec<(String, Vec<Vec<f32>>)>, floor: f32) -> Vec<EditPool> {
    edits
        .into_iter()
        .map(|(edited_path, seeds)| {
            let exclude = RelativePath::new(&edited_path);
            // `rows.len()` as top_k makes the heap unbounded in practice, so the
            // result is the full above-floor pool rather than a capped view.
            let pool = neighbors(rows, &seeds, &exclude, rows.len(), floor);
            EditPool { edited_path, pool }
        })
        .collect()
}

#[tokio::test]
#[ignore = "measures the working repo's live index; run by hand"]
async fn hint_distribution_over_the_live_index() {
    let project_root = std::env::var("CLAUDIX_MEASURE_ROOT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")));
    let config = config::load(&project_root).expect("config loads");
    let store = Store::new(&project_root, &config).expect("store opens");

    // Gate on the manifest the way the cross-repo loader does. Scores here are
    // raw cosine against an absolute floor, so vectors from a different model
    // would still produce numbers, just meaningless ones — and comparing two
    // repos is only valid when the printed identity matches.
    let Some(manifest) = store.read_manifest().expect("manifest reads") else {
        println!("no manifest in {}; skipping", project_root.display());
        return;
    };
    println!("\nrepo: {}", project_root.display());
    println!(
        "model: {} @ {} dims",
        manifest.embedding_model, manifest.dimensions
    );

    let rows = store.read_chunks().await.expect("chunks read");
    if rows.is_empty() {
        println!("no indexed chunks in {}; skipping", project_root.display());
        return;
    }

    let mut by_file: BTreeMap<String, Vec<&StoredChunk>> = BTreeMap::new();
    for row in &rows {
        by_file.entry(row.file_path.clone()).or_default().push(row);
    }
    for chunks in by_file.values_mut() {
        chunks.sort_by_key(|c| (c.line_start, c.byte_start));
    }

    let top_k = config.hooks.related_top_k;
    let floor = config.hooks.related_min_similarity;

    println!(
        "\nindex: {} files, {} chunks, floor {floor}, top_k {top_k}, pool depth {NEIGHBOR_CANDIDATE_DEPTH}",
        by_file.len(),
        rows.len()
    );
    let fallback_files = by_file.values().filter(|c| is_fallback_chunked(c)).count();
    println!(
        "sessions of {} edits, one edit per file, shuffled order; \
         {fallback_files} of {} files are fallback-chunked and stay file-wide",
        session_edits(),
        by_file.len()
    );

    println!("\n=== changed-chunk query, provider score scale ===");
    report_score_distribution(&rows, &by_file, top_k, floor);

    let order = shuffled_order(by_file.len());
    for mode in [Mode::WholeFile, Mode::ChangedChunk] {
        let pools = scan(&rows, build_edits(&by_file, mode), floor);

        println!("\n=== {} ===", mode.label());
        replay(&pools, top_k, false, &order).report("without cross-file dedup");
        replay(&pools, top_k, true, &order).report("with cross-file dedup (shipped)");
        report_pool_ceiling("one edit per file", &pools, top_k);
    }

    // Every chunk as its own edit: the per-file median pick above is one draw
    // from this, and the pool ceiling item 3 needs is the max over all of them.
    // Fallback-chunked files are excluded — production never seeds them from a
    // single chunk, so their entry here would be a shape that cannot occur.
    let all_chunk_edits: Vec<(String, Vec<Vec<f32>>)> = rows
        .iter()
        .filter(|c| c.language != Language::Unknown.as_str())
        .map(|c| (c.file_path.clone(), vec![c.vector.clone()]))
        .collect();
    let pools = scan(&rows, all_chunk_edits, floor);
    println!("\n=== changed-chunk query, every first-class chunk as an edit ===");
    report_pool_ceiling("every chunk", &pools, top_k);
}
