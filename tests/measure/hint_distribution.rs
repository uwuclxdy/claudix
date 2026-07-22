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

/// Candidate percentiles of the best-neighbor distribution to trial as the
/// corpus-relative floor. The shipped floor stores one such percentile in the
/// manifest, so this sweep is how that percentile gets picked: read the row
/// where volume drops to the target while gated precision holds.
const FLOOR_SWEEP_PERCENTILES: &[usize] = &[10, 15, 20, 25, 30, 35, 40];

/// Trial each candidate floor = P-th percentile of the best-neighbor-per-file
/// distribution, exactly what the shipped `similarity_floor` will store, and
/// report what that floor does to hint volume and to the precision of the hints
/// it admits.
///
/// A single model, choosing its own cutoff — so unlike [`report_symbol_precision`]
/// this gates precision at the floor. That is the point here: a good floor is the
/// highest one whose admitted hints stay as correct as the floor-free set while
/// volume falls toward the target. Raising a floor only removes weak neighbors,
/// so admitted precision should hold or rise; the cost of raising it too far is
/// coverage (symbols left with no hit at all), which the clear-rate column shows.
fn report_floor_sweep(
    rows: &[StoredChunk],
    by_file: &BTreeMap<String, Vec<&StoredChunk>>,
    references: &BTreeMap<u64, BTreeSet<String>>,
    top_k: usize,
) {
    // The distribution the floor is a percentile of: best neighbor per edit,
    // changed-chunk query, floor removed. Identical to the shipped computation.
    let pools = scan(rows, build_edits(by_file, Mode::ChangedChunk), 0.0);
    let mut best: Vec<f32> = pools
        .iter()
        .map(|edit| edit.pool.first().map_or(0.0, |n| n.score))
        .collect();
    best.sort_by(f32::total_cmp);
    let total_edits = pools.len().max(1);

    // Precompute each evaluable symbol's floor-free ranked hits plus its answer
    // set once. A higher floor only trims the tail of a floor-free top-k, so
    // filtering these by score is exact and avoids re-ranking per floor.
    struct Evaluable<'a> {
        hits: Vec<Neighbor>,
        mentioning: BTreeSet<&'a str>,
    }
    let mut evaluables: Vec<Evaluable> = Vec::new();
    for row in rows {
        let Some(symbol) = row.name.as_deref() else {
            continue;
        };
        if symbol.len() < 4 || GENERIC_SYMBOLS.contains(&symbol) {
            continue;
        }
        let mut mentioning: BTreeSet<&str> = BTreeSet::new();
        let mut other_files = 0usize;
        for (path, chunks) in by_file {
            if path == &row.file_path {
                continue;
            }
            other_files += 1;
            if chunks
                .iter()
                .any(|c| references_symbol(references, c, symbol))
            {
                mentioning.insert(path.as_str());
            }
        }
        if mentioning.is_empty() || other_files == 0 {
            continue;
        }
        let exclude = RelativePath::new(&row.file_path);
        let hits = neighbors(
            rows,
            std::slice::from_ref(&row.vector),
            &exclude,
            top_k,
            0.0,
        );
        if hits.is_empty() {
            continue;
        }
        evaluables.push(Evaluable { hits, mentioning });
    }
    let total_symbols = evaluables.len().max(1);

    println!("\n  corpus-relative floor sweep (single model, choosing a cutoff)");
    println!(
        "    {total_edits} edits, {} evaluable symbols, top_k {top_k}",
        evaluables.len()
    );
    println!("    pctl  floor  clear%  hints/edit  cover%  prec@{top_k}  prec@1");
    for &p in FLOOR_SWEEP_PERCENTILES {
        let floor = percentile(&best, p);

        // Volume: hits above the floor per edit, capped at top_k.
        let mut clearing = 0usize;
        let mut emitted = 0usize;
        for edit in &pools {
            let admitted = edit
                .pool
                .iter()
                .filter(|n| n.score >= floor)
                .take(top_k)
                .count();
            if admitted > 0 {
                clearing += 1;
            }
            emitted += admitted;
        }

        // Gated precision: of the hits still admitted at this floor, the share
        // that reference the edited symbol, averaged over symbols keeping a hit.
        let mut precision_sum = 0.0f64;
        let mut top1_hits = 0usize;
        let mut evaluated = 0usize;
        for symbol in &evaluables {
            let admitted: Vec<&Neighbor> =
                symbol.hits.iter().filter(|n| n.score >= floor).collect();
            let Some(first) = admitted.first() else {
                continue;
            };
            let relevant = admitted
                .iter()
                .filter(|hit| symbol.mentioning.contains(hit.file_path.as_str()))
                .count();
            precision_sum += relevant as f64 / admitted.len() as f64;
            if symbol.mentioning.contains(first.file_path.as_str()) {
                top1_hits += 1;
            }
            evaluated += 1;
        }
        let prec = if evaluated > 0 {
            precision_sum / evaluated as f64
        } else {
            0.0
        };
        let prec1 = if evaluated > 0 {
            top1_hits as f64 / evaluated as f64
        } else {
            0.0
        };

        println!(
            "    p{p:<3} {floor:.3}  {:>5.0}  {:>9.2}  {:>5.0}  {:.3}  {:.3}",
            100.0 * clearing as f64 / total_edits as f64,
            emitted as f64 / total_edits as f64,
            100.0 * evaluated as f64 / total_symbols as f64,
            prec,
            prec1,
        );
    }
}

fn is_identifier_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

/// Identifiers a chunk actually references in code, with comment and string
/// content excluded.
///
/// Plain text matching counts a symbol named in a doc comment, a log message,
/// or an unrelated prose mention as a reference, and those are exactly the
/// matches a related-code hint should not be rewarded for. Re-parsing with the
/// language's own grammar and keeping only identifier tokens removes them.
///
/// Returns `None` when the chunk cannot be parsed as code at all, which is the
/// signal to fall back rather than to treat the chunk as referencing nothing.
/// Chunk text is a fragment of a file, so the parse is expected to contain
/// error nodes; tree-sitter still tokenizes around them, which is all this
/// needs.
fn code_identifiers(content: &str, language: Language) -> Option<BTreeSet<String>> {
    let grammar = crate::chunking::grammar_for(language)?;
    let mut parser = tree_sitter::Parser::new();
    parser.set_language(&grammar).ok()?;
    let tree = parser.parse(content, None)?;

    let mut found = BTreeSet::new();
    let mut cursor = tree.walk();
    let mut pending = vec![tree.root_node()];

    while let Some(node) = pending.pop() {
        let kind = node.kind();
        // Whole subtree is prose or literal text, so nothing inside it is a
        // reference to anything.
        if kind.contains("comment") || kind.contains("string") || kind.contains("char_literal") {
            continue;
        }

        if kind.contains("identifier")
            && node.child_count() == 0
            && let Ok(text) = node.utf8_text(content.as_bytes())
        {
            found.insert(text.to_owned());
        }

        pending.extend(node.children(&mut cursor));
    }

    Some(found)
}

/// Every chunk's referenced identifiers, parsed once. Doing this inside the
/// query loop would re-parse the whole corpus for every symbol evaluated.
/// Chunks whose language has no grammar are absent and fall back to text.
fn index_references(rows: &[StoredChunk]) -> BTreeMap<u64, BTreeSet<String>> {
    rows.iter()
        .filter_map(|row| {
            let language = Language::from_storage(&row.language);
            code_identifiers(&row.content, language).map(|ids| (row.chunk_id, ids))
        })
        .collect()
}

/// Whether a chunk references `symbol` in code, falling back to whole-word text
/// matching for languages with no first-class grammar.
fn references_symbol(
    references: &BTreeMap<u64, BTreeSet<String>>,
    chunk: &StoredChunk,
    symbol: &str,
) -> bool {
    match references.get(&chunk.chunk_id) {
        Some(identifiers) => identifiers.contains(symbol),
        None => mentions_identifier(&chunk.content, symbol),
    }
}

/// Whether `content` references `symbol` as a whole identifier rather than as a
/// substring of a longer name.
fn mentions_identifier(content: &str, symbol: &str) -> bool {
    let bytes = content.as_bytes();
    let mut searched = 0;

    while let Some(offset) = content[searched..].find(symbol) {
        let start = searched + offset;
        let end = start + symbol.len();
        let clean_start = start == 0 || !is_identifier_byte(bytes[start - 1]);
        let clean_end = end == content.len() || !is_identifier_byte(bytes[end]);
        if clean_start && clean_end {
            return true;
        }
        searched = start + 1;
    }

    false
}

/// Symbols too generic to carry a relatedness signal. Every language here has
/// dozens of unrelated `new`/`fmt`/`main` definitions, so co-occurrence on one
/// says nothing about whether two chunks belong together.
const GENERIC_SYMBOLS: &[&str] = &[
    "new",
    "main",
    "default",
    "from",
    "into",
    "next",
    "drop",
    "clone",
    "eq",
    "hash",
    "cmp",
    "fmt",
    "get",
    "set",
    "run",
    "id",
    "len",
    "add",
    "build",
    "parse",
    "read",
    "write",
    "init",
    "test",
    "name",
    "value",
    "path",
    "start",
    "end",
    "close",
    "open",
    "to_string",
    "as_str",
];

/// Retrieval precision that does not depend on the score scale.
///
/// The rest of this harness counts hints, which rewards whichever model happens
/// to score tighter against a fixed floor. That says nothing about whether the
/// neighbors are the right ones, so it cannot rank two embedding models. This
/// can: after an edit to symbol `S`, a neighbor worth surfacing is one that
/// *references* `S`, and whether a chunk's text contains `S` is decided by the
/// text alone, identically for every model.
///
/// It is a proxy, not ground truth. A truly related chunk that never names the
/// symbol counts as a miss, and an incidental mention counts as a hit. Both
/// biases apply equally to every model, so the comparison between models holds
/// even though the absolute number means little on its own.
///
/// The base rate is the control and the only reason the number is readable: it
/// is the share of other files that mention the symbol at all, which is what
/// picking neighbors at random would score. Precision at or below the base rate
/// means the embedding contributed nothing.
///
/// Deliberately ranks with **no similarity floor**. Scoring only the queries
/// that clear one makes the metric conditional on the score scale: a model that
/// emits fewer, more confident hints is handed an easier question set and wins
/// on selection rather than on ranking. That silently reversed this metric's
/// verdict once already. With the floor off, every model answers the identical
/// symbol set, so `symbols evaluated` must match across two runs on one repo —
/// if it does not, the comparison is void.
fn report_symbol_precision(
    rows: &[StoredChunk],
    by_file: &BTreeMap<String, Vec<&StoredChunk>>,
    references: &BTreeMap<u64, BTreeSet<String>>,
    top_k: usize,
) {
    let mut precision_sum = 0.0f64;
    let mut base_sum = 0.0f64;
    let mut top1_hits = 0usize;
    let mut evaluated = 0usize;
    let mut skipped_no_target = 0usize;
    // A symbol referenced by half the repo is found by any ranking, so the
    // aggregate is dominated by cases no model can lose. These are the ones
    // that discriminate: few correct answers, most of the corpus wrong.
    let mut rare_precision_sum = 0.0f64;
    let mut rare_base_sum = 0.0f64;
    let mut rare_evaluated = 0usize;
    const RARE_MENTION_LIMIT: usize = 3;

    for row in rows {
        let Some(symbol) = row.name.as_deref() else {
            continue;
        };
        if symbol.len() < 4 || GENERIC_SYMBOLS.contains(&symbol) {
            continue;
        }

        // Only files other than the edited one can be retrieved, so the base
        // rate and the judgement must both be scoped to them.
        let mut other_files = 0usize;
        let mut mentioning: BTreeSet<&str> = BTreeSet::new();
        for (path, chunks) in by_file {
            if path == &row.file_path {
                continue;
            }
            other_files += 1;
            if chunks
                .iter()
                .any(|c| references_symbol(references, c, symbol))
            {
                mentioning.insert(path.as_str());
            }
        }
        let mentioning_files = mentioning.len();

        // A query with no correct answer anywhere scores zero for every model
        // and only dilutes the average.
        if mentioning_files == 0 || other_files == 0 {
            skipped_no_target += 1;
            continue;
        }

        let exclude = RelativePath::new(&row.file_path);
        let hits = neighbors(
            rows,
            std::slice::from_ref(&row.vector),
            &exclude,
            top_k,
            0.0,
        );
        if hits.is_empty() {
            continue;
        }

        let relevant = hits
            .iter()
            .filter(|hit| mentioning.contains(hit.file_path.as_str()))
            .count();

        let precision = relevant as f64 / hits.len() as f64;
        let base = mentioning_files as f64 / other_files as f64;

        precision_sum += precision;
        base_sum += base;
        if hits
            .first()
            .is_some_and(|hit| mentioning.contains(hit.file_path.as_str()))
        {
            top1_hits += 1;
        }
        evaluated += 1;

        if mentioning_files <= RARE_MENTION_LIMIT {
            rare_precision_sum += precision;
            rare_base_sum += base;
            rare_evaluated += 1;
        }
    }

    println!("\n  retrieval precision by symbol co-occurrence (model-independent)");
    if evaluated == 0 {
        println!("    no evaluable symbols in this index");
        return;
    }

    let precision = precision_sum / evaluated as f64;
    let base = base_sum / evaluated as f64;
    println!(
        "    symbols evaluated        {evaluated} ({skipped_no_target} skipped, referenced nowhere else)"
    );
    println!("    precision@{top_k}             {:.3}", precision);
    println!("    base rate (random)       {:.3}", base);
    println!(
        "    lift over random         {:.2}x",
        if base > 0.0 { precision / base } else { 0.0 }
    );
    println!(
        "    precision@1              {:.3}",
        top1_hits as f64 / evaluated as f64
    );

    if rare_evaluated > 0 {
        let rare_precision = rare_precision_sum / rare_evaluated as f64;
        let rare_base = rare_base_sum / rare_evaluated as f64;
        println!(
            "    rare symbols (<={RARE_MENTION_LIMIT} files) {rare_evaluated} evaluated, precision@{top_k} {:.3}, base {:.3}, lift {:.2}x",
            rare_precision,
            rare_base,
            if rare_base > 0.0 {
                rare_precision / rare_base
            } else {
                0.0
            }
        );
    }
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

/// Not ignored: the precision metric is only meaningful if this is exact, and
/// a substring match would silently inflate every model's score equally, which
/// is the kind of error a model comparison cannot reveal.
#[cfg(test)]
mod identifier_tests {
    use super::mentions_identifier;
    use crate::types::Language;

    #[test]
    fn matches_only_whole_identifiers() {
        assert!(mentions_identifier(
            "let x = parse_config();",
            "parse_config"
        ));
        assert!(mentions_identifier("parse_config", "parse_config"));
        assert!(mentions_identifier("(parse_config)", "parse_config"));

        assert!(!mentions_identifier("try_parse_config()", "parse_config"));
        assert!(!mentions_identifier("parse_config_inner()", "parse_config"));
        assert!(!mentions_identifier("xparse_configx", "parse_config"));
        assert!(!mentions_identifier("nothing here", "parse_config"));
    }

    /// The scanner must keep looking past a rejected substring match rather
    /// than giving up on the first hit.
    #[test]
    fn finds_a_real_match_after_a_rejected_substring() {
        assert!(mentions_identifier("my_render() then render()", "render"));
    }

    /// The reason for parsing instead of text-matching: a symbol named in
    /// prose or in a log line is not a reference to it, and counting it as one
    /// credits a model for retrieving a chunk that cannot need matching edits.
    #[test]
    fn code_references_exclude_comments_and_strings() {
        let source = r#"
            // calls parse_config to do the thing
            fn caller() {
                let msg = "parse_config failed";
                real_call();
            }
        "#;

        let found = super::code_identifiers(source, Language::Rust)
            .unwrap_or_else(|| unreachable!("rust grammar is available"));

        assert!(found.contains("real_call"), "code identifier must be found");
        assert!(
            !found.contains("parse_config"),
            "comment and string mentions must not count as references"
        );
    }

    /// A language with no first-class chunker has no grammar, and the caller
    /// needs to tell that apart from a chunk that references nothing.
    #[test]
    fn unknown_language_yields_no_reference_set() {
        assert!(super::code_identifiers("anything", Language::Unknown).is_none());
    }
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
    let references = index_references(&rows);
    println!(
        "reference labels: {} of {} chunks parsed with a grammar, rest fall back to text",
        references.len(),
        rows.len()
    );
    report_symbol_precision(&rows, &by_file, &references, top_k);
    report_floor_sweep(&rows, &by_file, &references, top_k);

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
