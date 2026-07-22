# Embedding findings

What we measured when tuning claudix's related-code hints, and what changed as a result. Published
because most of it applies to anything doing semantic code search, not just this project.

Every number here comes from `tests/measure/hint_distribution.rs`, an `#[ignore]`d test that
replays the real hook pipeline over a real index. Run it yourself:

```bash
cargo test --lib hint_distribution -- --ignored --nocapture
CLAUDIX_MEASURE_ROOT=/path/to/another/repo cargo test --lib hint_distribution -- --ignored --nocapture
```

## We were using the wrong pooling head

`BAAI/bge-small-en-v1.5` publishes `pooling_mode_cls_token: true, pooling_mode_mean_tokens: false`
in its `1_Pooling/config.json`, and its card says to "select the last hidden state of the first
token (i.e. `[CLS]`) as the sentence embedding". claudix took the attention-masked mean over all
tokens instead. Every bundled embedding used the wrong head from the first release until
`fix(embedding): read the pooling head the bundled model publishes`.

Nothing caught it, and nothing was going to. Indexing and querying both run through the same
pooling code, so a wrong head stays perfectly self-consistent: search still works, results still
look reasonable, and every round-trip test passes. The only thing that can catch this class of bug
is an assertion against what the model itself publishes, which is now what
`bundled_model_pools_the_head_its_model_card_specifies` does.

**If you run an ONNX embedding model directly, check its pooling config.** Loading a model through
`transformers`/`sentence-transformers` reads `1_Pooling/config.json` for you. Driving the ONNX
graph yourself, as any Rust, Go, or C++ port must, means that file is never consulted, and mean
pooling is the default everyone reaches for.

One caveat on that config file: `codefuse-ai/F2LLM-v2-80M` ships a `1_Pooling/config.json` whose
`word_embedding_dimension` (1024) contradicts its own `config.json` `hidden_size` (320), with no
projection module to reconcile them. Cross-check it against `config.json` and `modules.json`
rather than trusting it alone.

## Fixing the pooling barely changed retrieval, and doubled the noise

The surprise: correcting the head changed **what scores came out** far more than **what got
retrieved**.

| | mean pooling (wrong) | CLS pooling (correct) |
|---|---|---|
| best-neighbor similarity, median | 0.835 | 0.873 |
| edits clearing the 0.80 floor | 70% | 88% |
| hints emitted per edit | 1.77 | 3.33 |
| precision@5 | 0.445 | 0.440 |
| precision@1 | 0.543 | 0.541 |

Same corpus, same floor. Hint volume nearly doubled. Retrieval quality stayed flat, measured
across four repositories on identical symbol sets.

## An absolute similarity floor is not portable

claudix gates related-code hints on `related_min_similarity`, an absolute cosine defaulting to
0.80. That number turns out to describe a whole pipeline, not a model:

- **Across models.** `bge-small-en-v1.5` (384-dim) clears 0.80 on 64 of 91 edits where
  `Qwen3-Embedding-8B` (4096-dim) clears it on 47. The smaller model is the *noisier* one at a
  fixed cutoff, which is the opposite of what we assumed before measuring.
- **Across pooling heads.** Changing only the pooling head moved the median 0.835 → 0.873 and
  doubled hint volume, as above.

Neither shift tracks retrieval quality. If you gate on an absolute cosine anywhere, re-measure the
distribution after any change to the model, the pooling, or the provider, and prefer a cutoff
derived from your own corpus's score distribution over a constant.

That is what claudix now does. A full index stores the 30th percentile of its own best-neighbor
distribution as the hint floor, so ~70% of edits surface at least one hit on any model by
construction (measured 70% at both qwen3's p30 of 0.748 and gte-modernbert's 0.759, where the fixed
0.80 cleared 51% and 88% respectively). `related_min_similarity` stays as the fallback when no such
stat exists and as a hard minimum when a user sets it above the default. The percentile was picked
from a floor sweep on two models: precision of the admitted hints rose from ~0.43 floor-free to
~0.63-0.67 at p30 and plateaued there. Coverage kept falling past that point.

## Public code-retrieval benchmarks do not measure this use case

The question "which embedding model is best for finding code related to what I just edited" has no
public benchmark behind it.

- **CoIR** is the standard code-retrieval suite, but only 2 of its 10 tasks are code→code, and
  neither is our shape: `CodeSearchNetCCRetrieval` matches a function's prefix to its own suffix
  (a completion proxy), and the `CodeTransOcean` tasks are cross-language translation. The other
  ~80% is natural-language-query→code, a geometry a "related code" feature never exercises.
- **MTEB has no clone-detection tasks at all.** Code-to-code benchmarks that do exist (POJ-104,
  BigCloneBench) define similarity as *independently written functionally equivalent programs*,
  the opposite regime from related chunks inside one repository that share its vocabulary.
- **General ability does not predict code ability.** `gte-base-en-v1.5` beats `bge-base-en-v1.5`
  on general MTEB and loses to it on CoIR (36.75 vs 42.77). `nomic-embed-text-v1.5` edges
  bge-small on BEIR-15 and loses by 7.5 on CoIR. Rank correlation in the sub-200M band is close to
  nil.

So we built a repo-local metric instead, and treat CoIR only as a floor filter.

## How we measure retrieval quality locally

Counting hints cannot rank two models: it rewards whichever one happens to score tighter against a
fixed floor. The metric that can is precision by symbol reference.

For each indexed symbol, query with its chunk, take the top-k neighbor files, and count a neighbor
as correct when it **references that symbol in code**. References are extracted by re-parsing each
chunk with its own tree-sitter grammar and keeping identifier tokens, so a symbol named in a
comment or inside a string literal does not count. The label is computed from text alone, so it is
identical for every model being compared.

It is a proxy, not ground truth: a genuinely related chunk that never names the symbol scores as a
miss. That bias is the same for every model, so comparisons hold even though the absolute number
means little on its own. The **base rate**, the share of other files that reference the symbol at
all, is printed alongside. That is what random guessing would score.

Two traps worth repeating, because both produced confident wrong answers here first:

- **Do not score only the queries a model answered.** Gating evaluation on "the query cleared the
  floor" made mean pooling look 8 to 17 points better than CLS across four repos. The gap was
  entirely selection: CLS clears the floor more often, so it was being graded on harder questions.
  Removing the floor so both models face an identical question set collapsed the gap to 0.005.
  The tell was the control. The base rate is computed from text and *cannot* differ between two
  models on one corpus, yet it read 0.139 against 0.115.
- **A null result is worthless until the instrument has shown a non-null.** Before trusting
  "pooling doesn't matter", we checked the metric could separate models at all by running a much
  stronger one through it.

## The two bundled models, measured

`gte-modernbert-base` (149M params, 768-dim) is the default; `bge-small-en-v1.5` (33M, 384-dim)
stays selectable. Each repository below was indexed under both models, so the corpus and the
question set are identical and only the vectors differ. The base rate is the same for both by
construction, which is the check that the comparison is clean.

| repo (lang, chunks) | metric | bge-small | gte-modernbert |
|---|---|---|---|
| paru (rust, 485) | precision@5 | 0.319 | 0.314 |
| | precision@1 | 0.464 | 0.454 |
| | rare-symbol p@5 | 0.165 | **0.182** |
| agentgear (rust, 2057) | precision@5 | 0.608 | **0.636** |
| | precision@1 | 0.742 | **0.807** |
| | rare-symbol p@5 | 0.164 | **0.178** |
| osu-collect (c#, 3332) | precision@5 | 0.272 | **0.312** |
| | precision@1 | 0.471 | **0.523** |
| | rare-symbol p@5 | 0.148 | **0.167** |

gte wins on both realistic-sized repositories across every metric, most of all on precision@1
(agentgear 0.742 → 0.807). On the 34-file repo the aggregate metrics are a wash because there are
too few distractors to separate the models, but gte still takes the rare-symbol case, the one that
discriminates. Rare symbols, referenced in three files or fewer, are where a query has few correct
answers and most of the corpus is wrong: gte wins there in all three repositories.

## Scaling past that band pays much less

A code-adjacent model at 149M beats the 33M general model. Scaling the other direction, to a giant
general model, does not keep paying at the same rate. `Qwen3-Embedding-8B` (4096-dim) measured on
this repository scored precision@1 0.579 and rare-symbol p@5 0.189, in the same territory gte
reaches on comparable repositories despite roughly 50x the parameters. That is a cross-corpus
comparison, so read it as suggestive, not decisive, but it points the same way as the CoIR
observation above: for finding related chunks inside one repository, code-adjacency and a sensible size beat
raw scale for related-chunk retrieval inside one repository.

## Scope

Measured on a small number of repositories with one proxy metric. Each result pointed the same way
in every repository tested, but the magnitudes are not precise, and none of this substitutes for a
benchmark on your own corpus.
