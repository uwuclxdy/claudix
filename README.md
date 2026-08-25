<div align="center">

![claudix banner](media/claudix.png)

![GitHub Actions Workflow Status](https://shields.uwuclxdy.dev/github/actions/workflow/status/uwuclxdy/claudix/.github%2Fworkflows%2Frelease.yml?style=for-the-badge&cacheSeconds=60)
![GitHub Downloads (all assets, all releases)](https://shields.uwuclxdy.dev/github/downloads/uwuclxdy/claudix/total?style=for-the-badge&color=%2343ABE5&cacheSeconds=60)
![Claude Code](https://shields.uwuclxdy.dev/badge/Claude%20Code-D97757?style=for-the-badge)

# Claude Index: claudix

</div>

Copilot's Codebase Index but for Claude Code. Automatically indexes your repo, embeds with the embedding model of choice and provides semantic search through Claude's slash commands, MCP tools, and grep interception.

## What It Does

claudix is a Claude Code plugin that gives the agent local semantic search over any repository. A single Rust binary acts as MCP server, hook handler, and CLI. When Claude Code starts, claudix warmly bootstraps the index if missing. When you edit files, chunks are re-embedded automatically. When grep would be less useful than semantic search, the plugin intercepts and uses dense vectors instead. Configuration lives next to `settings.json` with optional per-project overrides. First-class language support covers Rust, Python, JavaScript, TypeScript, Go, Java, C, C++, C# and SQL.

Core design goal: never break the session, always recover gracefully.

## Requirements

- **macOS 11+ (Apple Silicon)** or **Linux x86_64** (glibc 2.28+ or musl) or **Windows 10+ x86_64**
- **Rust 1.91+**
- **Windows**: the [Visual C++ 2015-2022 x64 redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe). The MSVC-built prebuilt binary won't launch without it; a stock box may not have it preinstalled.
- **Optional**: LM Studio or Ollama for a custom embedding backend. The bundled `gte-modernbert-base` embedder is statically linked into the binary, so no separate ONNX runtime is needed.

## Installation

**Requires**: Claude Code 2.1.0+ (plugin skills in the slash menu).

claudix ships the bundled `gte-modernbert-base` embedder and uses it as the fallback when no embedding provider is configured. Set `embedding.provider = "http"` if you prefer LM Studio, Ollama, or another OpenAI-compatible embedding server.

### Linux / macOS

```bash
curl -fsSL https://raw.githubusercontent.com/uwuclxdy/claudix/mommy/install.sh | bash
```

### Windows

```powershell
irm https://raw.githubusercontent.com/uwuclxdy/claudix/mommy/install.bat | iex
```

### Manual

```bash
claude plugin marketplace add uwuclxdy/claudix
claude plugin install claudix@claudix
```

The native binary downloads on first session (~150MB). Restart Claude Code, then run `/claudix:doctor` to verify.

The downloader honors `GITHUB_TOKEN`/`GH_TOKEN` (lifts the anonymous rate limit) and, for mirrors or air-gapped installs, `CLAUDIX_RELEASE_API_URL`, `CLAUDIX_RELEASE_BASE_URL`, `CLAUDIX_RELEASE_WAIT_MS`, `CLAUDIX_RELEASE_RETRY_MS`.

Targets without a prebuilt (e.g. linux-aarch64, darwin-x86_64) fall back to `cargo install claudix@<version>`. Install [Rust](https://rustup.rs) first if you are on one of those.

## Configuration

Configuration lives in two TOML files (project overrides global):

- **Global**: `~/.claude/claudix.toml`
- **Project**: `<repo>/.claude/claudix.toml`

Both optional. If neither file sets an embedding provider, bundled defaults and the `gte-modernbert-base` embedder are used. Run `/claudix:doctor` to see active configuration.

### Full Schema (defaults)

```toml
watch = false                       # opt-in file watcher; default off (the PostToolUse hook covers edits)

[embedding]
provider = "bundled"                # bundled | http
endpoint = ""                       # required if provider = http (e.g., http://localhost:1234)
model = "gte-modernbert-base"
dimensions = 768
batch_size = 32
timeout_ms = 8000

[indexing]
respect_gitignore = true            # set false to also index gitignored files
follow_symlinks = false
max_file_size_kb = 512              # skip files larger than this
chunk_overlap_lines = 5             # lines each fallback chunk shares with the previous (force-indexed files)
reindex_after_hours = 24            # background reindex interval

[search]
top_k = 10                          # results per search
hybrid_weights = { dense = 0.55, bm25 = 0.30, rrf = 0.15 }  # hybrid retrieval weights
identifier_boost = 1.4              # boost exact identifier matches
similarity_threshold = 0.30         # minimum cosine similarity to keep a candidate
min_score = 0.05                    # minimum fused score to return a hit
cross_repos = []                    # extra already-indexed repo paths to search (read-only)

[hooks]
intercept_grep = true               # replace grep with semantic search when useful
auto_reembed_on_edit = true         # re-embed after Write/Edit
reindex_debounce_secs = 10          # watch=false: coalesce rapid edits, reindex after N idle secs
reindex_max_wait_secs = 60          # hard cap so a continuously-edited file still reindexes
auto_index_on_session_start = true  # background reindex check on session start
surface_related_on_edit = true      # surface semantically related files after an edit
surface_related_on_read = false     # surface related files after a ranged Read (opt-in)
related_top_k = 5                   # max related-code hits per edit or read
related_min_similarity = 0.80       # fallback cosine floor; active floor is corpus-relative

[paths]
index_dir = ".claudix/index"        # relative to repo root; committed to .gitignore
log_dir = ".claudix/logs"           # relative to repo root
```

Configuration is validated at every entry point (MCP, hook, CLI). Invalid config exits early with field-level error messages.

### Index Scope: `.indexignore` / `.indexinclude`

Two optional rule files control what gets indexed, using gitignore syntax (globs, `!` negation, comments). Place them at the repo root or in any subdirectory; patterns are relative to the rule file's own directory, like nested `.gitignore` files. Precedence per path: `.indexinclude` beats `.gitignore`, which beats `.indexignore`.

- `.indexignore` excludes tracked files from the index: test fixtures, vendored code, minified bundles.
- `.indexinclude` pulls gitignored paths into the index: internal `docs/`, generated code. A one-line `*` file inside the `docs/` directory indexes that whole tree. Files without a code chunker index as plain text.

A rule file buried two or more levels inside a gitignored subtree is not discovered; put it at the top of the gitignored directory or at the repo root. Edited rule files apply on the next index run. To index everything gitignored instead, set `[indexing] respect_gitignore = false`.

### Cross-Repo Search

Index each repo on its own first, with the same embedding model (per-repo mismatches surface in `repo_errors`). Extra repos are read-only. Include them persistently via `[search] cross_repos = ["/abs/path"]`, or per call via the `repos` argument on `search_code` and `find_duplicates` (both add to the active project, which is always scanned). Grouped results prefix each directory with its repo (`<repo> :: <dir>`).

## Skills & Commands

| Invocation | Kind | What it does |
|---|---|---|
| `/claudix:claudix` | skill | Teaches the agent the full feature surface: index rebuilds, `.indexignore`/`.indexinclude` scope, provider switches, cross-repo setup, hook tuning. Triggers on its own when a task touches those. |
| `/claudix:doctor` | command | Health check: binary, index, embedding provider |

Day-to-day operations are MCP tools the agent calls directly; ask in plain language ("where is auth handled?", "rebuild the index") and it routes to the right one:

| Tool | Purpose |
|---|---|
| `search_code` | Hybrid semantic search; `repos` adds other indexed repos |
| `reindex` | Full rebuild; `path` re-embeds one file, `force: true` wipes first |
| `find_duplicates` | Near-identical chunk pairs; `repos` adds other indexed repos |

Index status, a full teardown, and the per-directory map of what is actually indexed stay on the
CLI (`claudix status`, `claudix clear`, `claudix overview`) and `/claudix:doctor` — the session
start hook already reports counts and staleness, so spending agent context on tools for them bought
nothing.

### CLI

The same binary works as a CLI in a terminal (`claudix` on PATH on Linux/macOS; `node <plugin>/bin/claudix-bootstrap.js <subcommand>` on Windows):

| Subcommand | Flags |
|---|---|
| `index` | `--force`, `--progress` |
| `search <query...>` | `--top-k N`, `--language L` (repeatable), `--path-prefix P`, `--repo /abs/path` (repeatable) |
| `status` | |
| `overview` | `--path-prefix P` |
| `find-duplicates` | `--min-similarity N`, `--limit N`, `--repo /abs/path` (repeatable) |
| `reindex-file <path>` | |
| `clear` | |
| `doctor` | |
| `install` | |
| `watch` | Runs the opt-in file watcher (pairs with `watch = true`) |

## How It Works

### SessionStart Hook

On every session start, the hook:

1. Checks that plugin files and global config are present; tells the user to rerun the install script if not
2. Builds the first index in the background when none exists and notifies the conversation on completion
3. Reindexes in the background when the index is older than `reindex_after_hours` or the binary is newer than the stored schema
4. Tells the agent how search is set up: index freshness, `search_code` vs Grep routing, or a model-mismatch notice with the rebuild step

If anything fails, the hook exits 0 (fail-open): session continues unaffected.

### PostToolUse Hook (File Edit)

After `Write`, `Edit`, or `MultiEdit` tools, the hook:

1. Reads the edited file path(s) from the tool payload on stdin
2. Queues the paths; a background drain worker coalesces rapid edits (`reindex_debounce_secs` / `reindex_max_wait_secs`) so a burst re-embeds each file once
3. Atomically upserts new chunks, removes stale ones
4. Index stays live without a watcher

The same event (plus `UserPromptSubmit`) also surfaces background-indexing completion and semantically related code for the file just edited or read.

### PreToolUse Hook (Grep Intercept)

Before `Grep` or `Bash` tools (with `rg`, `grep`, `ag` commands), the hook:

1. Analyzes the query for regex patterns, globs, short length
2. Checks if the index is stale or missing
3. If the index is fresh and the query looks conceptual, denies the grep and answers it inline: the response carries the ranked semantic matches (file, line range, snippet, score) as `additionalContext` with a tip to call `search_code` directly next time
4. Otherwise grep proceeds untouched

Heuristics for passthrough: regex anchors/character classes, explicit file globs, <3 tokens, stale index, `intercept_grep = false`.

### Related-Code Surfacing

After an edit, claudix looks up code semantically related to the changed chunks and injects the locations into the conversation on the next hook event ("may need matching changes"). Source hits render under `related code:` and documentation files (markdown, rst, adoc, txt, org extensions, case-insensitive, and conventional extensionless basenames such as README or LICENSE) under `related docs:`; a label appears only when its group has hits. Ranged `Read`s get the same treatment when `surface_related_on_read = true` (opt-in). `related_top_k` caps hits per edit across both groups; the cosine floor is corpus-relative (each full index stores the p30 of its own score distribution), with `related_min_similarity` as the fallback and a hard minimum. A neighbor already surfaced this session is not repeated when it names the same lines, whichever file you were editing at the time; hints pointing at lines the session already Read are skipped too.

### MCP Tool: `search_code`

Claude invokes `claudix.search_code(query, language_filter, path_prefix, repos)` directly. Uses hybrid retrieval: dense vector (55%), BM25 (30%), reciprocal rank fusion (15%). Returns `{ groups: [ { directory, repo, hits: [...] } ], repo_errors: [...] }` — results are grouped by `(repo, directory)`, ordered by the best hit score in each group. Each hit carries file location, definition kind, name, line range, and score.

## Embedding Backends

### Bundled (Default)

`gte-modernbert-base` (768 dims, CLS pooling) runs via ONNX Runtime on CPU. No external dependencies.

`bge-small-en-v1.5` (384 dims) stays selectable via `model = "bge-small-en-v1.5"` with `dimensions = 384`. It is smaller and faster, at lower retrieval quality. Assets download once to `~/.claude/claudix/models`, pinned to a specific upstream revision and verified against a published sha256 before use. Both models' assets can coexist; only unversioned files from older releases are cleared.

Requires: `libonnxruntime` (Linux) or `onnxruntime.dll` (Windows) installed in system library path, or downloaded automatically.

### LM Studio

Local LLM inference server. Download [lm-studio.ai](https://lm-studio.ai), load an embedding model (e.g., `nomic-ai/nomic-embed-text-v1.5`), start the server.

```toml
[embedding]
provider = "http"
endpoint = "http://localhost:1234"
model = "your-model-name"
```

### Ollama

Local inference. Install [ollama.ai](https://ollama.ai), pull embedding model:

```bash
ollama pull nomic-embed-text
```

Configure:

```toml
[embedding]
provider = "http"
endpoint = "http://localhost:11434"
model = "nomic-embed-text"
```

Vector width follows the model: 768 for the bundled `gte-modernbert-base`, 384 for `bge-small-en-v1.5`, whatever an `http` model publishes. After any model or dimension change, rebuild with `claudix index` (SessionStart flags the mismatch but never wipes your index on its own).

### Choosing a Model

The bundled `gte-modernbert-base` is a general-purpose English text model with a long context window. It is a solid zero-setup default, but a code-specialized or larger embedder measurably improves retrieval on real codebases. For how the bundled models were measured against each other, why an absolute similarity floor is not portable across models (and how claudix calibrates a corpus-relative one per index), and why public code-retrieval benchmarks do not measure this use case, see [wiki/embedding-findings.md](wiki/embedding-findings.md).

The `http` provider speaks the OpenAI `/v1/embeddings` format and sends no authorization header, so it connects to keyless servers: LM Studio, Ollama, a local vLLM instance, or a local proxy such as LiteLLM. Hosted APIs that require a key (Voyage, OpenAI, Gemini) are reachable only by fronting them with a local proxy that injects the key. Set `dimensions` to the model's output size, or to a smaller Matryoshka size it supports; changing the dimension requires a `claudix index` rebuild.

| Pick | Model | Dimensions | Context | Access | Why |
|------|-------|-----------|---------|--------|-----|
| Best accuracy | `voyage-code-3` | 256 / 512 / 1024 / 2048 | 32K | Voyage API, $0.18 / 1M tokens (200M free) | Code-specialized. Beats OpenAI `text-embedding-3-large` by ~14% across 32 code-retrieval datasets, with int8/binary quantization for cheap storage. Needs a local proxy for the API key. |
| Best self-hosted | `Qwen3-Embedding` (0.6B / 4B / 8B) | 1024 / 2560 / 4096 | 32K | Apache-2.0, open weights | Runs keyless via Ollama, LM Studio, or vLLM. 0.6B is CPU-viable; 8B sits near the top of MTEB and code retrieval on a GPU. Matryoshka dimensions. |
| Best lightweight code model | `jina-code-embeddings` (0.5B / 1.5B) | 896 / 1536 | 32K | Open weights | Code-specialized, built on Qwen2.5-Coder. SOTA code retrieval for its size, cheap to self-host. Matryoshka dimensions. |

For a self-hosted default, Qwen3-Embedding 0.6B via Ollama is the easiest upgrade over the bundled model:

```toml
[embedding]
provider = "http"
endpoint = "http://localhost:11434"   # Ollama
model = "qwen3-embedding"             # use the id your server reports
dimensions = 1024
```

## Building from Source

Requires **Rust 1.91+** and **Cargo**.

```bash
git clone <repo>
cd claudix
cargo build --release
```

Test suite:

```bash
cargo test --lib                    # unit tests only
cargo test --test integration       # integration tests (~10s)
cargo test --all -- --include-ignored  # all tests including e2e (~30s)
```

Binary is `target/release/claudix`. Strip for distribution:

```bash
strip target/release/claudix
```

Linux musl builds are statically linked:

```bash
cargo build --release --target x86_64-unknown-linux-musl
```

## Troubleshooting

### Check Plugin Health

```
/claudix:doctor
```

Diagnostic output:

- Project root
- Index present (yes/no)
- Chunk and file counts
- Active embedding model
- Embedding provider and health (reachable or error)
- Configuration errors

### Index Missing or Stale

Ask the agent to rebuild the index (it calls the `reindex` MCP tool), or run `claudix index` in a terminal.

Full reindex takes ~1-5 minutes depending on repo size. Incremental updates (on file edit) take milliseconds.

### Embedding Endpoint Unreachable

If using LM Studio or Ollama:

1. Verify server is running: `curl http://localhost:1234/health` (LM Studio) or `curl http://localhost:11434/api/embeddings` (Ollama)
2. Check configuration: `/claudix:doctor` shows `endpoint` in use
3. Switch to bundled: set `provider = "bundled"` in `~/.claude/claudix.toml`, then rebuild with `claudix index` (the model changed)

### Hooks Don't Trigger or Fail Silently

SessionStart hook exits 0 even on error. Check logs:

```bash
tail -f .claudix/logs/index.log
```

Logs are created on first run. Enable debug logging:

```bash
RUST_LOG=debug claudix status  # or any other subcommand
```

### Schema Mismatch After Upgrade

If the binary is newer than indexed chunks, SessionStart triggers background reindex and emits `additionalContext`. You can manually rebuild with `claudix index`.

## Fail-Open Guarantee

**Hooks never fail the session.**

- Binary missing → hook exits 0, session proceeds
- Embedding endpoint down → hook exits 0, grep proceeds normally
- Index corrupted → hook exits 0, search MCP tool returns error with recovery hint
- Configuration invalid → caught at startup; MCP tool returns error message

MCP tool errors are structured JSON with a `recovery` field Claude can act on:

```json
{
  "error": "embedding_endpoint_unreachable",
  "message": "LM Studio at http://localhost:1234 did not respond.",
  "recovery": "Run /claudix:doctor to diagnose, or set [embedding] provider = \"bundled\" in ~/.claude/claudix.toml."
}
```

## Development

Architecture is domain-driven (chunking, embedding, store, search, mcp, hooks) rather than kind-driven. Each domain module owns types, traits, and tests. Cross-domain coupling goes through small interface traits.

Key modules:

- `src/chunking/` — tree-sitter code splitting
- `src/embedding/` — bundled (ONNX) and http (LM Studio, Ollama) providers
- `src/store/` — LanceDB for vector + FTS
- `src/search/` — hybrid retrieval
- `src/mcp/` — Model Context Protocol
- `src/hooks/` — SessionStart, PostToolUse, PreToolUse handlers
- `src/cli/` — slash command entry points

Tests live next to code (`#[cfg(test)]`) or in `tests/*.rs` with shared helpers in `tests/common/`.

## License

MIT

