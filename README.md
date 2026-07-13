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
- **Optional**: LM Studio or Ollama for a custom embedding backend. The bundled `bge-small-en-v1.5` embedder is statically linked into the binary, so no separate ONNX runtime is needed.

## Installation

**Requires**: Claude Code 2.0.12+.

claudix ships the bundled `bge-small-en-v1.5` embedder and uses it as the fallback when no embedding provider is configured. Set `embedding.provider = "http"` if you prefer LM Studio, Ollama, or another OpenAI-compatible embedding server.

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

Targets without a prebuilt (e.g. linux-aarch64, darwin-x86_64) fall back to `cargo install claudix@<version>`. Install [Rust](https://rustup.rs) first if you are on one of those.

## Configuration

Configuration lives in two TOML files (project overrides global):

- **Global**: `~/.claude/claudix.toml`
- **Project**: `<repo>/.claude/claudix.toml`

Both optional. If neither file sets an embedding provider, bundled defaults and the `bge-small-en-v1.5` embedder are used. Run `/claudix:doctor` to see active configuration.

### Full Schema (defaults)

```toml
[embedding]
provider = "bundled"                # bundled | http
endpoint = ""                       # required if provider = http (e.g., http://localhost:1234)
model = "bge-small-en-v1.5"
dimensions = 384
batch_size = 32
timeout_ms = 30000

[indexing]
respect_gitignore = true            # set false to also index gitignored files
follow_symlinks = false
max_file_size_kb = 512              # skip files larger than this
chunk_overlap_lines = 5             # overlap for fallback chunks only (force-indexed & unknown file types)
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
related_min_similarity = 0.80       # cosine floor for related-code hits (0.0-1.0)

[paths]
index_dir = ".claudix/index"        # relative to repo root; committed to .gitignore
log_dir = ".claudix/logs"           # relative to repo root
```

Configuration is validated at every entry point (MCP, hook, CLI). Invalid config exits early with field-level error messages.

## Slash Commands

| Command | Description | Arguments |
|---------|-------------|-----------|
| `/claudix:index` | Build or rebuild the index | `[--force]` |
| `/claudix:doctor` | Health check: binary, index, embedding provider | (none) |

Everything else is an MCP tool the agent calls directly: `search_code`, `get_index_status`, `overview`, `find_duplicates`, `reindex`, `reindex_file`, `clear_index`. Ask for a search in plain language ("where is auth handled?") and the agent routes it to `search_code`; the same CLI subcommands (`claudix search`, `claudix status`, ...) remain available in a terminal.

## How It Works

### SessionStart Hook

On first session or after plugin upgrade, the `SessionStart` hook:

1. Checks that plugin files and global config are present
2. Emits `claudix ready` via `systemMessage` when setup is complete
3. Tells the user to rerun the install script if setup is incomplete
4. Kicks off the background update check

If anything fails, the hook exits 0 (fail-open): session continues unaffected.

### PostToolUse Hook (File Edit)

After `Write`, `Edit`, or `MultiEdit` tools, the hook:

1. Reads edited file path from tool input
2. Invokes `claudix hook PostToolUse <path>`
3. Atomically upserts new chunks, removes stale ones
4. Index stays live without a watcher

### PreToolUse Hook (Grep Intercept)

Before `Grep` or `Bash` tools (with `rg`, `grep`, `ag` commands), the hook:

1. Analyzes query for regex patterns, globs, short length
2. Checks if index is stale or missing
3. If index is fresh and query looks conceptual, returns:
   ```json
   {
     "hookSpecificOutput": {
       "hookEventName": "PreToolUse",
       "permissionDecision": "deny",
       "permissionDecisionReason": "Use the claudix.search_code MCP tool for semantic queries; this query looks conceptual.",
       "additionalContext": "Original query was '<query>'. The claudix search index has <N> chunks across <M> files."
     }
   }
   ```
4. Otherwise allows grep to proceed

Heuristics for passthrough: regex anchors/character classes, explicit file globs, <3 tokens, stale index, `intercept_grep = false`.

### MCP Tool: `search_code`

Claude invokes `claudix.search_code(query, language_filter, path_prefix, repos)` directly. Uses hybrid retrieval: dense vector (55%), BM25 (30%), reciprocal rank fusion (15%). Returns `{ groups: [ { directory, repo, hits: [...] } ], repo_errors: [...] }` — results are grouped by `(repo, directory)`, ordered by the best hit score in each group. Each hit carries file location, definition kind, name, line range, and score.

## Embedding Backends

### Bundled (Default)

`bge-small-en-v1.5` runs via ONNX Runtime on CPU. No external dependencies. ~100ms/chunk on modern hardware.

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

All three backends return 384-dimensional vectors (for bge-small); other models may differ. Dimension mismatch triggers reindex.

### Choosing a Model

The bundled `bge-small-en-v1.5` is a small general-purpose English text model. It is a fine zero-setup default, but a code-specialized or larger embedder measurably improves retrieval on real codebases.

The `http` provider speaks the OpenAI `/v1/embeddings` format and sends no authorization header, so it connects to keyless servers: LM Studio, Ollama, a local vLLM instance, or a local proxy such as LiteLLM. Hosted APIs that require a key (Voyage, OpenAI, Gemini) are reachable only by fronting them with a local proxy that injects the key. Set `dimensions` to the model's output size, or to a smaller Matryoshka size it supports; changing the dimension triggers a full reindex.

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

```
/claudix:index
```

Full reindex takes ~1-5 minutes depending on repo size. Incremental updates (on file edit) take milliseconds.

### Embedding Endpoint Unreachable

If using LM Studio or Ollama:

1. Verify server is running: `curl http://localhost:1234/health` (LM Studio) or `curl http://localhost:11434/api/embeddings` (Ollama)
2. Check configuration: `/claudix:doctor` shows `endpoint` in use
3. Switch to bundled: set `provider = "bundled"` in `~/.claude/claudix.toml`, then run `/claudix:index` to reindex

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

If the binary is newer than indexed chunks, SessionStart triggers background reindex and emits `additionalContext`. You can manually rebuild with `/claudix:index --force`.

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

Tests live next to code (`#[cfg(test)]`) or in `tests/integration/` with fixtures in `tests/fixtures/`.

## License

MIT

