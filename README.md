<div align="center">

![claudix banner](media/claudix.png)

![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/uwuclxdy/claudix/.github%2Fworkflows%2Frelease.yml?style=for-the-badge&cacheSeconds=60)
![GitHub Downloads (all assets, all releases)](https://img.shields.io/github/downloads/uwuclxdy/claudix/total?style=for-the-badge&color=%2343ABE5&cacheSeconds=60)
![Claude Code](https://img.shields.io/badge/Claude%20Code-D97757?style=for-the-badge)

# Claude Index: claudix

</div>

Copilot's Codebase Index but for Claude Code. Automatically indexes your repo, embeds with the embedding model of choice and provides semantic search through Claude's slash commands, MCP tools, and grep interception.

## What It Does

claudix is a Claude Code plugin that gives the agent local semantic search over any repository. A single Rust binary acts as MCP server, hook handler, and CLI. When Claude Code starts, claudix warmly bootstraps the index if missing. When you edit files, chunks are re-embedded automatically. When grep would be less useful than semantic search, the plugin intercepts and uses dense vectors instead. Configuration lives next to `settings.json` with optional per-project overrides. First-class language support includes Python, JavaScript, Java, C, C++, C#, SQL, Rust, TypeScript, and Go.

Core design goal: never break the session, always recover gracefully.

## Requirements

- **macOS 11+ (Apple Silicon)** or **Linux x86_64** (glibc 2.28+ or musl) or **Windows 10+ x86_64**
- **Rust 1.91+**
- **Optional**: LM Studio or Ollama for custom embedding backends (
  - bundled `bge-small-en-v1.5` requires `libonnxruntime` or `onnxruntime.dll`)

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
respect_gitignore = true
follow_symlinks = false
max_file_size_kb = 512              # skip files larger than this
chunk_overlap_lines = 5             # lines overlapped between chunks
reindex_after_hours = 24            # background reindex interval

[search]
top_k = 10                          # results per search
hybrid_weights = { dense = 0.55, bm25 = 0.30, rrf = 0.15 }  # hybrid retrieval weights
identifier_boost = 1.4              # boost exact identifier matches
similarity_threshold = 0.30         # minimum relevance score

[hooks]
intercept_grep = true               # replace grep with semantic search when useful
auto_reembed_on_edit = true         # re-embed after Write/Edit
auto_index_on_session_start = true  # background reindex check on session start
surface_related_on_edit = true      # surface semantically related files after an edit
surface_related_on_read = false     # surface related files after a ranged Read (opt-in)
related_top_k = 5                   # max related-code hits per edit or read
related_min_similarity = 0.65       # cosine floor for related-code hits (0.0–1.0)

[paths]
index_dir = ".claudix/index"        # relative to repo root; committed to .gitignore
log_dir = "~/.claude/claudix/logs"
```

Configuration is validated at every entry point (MCP, hook, CLI). Invalid config exits early with field-level error messages.

## Slash Commands

All commands are available as `/claudix:<name>`:

| Command | Description | Arguments |
|---------|-------------|-----------|
| `/claudix:search` | Semantic code search | `<query> [--top-k N] [--language rust --language python] [--path-prefix src/]` |
| `/claudix:index` | Build or refresh index | (none) |
| `/claudix:status` | Show index metadata | (none) |
| `/claudix:doctor` | Diagnose health | (none) |
| `/claudix:reindex-file` | Re-embed one file | `<path>` |
| `/claudix:clear` | Delete index | (none) |

### Search Example

```
/claudix:search authentication flow --language rust --top-k 5
```

Results show file path, line range, language, definition kind, name, and relevance score.

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
tail -f ~/.claude/claudix/logs/claudix.log
```

Logs are created on first run. Enable debug logging:

```bash
RUST_LOG=debug /claudix:status  # or any other command
```

### Schema Mismatch After Upgrade

If the binary is newer than indexed chunks, SessionStart triggers background reindex and emits `additionalContext`. You can manually reindex with `/claudix:clear` then `/claudix:index`.

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

