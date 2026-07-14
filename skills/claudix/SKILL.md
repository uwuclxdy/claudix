---
description: This skill should be used when working with claudix semantic search beyond plain queries — building or rebuilding the index, controlling what gets indexed (indexing gitignored files, excluding paths, `.indexignore`/`.indexinclude` rules), switching embedding providers or models (bundled, LM Studio, Ollama), setting up cross-repo search, tuning or disabling grep interception and related-code surfacing, or recovering from a stale or corrupted index.
---

# claudix

One Rust binary serves the MCP tools, the hooks, and a CLI. Prefer `claudix` from PATH; when it is absent (Windows installs create no symlink), run the node bootstrap that ships with the plugin at `bin/claudix-bootstrap.js` under the plugin root, two directories above this skill's base directory:

```bash
claudix <subcommand> 2>&1
# without claudix on PATH:
node "<plugin-root>/bin/claudix-bootstrap.js" <subcommand> 2>&1
```

Configuration lives in `~/.claude/claudix.toml` (global) overridden by `<repo>/.claude/claudix.toml` (project). Both are optional; every entry point validates them and errors name the exact key to fix.

## Build or rebuild the index

Agent path: the `reindex` MCP tool. CLI path: `claudix index` (`--progress` streams per-file progress). Pass `force: true` / `--force` to wipe the store before rebuilding. Needed for:

- embedding model or dimension changes
- schema mismatch after a plugin upgrade
- resetting a corrupted index

Indexing may take a minute on large repositories; if it errors, run `/claudix:doctor` to diagnose the embedding provider.

## Control what gets indexed

`.indexignore` and `.indexinclude` files use gitignore syntax (globs, `!` negation, comments). Place them at the repo root or in any subdirectory; patterns are relative to the rule file's own directory, like nested `.gitignore` files. Precedence per path: `.indexinclude` beats `.gitignore`, which beats `.indexignore`.

- Exclude tracked files from the index: `.indexignore` (test fixtures, vendored code, minified bundles).
- Index gitignored paths: `.indexinclude` (internal `docs/`, generated code). A one-line `*` inside `docs/.indexinclude` pulls that whole tree. Files without a code chunker index as plain text.
- Placement gotcha: a rule file buried two or more levels inside a gitignored subtree is not discovered; put it at the top of the gitignored directory or at the repo root.
- `[indexing] respect_gitignore = false` indexes everything gitignored instead of selected subtrees.

Edited rule files apply on the next index run, so reindex after changing them.

## Embedding providers

`[embedding] provider = "bundled"` (default, `bge-small-en-v1.5`, zero setup) or `"http"` (any keyless OpenAI-`/v1/embeddings` server: LM Studio `http://localhost:1234`, Ollama `http://localhost:11434`, vLLM, a LiteLLM proxy). Set `model` to the id the server reports; set `dimensions` to the model's output size. Any model or dimension change requires a force rebuild (previous section). Keyed APIs (Voyage, OpenAI) need a local proxy that injects the key.

## Cross-repo search

Each repo must be indexed on its own first, with the same embedding model (mismatches surface per repo in `repo_errors`). Extra repos are read-only. Two ways to include them:

- persistent: `[search] cross_repos = ["/abs/path/other-repo"]` in the active repo's config
- per call: the `repos` arg on `search_code` (added to the active project) or on `find_duplicates` (replaces the active project: list everything to scan)

Grouped results prefix each directory with its repo (`<repo> :: <dir>`).

## Hook behavior tuning

All under `[hooks]`:

- `intercept_grep = false` disables the conceptual-Grep interception (anchored regexes, globs, short queries always pass through anyway).
- `auto_reembed_on_edit = false` stops re-embedding on Write/Edit; `reindex_debounce_secs` / `reindex_max_wait_secs` tune the edit-burst coalescing.
- `surface_related_on_edit` (default on) injects semantically related locations after an edit; `surface_related_on_read = true` (opt-in) does the same for ranged Reads. `related_top_k` / `related_min_similarity` control volume.
- top-level `watch = true` swaps the debounced hook reindex for a live file watcher (`claudix watch`).

## Verify and troubleshoot

`claudix status` shows file/chunk counts, the embedding model, staleness — SessionStart already reports these, so reach for it only mid-session. `/claudix:doctor` diagnoses the binary, index, provider health, config errors. Logs: `.claudix/logs/index.log`; `RUST_LOG=debug` on any CLI subcommand for verbose output.
