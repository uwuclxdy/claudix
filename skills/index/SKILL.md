---
description: This skill should be used when building or rebuilding the claudix semantic-search index, controlling what gets indexed (indexing gitignored files, excluding paths from the index, `.indexignore`/`.indexinclude` rules), switching embedding models, or recovering from a stale or corrupted index.
---

# claudix index management

## Build or rebuild the index

Prefer `claudix` from PATH. When it is absent (Windows installs create no symlink), run the node bootstrap that ships with the plugin: it lives at `bin/claudix-bootstrap.js` under the plugin root, two directories above this skill's base directory.

```bash
claudix index 2>&1
# without claudix on PATH:
node "<plugin-root>/bin/claudix-bootstrap.js" index 2>&1
```

Pass `--force` (also when this skill is invoked with `--force`) to wipe the store before rebuilding. Needed for:

- embedding model or dimension changes
- schema mismatch after a plugin upgrade
- resetting a corrupted index

`--progress` streams per-file progress. Indexing may take a minute on large repositories; if it errors, run `/claudix:doctor` to diagnose the embedding provider.

## Control what gets indexed

`.indexignore` and `.indexinclude` files use gitignore syntax (globs, `!` negation, comments). Place them at the repo root or in any subdirectory; patterns are relative to the rule file's own directory, like nested `.gitignore` files. Precedence per path: `.indexinclude` beats `.gitignore`, which beats `.indexignore`.

- Exclude tracked files from the index: `.indexignore` (test fixtures, vendored code, minified bundles).
- Index gitignored paths: `.indexinclude` (internal `docs/`, generated code). A one-line `*` inside `docs/.indexinclude` pulls that whole tree. Files without a code chunker index as plain text.
- Placement gotcha: a rule file buried two or more levels inside a gitignored subtree is not discovered; put it at the top of the gitignored directory or at the repo root.
- `[indexing] respect_gitignore = false` in `.claude/claudix.toml` indexes everything gitignored instead of selected subtrees.

Edited rule files apply on the next index run, so run the index command after changing them.

## Verify

`claudix status` (or the `get_index_status` MCP tool) shows file/chunk counts, the embedding model, and staleness. Search reflects the new scope as soon as indexing completes.
