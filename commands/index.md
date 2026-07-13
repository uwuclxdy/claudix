---
description: Build or rebuild the claudix index. Use when the index is empty or stale, after adding or deleting many files, or with --force after an embedding model switch or a corrupted index.
argument-hint: [--force]
allowed-tools: Bash(node:*)
---

Run through the node bootstrap so it resolves without `claudix` on PATH, with stderr merged so error messages stay visible:
!`node "${CLAUDE_PLUGIN_ROOT}/bin/claudix-bootstrap.js" index $ARGUMENTS 2>&1`

`--force` wipes the store before rebuilding; required after changing the embedding model, on a schema mismatch, or to reset a corrupted index.

This may take a minute for large repositories.
If it errors, run `/claudix:doctor` to diagnose the embedding provider.
