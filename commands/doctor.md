---
description: Diagnoses claudix binary, index, and embedding provider. Use when search errors, the index won't build, or the embedding provider is unreachable.
allowed-tools: Bash(node:*)
---

Run through the node bootstrap so it resolves without `claudix` on PATH, with stderr merged so action hints stay visible:
!`node "${CLAUDE_PLUGIN_ROOT}/bin/claudix-bootstrap.js" doctor 2>&1`

If `embedding_healthy: false`, tell the user to run `claudix install` to download the bundled model, or set `provider = "http"` in `~/.claude/claudix.toml` for LM Studio/Ollama.

If `embedding_model_mismatch: true` or `index_present: false`, build the index with the active model:
`node "${CLAUDE_PLUGIN_ROOT}/bin/claudix-bootstrap.js" index`

If `development_mode: true`, note that claudix is running the `cargo install` binary at `binary_path` (dev mode), not the downloaded release.
