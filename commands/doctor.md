---
description: Diagnose claudix binary, index, and embedding provider. Use when search errors, the index won't build, or the embedding provider is unreachable.
allowed-tools: Bash(claudix:*)
---

Run (stderr merged so action hints are visible):
!`claudix doctor 2>&1`

If `embedding_healthy: false`, tell the user to run `claudix install` to download the bundled model, or set `provider = "http"` in `~/.claude/claudix.toml` for LM Studio/Ollama.

If `index_present: false`, tell the user to run `/claudix:index`.
