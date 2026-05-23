---
description: Show index chunk count, file count, model, and staleness. Use when checking whether the index is fresh before relying on search.
allowed-tools: Bash(claudix:*)
---

Run (stderr merged so error messages are visible):
!`claudix status 2>&1`

If `chunks: 0`, the index is empty — tell the user to run `/claudix:index`.
If the command errors, the embedding provider may be unavailable — tell the user to run `/claudix:doctor`.
