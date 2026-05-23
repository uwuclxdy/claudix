---
description: Show a structural map of indexed files grouped by directory.
allowed-tools: Bash(claudix:*)
---

Run (stderr merged so error messages are visible):
!`claudix overview 2>&1`

To narrow the output to a specific subtree, pass `--path-prefix`:
!`claudix overview --path-prefix src/hooks 2>&1`

If `chunks: 0`, the index is empty — tell the user to run `/claudix:index`.
If the command errors, the embedding provider may be unavailable — tell the user to run `/claudix:doctor`.
