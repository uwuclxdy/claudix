---
description: Delete all indexed chunks for the active project. Use before switching embedding models or resetting a corrupted index.
allowed-tools: Bash(claudix:*)
---

Run:
!`claudix clear 2>&1`

After clearing, run `/claudix:index` to rebuild the index.
