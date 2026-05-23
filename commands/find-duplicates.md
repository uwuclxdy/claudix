---
description: Find near-duplicate or copy-pasted code chunks within the active repo or across indexed repos.
allowed-tools: Bash(claudix:*)
---

Run (stderr merged so error messages are visible):
!`claudix find-duplicates 2>&1`

To raise or lower the similarity threshold (default 0.85):
!`claudix find-duplicates --min-similarity 0.95 2>&1`

To cap the number of pairs returned (default 50):
!`claudix find-duplicates --limit 10 2>&1`

To scan additional already-indexed repos alongside the active project (active project is NOT auto-added when --repo is used — list everything you want):
!`claudix find-duplicates --repo /path/to/other-repo --repo /path/to/another-repo 2>&1`

If `no near-duplicate pairs found`, the threshold may be too high — lower `--min-similarity`.
If any repo shows a warning line, that repo is not indexed or uses a different embedding model — run `/claudix:index` in that repo first.
