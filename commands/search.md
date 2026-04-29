---
description: Search code semantically with claudix.
argument-hint: <query> [--top-k N] [--language rust --language python] [--path-prefix src/]
allowed-tools: Bash(${CLAUDE_PLUGIN_ROOT}/bin/claudix:*)
---

Run the claudix semantic search CLI and present the results:

!`${CLAUDE_PLUGIN_ROOT}/bin/claudix search $ARGUMENTS`

Each result shows `file:line_start-line_end [language] kind name score`. Open the file at the indicated line range to see the full definition.

Valid `--language` values: `rust`, `python`, `javascript`, `typescript`, `go`, `java`, `c`, `cpp`.

If the index is missing, tell the user to run `/claudix:index` first.
