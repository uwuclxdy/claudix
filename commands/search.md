---
description: Search code semantically with claudix.
argument-hint: <query words...> [--top-k N] [--language rust --language python] [--path-prefix src/]
allowed-tools: Bash(claudix:*)
---

Run the claudix semantic search CLI and present the results.
Multi-word queries work without quoting: `/claudix:search where is auth handled`

!`claudix search $ARGUMENTS`

Each result shows `file:line_start-line_end [language] kind name score`. Open the file at the indicated line range to see the full definition.

Valid `--language` values: `rust`, `python`, `javascript`, `typescript`, `go`, `java`, `c`, `cpp`.

Prefer the `search_code` MCP tool directly for programmatic use — this command is for interactive lookup only.

If the index is missing, tell the user to run `/claudix:index` first.
