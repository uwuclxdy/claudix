---
description: Semantic code search by meaning or identifier. Use before implementing to check if logic already exists, or when Grep won't find it.
argument-hint: <query words...> [--top-k N] [--language rust --language python] [--path-prefix src/] [--repo /abs/path]
allowed-tools: Bash(claudix:*)
---

Run the claudix semantic search CLI and present the results.
Multi-word queries work without quoting: `/claudix:search where is auth handled`

!`claudix search $ARGUMENTS`

Each result shows `file:line_start-line_end [language] kind name score`. Open the file at the indicated line range to see the full definition.

Valid `--language` values: `rust`, `python`, `javascript`, `typescript`, `go`, `java`, `c`, `cpp`.

Pass `--repo /absolute/path` (repeatable) to include other already-indexed repos read-only; the active project is always included. When results span more than one repo, each directory group is prefixed with its repo path as `<repo> :: <directory>:`.

Prefer the `search_code` MCP tool directly for programmatic use — this command is for interactive lookup only.

If the index is missing, tell the user to run `/claudix:index` first.
