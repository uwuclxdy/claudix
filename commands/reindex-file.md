---
description: Re-embed one file in the claudix index.
argument-hint: <path>
allowed-tools: Bash(${CLAUDE_PLUGIN_ROOT}/bin/claudix:*)
---

Run:
!`${CLAUDE_PLUGIN_ROOT}/bin/claudix reindex-file $ARGUMENTS`
