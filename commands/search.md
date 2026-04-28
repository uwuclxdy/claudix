---
description: Search code semantically with claudix.
argument-hint: <query> [--top-k N] [--language rust --language python]
allowed-tools: Bash(${CLAUDE_PLUGIN_ROOT}/bin/claudix:*)
---

Run:
!`${CLAUDE_PLUGIN_ROOT}/bin/claudix search $ARGUMENTS`
