---
description: Build or refresh the claudix index.
allowed-tools: Bash(claudix:*)
---

Run (stderr merged so error messages are visible):
!`claudix index 2>&1`

This may take a minute for large repositories.
If it errors, run `/claudix:doctor` to diagnose the embedding provider.
