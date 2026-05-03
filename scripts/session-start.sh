#!/usr/bin/env bash

"${CLAUDE_PLUGIN_ROOT}/bin/claudix" hook SessionStart || true

CLAUDIX_DATA="${CLAUDIX_HOME:-${XDG_DATA_HOME:-${HOME}/.local/share}/claudix}"
mkdir -p "$CLAUDIX_DATA" || true
nohup "${CLAUDE_PLUGIN_ROOT}/scripts/check-update.sh" >/dev/null 2>&1 &
disown 2>/dev/null || true
exit 0
