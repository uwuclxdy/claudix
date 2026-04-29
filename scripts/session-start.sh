#!/usr/bin/env bash
set -euo pipefail

"${CLAUDE_PLUGIN_ROOT}/bin/claudix" hook SessionStart || exit 0

CLAUDIX_DATA="${CLAUDIX_HOME:-${XDG_DATA_HOME:-${HOME}/.local/share}/claudix}"
mkdir -p "$CLAUDIX_DATA"
nohup "${CLAUDE_PLUGIN_ROOT}/scripts/check-update.sh" >/dev/null 2>&1 &
disown 2>/dev/null || true
