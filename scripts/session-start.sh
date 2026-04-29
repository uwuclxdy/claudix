#!/usr/bin/env bash
set -euo pipefail

"${CLAUDE_PLUGIN_ROOT}/bin/claudix" hook SessionStart || exit 0
