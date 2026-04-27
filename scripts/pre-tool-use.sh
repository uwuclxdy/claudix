#!/usr/bin/env bash
set -euo pipefail

binary="${CLAUDE_PLUGIN_ROOT}/bin/claudix"
if [[ ! -x "$binary" ]]; then
  exit 0
fi

"$binary" hook PreToolUse || exit 0
