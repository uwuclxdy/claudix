#!/usr/bin/env bash
set -euo pipefail

binary="${CLAUDE_PLUGIN_ROOT}/bin/claudix"
if [[ ! -x "$binary" ]]; then
  printf '%s\n' 'claudix: binary not yet installed; run /claudix:doctor' >&2
  exit 0
fi

"$binary" hook SessionStart || exit 0
