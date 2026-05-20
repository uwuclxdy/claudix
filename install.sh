#!/usr/bin/env bash
set -euo pipefail

if ! command -v claude >/dev/null 2>&1; then
  printf 'claudix: claude CLI not found. Open Claude Code and run:\n'
  printf '  /plugin marketplace add uwuclxdy/claudix\n'
  printf '  /plugin install claudix@claudix\n'
  printf 'Then restart Claude Code; the binary downloads on first session.\n'
  exit 0
fi

claude plugin uninstall claudix@claudix || true
claude plugin marketplace rm claudix || true
claude plugin marketplace add uwuclxdy/claudix
claude plugin install claudix@claudix

# Prime the binary cache from the local checkout so the first session
# does not stall on download. Uses the documented plugin-data path,
# avoiding any dependency on `node` or `jq`.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
plugin_data="${HOME}/.claude/plugins/data/claudix-claudix"
mkdir -p "$plugin_data"
CLAUDE_PLUGIN_ROOT="$script_dir" CLAUDE_PLUGIN_DATA="$plugin_data" \
  bash "$script_dir/scripts/ensure-binary.sh" --install >/dev/null \
  || printf 'claudix: binary install will retry on first session\n' >&2

printf '\nclaudix installed. Restart Claude Code to activate.\n'
