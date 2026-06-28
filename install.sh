#!/usr/bin/env bash
set -euo pipefail

if ! command -v claude >/dev/null 2>&1; then
  printf 'claudix: claude CLI not found. Open Claude Code and run:\n'
  printf '  /plugin marketplace add uwuclxdy/claudix\n'
  printf '  /plugin install claudix@claudix\n'
  printf 'Then restart Claude Code; the binary downloads on first session.\n'
  exit 0
fi

# refresh the marketplace source (add if missing, update if present), then
# install or update without uninstalling an existing install — preserves
# plugin-local state instead of nuking + reinstalling on every run.
claude plugin marketplace add uwuclxdy/claudix 2>/dev/null \
  || claude plugin marketplace update claudix 2>/dev/null || true
if claude plugin list 2>/dev/null | grep -q 'claudix@claudix'; then
  claude plugin update claudix@claudix || true
else
  claude plugin install claudix@claudix
fi

# Prime the binary cache from the local checkout so the first session
# does not stall on download. The runtime bootstrap is node, so priming uses
# node too; if it is not on PATH here the first session primes it via Claude
# Code's node instead.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
plugin_data="${HOME}/.claude/plugins/data/claudix-claudix"
mkdir -p "$plugin_data"
if command -v node >/dev/null 2>&1; then
  CLAUDE_PLUGIN_ROOT="$script_dir" CLAUDE_PLUGIN_DATA="$plugin_data" \
    node "$script_dir/bin/claudix-bootstrap.js" --install >/dev/null \
    || printf 'claudix: binary install will retry on first session\n' >&2
fi

printf '\nclaudix installed. Restart Claude Code to activate.\n'
