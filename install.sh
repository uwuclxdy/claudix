#!/usr/bin/env bash
set -euo pipefail

if ! command -v cargo >/dev/null 2>&1; then
  printf 'claudix: cargo is required. Install Rust from https://rustup.rs/\n' >&2
  exit 1
fi

REPO="https://github.com/uwuclxdy/claudix"

cargo install --git "$REPO"

if command -v claude >/dev/null 2>&1; then
  claude plugin uninstall claudix@claudix || true
  claude plugin marketplace rm uwuclxdy/claudix || true
  claude plugin marketplace add uwuclxdy/claudix
  claude plugin install claudix@claudix
  printf '\nclaudix installed. Restart Claude Code to activate.\n'
else
  printf '\nclaudix binary installed. Open Claude Code and run:\n'
  printf '  /plugin marketplace add uwuclxdy/claudix\n'
  printf '  /plugin install claudix@claudix\n'
  printf 'Then restart Claude Code.\n'
fi
