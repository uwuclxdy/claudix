#!/usr/bin/env bash
set -euo pipefail

BUNDLED=false
for arg in "$@"; do
  case "$arg" in
    --bundled) BUNDLED=true ;;
  esac
done

if ! command -v cargo >/dev/null 2>&1; then
  printf 'claudix: cargo is required. Install Rust from https://rustup.rs/\n' >&2
  exit 1
fi

if [[ "$BUNDLED" == "true" ]]; then
  cargo install claudix
else
  cargo install claudix --no-default-features
fi

if command -v claude >/dev/null 2>&1; then
  claude plugin marketplace add uwuclxdy/claudix
  claude plugin install claudix@uwuclxdy
  printf '\nclaudix installed. Restart Claude Code to activate.\n'
else
  printf '\nclaudix binary installed. Open Claude Code and run:\n'
  printf '  claude plugin marketplace add uwuclxdy/claudix\n'
  printf '  claude plugin install claudix@uwuclxdy\n'
  printf 'Then restart Claude Code.\n'
fi
