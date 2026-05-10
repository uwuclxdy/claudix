#!/usr/bin/env bash
# Runs in the background from session-start.sh.
# Checks GitHub releases once per day; installs via cargo if a newer
# version exists; writes a pending-restart flag for the next SessionStart.
set -uo pipefail

CLAUDIX_DATA="${CLAUDIX_HOME:-${XDG_DATA_HOME:-${HOME}/.local/share}/claudix}"
CHECK_CACHE="${CLAUDIX_DATA}/update-check"
PENDING_RESTART="${CLAUDIX_DATA}/pending-restart"
CARGO_BIN="${CARGO_HOME:-${HOME}/.cargo}/bin/claudix"

command -v cargo >/dev/null 2>&1 || exit 0
[[ -x "$CARGO_BIN" ]] || exit 0

# Rate-limit: skip if already checked within the last 24 hours
if [[ -f "$CHECK_CACHE" ]]; then
  now="$(date +%s)"
  mtime="$(stat -c %Y "$CHECK_CACHE" 2>/dev/null || stat -f %m "$CHECK_CACHE" 2>/dev/null || echo 0)"
  age=$(( now - mtime ))
  (( age < 86400 )) && exit 0
fi

installed_ver="$("$CARGO_BIN" -V 2>/dev/null | awk '{print $2}' || echo '')"
[[ -n "$installed_ver" ]] || exit 0

latest_ver="$(curl --fail --silent --max-time 10 \
  -H 'User-Agent: claudix-update-check/1 (https://github.com/uwuclxdy/claudix)' \
  'https://api.github.com/repos/uwuclxdy/claudix/releases/latest' 2>/dev/null \
  | grep '"tag_name"' \
  | sed 's/.*"v\([^"]*\)".*/\1/' | head -1 || echo '')"
[[ -n "$latest_ver" ]] || exit 0

mkdir -p "$CLAUDIX_DATA"
touch "$CHECK_CACHE"

if [[ "$latest_ver" != "$installed_ver" ]]; then
  if cargo install "claudix@${latest_ver}" --quiet 2>/dev/null; then
    printf '%s\n' "$latest_ver" > "$PENDING_RESTART"
  fi
fi
