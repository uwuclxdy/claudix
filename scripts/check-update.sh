#!/usr/bin/env bash
# Runs in the background from session-start.sh.
# Checks GitHub releases once per day; installs via cargo if a newer
# version exists; writes a pending-restart flag for the next SessionStart.
set -uo pipefail

CLAUDIX_DATA="${CLAUDIX_HOME:-${XDG_DATA_HOME:-${HOME}/.local/share}/claudix}"
CHECK_CACHE="${CLAUDIX_DATA}/update-check"
PENDING_RESTART="${CLAUDIX_DATA}/pending-restart"
UPDATE_LOG="${CLAUDIX_DATA}/update.log"
CARGO_HOME_DIR="${CARGO_HOME:-${HOME}/.cargo}"

command -v cargo >/dev/null 2>&1 || exit 0

# Resolve the active claudix on PATH so we update the binary the user
# actually runs, not whatever happens to live at ~/.cargo/bin. Auto-update
# only makes sense for cargo-installed binaries; skip silently for distro
# packages, homebrew, or other deployments.
ACTIVE_BIN="$(command -v claudix 2>/dev/null || true)"
[[ -n "${ACTIVE_BIN}" ]] || exit 0

resolved_bin="$(readlink -f "${ACTIVE_BIN}" 2>/dev/null || echo "${ACTIVE_BIN}")"
resolved_cargo_home="$(readlink -f "${CARGO_HOME_DIR}" 2>/dev/null || echo "${CARGO_HOME_DIR}")"
case "${resolved_bin}" in
  "${resolved_cargo_home}"/bin/*) ;;
  *) exit 0 ;;
esac

mkdir -p "${CLAUDIX_DATA}"

log_failure() {
  printf '%s claudix auto-update: %s\n' "$(date -Iseconds 2>/dev/null || date)" "$*" \
    >>"${UPDATE_LOG}" 2>/dev/null || true
}

# Rate-limit: skip if already checked within the last 24 hours
if [[ -f "${CHECK_CACHE}" ]]; then
  now="$(date +%s)"
  mtime="$(stat -c %Y "${CHECK_CACHE}" 2>/dev/null || stat -f %m "${CHECK_CACHE}" 2>/dev/null || echo 0)"
  age=$(( now - mtime ))
  (( age < 86400 )) && exit 0
fi

installed_ver="$("${ACTIVE_BIN}" -V 2>/dev/null | awk '{print $2}' || echo '')"
[[ -n "${installed_ver}" ]] || exit 0

latest_ver="$(curl --fail --silent --max-time 10 \
  -H 'User-Agent: claudix-update-check/1 (https://github.com/uwuclxdy/claudix)' \
  'https://api.github.com/repos/uwuclxdy/claudix/releases/latest' 2>/dev/null \
  | grep '"tag_name"' \
  | sed 's/.*"v\([^"]*\)".*/\1/' | head -1 || echo '')"
[[ -n "${latest_ver}" ]] || exit 0

if [[ "${latest_ver}" == "${installed_ver}" ]]; then
  # Up-to-date: record the success so we don't recheck for 24h.
  touch "${CHECK_CACHE}"
  exit 0
fi

# Capture cargo output so a failure surfaces in the log instead of
# silently being swallowed alongside the rate-limit touch.
if cargo_output="$(cargo install "claudix@${latest_ver}" 2>&1)"; then
  touch "${CHECK_CACHE}"
  printf '%s\n' "${latest_ver}" > "${PENDING_RESTART}"
else
  log_failure "cargo install claudix@${latest_ver} failed: ${cargo_output}"
  exit 0
fi
