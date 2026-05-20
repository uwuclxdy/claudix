#!/usr/bin/env bash
# resolve, download, verify, and cache the claudix native binary
# version is pinned to .claude-plugin/plugin.json; updates flow through
# claude code's plugin-upgrade mechanism, not a github-poll loop
set -uo pipefail

PLUGIN_NAME="claudix"
GITHUB_REPO="${CLAUDIX_GITHUB_REPO:-uwuclxdy/claudix}"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
plugin_root_default="$(cd -- "${script_dir}/.." && pwd)"
PLUGIN_ROOT="${CLAUDE_PLUGIN_ROOT:-${plugin_root_default}}"

# stable cache dir survives plugin upgrades
# prefer claude code's official per-plugin data dir; fall back for non-claude contexts
CACHE_DIR="${CLAUDE_PLUGIN_DATA:-${CLAUDIX_HOME:-${XDG_DATA_HOME:-${HOME}/.local/share}/claudix}}"
BIN_DIR="${CACHE_DIR}/bin"
LOCK_DIR="${CACHE_DIR}/install.lock"
CLAUDIX_RELEASE_BASE_URL="${CLAUDIX_RELEASE_BASE_URL:-}"
TEMP_DIR=""

MODE="install"
case "${1:-}" in
  --check-only) MODE="check-only" ;;
  --install)    MODE="install"    ;;
esac

log()  { printf 'claudix: %s\n' "$*" >&2; }
fail() { log "$*"; exit 1; }

read_wanted_version() {
  local manifest="${PLUGIN_ROOT}/.claude-plugin/plugin.json"
  [[ -f "$manifest" ]] || fail "plugin manifest missing: ${manifest}"
  # tolerate single or double-quoted version key, any whitespace
  local v
  v="$(grep -E '"version"[[:space:]]*:' "$manifest" \
       | head -1 \
       | sed -E 's/.*"version"[[:space:]]*:[[:space:]]*"([^"]+)".*/\1/')"
  [[ -n "$v" ]] || fail "could not read version from ${manifest}"
  printf '%s\n' "$v"
}

detect_platform() {
  if [[ -n "${CLAUDIX_PLATFORM_OVERRIDE:-}" ]]; then
    printf '%s\n' "$CLAUDIX_PLATFORM_OVERRIDE"
    return
  fi
  local os arch
  case "$(uname -s)" in
    Linux*)               os="linux"   ;;
    Darwin*)              os="darwin"  ;;
    MINGW*|MSYS*|CYGWIN*) os="windows" ;;
    *) fail "unsupported os: $(uname -s)" ;;
  esac
  case "$(uname -m)" in
    x86_64|amd64)  arch="x86_64"  ;;
    arm64|aarch64) arch="aarch64" ;;
    *) fail "unsupported arch: $(uname -m)" ;;
  esac
  printf '%s-%s\n' "$os" "$arch"
}

asset_name() {
  local platform="$1"
  if [[ "$platform" == windows-* ]]; then
    printf '%s-%s.exe\n' "$PLUGIN_NAME" "$platform"
  else
    printf '%s-%s\n' "$PLUGIN_NAME" "$platform"
  fi
}

binary_path_for() {
  local version="$1" platform="$2"
  printf '%s/%s-v%s-%s\n' "$BIN_DIR" "$PLUGIN_NAME" "$version" "$platform"
}

cargo_bin_path() {
  printf '%s/bin/%s\n' "${CARGO_HOME:-${HOME}/.cargo}" "$PLUGIN_NAME"
}

prebuilt_supported() {
  local platform="$1"
  case "$platform" in
    linux-x86_64|darwin-aarch64|windows-x86_64) return 0 ;;
    *) return 1 ;;
  esac
}

# returns 0 (and prints path) if a usable binary is already cached or a cargo binary exists
# priority: release-cached for the wanted version → cargo binary iff its version matches
installed_path() {
  local version platform binary
  version="$(read_wanted_version 2>/dev/null)" || return 1
  platform="$(detect_platform 2>/dev/null)" || return 1
  binary="$(binary_path_for "$version" "$platform")"
  if [[ -x "$binary" ]]; then
    printf '%s\n' "$binary"
    return 0
  fi
  local cb cb_version
  cb="$(cargo_bin_path)"
  if [[ -x "$cb" ]]; then
    cb_version="$("$cb" -V 2>/dev/null | awk '{print $2}')"
    if [[ "$cb_version" == "$version" ]]; then
      printf '%s\n' "$cb"
      return 0
    fi
  fi
  return 1
}

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    fail "sha256sum or shasum is required to verify the binary"
  fi
}

fetch_file() {
  local url="$1" dest="$2"
  if [[ "$url" == file://* ]]; then
    cp "${url#file://}" "$dest"
    return
  fi
  if command -v curl >/dev/null 2>&1; then
    curl --fail --location --silent --show-error --output "$dest" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget --quiet --output-document="$dest" "$url"
  else
    fail "curl or wget is required to download the binary"
  fi
}

acquire_lock() {
  local attempts=0
  mkdir -p "$CACHE_DIR"
  local pid_file="${LOCK_DIR}.pid"
  while ! mkdir "$LOCK_DIR" 2>/dev/null; do
    if [[ -f "$pid_file" ]] && ! kill -0 "$(cat "$pid_file" 2>/dev/null)" 2>/dev/null; then
      rm -rf "$LOCK_DIR" "$pid_file" 2>/dev/null || true
      continue
    fi
    (( ++attempts > 120 )) && fail "install lock stuck; remove ${LOCK_DIR} and retry"
    sleep 1
  done
  printf '%s\n' "$$" > "$pid_file"
  trap 'rm -rf "${TEMP_DIR:-}" 2>/dev/null || true; rm -f "${LOCK_DIR}.pid" 2>/dev/null || true; rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT
}

# drop all but the latest two cached versions for this platform
gc_old_versions() {
  local platform="$1" keep=2
  [[ -d "$BIN_DIR" ]] || return 0
  # mtime-sorted newest first; everything past `keep` is removed
  # shellcheck disable=SC2012
  local victims
  victims="$(ls -1t "${BIN_DIR}/${PLUGIN_NAME}-v"*"-${platform}" 2>/dev/null \
            | tail -n +$((keep + 1)) || true)"
  [[ -z "$victims" ]] && return 0
  while IFS= read -r f; do
    [[ -n "$f" ]] && rm -f "$f" 2>/dev/null || true
  done <<< "$victims"
}

install_from_release() {
  local version="$1" platform="$2"
  local asset binary base_url
  asset="$(asset_name "$platform")"
  binary="$(binary_path_for "$version" "$platform")"
  base_url="${CLAUDIX_RELEASE_BASE_URL:-https://github.com/${GITHUB_REPO}/releases/download/v${version}}"

  mkdir -p "$BIN_DIR"
  TEMP_DIR="$(mktemp -d "${CACHE_DIR}/download.XXXXXX")"

  local asset_path="${TEMP_DIR}/${asset}" sums_path="${TEMP_DIR}/SHA256SUMS"
  log "downloading ${asset} v${version}"
  fetch_file "${base_url}/${asset}" "$asset_path"
  fetch_file "${base_url}/SHA256SUMS" "$sums_path"

  local expected actual
  expected="$(awk -v a="$asset" '$2 == a { print $1 }' "$sums_path")"
  [[ -n "$expected" ]] || fail "SHA256SUMS does not contain ${asset}"
  actual="$(sha256_file "$asset_path")"
  [[ "$actual" == "$expected" ]] || fail "checksum mismatch for ${asset}"

  chmod 755 "$asset_path"
  mv "$asset_path" "$binary"
  log "installed claudix v${version} for ${platform}"
  gc_old_versions "$platform"
  printf '%s\n' "$binary"
}

try_cargo() {
  local version="$1"
  command -v cargo >/dev/null 2>&1 || return 1
  log "no prebuilt for this target; falling back to cargo install"
  cargo install "${PLUGIN_NAME}@${version}" --quiet 2>/dev/null || return 1
  local cb
  cb="$(cargo_bin_path)"
  [[ -x "$cb" ]] || return 1
  printf '%s\n' "$cb"
}

do_install() {
  local version platform binary
  version="$(read_wanted_version)"
  platform="$(detect_platform)"

  acquire_lock

  # re-check after lock: another invocation may have finished while we waited
  if binary="$(installed_path 2>/dev/null)"; then
    printf '%s\n' "$binary"
    return 0
  fi

  if prebuilt_supported "$platform"; then
    install_from_release "$version" "$platform"
    return 0
  fi

  if binary="$(try_cargo "$version")"; then
    printf '%s\n' "$binary"
    return 0
  fi

  fail "no prebuilt for ${platform} and cargo install failed; install rust from https://rustup.rs and retry"
}

# unconditional cleanup of orphan temp dirs and dead locks
if [[ -d "$CACHE_DIR" ]]; then
  for _stale in "${CACHE_DIR}"/download.*; do
    [[ -d "$_stale" ]] && rm -rf "$_stale" 2>/dev/null || true
  done
  _pid_file="${LOCK_DIR}.pid"
  if [[ -d "$LOCK_DIR" && -f "$_pid_file" ]] \
      && ! kill -0 "$(cat "$_pid_file" 2>/dev/null)" 2>/dev/null; then
    rm -rf "$LOCK_DIR" "$_pid_file" 2>/dev/null || true
  fi
fi

case "$MODE" in
  check-only)
    if binary="$(installed_path 2>/dev/null)"; then
      printf '%s\n' "$binary"
      exit 0
    fi
    exit 1
    ;;
  install)
    if binary="$(installed_path 2>/dev/null)"; then
      printf '%s\n' "$binary"
      exit 0
    fi
    do_install
    ;;
esac
