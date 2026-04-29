#!/usr/bin/env bash
set -euo pipefail

PLUGIN_NAME="claudix"
GITHUB_REPO="${CLAUDIX_GITHUB_REPO:-uwuclxdy/claudix}"
# Stable dir independent of plugin version — survives plugin upgrades
STABLE_DIR="${CLAUDIX_HOME:-${XDG_DATA_HOME:-${HOME}/.local/share}/claudix}"
BIN_DIR="${STABLE_DIR}/bin"
VERSION_DIR="${STABLE_DIR}/versions"
VERSION_CHECK_CACHE="${STABLE_DIR}/latest-version"
LOCK_DIR="${STABLE_DIR}/install.lock"
# Allow override for tests (file:// URLs work too)
CLAUDIX_RELEASE_BASE_URL="${CLAUDIX_RELEASE_BASE_URL:-}"
TEMP_DIR=""

MODE="install"
case "${1:-}" in
  --check-only) MODE="check-only" ;;
  --update)     MODE="update"     ;;
  --print-path) MODE="install"    ;;  # legacy alias
esac

log()  { printf 'claudix: %s\n' "$*" >&2; }
fail() { log "$*"; exit 1; }

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    fail "sha256sum or shasum is required to verify the claudix binary"
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
    fail "curl or wget is required to download the claudix binary"
  fi
}

fetch_text() {
  local url="$1"
  if command -v curl >/dev/null 2>&1; then
    curl --fail --location --silent "$url" 2>/dev/null || true
  elif command -v wget >/dev/null 2>&1; then
    wget --quiet --output-document=- "$url" 2>/dev/null || true
  fi
}

detect_platform() {
  if [[ -n "${CLAUDIX_PLATFORM_OVERRIDE:-}" ]]; then
    printf '%s\n' "$CLAUDIX_PLATFORM_OVERRIDE"
    return
  fi
  local os arch
  case "$(uname -s)" in
    Linux*)              os="linux"   ;;
    Darwin*)             os="darwin"  ;;
    MINGW*|MSYS*|CYGWIN*) os="windows" ;;
    *) fail "unsupported operating system: $(uname -s)" ;;
  esac
  case "$(uname -m)" in
    x86_64|amd64)   arch="x86_64"  ;;
    arm64|aarch64)  arch="aarch64" ;;
    *) fail "unsupported CPU architecture: $(uname -m)" ;;
  esac
  local platform="${os}-${arch}"
  case "$platform" in
    linux-x86_64|darwin-aarch64|windows-x86_64) ;;
    darwin-x86_64)
      fail "no prebuilt binary for macOS Intel; install with: cargo install claudix (https://github.com/${GITHUB_REPO})"
      ;;
    *)
      fail "no prebuilt binary for ${platform}; install with: cargo install claudix (https://github.com/${GITHUB_REPO})"
      ;;
  esac
  printf '%s\n' "$platform"
}

asset_name() {
  local platform="$1"
  if [[ "$platform" == windows-* ]]; then
    printf '%s-%s.exe\n' "$PLUGIN_NAME" "$platform"
  else
    printf '%s-%s\n' "$PLUGIN_NAME" "$platform"
  fi
}

semver_gt() {
  # Returns 0 (true) if $1 strictly greater than $2, 1 otherwise
  [[ -z "${1:-}" || -z "${2:-}" ]] && return 1
  [[ "$1" == "$2" ]] && return 1
  local IFS=.
  local -a a=($1) b=($2)
  local i
  for i in 0 1 2; do
    local ai=${a[$i]:-0} bi=${b[$i]:-0}
    (( ai > bi )) && return 0
    (( ai < bi )) && return 1
  done
  return 1
}

fetch_latest_version() {
  # Returns latest GitHub release version; cached for 24 hours to avoid API rate limits
  if [[ -f "$VERSION_CHECK_CACHE" ]]; then
    local now mtime age
    now="$(date +%s)"
    mtime="$(stat -c %Y "$VERSION_CHECK_CACHE" 2>/dev/null \
          || stat -f %m "$VERSION_CHECK_CACHE" 2>/dev/null \
          || echo 0)"
    age=$(( now - mtime ))
    if (( age < 86400 )); then
      cat "$VERSION_CHECK_CACHE"
      return
    fi
  fi
  local version
  version="$(fetch_text "https://api.github.com/repos/${GITHUB_REPO}/releases/latest" \
    | grep '"tag_name"' \
    | sed 's/.*"v\([^"]*\)".*/\1/' \
    | head -1 || true)"
  if [[ -n "$version" ]]; then
    mkdir -p "$STABLE_DIR"
    printf '%s\n' "$version" > "$VERSION_CHECK_CACHE"
    printf '%s\n' "$version"
  fi
}

installed_path() {
  # Prints path of installed binary; exits 1 if not found
  # Priority 1: cargo-installed binary already in PATH
  if command -v "$PLUGIN_NAME" >/dev/null 2>&1; then
    printf '%s\n' "$(command -v "$PLUGIN_NAME")"
    return 0
  fi
  # Priority 2: previously downloaded release binary
  local platform version_file installed_version binary
  platform="$(detect_platform 2>/dev/null)" || return 1
  version_file="${VERSION_DIR}/${PLUGIN_NAME}-${platform}.version"
  [[ -f "$version_file" ]] || return 1
  installed_version="$(cat "$version_file")"
  binary="${BIN_DIR}/${PLUGIN_NAME}-v${installed_version}-${platform}"
  [[ -x "$binary" ]] || return 1
  printf '%s\n' "$binary"
}

acquire_lock() {
  local attempts=0
  mkdir -p "$STABLE_DIR"
  local pid_file="${LOCK_DIR}.pid"
  while ! mkdir "$LOCK_DIR" 2>/dev/null; do
    # Stale lock: if owning PID no longer exists, remove and retry
    if [[ -f "$pid_file" ]] && ! kill -0 "$(cat "$pid_file" 2>/dev/null)" 2>/dev/null; then
      rm -rf "$LOCK_DIR" "$pid_file" 2>/dev/null || true
      continue
    fi
    (( ++attempts > 90 )) && fail "install lock stuck; remove ${LOCK_DIR} and retry"
    sleep 1
  done
  printf '%s\n' "$$" > "$pid_file"
  trap 'rm -rf "${TEMP_DIR:-}" "$pid_file"; rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT
}

try_cargo() {
  local version="${1:-}"
  command -v cargo >/dev/null 2>&1 || return 1
  local pkg="$PLUGIN_NAME"
  [[ -n "$version" ]] && pkg="${PLUGIN_NAME}@${version}"
  log "installing via: cargo install ${pkg}"
  cargo install "$pkg" --quiet 2>/dev/null && return 0
  log "cargo install failed; falling back to GitHub release"
  return 1
}

install_from_release() {
  local platform="$1" version="$2"
  local asset bin_path version_file base_url
  asset="$(asset_name "$platform")"
  bin_path="${BIN_DIR}/${PLUGIN_NAME}-v${version}-${platform}"
  version_file="${VERSION_DIR}/${PLUGIN_NAME}-${platform}.version"
  base_url="${CLAUDIX_RELEASE_BASE_URL:-https://github.com/${GITHUB_REPO}/releases/download/v${version}}"

  mkdir -p "$BIN_DIR" "$VERSION_DIR"
  TEMP_DIR="$(mktemp -d "${STABLE_DIR}/download.XXXXXX")"

  local asset_path="${TEMP_DIR}/${asset}" sums_path="${TEMP_DIR}/SHA256SUMS"
  log "downloading ${asset} ${version}"
  fetch_file "${base_url}/${asset}" "$asset_path"
  fetch_file "${base_url}/SHA256SUMS" "$sums_path"

  local expected actual
  expected="$(awk -v a="$asset" '$2 == a { print $1 }' "$sums_path")"
  [[ -n "$expected" ]] || fail "SHA256SUMS does not contain ${asset}"
  actual="$(sha256_file "$asset_path")"
  [[ "$actual" == "$expected" ]] || fail "checksum mismatch for ${asset}"

  chmod 755 "$asset_path"
  mv "$asset_path" "$bin_path"
  printf '%s\n' "$version" > "$version_file"
  log "installed claudix ${version} for ${platform}"
  printf '%s\n' "$bin_path"
}

do_install() {
  local platform
  platform="$(detect_platform)"
  acquire_lock

  # Re-check after acquiring lock — a concurrent install may have finished
  if binary="$(installed_path 2>/dev/null)"; then
    printf '%s\n' "$binary"
    return 0
  fi

  # Prefer cargo (builds from source, always up to date)
  if try_cargo; then
    printf '%s\n' "$(command -v "$PLUGIN_NAME")"
    return 0
  fi

  # Fall back to prebuilt GitHub release
  local version
  version="$(fetch_latest_version 2>/dev/null || true)"
  [[ -n "$version" ]] || version="${CLAUDIX_VERSION:-0.1.0}"
  install_from_release "$platform" "$version"
}

do_update() {
  local platform
  platform="$(detect_platform 2>/dev/null)" || return 0

  local latest
  latest="$(fetch_latest_version 2>/dev/null || true)"
  [[ -n "$latest" ]] || return 0  # can't reach GitHub, skip silently

  if command -v "$PLUGIN_NAME" >/dev/null 2>&1; then
    local installed
    installed="$("$PLUGIN_NAME" --version 2>/dev/null \
      | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || echo '0.0.0')"
    semver_gt "$latest" "$installed" || return 0
    log "updating claudix ${installed} → ${latest}"
    acquire_lock
    try_cargo "$latest" || install_from_release "$platform" "$latest" >/dev/null
    return 0
  fi

  local version_file="${VERSION_DIR}/${PLUGIN_NAME}-${platform}.version"
  [[ -f "$version_file" ]] || return 0
  local installed
  installed="$(cat "$version_file")"
  semver_gt "$latest" "$installed" || return 0
  log "updating claudix ${installed} → ${latest}"
  acquire_lock
  install_from_release "$platform" "$latest" >/dev/null
}

case "$MODE" in
  check-only)
    if binary="$(installed_path 2>/dev/null)"; then
      printf '%s\n' "$binary"
      exit 0
    fi
    exit 1
    ;;
  update)
    do_update
    ;;
  install)
    if binary="$(installed_path 2>/dev/null)"; then
      printf '%s\n' "$binary"
      exit 0
    fi
    do_install
    ;;
esac
