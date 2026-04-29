#!/usr/bin/env bash
set -euo pipefail

PLUGIN_NAME="claudix"
DEFAULT_VERSION="0.1.0"
WANT_VERSION="${CLAUDIX_VERSION:-$DEFAULT_VERSION}"
GITHUB_REPO="${CLAUDIX_GITHUB_REPO:-uwuclxdy/claudix}"
RELEASE_BASE_URL="${CLAUDIX_RELEASE_BASE_URL:-https://github.com/${GITHUB_REPO}/releases/download/v${WANT_VERSION}}"
PLUGIN_DATA="${CLAUDE_PLUGIN_DATA:?CLAUDE_PLUGIN_DATA is required}"
BIN_DIR="${PLUGIN_DATA}/bin"
LOCK_DIR="${PLUGIN_DATA}/install.lock"
VERSION_DIR="${PLUGIN_DATA}/versions"
PRINT_PATH=false
TEMP_DIR=""

if [[ "${1:-}" == "--print-path" ]]; then
  PRINT_PATH=true
fi

log() {
  printf 'claudix: %s\n' "$*" >&2
}

fail() {
  log "$*"
  exit 1
}

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    fail "sha256sum or shasum is required to verify the claudix binary"
  fi
}

download() {
  local url="$1"
  local destination="$2"

  if [[ "$url" == file://* ]]; then
    cp "${url#file://}" "$destination"
    return
  fi

  if command -v curl >/dev/null 2>&1; then
    curl --fail --location --silent --show-error --output "$destination" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget --quiet --output-document="$destination" "$url"
  else
    fail "curl or wget is required to download the claudix binary"
  fi
}

detect_platform() {
  if [[ -n "${CLAUDIX_PLATFORM_OVERRIDE:-}" ]]; then
    printf '%s\n' "$CLAUDIX_PLATFORM_OVERRIDE"
    return
  fi

  local os arch
  case "$(uname -s)" in
    Linux*) os="linux" ;;
    Darwin*) os="darwin" ;;
    MINGW*|MSYS*|CYGWIN*) os="windows" ;;
    *) fail "unsupported operating system: $(uname -s)" ;;
  esac

  case "$(uname -m)" in
    x86_64|amd64) arch="x86_64" ;;
    arm64|aarch64) arch="aarch64" ;;
    *) fail "unsupported CPU architecture: $(uname -m)" ;;
  esac

  printf '%s-%s\n' "$os" "$arch"
}

binary_name_for_platform() {
  local platform="$1"
  if [[ "$platform" == windows-* ]]; then
    printf '%s-%s.exe\n' "$PLUGIN_NAME" "$platform"
  else
    printf '%s-%s\n' "$PLUGIN_NAME" "$platform"
  fi
}

acquire_lock() {
  local attempts=0
  mkdir -p "$PLUGIN_DATA"
  while ! mkdir "$LOCK_DIR" 2>/dev/null; do
    attempts=$((attempts + 1))
    if [[ $attempts -gt 90 ]]; then
      fail "another install appears stuck; remove ${LOCK_DIR} and retry"
    fi
    sleep 1
  done
  trap 'rm -rf "$TEMP_DIR"; rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT
}

install_binary() {
  local platform="$1"
  local asset="$2"
  local binary_path="$3"
  local version_file="$4"

  mkdir -p "$BIN_DIR" "$VERSION_DIR"
  TEMP_DIR="$(mktemp -d "${PLUGIN_DATA}/download.XXXXXX")"

  local asset_path="${TEMP_DIR}/${asset}"
  local sums_path="${TEMP_DIR}/SHA256SUMS"

  log "downloading ${asset} from ${RELEASE_BASE_URL}"
  download "${RELEASE_BASE_URL}/${asset}" "$asset_path"
  download "${RELEASE_BASE_URL}/SHA256SUMS" "$sums_path"

  local expected actual
  expected="$(awk -v asset="$asset" '$2 == asset { print $1 }' "$sums_path")"
  if [[ -z "$expected" ]]; then
    fail "SHA256SUMS does not contain ${asset}"
  fi
  actual="$(sha256_file "$asset_path")"
  if [[ "$actual" != "$expected" ]]; then
    fail "checksum mismatch for ${asset}"
  fi

  chmod 755 "$asset_path"
  mv "$asset_path" "$binary_path"
  printf '%s\n' "$WANT_VERSION" > "$version_file"
  log "installed claudix ${WANT_VERSION} for ${platform}"
}

platform="$(detect_platform)"
asset="$(binary_name_for_platform "$platform")"
binary_path="${BIN_DIR}/${PLUGIN_NAME}-v${WANT_VERSION}-${platform}"
version_file="${VERSION_DIR}/${PLUGIN_NAME}-${platform}.version"

if [[ -x "$binary_path" && -f "$version_file" && "$(cat "$version_file")" == "$WANT_VERSION" ]]; then
  if [[ "$PRINT_PATH" == true ]]; then
    printf '%s\n' "$binary_path"
  fi
  exit 0
fi

acquire_lock

if [[ -x "$binary_path" && -f "$version_file" && "$(cat "$version_file")" == "$WANT_VERSION" ]]; then
  if [[ "$PRINT_PATH" == true ]]; then
    printf '%s\n' "$binary_path"
  fi
  exit 0
fi

install_binary "$platform" "$asset" "$binary_path" "$version_file"

if [[ "$PRINT_PATH" == true ]]; then
  printf '%s\n' "$binary_path"
fi
