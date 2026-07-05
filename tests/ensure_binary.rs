//! Drives `bin/claudix-bootstrap.js` (via node) to verify `development_mode`
//! binary resolution, the cache-hit symlink relink, the warn-not-delete cargo
//! behavior, and that download failures are recorded to install.log.
#![cfg(unix)]

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn bootstrap_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("bin")
        .join("claudix-bootstrap.js")
}

fn node_available() -> bool {
    Command::new("node")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn temp_dir() -> tempfile::TempDir {
    let temp = tempfile::tempdir();
    assert!(temp.is_ok());
    temp.ok().unwrap_or_else(|| unreachable!())
}

fn write_config(dir: &Path, body: &str) {
    let claude_dir = dir.join(".claude");
    assert!(fs::create_dir_all(&claude_dir).is_ok());
    assert!(fs::write(claude_dir.join("claudix.toml"), body).is_ok());
}

fn write_executable(path: &Path, body: &str) {
    if let Some(parent) = path.parent() {
        assert!(fs::create_dir_all(parent).is_ok());
    }
    assert!(fs::write(path, body).is_ok());
    let metadata = fs::metadata(path);
    assert!(metadata.is_ok());
    let mut perms = metadata
        .ok()
        .unwrap_or_else(|| unreachable!())
        .permissions();
    perms.set_mode(0o755);
    assert!(fs::set_permissions(path, perms).is_ok());
}

/// Run `node claudix-bootstrap.js --check-only` with an isolated HOME, project dir, and
/// CARGO_HOME so the real environment is never touched.
fn run_check_only(home: &Path, project_dir: &Path, cargo_home: &Path) -> Output {
    let output = Command::new("node")
        .arg(bootstrap_path())
        .arg("--check-only")
        .env("HOME", home)
        .env("CLAUDE_PROJECT_DIR", project_dir)
        .env("CARGO_HOME", cargo_home)
        .env("CLAUDIX_HOME", home.join("cache"))
        .env_remove("CLAUDE_PLUGIN_DATA")
        .output();
    assert!(output.is_ok());
    output.ok().unwrap_or_else(|| unreachable!())
}

fn stderr_of(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn stdout_of(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn write_manifest(plugin_root: &Path, version: &str) {
    let dir = plugin_root.join(".claude-plugin");
    assert!(fs::create_dir_all(&dir).is_ok());
    let body = format!(
        "{{\n  \"name\": \"claudix\",\n  \"version\": \"{}\"\n}}\n",
        version
    );
    assert!(fs::write(dir.join("plugin.json"), body).is_ok());
}

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// The version the bootstrap will actually request. Bootstrap ignores the temp
/// plugin root the test sets (no `bin/claudix-bootstrap.js` under it) and reads the
/// committed manifest instead, so the test stub's version is not what gets fetched.
fn committed_plugin_version() -> Option<String> {
    let plugin_json = manifest_dir()
        .join(".claude-plugin")
        .join("plugin.json")
        .to_string_lossy()
        .into_owned();
    let out = Command::new("node")
        .args([
            "-e",
            "console.log(require(process.argv[1]).version)",
            &plugin_json,
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let v = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if v.is_empty() { None } else { Some(v) }
}

/// Run `node claudix-bootstrap.js --install` with an isolated HOME, plugin root, cargo
/// home, and symlink dir so the real environment is never touched. Forces a
/// deterministic platform so the cached-binary path is stable on any unix host.
fn run_install(
    home: &Path,
    project_dir: &Path,
    cargo_home: &Path,
    plugin_root: &Path,
    local_bin: &Path,
) -> Output {
    let output = Command::new("node")
        .arg(bootstrap_path())
        .arg("--install")
        .env("HOME", home)
        .env("CLAUDE_PROJECT_DIR", project_dir)
        .env("CARGO_HOME", cargo_home)
        .env("CLAUDE_PLUGIN_ROOT", plugin_root)
        .env("CLAUDE_PLUGIN_DATA", home.join("cache"))
        .env("CLAUDIX_LOCAL_BIN", local_bin)
        .env("CLAUDIX_PLATFORM_OVERRIDE", "linux-x86_64")
        .output();
    assert!(output.is_ok());
    output.ok().unwrap_or_else(|| unreachable!())
}

#[test]
fn development_mode_resolves_cargo_binary() {
    if !node_available() {
        return;
    }
    let temp = temp_dir();
    let home = temp.path().join("home");
    let project = temp.path().join("project");
    let cargo_home = temp.path().join("cargo");
    assert!(fs::create_dir_all(&project).is_ok());

    write_config(&home, "development_mode = true\n");
    let cargo_bin = cargo_home.join("bin").join("claudix");
    write_executable(&cargo_bin, "#!/bin/sh\necho stub\n");

    let output = run_check_only(&home, &project, &cargo_home);
    assert!(
        output.status.success(),
        "expected success, stderr: {}",
        stderr_of(&output)
    );
    assert_eq!(stdout_of(&output).trim(), cargo_bin.to_string_lossy());
}

#[test]
fn development_mode_enabled_only_in_project_config() {
    if !node_available() {
        return;
    }
    let temp = temp_dir();
    let home = temp.path().join("home");
    let project = temp.path().join("project");
    let cargo_home = temp.path().join("cargo");
    assert!(fs::create_dir_all(&project).is_ok());

    // No global config at all; dev mode comes solely from the project file.
    write_config(&project, "development_mode = true\n");
    let cargo_bin = cargo_home.join("bin").join("claudix");
    write_executable(&cargo_bin, "#!/bin/sh\necho stub\n");

    let output = run_check_only(&home, &project, &cargo_home);
    assert!(
        output.status.success(),
        "expected success, stderr: {}",
        stderr_of(&output)
    );
    assert_eq!(stdout_of(&output).trim(), cargo_bin.to_string_lossy());
}

#[test]
fn development_mode_missing_cargo_binary_exits_3() {
    if !node_available() {
        return;
    }
    let temp = temp_dir();
    let home = temp.path().join("home");
    let project = temp.path().join("project");
    let cargo_home = temp.path().join("cargo");
    assert!(fs::create_dir_all(&project).is_ok());

    write_config(&home, "development_mode = true\n");

    let output = run_check_only(&home, &project, &cargo_home);
    assert_eq!(
        output.status.code(),
        Some(3),
        "expected exit 3 for missing cargo binary, stderr: {}",
        stderr_of(&output)
    );
    assert!(
        stderr_of(&output).contains("cargo install --path ."),
        "expected install hint in stderr"
    );
    assert!(stdout_of(&output).trim().is_empty());
}

#[test]
fn project_config_disables_global_development_mode() {
    if !node_available() {
        return;
    }
    let temp = temp_dir();
    let home = temp.path().join("home");
    let project = temp.path().join("project");
    let cargo_home = temp.path().join("cargo");
    assert!(fs::create_dir_all(&project).is_ok());

    // Global turns it on, project turns it off: the dev branch must be skipped.
    write_config(&home, "development_mode = true\n");
    write_config(&project, "development_mode = false\n");
    write_executable(
        &cargo_home.join("bin").join("claudix"),
        "#!/bin/sh\necho stub\n",
    );

    let output = run_check_only(&home, &project, &cargo_home);
    // Dev branch skipped -> normal resolution finds no cached release -> exit 1,
    // never exit 3 (the dev-missing signal) and never the cargo path.
    assert_eq!(
        output.status.code(),
        Some(1),
        "expected normal cache-miss exit 1, stderr: {}",
        stderr_of(&output)
    );
    assert!(
        !stderr_of(&output).contains("cargo install --path ."),
        "dev branch leaked despite project override"
    );
}

#[test]
fn commented_development_mode_key_is_ignored() {
    if !node_available() {
        return;
    }
    let temp = temp_dir();
    let home = temp.path().join("home");
    let project = temp.path().join("project");
    let cargo_home = temp.path().join("cargo");
    assert!(fs::create_dir_all(&project).is_ok());

    write_config(&home, "# development_mode = true\n");
    write_executable(
        &cargo_home.join("bin").join("claudix"),
        "#!/bin/sh\necho stub\n",
    );

    let output = run_check_only(&home, &project, &cargo_home);
    assert_eq!(
        output.status.code(),
        Some(1),
        "commented key should leave dev mode off -> normal cache-miss exit 1, stderr: {}",
        stderr_of(&output)
    );
    assert!(
        !stderr_of(&output).contains("cargo install --path ."),
        "commented key wrongly triggered the dev branch"
    );
}

#[test]
fn install_relinks_cached_binary_on_fast_path() {
    if !node_available() {
        return;
    }
    let temp = temp_dir();
    let home = temp.path().join("home");
    let project = temp.path().join("project");
    let cargo_home = temp.path().join("cargo");
    let plugin_root = temp.path().join("plugin");
    let local_bin = temp.path().join("bin");
    assert!(fs::create_dir_all(&project).is_ok());

    // bootstrap ignores the temp plugin root (no `bin/claudix-bootstrap.js` under
    // it) and reads the committed manifest, so the cache file must match that
    // version or this turns into a download instead of a fast-path hit.
    let version = committed_plugin_version().expect("committed plugin.json version");
    write_manifest(&plugin_root, &version);

    // a cached release binary for the wanted version + platform — the fast-path
    // hit. `--install` must repoint the symlink even when nothing is downloaded.
    let cache_bin = home
        .join("cache")
        .join("bin")
        .join(format!("claudix-v{version}-linux-x86_64"));
    write_executable(&cache_bin, "#!/bin/sh\nexit 0\n");

    let output = run_install(&home, &project, &cargo_home, &plugin_root, &local_bin);
    assert!(
        output.status.success(),
        "expected success, stderr: {}",
        stderr_of(&output)
    );

    let target = fs::read_link(local_bin.join("claudix"));
    assert!(
        target.is_ok(),
        "expected ~/.local/bin/claudix symlink on cache-hit, stderr: {}",
        stderr_of(&output)
    );
    assert_eq!(
        target
            .ok()
            .unwrap_or_else(|| unreachable!())
            .to_string_lossy(),
        cache_bin.to_string_lossy()
    );
}

#[test]
fn install_leaves_intentional_cargo_binary_in_place() {
    if !node_available() {
        return;
    }
    let temp = temp_dir();
    let home = temp.path().join("home");
    let project = temp.path().join("project");
    let cargo_home = temp.path().join("cargo");
    let plugin_root = temp.path().join("plugin");
    let local_bin = temp.path().join("bin");
    assert!(fs::create_dir_all(&project).is_ok());

    let version = committed_plugin_version().expect("committed plugin.json version");
    write_manifest(&plugin_root, &version);

    // cached release binary for the wanted version — release wins on the fast path.
    let cache_bin = home
        .join("cache")
        .join("bin")
        .join(format!("claudix-v{version}-linux-x86_64"));
    write_executable(&cache_bin, "#!/bin/sh\nexit 0\n");

    // a cargo-installed claudix at a different version — intentionally installed,
    // must be left in place (warned, not deleted).
    let cargo_bin = cargo_home.join("bin").join("claudix");
    write_executable(
        &cargo_bin,
        "#!/bin/sh\n[ \"$1\" = \"-V\" ] && echo \"claudix 0.1.5\" || true\n",
    );

    let output = run_install(&home, &project, &cargo_home, &plugin_root, &local_bin);
    assert!(
        output.status.success(),
        "expected success, stderr: {}",
        stderr_of(&output)
    );

    assert!(
        cargo_bin.exists(),
        "cargo-installed claudix was deleted; stderr: {}",
        stderr_of(&output)
    );
    assert!(
        stderr_of(&output).contains("left in place"),
        "expected left-in-place warning, stderr: {}",
        stderr_of(&output)
    );
}

#[test]
fn install_log_records_error_on_failed_download() {
    if !node_available() {
        return;
    }
    let temp = temp_dir();
    let home = temp.path().join("home");
    let project = temp.path().join("project");
    let cargo_home = temp.path().join("cargo"); // empty -> no cargo fallback
    let plugin_root = temp.path().join("plugin");
    let local_bin = temp.path().join("bin");
    assert!(fs::create_dir_all(&project).is_ok());
    write_manifest(&plugin_root, "0.1.6");

    // point the release URL at a path nothing serves -> download fails. AU-1: the
    // failure must be recorded to install.log so /doctor can surface it instead of
    // leaving MCP silently dead across sessions.
    let output = Command::new("node")
        .arg(bootstrap_path())
        .arg("--install")
        .env("HOME", &home)
        .env("CLAUDE_PROJECT_DIR", &project)
        .env("CARGO_HOME", &cargo_home)
        .env("CLAUDE_PLUGIN_ROOT", &plugin_root)
        .env("CLAUDE_PLUGIN_DATA", home.join("cache"))
        .env("CLAUDIX_LOCAL_BIN", &local_bin)
        .env("CLAUDIX_PLATFORM_OVERRIDE", "linux-x86_64")
        .env(
            "CLAUDIX_RELEASE_BASE_URL",
            "file:///nonexistent/claudix-release",
        )
        .output();
    let output = output.expect("node spawn");
    assert!(
        !output.status.success(),
        "expected install to fail on dead URL, stderr: {}",
        stderr_of(&output)
    );

    let log = fs::read_to_string(home.join("cache").join("install.log"))
        .expect("install.log should exist after a failed download");
    assert!(
        log.lines().any(|l| l.starts_with("error:")),
        "expected an `error:` line in install.log, got: {}",
        log
    );
}

#[test]
fn bootstrap_source_never_corrupts_mcp_stdio() {
    // Static guard: the bootstrap spawns the native binary with stdio:'inherit' for MCP,
    // so it must never write to stdout (console.log) or read process.stdin itself —
    // either would corrupt the JSON-RPC handshake.
    let src = fs::read_to_string(bootstrap_path()).expect("bootstrap source readable");
    assert!(
        !src.contains("console.log"),
        "bootstrap writes to stdout via console.log (MCP stdio corruption risk)"
    );
    assert!(
        !src.contains("process.stdin."),
        "bootstrap reads process.stdin (MCP handshake corruption risk)"
    );
}
