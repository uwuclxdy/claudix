//! Drives `scripts/ensure-binary.sh` to verify `development_mode` binary
//! resolution: cargo binary when present, a clear exit-3 hint when missing,
//! and project-config precedence over global.
#![cfg(unix)]

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn script_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("scripts/ensure-binary.sh")
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

/// Run `ensure-binary.sh --check-only` with an isolated HOME, project dir, and
/// CARGO_HOME so the real environment is never touched.
fn run_check_only(home: &Path, project_dir: &Path, cargo_home: &Path) -> Output {
    let output = Command::new("bash")
        .arg(script_path())
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

#[test]
fn development_mode_resolves_cargo_binary() {
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
    let temp = temp_dir();
    let home = temp.path().join("home");
    let project = temp.path().join("project");
    let cargo_home = temp.path().join("cargo");
    assert!(fs::create_dir_all(&project).is_ok());

    // No global config at all; dev mode comes solely from the project file —
    // the common contributor case, since `development_mode_on` reads project first.
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
    let temp = temp_dir();
    let home = temp.path().join("home");
    let project = temp.path().join("project");
    let cargo_home = temp.path().join("cargo");
    assert!(fs::create_dir_all(&project).is_ok());

    // Global turns it on, project turns it off: the dev branch must be skipped.
    write_config(&home, "development_mode = true\n");
    write_config(&project, "development_mode = false\n");
    // A cargo binary exists; dev mode being off means it must NOT be chosen here.
    write_executable(
        &cargo_home.join("bin").join("claudix"),
        "#!/bin/sh\necho stub\n",
    );

    let output = run_check_only(&home, &project, &cargo_home);
    // Dev branch skipped → normal resolution finds no cached release → exit 1,
    // never exit 3 (the dev-missing signal) and never the cargo path.
    assert_eq!(
        output.status.code(),
        Some(1),
        "expected normal cache-miss exit 1, stderr: {}",
        stderr_of(&output)
    );
    // Unambiguous: the dev branch must not have run at all, so its hint is absent.
    assert!(
        !stderr_of(&output).contains("cargo install --path ."),
        "dev branch leaked despite project override"
    );
}

#[test]
fn commented_development_mode_key_is_ignored() {
    let temp = temp_dir();
    let home = temp.path().join("home");
    let project = temp.path().join("project");
    let cargo_home = temp.path().join("cargo");
    assert!(fs::create_dir_all(&project).is_ok());

    // A commented key must never enable dev mode, even with a cargo binary present.
    write_config(&home, "# development_mode = true\n");
    write_executable(
        &cargo_home.join("bin").join("claudix"),
        "#!/bin/sh\necho stub\n",
    );

    let output = run_check_only(&home, &project, &cargo_home);
    assert_eq!(
        output.status.code(),
        Some(1),
        "commented key should leave dev mode off → normal cache-miss exit 1, stderr: {}",
        stderr_of(&output)
    );
    assert!(
        !stderr_of(&output).contains("cargo install --path ."),
        "commented key wrongly triggered the dev branch"
    );
}
