use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use tempfile::TempDir;

pub struct TestFixture {
    _tempdir: TempDir,
    root: PathBuf,
}

impl TestFixture {
    pub fn new(name: &str) -> std::io::Result<Self> {
        let source = fixture_source(name);
        let tempdir = tempfile::tempdir()?;
        let root = tempdir.path().join(name);
        copy_dir_recursive(&source, &root)?;
        // Mirror production: `Store::new` canonicalizes the project root, so a
        // test comparing a path against this root must use the same canonical
        // spelling. On macOS (/var → /private/var) and Windows (\\?\ verbatim +
        // 8.3 short names) the raw tempdir path diverges from its canonical form
        // and breaks strip_prefix / substring checks that pass on plain /tmp.
        let root = root.canonicalize()?;
        init_git_repo(&root)?;

        Ok(Self {
            _tempdir: tempdir,
            root,
        })
    }

    /// Like `new`, but skips the 5 git subprocess calls. Use this in tests that
    /// don't exercise git enumeration — saves ~50-100 ms per fixture construction.
    // This file is `include!`d into several test modules; not every copy uses
    // every helper.
    #[allow(dead_code)]
    pub fn without_git(name: &str) -> std::io::Result<Self> {
        let source = fixture_source(name);
        let tempdir = tempfile::tempdir()?;
        let root = tempdir.path().join(name);
        copy_dir_recursive(&source, &root)?;
        // Canonicalize to match production's canonical root (see `new`).
        let root = root.canonicalize()?;

        Ok(Self {
            _tempdir: tempdir,
            root,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Decompose into `(TempDir, root_path)` so the caller can store the
    /// `TempDir` guard without keeping the whole `TestFixture` alive.
    #[allow(dead_code)]
    pub fn into_parts(self) -> (TempDir, PathBuf) {
        (self._tempdir, self.root)
    }
}

fn fixture_source(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

fn copy_dir_recursive(source: &Path, destination: &Path) -> std::io::Result<()> {
    fs::create_dir_all(destination)?;

    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let source_path = entry.path();
        let destination_path = destination.join(entry.file_name());
        let file_type = entry.file_type()?;

        if file_type.is_dir() {
            copy_dir_recursive(&source_path, &destination_path)?;
        } else {
            fs::copy(&source_path, &destination_path)?;
        }
    }

    Ok(())
}

fn init_git_repo(root: &Path) -> std::io::Result<()> {
    run(root, ["init"])?;
    run(root, ["config", "user.name", "Test User"])?;
    run(root, ["config", "user.email", "test@example.com"])?;
    run(root, ["add", "."])?;
    run(root, ["commit", "-m", "fixture"])?;
    Ok(())
}

fn run<const N: usize>(root: &Path, args: [&str; N]) -> std::io::Result<()> {
    // Neutralise the developer's global/system git config for the throwaway
    // fixture repo. The user's `commit.gpgsign = true` otherwise routes every
    // fixture commit through gpg-agent (eddsa signing), adding 10-30s per
    // fixture and intermittently failing under parallel test load. Pointing
    // both config scopes at a path that does not exist makes git read them as
    // empty (portable across platforms); the local `user.name`/`user.email`
    // we set still land in `.git/config`.
    let empty_config = root.join(".claudix-test-empty-gitconfig");
    // `output()` (not `status()`) so fixture git chatter — init hints, commit
    // summaries — never pollutes test output; stderr surfaces only on failure.
    let output = Command::new("git")
        .current_dir(root)
        .env("GIT_CONFIG_GLOBAL", &empty_config)
        .env("GIT_CONFIG_SYSTEM", &empty_config)
        .args(args)
        .output()?;
    if output.status.success() {
        return Ok(());
    }

    Err(std::io::Error::other(format!(
        "git command failed: {:?}: {}",
        args,
        String::from_utf8_lossy(&output.stderr)
    )))
}
