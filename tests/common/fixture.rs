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

    /// Like `new`, but skips the fixture repo init chain (4 git calls plus the
    /// commit). Use this in tests that don't exercise git enumeration — saves
    /// ~50-100 ms per fixture construction.
    // This file is `include!`d into several test modules; not every copy uses
    // every helper.
    #[allow(dead_code)]
    pub fn without_git(name: &str) -> std::io::Result<Self> {
        let source = fixture_source(name);
        let tempdir = Self::tempdir_outside_git()?;
        let root = tempdir.path().join(name);
        copy_dir_recursive(&source, &root)?;
        // Canonicalize to match production's canonical root (see `new`).
        let root = root.canonicalize()?;

        Ok(Self {
            _tempdir: tempdir,
            root,
        })
    }

    /// The non-git premise is ambient: `TMPDIR` may point inside a git repo
    /// (measured: a foreign `/mnt/scratch/.git` on the dev box), and
    /// `tempfile::tempdir` then yields a dir the hook's walk-up reads as a
    /// repo, silently redding every test that asked for a non-git fixture.
    /// Own the premise: probe each candidate root with the same marker walk
    /// the hook runs, take the first clean root, and refuse loudly when none
    /// qualifies.
    #[allow(dead_code)]
    fn tempdir_outside_git() -> std::io::Result<TempDir> {
        let mut candidates = vec![std::env::temp_dir()];
        #[cfg(unix)]
        candidates.push(PathBuf::from("/tmp"));

        let mut last_create_error = None;
        for root in candidates {
            match tempfile::Builder::new().tempdir_in(&root) {
                Ok(dir) if !Self::inside_git_repo(dir.path()) => return Ok(dir),
                Ok(_) => {}
                Err(error) => last_create_error = Some(error),
            }
        }

        Err(last_create_error.unwrap_or_else(|| {
            std::io::Error::other(
                "every temp root sits inside a git repo; the fixture cannot honor \
                 its non-git premise (unset TMPDIR or point it outside any git repo)",
            )
        }))
    }

    /// Mirror of `enumeration::is_git_repo`: this file compiles under both
    /// `crate::` (lib tests) and `claudix::` (external test crates) path
    /// roots, so it cannot call the crate's predicate directly. A divergence
    /// that flips a verdict reds `non_git_dir_edit_creates_no_state` on any
    /// box whose temp root is polluted, which is exactly the guard this
    /// fixture needs; the two walks agree silently everywhere else.
    fn inside_git_repo(path: &Path) -> bool {
        let mut current = path;
        loop {
            if let Some(valid) = Self::valid_git_marker(current) {
                return valid;
            }
            match current.parent() {
                Some(parent) => current = parent,
                None => return false,
            }
        }
    }

    /// Mirror of `enumeration::valid_git_marker`: a `.git` *file* is a
    /// `gitdir:` pointer (submodules, worktrees) — trust presence; a
    /// directory (or symlink to one) must hold `HEAD` and `objects/`.
    fn valid_git_marker(dir: &Path) -> Option<bool> {
        let dot_git = dir.join(".git");
        let metadata = fs::symlink_metadata(&dot_git).ok()?;
        if metadata.is_file() {
            return Some(true);
        }
        Some(dot_git.join("HEAD").exists() && dot_git.join("objects").exists())
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
        // A gated commit runs the suite inside git's hook environment, which
        // hands GIT_INDEX_FILE to every test subprocess; without this scrub
        // the fixture's git calls retarget the real repo (measured: `git add
        // .` rewrote the checkout's own index, and parallel fixture inits
        // collided on its lock). The fixture runs git in its own throwaway
        // repo, so every redirecting git env var goes.
        .env_remove("GIT_DIR")
        .env_remove("GIT_WORK_TREE")
        .env_remove("GIT_INDEX_FILE")
        .env_remove("GIT_OBJECT_DIRECTORY")
        .env_remove("GIT_COMMON_DIR")
        .env_remove("GIT_CEILING_DIRECTORIES")
        .env_remove("GIT_CONFIG")
        .env_remove("GIT_CONFIG_COUNT")
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
