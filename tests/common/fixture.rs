use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use tempfile::TempDir;

pub struct TestFixture {
    tempdir: TempDir,
    root: PathBuf,
}

impl TestFixture {
    pub fn new(name: &str) -> std::io::Result<Self> {
        let source = fixture_source(name);
        let tempdir = tempfile::tempdir()?;
        let root = tempdir.path().join(name);
        copy_dir_recursive(&source, &root)?;
        init_git_repo(&root)?;

        Ok(Self { tempdir, root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn tempdir_path(&self) -> &Path {
        self.tempdir.path()
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
    let status = Command::new("git").current_dir(root).args(args).status()?;
    if status.success() {
        return Ok(());
    }

    Err(std::io::Error::other(format!("git command failed: {:?}", args)))
}
