mod filters;
mod git;

use std::fs;
use std::path::{Component, Path, PathBuf};

use crate::config::Config;
use crate::error::{ClaudixError, RecoveryHint, Result};
use crate::types::{FileHash, Language, RelativePath};

pub use filters::PathFilters;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnumeratedFile {
    pub absolute_path: PathBuf,
    pub relative_path: RelativePath,
    pub language: Language,
    pub file_hash: FileHash,
}

#[derive(Debug, Clone)]
pub struct FileEnumerator {
    project_root: PathBuf,
    config: Config,
}

impl FileEnumerator {
    pub fn new(project_root: PathBuf, config: Config) -> Result<Self> {
        let canonical_root = project_root.canonicalize()?;
        Ok(Self {
            project_root: canonical_root,
            config,
        })
    }

    pub fn enumerate(&self) -> Result<Vec<EnumeratedFile>> {
        let repo = git::discover_repository(&self.project_root)?;
        let tracked_and_untracked = git::list_candidate_paths(&repo)?;
        let filters = PathFilters::load(&self.project_root)?;

        let mut files = Vec::new();
        for relative_path in tracked_and_untracked {
            if !filters.is_included(&relative_path) {
                continue;
            }

            if let Some(file) = self.enumerate_one(relative_path)? {
                files.push(file);
            }
        }

        Ok(files)
    }

    fn enumerate_one(&self, relative_path: RelativePath) -> Result<Option<EnumeratedFile>> {
        let absolute_path = self.resolve_relative_path(&relative_path)?;
        let metadata = fs::symlink_metadata(&absolute_path)?;

        if metadata.file_type().is_symlink() && !self.config.indexing.follow_symlinks {
            return Ok(None);
        }

        if metadata.len() > self.max_file_size_bytes() {
            return Ok(None);
        }

        let contents = fs::read(&absolute_path)?;
        let file_hash = hash_bytes(&contents);
        let language = language_for_path(&absolute_path);

        Ok(Some(EnumeratedFile {
            absolute_path,
            relative_path,
            language,
            file_hash,
        }))
    }

    fn resolve_relative_path(&self, relative_path: &RelativePath) -> Result<PathBuf> {
        reject_path_escape(relative_path)?;
        let joined = self.project_root.join(relative_path.to_path_buf());
        ensure_within_root(&self.project_root, &joined)?;
        Ok(joined)
    }

    fn max_file_size_bytes(&self) -> u64 {
        self.config.indexing.max_file_size_kb.saturating_mul(1024)
    }
}

fn language_for_path(path: &Path) -> Language {
    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or_default();
    Language::from_extension(extension)
}

fn hash_bytes(bytes: &[u8]) -> FileHash {
    let digest = xxhash_rust::xxh3::xxh3_128(bytes);
    FileHash(digest.to_be_bytes())
}

fn reject_path_escape(relative_path: &RelativePath) -> Result<()> {
    let path = relative_path.to_path_buf();
    if path.is_absolute() {
        return Err(ClaudixError::PathTraversal {
            path,
            recovery: RecoveryHint("Only enumerate files inside $CLAUDE_PROJECT_DIR"),
        });
    }

    for component in path.components() {
        if matches!(
            component,
            Component::ParentDir | Component::RootDir | Component::Prefix(_)
        ) {
            return Err(ClaudixError::PathTraversal {
                path: relative_path.to_path_buf(),
                recovery: RecoveryHint("Only enumerate files inside $CLAUDE_PROJECT_DIR"),
            });
        }
    }

    Ok(())
}

fn ensure_within_root(root: &Path, path: &Path) -> Result<()> {
    if path.starts_with(root) {
        return Ok(());
    }

    Err(ClaudixError::PathTraversal {
        path: path.to_path_buf(),
        recovery: RecoveryHint("Only enumerate files inside $CLAUDE_PROJECT_DIR"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::types::RelativePath;
    use std::collections::BTreeSet;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::process::Command;
    use tempfile::TempDir;

    struct TestFixture {
        _tempdir: TempDir,
        root: PathBuf,
    }

    impl TestFixture {
        fn new(name: &str) -> std::io::Result<Self> {
            let source = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests")
                .join("fixtures")
                .join(name);
            let tempdir = tempfile::tempdir()?;
            let root = tempdir.path().join(name);
            copy_dir_recursive(&source, &root)?;
            init_git_repo(&root)?;
            Ok(Self {
                _tempdir: tempdir,
                root,
            })
        }

        fn root(&self) -> &Path {
            &self.root
        }
    }

    #[test]
    fn enumerates_tracked_and_untracked_files() {
        let fixture = TestFixture::new("small_rust");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());

        let target_dir = fixture.root().join("target");
        assert!(fs::create_dir_all(&target_dir).is_ok());
        assert!(fs::write(target_dir.join("debug.log"), "ignore me\n").is_ok());
        assert!(
            fs::write(
                fixture.root().join("src/untracked.rs"),
                "pub fn temp() {}\n"
            )
            .is_ok()
        );

        let enumerator = FileEnumerator::new(fixture.root().to_path_buf(), Config::default());
        assert!(enumerator.is_ok());
        let enumerator = enumerator.ok().unwrap_or_else(|| unreachable!());

        let files = enumerator.enumerate();
        assert!(files.is_ok());
        let files = files.ok().unwrap_or_else(|| unreachable!());

        let paths: BTreeSet<_> = files
            .iter()
            .map(|file| file.relative_path.as_str().to_owned())
            .collect();
        assert!(paths.contains("src/lib.rs"));
        assert!(paths.contains("src/untracked.rs"));
        assert!(!paths.contains("target/debug.log"));
    }

    #[test]
    fn indexignore_excludes_and_indexinclude_readds() {
        let fixture = TestFixture::new("ignore_overrides");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());

        let enumerator = FileEnumerator::new(fixture.root().to_path_buf(), Config::default());
        assert!(enumerator.is_ok());
        let enumerator = enumerator.ok().unwrap_or_else(|| unreachable!());

        let files = enumerator.enumerate();
        assert!(files.is_ok());
        let files = files.ok().unwrap_or_else(|| unreachable!());

        let paths: BTreeSet<_> = files
            .iter()
            .map(|file| file.relative_path.as_str().to_owned())
            .collect();
        assert!(paths.contains("src/keep.rs"));
        assert!(paths.contains("src/reinclude.rs"));
        assert!(!paths.contains("src/skip.rs"));
    }

    #[test]
    fn path_traversal_is_rejected() {
        let root = std::env::temp_dir().join("claudix-path-check-root");
        assert!(fs::create_dir_all(&root).is_ok());

        let enumerator = FileEnumerator::new(root.clone(), Config::default());
        assert!(enumerator.is_ok());
        let enumerator = enumerator.ok().unwrap_or_else(|| unreachable!());

        let escaped = RelativePath::new("../escape.rs");
        let error = enumerator.resolve_relative_path(&escaped);
        assert!(matches!(error, Err(ClaudixError::PathTraversal { .. })));
    }

    #[test]
    fn oversized_files_are_skipped() {
        let fixture = TestFixture::new("small_rust");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());

        let oversized_path = fixture.root().join("src/oversized.rs");
        assert!(fs::write(&oversized_path, vec![b'x'; 2048]).is_ok());

        let mut config = Config::default();
        config.indexing.max_file_size_kb = 1;

        let enumerator = FileEnumerator::new(fixture.root().to_path_buf(), config);
        assert!(enumerator.is_ok());
        let enumerator = enumerator.ok().unwrap_or_else(|| unreachable!());

        let files = enumerator.enumerate();
        assert!(files.is_ok());
        let files = files.ok().unwrap_or_else(|| unreachable!());

        let paths: BTreeSet<_> = files
            .iter()
            .map(|file| file.relative_path.as_str().to_owned())
            .collect();
        assert!(!paths.contains("src/oversized.rs"));
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
        run_git(root, ["init"])?;
        run_git(root, ["config", "user.name", "Test User"])?;
        run_git(root, ["config", "user.email", "test@example.com"])?;
        run_git(root, ["add", "."])?;
        run_git(root, ["commit", "-m", "fixture"])?;
        Ok(())
    }

    fn run_git<const N: usize>(root: &Path, args: [&str; N]) -> std::io::Result<()> {
        let status = Command::new("git").current_dir(root).args(args).status()?;
        if status.success() {
            return Ok(());
        }

        Err(std::io::Error::other(format!(
            "git command failed: {:?}",
            args
        )))
    }
}
