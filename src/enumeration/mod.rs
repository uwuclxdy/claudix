mod filters;
mod git;

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use crate::config::Config;
use crate::error::{ClaudixError, RecoveryHint, Result};
use crate::types::{FileHash, Language, RelativePath};
use crate::{IndexFileStatus, IndexProgress};

pub use filters::{PathFilters, WatchFilter};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnumeratedFile {
    pub absolute_path: PathBuf,
    pub relative_path: RelativePath,
    pub language: Language,
    pub file_hash: FileHash,
    pub force_indexed: bool,
    /// Raw file bytes pre-read by the caller; `None` in the bulk-index path.
    /// When present, `enumerate_one` skips the disk read entirely.
    pub content: Option<Vec<u8>>,
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

    pub fn enumerate(&self, progress: &mut dyn IndexProgress) -> Result<Vec<EnumeratedFile>> {
        let repo = git::discover_repository(&self.project_root)?;
        let filters = PathFilters::load(&self.project_root)?;

        let mut candidates: BTreeSet<RelativePath> =
            git::list_candidate_paths(&repo)?.into_iter().collect();

        // `.indexinclude` can re-include paths the gitignore-aware walk pruned
        // before any filter saw them (e.g. a gitignored `docs/` tree). Add back
        // the ones an include rule matches from a gitignore-blind deep walk;
        // skipped when no include rule exists so the common case pays nothing.
        if filters.has_includes() {
            for relative_path in git::list_all_paths(&repo)? {
                if filters.is_force_included(&relative_path) {
                    candidates.insert(relative_path);
                }
            }
        }

        let mut files = Vec::new();
        for relative_path in candidates {
            if !filters.is_included(&relative_path) {
                progress.file(
                    &relative_path,
                    IndexFileStatus::Skipped("excluded by index filters"),
                )?;
                continue;
            }

            let force_indexed = filters.is_force_included(&relative_path);
            match self.enumerate_one(relative_path.clone(), force_indexed)? {
                Some(file) => files.push(file),
                None => {
                    progress.file(
                        &relative_path,
                        IndexFileStatus::Skipped("not an indexable file"),
                    )?;
                }
            }
        }

        Ok(files)
    }

    pub(crate) fn enumerate_one(
        &self,
        relative_path: RelativePath,
        force_indexed: bool,
    ) -> Result<Option<EnumeratedFile>> {
        self.enumerate_one_with_bytes(relative_path, force_indexed, None)
    }

    /// Like `enumerate_one` but accepts pre-read bytes to skip the disk read.
    /// Used by `reindex_file` to thread through bytes already read for hashing,
    /// so the file is read at most once per `reindex_file` call.
    pub(crate) fn enumerate_one_with_bytes(
        &self,
        relative_path: RelativePath,
        force_indexed: bool,
        preread_bytes: Option<Vec<u8>>,
    ) -> Result<Option<EnumeratedFile>> {
        let absolute_path = self.resolve_relative_path(&relative_path)?;
        let metadata = match fs::symlink_metadata(&absolute_path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error.into()),
        };

        let read_path = if metadata.file_type().is_symlink() {
            if !self.config.indexing.follow_symlinks {
                return Ok(None);
            }
            let target_path = match absolute_path.canonicalize() {
                Ok(path) => path,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
                Err(error) => return Err(error.into()),
            };
            if !target_path.starts_with(&self.project_root) {
                return Ok(None);
            }
            let target_metadata = match fs::metadata(&target_path) {
                Ok(m) => m,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
                Err(error) => return Err(error.into()),
            };
            if !target_metadata.is_file() || target_metadata.len() > self.max_file_size_bytes() {
                return Ok(None);
            }
            target_path
        } else {
            if !metadata.is_file() || metadata.len() > self.max_file_size_bytes() {
                return Ok(None);
            }
            absolute_path.clone()
        };

        let contents = if let Some(bytes) = preread_bytes {
            // Caller already read the file; reuse to avoid a second disk read.
            bytes
        } else {
            match fs::read(&read_path) {
                Ok(contents) => contents,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
                Err(error) => return Err(error.into()),
            }
        };
        let file_hash = hash_bytes(&contents);
        let language = language_for_path(&absolute_path);

        Ok(Some(EnumeratedFile {
            absolute_path,
            relative_path,
            language,
            file_hash,
            force_indexed,
            content: Some(contents),
        }))
    }

    fn resolve_relative_path(&self, relative_path: &RelativePath) -> Result<PathBuf> {
        relative_path.reject_escape("Only enumerate files inside $CLAUDE_PROJECT_DIR")?;
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

pub(crate) fn hash_bytes(bytes: &[u8]) -> FileHash {
    let digest = xxhash_rust::xxh3::xxh3_128(bytes);
    FileHash(digest.to_be_bytes())
}

/// Walk up from `path` looking for a `.git` directory; return `true` if any
/// ancestor contains one. Cheaper than constructing a `gix::Repository`, so
/// hot paths (hooks, status checks) call this instead of `discover_repository`.
pub fn is_git_repo(path: &Path) -> bool {
    let mut current = path;
    loop {
        if current.join(".git").exists() {
            return true;
        }
        match current.parent() {
            Some(parent) => current = parent,
            None => return false,
        }
    }
}

fn ensure_within_root(root: &Path, path: &Path) -> Result<()> {
    if path.starts_with(root) {
        return Ok(());
    }

    Err(ClaudixError::PathTraversal {
        path: path.to_path_buf(),
        recovery: RecoveryHint(crate::prompts::hints::ENUMERATE_INSIDE_PROJECT_DIR),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::types::RelativePath;
    use std::collections::BTreeSet;
    use std::fs;
    #[cfg(unix)]
    use std::os::unix::fs as unix_fs;

    mod fixture {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/common/fixture.rs"
        ));
    }

    use fixture::TestFixture;

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

        let files = enumerator.enumerate(&mut ());
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

        let files = enumerator.enumerate(&mut ());
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
    fn indexinclude_sets_force_indexed_for_unknown_language_files() {
        let fixture = TestFixture::new("small_rust");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());

        let readme = fixture.root().join("README.md");
        assert!(fs::write(&readme, "# hello\nsome docs\n").is_ok());
        let indexinclude = fixture.root().join(".indexinclude");
        assert!(fs::write(&indexinclude, "*.md\n").is_ok());

        let enumerator = FileEnumerator::new(fixture.root().to_path_buf(), Config::default());
        assert!(enumerator.is_ok());
        let enumerator = enumerator.ok().unwrap_or_else(|| unreachable!());

        let files = enumerator.enumerate(&mut ());
        assert!(files.is_ok());
        let files = files.ok().unwrap_or_else(|| unreachable!());

        let readme_file = files
            .iter()
            .find(|f| f.relative_path.as_str() == "README.md");
        assert!(readme_file.is_some());
        let readme_file = readme_file.unwrap_or_else(|| unreachable!());
        assert_eq!(readme_file.language, Language::Unknown);
        assert!(readme_file.force_indexed);

        let rs_file = files
            .iter()
            .find(|f| f.relative_path.as_str() == "src/lib.rs");
        assert!(rs_file.is_some());
        assert!(!rs_file.unwrap_or_else(|| unreachable!()).force_indexed);
    }

    #[test]
    fn indexinclude_reincludes_gitignored_directory() {
        let fixture = TestFixture::new("gitignored_docs");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());

        let enumerator = FileEnumerator::new(fixture.root().to_path_buf(), Config::default());
        assert!(enumerator.is_ok());
        let enumerator = enumerator.ok().unwrap_or_else(|| unreachable!());

        let files = enumerator.enumerate(&mut ());
        assert!(files.is_ok());
        let files = files.ok().unwrap_or_else(|| unreachable!());

        let paths: BTreeSet<_> = files
            .iter()
            .map(|file| file.relative_path.as_str().to_owned())
            .collect();
        // `docs/` is gitignored and untracked; only the root `.indexinclude`
        // (`docs/**`) rescues it from the gitignore-aware walk.
        assert!(paths.contains("src/lib.rs"));
        assert!(
            paths.contains("docs/guide.md"),
            "missing docs/guide.md: {paths:?}"
        );
        assert!(
            paths.contains("docs/sub/api.md"),
            "missing nested doc: {paths:?}"
        );

        // Unknown-language docs only chunk when force-indexed.
        let guide = files
            .iter()
            .find(|f| f.relative_path.as_str() == "docs/guide.md")
            .unwrap_or_else(|| unreachable!());
        assert_eq!(guide.language, Language::Unknown);
        assert!(guide.force_indexed);
    }

    #[test]
    fn gitignored_directory_stays_excluded_without_indexinclude() {
        let fixture = TestFixture::new("gitignored_docs");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());
        // Drop the re-include rule: the gitignored docs tree must vanish again.
        assert!(fs::remove_file(fixture.root().join(".indexinclude")).is_ok());

        let enumerator = FileEnumerator::new(fixture.root().to_path_buf(), Config::default());
        assert!(enumerator.is_ok());
        let enumerator = enumerator.ok().unwrap_or_else(|| unreachable!());

        let files = enumerator.enumerate(&mut ());
        assert!(files.is_ok());
        let files = files.ok().unwrap_or_else(|| unreachable!());

        let paths: BTreeSet<_> = files
            .iter()
            .map(|file| file.relative_path.as_str().to_owned())
            .collect();
        assert!(paths.contains("src/lib.rs"));
        assert!(
            !paths.iter().any(|p| p.starts_with("docs/")),
            "docs leaked: {paths:?}"
        );
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

    #[cfg(unix)]
    #[test]
    fn symlink_targets_outside_project_are_skipped() {
        let fixture = TestFixture::new("small_rust");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());

        let outside_dir = tempfile::tempdir();
        assert!(outside_dir.is_ok());
        let outside_dir = outside_dir.ok().unwrap_or_else(|| unreachable!());
        let outside_file = outside_dir.path().join("outside.rs");
        assert!(fs::write(&outside_file, "pub fn outside() {}\n").is_ok());
        assert!(unix_fs::symlink(&outside_file, fixture.root().join("src/outside.rs")).is_ok());

        let mut config = Config::default();
        config.indexing.follow_symlinks = true;
        let enumerator = FileEnumerator::new(fixture.root().to_path_buf(), config);
        assert!(enumerator.is_ok());
        let enumerator = enumerator.ok().unwrap_or_else(|| unreachable!());

        let files = enumerator.enumerate(&mut ());
        assert!(files.is_ok());
        let files = files.ok().unwrap_or_else(|| unreachable!());
        let paths: BTreeSet<_> = files
            .iter()
            .map(|file| file.relative_path.as_str().to_owned())
            .collect();

        assert!(!paths.contains("src/outside.rs"));
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_directories_are_skipped() {
        let fixture = TestFixture::new("small_rust");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());
        assert!(
            unix_fs::symlink(
                fixture.root().join("src"),
                fixture.root().join("src/link.rs")
            )
            .is_ok()
        );

        let mut config = Config::default();
        config.indexing.follow_symlinks = true;
        let enumerator = FileEnumerator::new(fixture.root().to_path_buf(), config);
        assert!(enumerator.is_ok());
        let enumerator = enumerator.ok().unwrap_or_else(|| unreachable!());

        let files = enumerator.enumerate(&mut ());
        assert!(files.is_ok());
        let files = files.ok().unwrap_or_else(|| unreachable!());
        let paths: BTreeSet<_> = files
            .iter()
            .map(|file| file.relative_path.as_str().to_owned())
            .collect();

        assert!(!paths.contains("src/link.rs"));
    }

    #[test]
    fn directories_are_skipped() {
        let fixture = TestFixture::new("small_rust");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());
        let directory_path = fixture.root().join("src/directory.rs");
        assert!(fs::create_dir_all(&directory_path).is_ok());

        let enumerator = FileEnumerator::new(fixture.root().to_path_buf(), Config::default());
        assert!(enumerator.is_ok());
        let enumerator = enumerator.ok().unwrap_or_else(|| unreachable!());

        let directory = RelativePath::new("src/directory.rs");
        let file = enumerator.enumerate_one(directory, false);

        assert!(file.is_ok());
        assert!(file.ok().unwrap_or_else(|| unreachable!()).is_none());
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

        let files = enumerator.enumerate(&mut ());
        assert!(files.is_ok());
        let files = files.ok().unwrap_or_else(|| unreachable!());

        let paths: BTreeSet<_> = files
            .iter()
            .map(|file| file.relative_path.as_str().to_owned())
            .collect();
        assert!(!paths.contains("src/oversized.rs"));
    }
}
