mod filters;
mod git;

use std::collections::BTreeSet;
use std::env;
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
        let tracked = git::list_candidate_paths(&repo, self.config.indexing.respect_gitignore)?;
        let mut candidates: BTreeSet<RelativePath> = tracked.iter().cloned().collect();

        // Nested `.indexinclude`/`.indexignore` rules — including ones living
        // inside an otherwise gitignored subtree — only surface from a
        // gitignore-blind deep walk. Pay for it only when a rule plausibly
        // exists; otherwise the common no-rule repo keeps the cheap walk alone.
        let filters = if index_rules_present(&self.project_root, &tracked) {
            let all_paths = git::list_all_paths(&repo)?;
            let filters = PathFilters::from_paths(&self.project_root, &all_paths)?;
            // `.indexinclude` re-includes paths the gitignore-aware walk pruned
            // before any filter saw them (e.g. a gitignored `docs/` tree).
            if filters.has_includes() {
                for relative_path in all_paths {
                    if filters.is_force_included(&relative_path) {
                        candidates.insert(relative_path);
                    }
                }
            }
            filters
        } else {
            PathFilters::default()
        };

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

/// Cheap pre-check gating the gitignore-blind deep walk: return true when any
/// `.indexinclude`/`.indexignore` rule plausibly exists. Catches root rules,
/// nested rules among tracked files, and a rule at the top of a gitignored
/// directory (e.g. `docs/.indexinclude`). A rule buried deeper inside a
/// gitignored subtree with no shallower rule is the one gap — place such a rule
/// at the gitignored directory's top level or at the repo root.
fn index_rules_present(project_root: &Path, tracked: &[RelativePath]) -> bool {
    if project_root.join(".indexinclude").is_file() || project_root.join(".indexignore").is_file() {
        return true;
    }
    if tracked
        .iter()
        .any(|path| is_index_rule_file(&path.to_path_buf()))
    {
        return true;
    }
    let Ok(entries) = fs::read_dir(project_root) else {
        return false;
    };
    for entry in entries.flatten() {
        if entry.file_type().is_ok_and(|kind| kind.is_dir()) {
            let dir = entry.path();
            if dir.join(".indexinclude").is_file() || dir.join(".indexignore").is_file() {
                return true;
            }
        }
    }
    false
}

fn is_index_rule_file(path: &Path) -> bool {
    matches!(
        path.file_name().and_then(|name| name.to_str()),
        Some(".indexinclude") | Some(".indexignore")
    )
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

/// The first valid git repo root at or above `path`, walking up one directory
/// at a time. Cheaper than constructing a `gix::Repository`, so hot paths
/// (hooks, status checks) call this instead of `discover_repository`.
/// Validates the discovered `.git` is a real repo, not just present: an orphan
/// empty `.git` directory left in a shared parent (e.g. an interrupted
/// `git init`) stops the walk with `None` so it cannot fool it into reporting
/// every sibling as part of a repo.
pub fn git_repo_root(path: &Path) -> Option<PathBuf> {
    let mut current = path;
    loop {
        match valid_git_marker(current) {
            Some(true) => return Some(current.to_path_buf()),
            Some(false) => return None,
            None => {}
        }
        current = current.parent()?;
    }
}

/// Whether `path` sits inside a git repository (walking up, like
/// `git rev-parse --show-toplevel`). A verdict of [`git_repo_root`].
pub fn is_git_repo(path: &Path) -> bool {
    git_repo_root(path).is_some()
}

/// Classify the `.git` at `dir`: `Some(true)` for a usable repo, `Some(false)`
/// for a partial/orphan state, `None` when no `.git` is present here. A `.git`
/// *file* is a `gitdir:` pointer for submodules and
/// worktrees — trust presence, resolving it cross-platform is more than this
/// hot-path guard owes (and the orphan case is always an empty directory, never
/// a pointer file). For a directory (or a symlink to one), `git init` always
/// writes `HEAD` and `objects/`, so require both; `Path::join` follows the
/// symlink for those inner checks so no special case is needed.
fn valid_git_marker(dir: &Path) -> Option<bool> {
    let dot_git = dir.join(".git");
    let metadata = fs::symlink_metadata(&dot_git).ok()?;
    if metadata.is_file() {
        return Some(true);
    }
    Some(dot_git.join("HEAD").exists() && dot_git.join("objects").exists())
}

/// Resolve the project root a process entry acts on: the session's
/// `CLAUDE_PROJECT_DIR` when set, else the process cwd, walked up to the
/// enclosing git repo root. A non-git start dir resolves to itself so the
/// existing non-git gates (hook passthrough, `require_git_repo`) keep their
/// behavior unchanged. Ruling 2026-08-25: claudix never creates `.claudix/`
/// outside a repo root.
pub fn active_project_root() -> std::io::Result<PathBuf> {
    let start = match env::var_os("CLAUDE_PROJECT_DIR") {
        Some(path) => PathBuf::from(path),
        None => env::current_dir()?,
    };
    Ok(resolve_project_root(&start))
}

/// [`git_repo_root`] with the non-git fallback: the start dir itself. The
/// start is canonicalized first, so a relative `CLAUDE_PROJECT_DIR` (e.g.
/// `.` from a shell) resolves against the process cwd — a lexical walk from a
/// relative path never leaves the relative namespace and would fall back to
/// the start dir itself, recreating the nested-store shape. A non-existent
/// start falls back to its raw spelling and keeps the non-git fallback.
pub fn resolve_project_root(start: &Path) -> PathBuf {
    let start = start.canonicalize().unwrap_or_else(|_| start.to_path_buf());
    git_repo_root(&start).unwrap_or(start)
}

/// Delete `.claudix/` dirs pre-fix binaries left in directories between
/// `start` and the resolved repo `root` (ruling 2026-08-25: such folders must
/// never exist and are deleted automatically). Only dirs holding
/// claudix-created content are touched; a nested repo's own store is legal
/// and is skipped while the walk continues above it, and a store beside an
/// orphan `.git` is litter and is deleted. Fail-open: a failed deletion logs
/// and the caller continues.
pub fn delete_nested_stores_between(start: &Path, root: &Path) {
    // Canonicalize so the boundary checks hold under symlinked paths (macOS
    // `/var`, the gate's symlinked TMPDIR); a raw spelling falls back intact.
    let start = start.canonicalize().unwrap_or_else(|_| start.to_path_buf());
    let root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());

    let mut current = Some(start.as_path());
    while let Some(dir) = current {
        if dir == root || !dir.starts_with(&root) {
            // Never touch the resolved root's own store, and never walk
            // outside the resolved repo tree: cwd and CLAUDE_PROJECT_DIR can
            // name unrelated directories.
            return;
        }
        match valid_git_marker(dir) {
            // A nested repo (worktree, submodule) is its own root: its store
            // is legal, but the walk continues above it — dirs between it and
            // the resolved root are the outer repo's subdirs and still swept.
            Some(true) => {}
            // No marker, or an orphan one (not a repo): a store here is
            // litter.
            None | Some(false) => {
                let store = dir.join(".claudix");
                if is_claudix_store(&store) {
                    match fs::remove_dir_all(&store) {
                        Ok(()) => {
                            tracing::warn!("deleted nested claudix store {}", store.display());
                        }
                        Err(error) => {
                            tracing::warn!(
                                "could not delete nested claudix store {}: {error}",
                                store.display()
                            );
                        }
                    }
                }
            }
        }
        current = dir.parent();
    }
}

/// The start dirs a pre-fix nested store can sit under: the process cwd and
/// `CLAUDE_PROJECT_DIR` (they may differ, and either may be a subdir).
pub fn nested_store_start_dirs() -> Vec<PathBuf> {
    let mut starts = Vec::new();
    if let Ok(cwd) = env::current_dir() {
        starts.push(cwd);
    }
    if let Some(dir) = env::var_os("CLAUDE_PROJECT_DIR") {
        starts.push(PathBuf::from(dir));
    }
    starts
}

/// [`delete_nested_stores_between`] over both directories a session can start
/// from (the process cwd and `CLAUDE_PROJECT_DIR`).
pub fn delete_nested_stores(root: &Path) {
    delete_nested_stores_from(root, nested_store_start_dirs());
}

/// [`delete_nested_stores_between`] over explicit start dirs; the
/// env-derived entry is [`delete_nested_stores`]. The explicit form lets a
/// write-path test plant a store under a fixture start dir without touching
/// the process environment.
pub fn delete_nested_stores_from(root: &Path, starts: impl IntoIterator<Item = PathBuf>) {
    for start in starts {
        delete_nested_stores_between(&start, root);
    }
}

/// A `.claudix/` counts as claudix-created when it holds any shape the
/// binary's write paths lay down: `ensure_layout` output (`manifest.json`,
/// `index/`, `.gitignore`) or the bare husks pre-fix sessions left without
/// one (`index.lock`, `reindex-queue`, `reindex-queue.lock` — each
/// `create_dir_all`s the state dir directly). Anything else under that name
/// belongs to the user.
fn is_claudix_store(store_dir: &Path) -> bool {
    store_dir.join("manifest.json").is_file()
        || store_dir.join("index").is_dir()
        || store_dir.join(".gitignore").is_file()
        || store_dir.join("index.lock").is_file()
        || store_dir.join("reindex-queue").is_file()
        || store_dir.join("reindex-queue.lock").is_file()
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
    fn respect_gitignore_false_includes_gitignored_files() {
        let fixture = TestFixture::new("gitignored_docs");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());
        // Drop the re-include rule so the docs tree reaches candidates via the
        // candidate walk itself, not the `.indexinclude` force-include path.
        assert!(fs::remove_file(fixture.root().join(".indexinclude")).is_ok());

        let mut config = Config::default();
        config.indexing.respect_gitignore = false;

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
        assert!(paths.contains("src/lib.rs"));
        assert!(
            paths.contains("docs/guide.md"),
            "docs/guide.md missing with respect_gitignore=false: {paths:?}"
        );
        assert!(
            paths.contains("docs/sub/api.md"),
            "docs/sub/api.md missing with respect_gitignore=false: {paths:?}"
        );
    }

    #[test]
    fn nested_indexinclude_reincludes_its_subtree() {
        let fixture = TestFixture::new("nested_indexinclude");
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
        // No root rule: `docs/.indexinclude` (`*`) alone rescues the gitignored
        // subtree, patterns relative to its own directory.
        assert!(paths.contains("src/lib.rs"));
        assert!(
            paths.contains("docs/guide.md"),
            "missing docs/guide.md: {paths:?}"
        );
        assert!(
            paths.contains("docs/sub/api.md"),
            "missing nested doc: {paths:?}"
        );
    }

    #[test]
    fn nested_indexignore_excludes_its_subtree() {
        let fixture = TestFixture::new("nested_indexignore");
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
        // `src/gen/.indexignore` (`*`) applies only to its own subtree.
        assert!(paths.contains("src/keep.rs"));
        assert!(
            !paths.contains("src/gen/gen.rs"),
            "nested ignore leaked: {paths:?}"
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

    /// Mirror `tests/common/fixture.rs`'s empty-config trick and git env
    /// scrub inlined, so the orphan/real-repo tests below stay inside this
    /// module without importing the cross-module helper.
    fn init_real_git_repo(root: &Path) {
        // Neutralise the developer's global/system git config and every
        // redirecting git env var for the throwaway repo (same rationale as
        // `tests/common/fixture.rs`: a gated commit runs the suite inside
        // git's hook environment, which hands GIT_INDEX_FILE to every test
        // subprocess).
        let empty_config = root.join(".claudix-test-empty-gitconfig");
        let output = std::process::Command::new("git")
            .current_dir(root)
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
            .args(["init"])
            .output()
            .ok()
            .unwrap_or_else(|| unreachable!());
        assert!(
            output.status.success(),
            "git init failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        // `git init` always writes both; pin them so a future git layout change
        // doesn't silently flip these tests green for the wrong reason.
        assert!(
            root.join(".git/HEAD").exists(),
            "git init must write .git/HEAD"
        );
        assert!(
            root.join(".git/objects").exists(),
            "git init must write .git/objects"
        );
    }

    /// Regression: an orphan empty `.git` directory left in a shared parent
    /// (e.g. from an interrupted `git init /tmp/scratch`) used to fool the
    /// walk-up into reporting every sibling as a git repo. Constructed inside a
    /// tempdir we own so `$TMPDIR` stays clean.
    #[test]
    fn is_git_repo_false_for_orphan_empty_dot_git_in_parent() {
        let outer = tempfile::tempdir();
        assert!(outer.is_ok());
        let outer = outer.ok().unwrap_or_else(|| unreachable!());
        assert!(fs::create_dir_all(outer.path().join(".git")).is_ok());

        let inner = outer.path().join("workspace");
        assert!(fs::create_dir_all(&inner).is_ok());

        assert!(
            !is_git_repo(&inner),
            "orphan empty `.git` in a parent must not register as a repo"
        );
    }

    #[test]
    fn is_git_repo_true_for_real_repo_subdir() {
        let root = tempfile::tempdir();
        assert!(root.is_ok());
        let root = root.ok().unwrap_or_else(|| unreachable!());
        init_real_git_repo(root.path());

        let subdir = root.path().join("subdir");
        assert!(fs::create_dir_all(&subdir).is_ok());

        assert!(
            is_git_repo(&subdir),
            "a real git repo must be detected from a subdir"
        );
    }

    #[test]
    fn is_git_repo_false_when_no_dot_git_in_walk_up() {
        let root = tempfile::tempdir();
        assert!(root.is_ok());
        let root = root.ok().unwrap_or_else(|| unreachable!());
        // Skip if any ancestor holds a `.git`: a repo above the tempdir (e.g.
        // `TMPDIR=~/code/tmp`) makes the assertion red for environment reasons
        // unrelated to the fix. The orphan and real-repo tests carry the
        // mutation red on every system; this one only pins the no-`.git`
        // baseline.
        let mut current = root.path();
        let blocked_by_ancestor = loop {
            if current.join(".git").exists() {
                break true;
            }
            match current.parent() {
                Some(parent) => current = parent,
                None => break false,
            }
        };
        if blocked_by_ancestor {
            eprintln!(
                "skipped: a `.git` ancestor exists above {}",
                root.path().display()
            );
            return;
        }

        assert!(
            !is_git_repo(root.path()),
            "a directory with no `.git` ancestor must not register as a repo"
        );
    }

    #[test]
    fn git_repo_root_resolves_subdir_to_repo_root() {
        let root = tempfile::tempdir();
        assert!(root.is_ok());
        let root = root.ok().unwrap_or_else(|| unreachable!());
        init_real_git_repo(root.path());

        let subdir = root.path().join("a/b");
        assert!(fs::create_dir_all(&subdir).is_ok());

        assert_eq!(
            git_repo_root(&subdir),
            Some(root.path().to_path_buf()),
            "the walk must resolve a subdir to its repo root"
        );
    }

    #[test]
    fn git_repo_root_stops_at_nested_repo() {
        let outer = tempfile::tempdir();
        assert!(outer.is_ok());
        let outer = outer.ok().unwrap_or_else(|| unreachable!());
        init_real_git_repo(outer.path());

        let inner = outer.path().join("inner");
        assert!(fs::create_dir_all(&inner).is_ok());
        init_real_git_repo(&inner);

        let subdir = inner.join("deep");
        assert!(fs::create_dir_all(&subdir).is_ok());

        assert_eq!(
            git_repo_root(&subdir),
            Some(inner.clone()),
            "a nested repo is its own root; the walk must stop there"
        );
        assert_eq!(
            git_repo_root(outer.path()),
            Some(outer.path().to_path_buf()),
            "the outer repo still resolves for its own subtree"
        );
    }

    #[test]
    fn git_repo_root_none_for_orphan_dot_git_parent() {
        let outer = tempfile::tempdir();
        assert!(outer.is_ok());
        let outer = outer.ok().unwrap_or_else(|| unreachable!());
        assert!(fs::create_dir_all(outer.path().join(".git")).is_ok());

        let inner = outer.path().join("workspace");
        assert!(fs::create_dir_all(&inner).is_ok());

        assert_eq!(
            git_repo_root(&inner),
            None,
            "an orphan empty `.git` in a parent must not resolve to a root"
        );
    }

    #[test]
    fn git_repo_root_none_and_fallback_without_repo() {
        let root = tempfile::tempdir();
        assert!(root.is_ok());
        let root = root.ok().unwrap_or_else(|| unreachable!());
        // Skip when any ancestor holds a `.git`: an ambient repo above the
        // tempdir (e.g. `TMPDIR=~/code/tmp`) makes both assertions red for
        // environment reasons unrelated to the resolver (same guard as the
        // `is_git_repo` no-`.git` baseline test). Check both the raw and the
        // canonical chain — resolution canonicalizes first, and the gate's
        // symlinked TMPDIR makes the two spellings diverge.
        let blocked_by_ancestor = [
            root.path().to_path_buf(),
            root.path().canonicalize().unwrap_or_default(),
        ]
        .iter()
        .any(|start| {
            let mut current = start.as_path();
            loop {
                if current.join(".git").exists() {
                    return true;
                }
                match current.parent() {
                    Some(parent) => current = parent,
                    None => return false,
                }
            }
        });
        if blocked_by_ancestor {
            eprintln!(
                "skipped: a `.git` ancestor exists above {}",
                root.path().display()
            );
            return;
        }

        assert_eq!(git_repo_root(root.path()), None);
        assert_eq!(
            resolve_project_root(root.path()),
            root.path()
                .canonicalize()
                .unwrap_or_else(|_| root.path().to_path_buf()),
            "a non-git start dir must resolve to itself (canonical spelling), preserving the non-git gates"
        );
    }

    #[test]
    fn delete_nested_stores_removes_only_claudix_created_dirs_on_the_chain() {
        let fixture = TestFixture::new("small_rust");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());
        let root = fixture.root().to_path_buf();

        // The pre-fix bug shapes on one ancestor chain: a full store, bare
        // reindex-queue and reindex-queue.lock husks (queue append
        // `create_dir_all`s the state dir and takes the queue lock with no
        // `ensure_layout`), a markerless user dir, and the legal root store.
        let start = root.join("website/worker/deep/lockhusk");
        assert!(fs::create_dir_all(start.join(".claudix")).is_ok());
        assert!(fs::write(start.join(".claudix/reindex-queue.lock"), "").is_ok());

        let queue_husk = root.join("website/worker/deep");
        assert!(fs::create_dir_all(queue_husk.join(".claudix")).is_ok());
        assert!(fs::write(queue_husk.join(".claudix/reindex-queue"), "").is_ok());

        let store_at_worker = root.join("website/worker");
        assert!(fs::create_dir_all(store_at_worker.join(".claudix")).is_ok());
        assert!(fs::write(store_at_worker.join(".claudix/manifest.json"), "{}").is_ok());

        let middle = root.join("website");
        assert!(fs::create_dir_all(middle.join(".claudix")).is_ok());
        assert!(fs::write(middle.join(".claudix/notes.txt"), "mine").is_ok());

        assert!(fs::create_dir_all(root.join(".claudix")).is_ok());
        assert!(fs::write(root.join(".claudix/manifest.json"), "{}").is_ok());

        delete_nested_stores_between(&start, &root);

        assert!(
            !start.join(".claudix").exists(),
            "a bare reindex-queue.lock husk is claudix litter and must be deleted"
        );
        assert!(
            !queue_husk.join(".claudix").exists(),
            "a bare reindex-queue husk is claudix litter and must be deleted"
        );
        assert!(
            !store_at_worker.join(".claudix").exists(),
            "a store-shaped nested dir must be deleted"
        );
        assert!(
            middle.join(".claudix").exists(),
            "a markerless `.claudix` dir belongs to the user and must survive"
        );
        assert!(
            root.join(".claudix").exists(),
            "the resolved root's own store is legal and must survive"
        );
    }

    #[test]
    fn delete_nested_stores_stops_at_nested_repo_boundary() {
        let fixture = TestFixture::new("small_rust");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());
        let root = fixture.root().to_path_buf();

        let inner = root.join("vendor/upstream");
        assert!(fs::create_dir_all(&inner).is_ok());
        init_real_git_repo(&inner);

        // The nested repo's store is legal (its own root); the start dir
        // below it may hold pre-fix litter, and so may the dirs between the
        // nested root and the resolved outer root — the walk must delete both
        // while skipping the nested store itself.
        let start = inner.join("sub");
        assert!(fs::create_dir_all(start.join(".claudix")).is_ok());
        assert!(fs::write(start.join(".claudix/manifest.json"), "{}").is_ok());
        assert!(fs::create_dir_all(inner.join(".claudix")).is_ok());
        assert!(fs::write(inner.join(".claudix/manifest.json"), "{}").is_ok());

        let above = root.join("vendor");
        assert!(fs::create_dir_all(above.join(".claudix")).is_ok());
        assert!(fs::write(above.join(".claudix/manifest.json"), "{}").is_ok());

        delete_nested_stores_between(&start, &root);

        assert!(
            !start.join(".claudix").exists(),
            "litter below a nested repo root must be deleted"
        );
        assert!(
            inner.join(".claudix").exists(),
            "a nested repo's own store is legal and must survive"
        );
        assert!(
            !above.join(".claudix").exists(),
            "litter above a nested repo root must be deleted"
        );
    }

    #[test]
    fn delete_nested_stores_deletes_store_at_orphan_marker_dir() {
        let fixture = TestFixture::new("small_rust");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());
        let root = fixture.root().to_path_buf();

        // An orphan empty `.git` does not make its dir a repo root, so a
        // store beside it is litter per the ruling (e.g. the repo was
        // indexed, then its `.git` got emptied).
        let orphan = root.join("broken");
        assert!(fs::create_dir_all(orphan.join(".git")).is_ok());
        let store = orphan.join(".claudix");
        assert!(fs::create_dir_all(&store).is_ok());
        assert!(fs::write(store.join("manifest.json"), "{}").is_ok());
        let start = orphan.join("sub");
        assert!(fs::create_dir_all(&start).is_ok());

        delete_nested_stores_between(&start, &root);

        assert!(
            !store.exists(),
            "a store beside an orphan `.git` is litter and must be deleted"
        );
    }

    #[test]
    fn resolve_project_root_handles_relative_start() {
        // A relative `CLAUDE_PROJECT_DIR` (e.g. `.` from a shell) must
        // resolve against the process cwd and walk up to the repo root — a
        // lexical walk from a relative path never leaves the relative
        // namespace and would fall back to the start dir itself. Cargo runs
        // tests with cwd = the package root, which is a git repo, so the
        // resolved root is the canonical package root (canonicalized both
        // sides: windows `canonicalize` returns `\\?\` verbatim paths).
        let expected = env::current_dir();
        assert!(expected.is_ok());
        let expected = expected
            .ok()
            .unwrap_or_else(|| unreachable!())
            .canonicalize()
            .unwrap_or_else(|_| env::current_dir().unwrap_or_default());
        assert!(
            git_repo_root(&expected).is_some(),
            "the package root must be a git repo for this test"
        );
        for relative in [".", "./"] {
            let resolved = resolve_project_root(Path::new(relative));
            assert!(
                resolved.is_absolute(),
                "`{relative}` must resolve to an absolute root, got {resolved:?}"
            );
            assert_eq!(
                resolved.canonicalize().unwrap_or_else(|_| resolved.clone()),
                expected
            );
        }
    }

    #[test]
    fn delete_nested_stores_never_walks_outside_the_resolved_root() {
        let fixture = TestFixture::new("small_rust");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());
        let root = fixture.root().to_path_buf();

        // A sibling tree (not under the resolved root) holding a store-shaped
        // dir: cwd and CLAUDE_PROJECT_DIR can name unrelated directories, and
        // neither walk may delete anything outside the resolved repo tree.
        let sibling = root
            .parent()
            .map(|parent| parent.join("sibling"))
            .unwrap_or_else(|| unreachable!());
        assert!(fs::create_dir_all(sibling.join(".claudix")).is_ok());
        assert!(fs::write(sibling.join(".claudix/manifest.json"), "{}").is_ok());

        delete_nested_stores_between(&sibling, &root);

        assert!(
            sibling.join(".claudix").exists(),
            "nothing outside the resolved repo tree may be deleted"
        );
    }

    #[cfg(unix)]
    #[test]
    fn delete_nested_stores_fails_open_on_delete_error() {
        use std::os::unix::fs::PermissionsExt;

        let fixture = TestFixture::new("small_rust");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());
        let root = fixture.root().to_path_buf();

        let start = root.join("sub");
        let store = start.join(".claudix");
        assert!(fs::create_dir_all(&store).is_ok());
        assert!(fs::write(store.join("manifest.json"), "{}").is_ok());
        // Read-only store dir: unlinking `manifest.json` needs write on its
        // parent, so `remove_dir_all` fails with EACCES. Restore afterwards so
        // the tempdir can be cleaned up.
        assert!(fs::set_permissions(&store, fs::Permissions::from_mode(0o500)).is_ok());

        delete_nested_stores_between(&start, &root);

        assert!(
            store.exists(),
            "a failed deletion must not abort the caller (fail-open)"
        );
        assert!(fs::set_permissions(&store, fs::Permissions::from_mode(0o700)).is_ok());
    }
}
