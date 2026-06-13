use std::path::{Component, Path, PathBuf};

use ignore::gitignore::{Gitignore, GitignoreBuilder};

use crate::error::Result;
use crate::types::RelativePath;

/// Decide whether a file-watch event should trigger a reindex. Wraps the
/// project's `.gitignore`, `.indexignore`, and `.indexinclude` matchers so
/// build/cache trees (`target/`, `node_modules/`, …) don't churn the index.
#[derive(Debug)]
pub struct WatchFilter {
    gitignore: Gitignore,
    indexignore: Gitignore,
    indexinclude: Gitignore,
}

impl WatchFilter {
    pub fn load(project_root: &Path) -> Result<Self> {
        Ok(Self {
            gitignore: load_matcher(project_root, ".gitignore")?,
            indexignore: load_matcher(project_root, ".indexignore")?,
            indexinclude: load_matcher(project_root, ".indexinclude")?,
        })
    }

    pub fn is_watchable(&self, relative: &Path) -> bool {
        let Some(first) = relative.components().next() else {
            return false;
        };
        if matches!(first, Component::Normal(name) if name == ".git" || name == ".claudix") {
            return false;
        }

        // Explicit re-include via .indexinclude wins over both ignore lists.
        if self
            .indexinclude
            .matched_path_or_any_parents(relative, false)
            .is_ignore()
        {
            return true;
        }
        if self
            .gitignore
            .matched_path_or_any_parents(relative, false)
            .is_ignore()
        {
            return false;
        }
        if self
            .indexignore
            .matched_path_or_any_parents(relative, false)
            .is_ignore()
        {
            return false;
        }
        true
    }
}

/// A single `.indexinclude`/`.indexignore` file scoped to the directory that
/// owns it. Patterns are interpreted relative to that directory, matching
/// nested-`.gitignore` semantics, so the owning prefix is stripped from a
/// candidate path before the file's globs are consulted.
#[derive(Debug)]
struct ScopedMatcher {
    /// Owning directory relative to the project root; empty for a root rule.
    base: PathBuf,
    matcher: Gitignore,
}

impl ScopedMatcher {
    /// Build from a rule file path relative to `project_root`.
    fn build(project_root: &Path, rule_file: &Path) -> Result<Self> {
        let base = rule_file
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_default();
        let mut builder = GitignoreBuilder::new(project_root.join(&base));
        builder.add(project_root.join(rule_file));
        Ok(Self {
            base,
            matcher: builder.build()?,
        })
    }

    fn matches(&self, root_relative: &Path) -> bool {
        let scoped = if self.base.as_os_str().is_empty() {
            root_relative
        } else {
            match root_relative.strip_prefix(&self.base) {
                Ok(scoped) => scoped,
                Err(_) => return false,
            }
        };
        if scoped.as_os_str().is_empty() {
            return false;
        }
        self.matcher
            .matched_path_or_any_parents(scoped, false)
            .is_ignore()
    }
}

/// Nested `.indexignore`/`.indexinclude` rules. Each rule file applies to its
/// own subtree (root + any subdirectory), and `.indexinclude` wins over
/// `.indexignore` just as the watch path resolves the two.
#[derive(Debug, Default)]
pub struct PathFilters {
    indexignore: Vec<ScopedMatcher>,
    indexinclude: Vec<ScopedMatcher>,
}

impl PathFilters {
    /// Build from every rule file present in `paths` (a gitignore-blind path
    /// list), so rule files nested anywhere — including inside otherwise
    /// gitignored subtrees — are honored.
    pub fn from_paths(project_root: &Path, paths: &[RelativePath]) -> Result<Self> {
        let mut filters = Self::default();
        for path in paths {
            let path_buf = path.to_path_buf();
            match path_buf.file_name().and_then(|name| name.to_str()) {
                Some(".indexignore") => filters
                    .indexignore
                    .push(ScopedMatcher::build(project_root, &path_buf)?),
                Some(".indexinclude") => filters
                    .indexinclude
                    .push(ScopedMatcher::build(project_root, &path_buf)?),
                _ => {}
            }
        }
        Ok(filters)
    }

    /// Build only from the rule files on the ancestor chain of `path` (root plus
    /// each parent directory). Cheap enough for the per-file reindex path, which
    /// cannot afford a full-tree discovery walk on every edit.
    pub fn for_path(project_root: &Path, path: &RelativePath) -> Result<Self> {
        let mut filters = Self::default();
        let mut directory = PathBuf::new();
        let mut directories = vec![directory.clone()];
        if let Some(parent) = path.to_path_buf().parent() {
            for component in parent.components() {
                directory = directory.join(component);
                directories.push(directory.clone());
            }
        }
        for dir in directories {
            let ignore_file = dir.join(".indexignore");
            if project_root.join(&ignore_file).is_file() {
                filters
                    .indexignore
                    .push(ScopedMatcher::build(project_root, &ignore_file)?);
            }
            let include_file = dir.join(".indexinclude");
            if project_root.join(&include_file).is_file() {
                filters
                    .indexinclude
                    .push(ScopedMatcher::build(project_root, &include_file)?);
            }
        }
        Ok(filters)
    }

    pub fn is_included(&self, path: &RelativePath) -> bool {
        let path_buf = path.to_path_buf();
        if self.indexinclude.iter().any(|rule| rule.matches(&path_buf)) {
            return true;
        }
        !self.indexignore.iter().any(|rule| rule.matches(&path_buf))
    }

    pub fn is_force_included(&self, path: &RelativePath) -> bool {
        let path_buf = path.to_path_buf();
        self.indexinclude.iter().any(|rule| rule.matches(&path_buf))
    }

    /// Whether any `.indexinclude` rule is in effect. Gates the gitignore-blind
    /// deep walk: with no re-include rule there is nothing to rescue from an
    /// ignored subtree, so the enumerator skips the extra walk entirely.
    pub fn has_includes(&self) -> bool {
        !self.indexinclude.is_empty()
    }
}

fn load_matcher(project_root: &Path, file_name: &str) -> Result<Gitignore> {
    let mut builder = GitignoreBuilder::new(project_root);
    let file_path = project_root.join(file_name);

    if file_path.exists() {
        builder.add(&file_path);
    }

    builder.build().map_err(Into::into)
}
