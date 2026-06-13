use std::path::{Component, Path};

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

#[derive(Debug)]
pub struct PathFilters {
    indexignore: Gitignore,
    indexinclude: Gitignore,
}

impl PathFilters {
    pub fn load(project_root: &Path) -> Result<Self> {
        Ok(Self {
            indexignore: load_matcher(project_root, ".indexignore")?,
            indexinclude: load_matcher(project_root, ".indexinclude")?,
        })
    }

    pub fn is_included(&self, path: &RelativePath) -> bool {
        let path_buf = path.to_path_buf();
        let ignored = self.indexignore.matched(&path_buf, false).is_ignore();
        let re_included = self.indexinclude.matched(&path_buf, false).is_ignore();

        if re_included {
            return true;
        }

        !ignored
    }

    pub fn is_force_included(&self, path: &RelativePath) -> bool {
        self.indexinclude
            .matched(path.to_path_buf(), false)
            .is_ignore()
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
