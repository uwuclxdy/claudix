use std::path::Path;

use ignore::gitignore::{Gitignore, GitignoreBuilder};

use crate::error::Result;
use crate::types::RelativePath;

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
}

fn load_matcher(project_root: &Path, file_name: &str) -> Result<Gitignore> {
    let mut builder = GitignoreBuilder::new(project_root);
    let file_path = project_root.join(file_name);

    if file_path.exists() {
        builder.add(&file_path);
    }

    builder.build().map_err(Into::into)
}
