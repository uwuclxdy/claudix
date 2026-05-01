use std::collections::BTreeSet;
use std::path::Path;

use gix::Repository;
use gix::bstr::BStr;
use ignore::WalkBuilder;

use crate::error::{ClaudixError, Result};
use crate::types::RelativePath;

pub fn discover_repository(project_root: &Path) -> Result<Repository> {
    gix::discover(project_root).map_err(|error| ClaudixError::Git(error.to_string()))
}

pub fn list_candidate_paths(repo: &Repository) -> Result<Vec<RelativePath>> {
    let workdir = repo
        .work_dir()
        .ok_or_else(|| ClaudixError::Git("bare repositories are not supported".into()))?;
    let mut paths = BTreeSet::new();

    let index = repo
        .index_or_empty()
        .map_err(|error| ClaudixError::Git(error.to_string()))?;
    for (path, _) in index.entries_with_paths_by_filter_map(|_path: &BStr, _entry| Some(())) {
        paths.insert(RelativePath::new(path.to_string()));
    }

    for relative_path in list_untracked_paths(workdir)? {
        paths.insert(relative_path);
    }

    Ok(paths.into_iter().collect())
}

fn list_untracked_paths(workdir: &Path) -> Result<Vec<RelativePath>> {
    let mut paths = Vec::new();
    let mut builder = WalkBuilder::new(workdir);
    builder.hidden(false);
    builder.git_ignore(true);
    builder.git_exclude(true);
    builder.git_global(true);
    builder.parents(true);

    for entry in builder.build() {
        let entry = entry?;
        if !entry.file_type().is_some_and(|kind| kind.is_file()) {
            continue;
        }

        let absolute_path = entry.into_path();
        let relative_path = absolute_path
            .strip_prefix(workdir)
            .map_err(|error| ClaudixError::Git(error.to_string()))?;

        if is_tracked_artifact(relative_path) {
            continue;
        }

        paths.push(RelativePath::from_path(relative_path));
    }

    Ok(paths)
}

fn is_tracked_artifact(relative_path: &Path) -> bool {
    let path = path_to_slash_string(relative_path);
    path == ".git" || path.starts_with(".git/")
}

fn path_to_slash_string(path: &Path) -> String {
    path.components()
        .map(|component| component.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join("/")
}
