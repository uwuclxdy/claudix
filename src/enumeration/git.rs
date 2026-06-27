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

pub fn list_candidate_paths(
    repo: &Repository,
    respect_gitignore: bool,
) -> Result<Vec<RelativePath>> {
    let workdir = repo
        .workdir()
        .ok_or_else(|| ClaudixError::Git("bare repositories are not supported".into()))?;
    let mut paths = BTreeSet::new();

    let index = repo
        .index_or_empty()
        .map_err(|error| ClaudixError::Git(error.to_string()))?;
    for (path, _) in index.entries_with_paths_by_filter_map(|_path: &BStr, _entry| Some(())) {
        paths.insert(RelativePath::new(path.to_string()));
    }

    for relative_path in list_untracked_paths(workdir, respect_gitignore)? {
        paths.insert(relative_path);
    }

    Ok(paths.into_iter().collect())
}

/// Walk every file under the work tree ignoring `.gitignore` entirely, so files
/// inside gitignored directories (e.g. a doc tree) are visible. Never descends
/// into `.git` or `.claudix`. The gitignore-aware [`list_candidate_paths`] walk
/// prunes ignored subtrees before any `.indexinclude` rule can re-include them;
/// the enumerator consults this list to add back the paths an include rule
/// matches and to discover nested rule files that live inside ignored subtrees.
///
/// Full-reindex only, and only when an include rule is present — it reads
/// directory entries, not file contents. Paths still pass through the normal
/// `.indexinclude`/`.indexignore` filters afterwards, so this never widens the
/// indexed set on its own.
pub fn list_all_paths(repo: &Repository) -> Result<Vec<RelativePath>> {
    let workdir = repo
        .workdir()
        .ok_or_else(|| ClaudixError::Git("bare repositories are not supported".into()))?;
    let mut paths = Vec::new();
    let mut builder = WalkBuilder::new(workdir);
    builder.hidden(false);
    builder.git_ignore(false);
    builder.git_exclude(false);
    builder.git_global(false);
    builder.parents(false);
    builder.require_git(false);
    builder.filter_entry(|entry| {
        !matches!(entry.file_name().to_str(), Some(".git") | Some(".claudix"))
    });

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

fn list_untracked_paths(workdir: &Path, respect_gitignore: bool) -> Result<Vec<RelativePath>> {
    let mut paths = Vec::new();
    let mut builder = WalkBuilder::new(workdir);
    builder.hidden(false);
    builder.git_ignore(respect_gitignore);
    builder.git_exclude(respect_gitignore);
    builder.git_global(respect_gitignore);
    builder.parents(respect_gitignore);

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
