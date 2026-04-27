use std::fs;
use std::path::{Component, Path, PathBuf};

use crate::config::Config;
use crate::error::{ClaudixError, RecoveryHint, Result};

pub const SCHEMA_VERSION: u32 = 1;
const MANIFEST_FILE_NAME: &str = "manifest.json";
const GITIGNORE_FILE_NAME: &str = ".gitignore";
const GITIGNORE_CONTENTS: &str = "*\n";

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Manifest {
    pub schema_version: u32,
    pub embedding_model: String,
    pub dimensions: u16,
    pub last_full_index_at: Option<String>,
    pub last_incremental_at: Option<String>,
    pub chunk_count: u64,
    pub file_count: u64,
}

impl Manifest {
    pub fn new(embedding_model: impl Into<String>, dimensions: u16) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            embedding_model: embedding_model.into(),
            dimensions,
            last_full_index_at: None,
            last_incremental_at: None,
            chunk_count: 0,
            file_count: 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StorePaths {
    state_dir: PathBuf,
    index_dir: PathBuf,
    manifest_path: PathBuf,
    gitignore_path: PathBuf,
}

impl StorePaths {
    pub fn state_dir(&self) -> &Path {
        &self.state_dir
    }

    pub fn index_dir(&self) -> &Path {
        &self.index_dir
    }

    pub fn manifest_path(&self) -> &Path {
        &self.manifest_path
    }

    pub fn gitignore_path(&self) -> &Path {
        &self.gitignore_path
    }
}

#[derive(Debug, Clone)]
pub struct Store {
    project_root: PathBuf,
    paths: StorePaths,
}

impl Store {
    pub fn new(project_root: impl AsRef<Path>, config: &Config) -> Result<Self> {
        let project_root = project_root.as_ref().canonicalize()?;
        let index_dir = resolve_project_path(&project_root, &config.paths.index_dir)?;
        let state_dir = index_dir
            .parent()
            .ok_or_else(|| ClaudixError::Store("index path has no parent directory".to_owned()))?
            .to_path_buf();

        let paths = StorePaths {
            manifest_path: state_dir.join(MANIFEST_FILE_NAME),
            gitignore_path: state_dir.join(GITIGNORE_FILE_NAME),
            state_dir,
            index_dir,
        };

        Ok(Self {
            project_root,
            paths,
        })
    }

    pub fn project_root(&self) -> &Path {
        &self.project_root
    }

    pub fn paths(&self) -> &StorePaths {
        &self.paths
    }

    pub fn ensure_layout(&self) -> Result<()> {
        fs::create_dir_all(&self.paths.index_dir)?;
        fs::write(&self.paths.gitignore_path, GITIGNORE_CONTENTS)?;
        Ok(())
    }

    pub fn read_manifest(&self) -> Result<Option<Manifest>> {
        if !self.paths.manifest_path.exists() {
            return Ok(None);
        }

        let text = fs::read_to_string(&self.paths.manifest_path)?;
        let manifest = serde_json::from_str(&text)?;
        Ok(Some(manifest))
    }

    pub fn write_manifest(&self, manifest: &Manifest) -> Result<()> {
        self.ensure_layout()?;

        let temp_path = self.paths.manifest_path.with_extension("json.tmp");
        let bytes = serde_json::to_vec_pretty(manifest)?;
        fs::write(&temp_path, bytes)?;
        fs::rename(temp_path, &self.paths.manifest_path)?;
        Ok(())
    }

    pub fn validate_manifest_compatibility(
        &self,
        expected_model: &str,
        expected_dimensions: u16,
    ) -> Result<Option<Manifest>> {
        self.read_manifest()?
            .map(|manifest| {
                validate_manifest_compatibility(manifest, expected_model, expected_dimensions)
            })
            .transpose()
    }
}

fn validate_manifest_compatibility(
    manifest: Manifest,
    expected_model: &str,
    expected_dimensions: u16,
) -> Result<Manifest> {
    if manifest.schema_version != SCHEMA_VERSION {
        return Err(ClaudixError::SchemaMismatch {
            store: manifest.schema_version,
            binary: SCHEMA_VERSION,
            recovery: RecoveryHint(
                "Reindex the project to rebuild the store with the current schema version",
            ),
        });
    }

    if manifest.embedding_model != expected_model {
        return Err(ClaudixError::EmbeddingModelMismatch {
            store_model: manifest.embedding_model,
            active_model: expected_model.to_owned(),
            recovery: RecoveryHint(
                "Reindex the project after changing the configured embedding model",
            ),
        });
    }

    if manifest.dimensions != expected_dimensions {
        return Err(ClaudixError::DimensionMismatch {
            store_dim: manifest.dimensions,
            model_dim: expected_dimensions,
            recovery: RecoveryHint(
                "Reindex the project after changing the configured embedding dimensions",
            ),
        });
    }

    Ok(manifest)
}

fn resolve_project_path(project_root: &Path, relative_path: &Path) -> Result<PathBuf> {
    reject_path_escape(relative_path)?;

    let resolved = project_root.join(relative_path);
    if resolved.starts_with(project_root) {
        return Ok(resolved);
    }

    Err(ClaudixError::PathTraversal {
        path: resolved,
        recovery: RecoveryHint("Only use store paths inside $CLAUDE_PROJECT_DIR"),
    })
}

fn reject_path_escape(path: &Path) -> Result<()> {
    if path.is_absolute() {
        return Err(ClaudixError::PathTraversal {
            path: path.to_path_buf(),
            recovery: RecoveryHint("Only use store paths inside $CLAUDE_PROJECT_DIR"),
        });
    }

    for component in path.components() {
        if matches!(
            component,
            Component::ParentDir | Component::RootDir | Component::Prefix(_)
        ) {
            return Err(ClaudixError::PathTraversal {
                path: path.to_path_buf(),
                recovery: RecoveryHint("Only use store paths inside $CLAUDE_PROJECT_DIR"),
            });
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn manifest_with_schema(
        schema_version: u32,
        embedding_model: &str,
        dimensions: u16,
    ) -> Manifest {
        let mut manifest = Manifest::new(embedding_model, dimensions);
        manifest.schema_version = schema_version;
        manifest
    }

    #[test]
    fn manifest_new_uses_schema_defaults() {
        let manifest = Manifest::new("stub-model", 512);

        assert_eq!(manifest.schema_version, SCHEMA_VERSION);
        assert_eq!(manifest.embedding_model, "stub-model");
        assert_eq!(manifest.dimensions, 512);
        assert_eq!(manifest.chunk_count, 0);
        assert_eq!(manifest.file_count, 0);
        assert!(manifest.last_full_index_at.is_none());
        assert!(manifest.last_incremental_at.is_none());
    }

    #[test]
    fn store_resolves_default_layout_inside_project() {
        let project_root = tempdir();
        assert!(project_root.is_ok());
        let project_root = project_root.ok().unwrap_or_else(|| unreachable!());

        let store = Store::new(project_root.path(), &Config::default());
        assert!(store.is_ok());
        let store = store.ok().unwrap_or_else(|| unreachable!());

        assert_eq!(store.project_root(), project_root.path());
        assert_eq!(
            store.paths().state_dir(),
            project_root.path().join(".claudix").as_path()
        );
        assert_eq!(
            store.paths().index_dir(),
            project_root.path().join(".claudix/index").as_path()
        );
        assert_eq!(
            store.paths().manifest_path(),
            project_root.path().join(".claudix/manifest.json").as_path()
        );
    }

    #[test]
    fn ensure_layout_creates_index_state_and_gitignore() {
        let project_root = tempdir();
        assert!(project_root.is_ok());
        let project_root = project_root.ok().unwrap_or_else(|| unreachable!());

        let store = Store::new(project_root.path(), &Config::default());
        assert!(store.is_ok());
        let store = store.ok().unwrap_or_else(|| unreachable!());

        assert!(store.ensure_layout().is_ok());
        assert!(store.paths().state_dir().exists());
        assert!(store.paths().index_dir().exists());

        let gitignore = fs::read_to_string(store.paths().gitignore_path());
        assert!(gitignore.is_ok());
        assert_eq!(gitignore.ok().unwrap_or_else(|| unreachable!()), "*\n");
    }

    #[test]
    fn manifest_round_trips_through_disk() {
        let project_root = tempdir();
        assert!(project_root.is_ok());
        let project_root = project_root.ok().unwrap_or_else(|| unreachable!());

        let store = Store::new(project_root.path(), &Config::default());
        assert!(store.is_ok());
        let store = store.ok().unwrap_or_else(|| unreachable!());

        let mut manifest = Manifest::new("stub-model", 384);
        manifest.last_full_index_at = Some("2026-04-27T12:00:00Z".to_owned());
        manifest.chunk_count = 42;
        manifest.file_count = 7;

        assert!(store.write_manifest(&manifest).is_ok());

        let loaded = store.read_manifest();
        assert!(loaded.is_ok());
        assert_eq!(
            loaded.ok().unwrap_or_else(|| unreachable!()),
            Some(manifest)
        );
    }

    #[test]
    fn missing_manifest_returns_none() {
        let project_root = tempdir();
        assert!(project_root.is_ok());
        let project_root = project_root.ok().unwrap_or_else(|| unreachable!());

        let store = Store::new(project_root.path(), &Config::default());
        assert!(store.is_ok());
        let store = store.ok().unwrap_or_else(|| unreachable!());

        let manifest = store.read_manifest();
        assert!(manifest.is_ok());
        assert!(manifest.ok().unwrap_or_else(|| unreachable!()).is_none());
    }

    #[test]
    fn store_rejects_escape_paths() {
        let project_root = tempdir();
        assert!(project_root.is_ok());
        let project_root = project_root.ok().unwrap_or_else(|| unreachable!());

        let mut config = Config::default();
        config.paths.index_dir = PathBuf::from("../outside/index");

        let store = Store::new(project_root.path(), &config);
        assert!(matches!(store, Err(ClaudixError::PathTraversal { .. })));
    }

    #[test]
    fn manifest_compatibility_accepts_matching_state() {
        let project_root = tempdir();
        assert!(project_root.is_ok());
        let project_root = project_root.ok().unwrap_or_else(|| unreachable!());

        let store = Store::new(project_root.path(), &Config::default());
        assert!(store.is_ok());
        let store = store.ok().unwrap_or_else(|| unreachable!());

        let manifest = Manifest::new("stub-model", 512);
        assert!(store.write_manifest(&manifest).is_ok());

        let loaded = store.validate_manifest_compatibility("stub-model", 512);
        assert!(loaded.is_ok());
        assert_eq!(
            loaded.ok().unwrap_or_else(|| unreachable!()),
            Some(manifest)
        );
    }

    #[test]
    fn manifest_compatibility_rejects_schema_mismatch() {
        let project_root = tempdir();
        assert!(project_root.is_ok());
        let project_root = project_root.ok().unwrap_or_else(|| unreachable!());

        let store = Store::new(project_root.path(), &Config::default());
        assert!(store.is_ok());
        let store = store.ok().unwrap_or_else(|| unreachable!());

        let manifest = manifest_with_schema(SCHEMA_VERSION + 1, "stub-model", 512);
        assert!(store.write_manifest(&manifest).is_ok());

        let error = store.validate_manifest_compatibility("stub-model", 512);
        assert!(matches!(error, Err(ClaudixError::SchemaMismatch { .. })));
    }

    #[test]
    fn manifest_compatibility_rejects_model_mismatch() {
        let project_root = tempdir();
        assert!(project_root.is_ok());
        let project_root = project_root.ok().unwrap_or_else(|| unreachable!());

        let store = Store::new(project_root.path(), &Config::default());
        assert!(store.is_ok());
        let store = store.ok().unwrap_or_else(|| unreachable!());

        let manifest = Manifest::new("old-model", 512);
        assert!(store.write_manifest(&manifest).is_ok());

        let error = store.validate_manifest_compatibility("new-model", 512);
        assert!(matches!(
            error,
            Err(ClaudixError::EmbeddingModelMismatch { .. })
        ));
    }

    #[test]
    fn manifest_compatibility_rejects_dimension_mismatch() {
        let project_root = tempdir();
        assert!(project_root.is_ok());
        let project_root = project_root.ok().unwrap_or_else(|| unreachable!());

        let store = Store::new(project_root.path(), &Config::default());
        assert!(store.is_ok());
        let store = store.ok().unwrap_or_else(|| unreachable!());

        let manifest = Manifest::new("stub-model", 384);
        assert!(store.write_manifest(&manifest).is_ok());

        let error = store.validate_manifest_compatibility("stub-model", 512);
        assert!(matches!(error, Err(ClaudixError::DimensionMismatch { .. })));
    }
}
