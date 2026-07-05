use std::path::{Path, PathBuf};
use std::sync::Arc;

use tokio::fs;

use crate::Claudix;
use crate::config;
use crate::error::{ClaudixError, RecoveryHint, Result};
use crate::prompts::hints;

use super::{InstallOutput, SetupState, canonical_project_root};

pub async fn run_install(project_root: impl AsRef<Path>) -> Result<InstallOutput> {
    let project_root = canonical_project_root(project_root.as_ref())?;
    let source_root = install_source_root(&project_root)?;
    let plugin_root = plugin_root_from_env(&project_root, std::env::var_os("CLAUDE_PLUGIN_ROOT"))?;
    let binary_path = plugin_root.join("bin").join("claudix");
    let config_path = global_config_path()?;

    install_plugin_assets(&source_root, &plugin_root).await?;

    let wrote_config = ensure_global_config(&config_path).await?;
    let config = config::load(&project_root)?;
    let claudix = Claudix::new(project_root, Arc::new(config)).await?;
    let embedding_healthy = claudix.embedder_health_check().await.is_ok();

    Ok(InstallOutput {
        plugin_root: plugin_root.display().to_string(),
        binary_path: binary_path.display().to_string(),
        config_path: config_path.display().to_string(),
        wrote_config,
        embedding_healthy,
    })
}

pub async fn setup_state(project_root: impl AsRef<Path>) -> SetupState {
    let project_root = project_root.as_ref();
    let mut missing = Vec::new();

    if plugin_root_from_env(project_root, std::env::var_os("CLAUDE_PLUGIN_ROOT")).is_err() {
        missing.push("plugin files");
    }
    match global_config_path() {
        Ok(config_path) if config_path.try_exists().unwrap_or(false) => {}
        _ => missing.push("global config"),
    }
    if config::load(project_root).is_err() {
        missing.push("valid config");
    }

    if missing.is_empty() {
        SetupState::Ready
    } else {
        SetupState::Missing(missing)
    }
}

async fn install_plugin_assets(project_root: &Path, plugin_root: &Path) -> Result<bool> {
    let mut changed = false;
    changed |= copy_plugin_asset(
        project_root,
        ".claude-plugin/plugin.json",
        plugin_root.join(".claude-plugin").join("plugin.json"),
    )
    .await?;
    changed |= copy_plugin_asset(
        project_root,
        "hooks/hooks.json",
        plugin_root.join("hooks").join("hooks.json"),
    )
    .await?;
    changed |= copy_plugin_asset(
        project_root,
        "bin/claudix-bootstrap.js",
        plugin_root.join("bin").join("claudix-bootstrap.js"),
    )
    .await?;
    changed |=
        copy_plugin_directory(project_root, "commands", plugin_root.join("commands")).await?;
    Ok(changed)
}

async fn copy_plugin_asset(
    project_root: &Path,
    source_relative: &str,
    destination: PathBuf,
) -> Result<bool> {
    let source = required_plugin_asset(project_root, source_relative).await?;

    if source == destination || files_match(&source, &destination).await? {
        return Ok(false);
    }

    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent).await?;
    }
    fs::copy(source, &destination).await?;
    Ok(true)
}

async fn copy_plugin_directory(
    project_root: &Path,
    source_relative: &str,
    destination: PathBuf,
) -> Result<bool> {
    let source = required_plugin_asset(project_root, source_relative).await?;
    if source == destination || directories_match(&source, &destination).await? {
        return Ok(false);
    }

    if fs::try_exists(&destination).await? {
        fs::remove_dir_all(&destination).await?;
    }
    fs::create_dir_all(&destination).await?;

    let mut entries = fs::read_dir(source).await?;
    while let Some(entry) = entries.next_entry().await? {
        let file_type = entry.file_type().await?;
        if file_type.is_file() {
            let destination_file = destination.join(entry.file_name());
            fs::copy(entry.path(), &destination_file).await?;
            if destination_file
                .extension()
                .is_some_and(|extension| extension == "sh")
            {
                make_executable(&destination_file).await?;
            }
        }
    }

    Ok(true)
}

async fn files_match(left: &Path, right: &Path) -> Result<bool> {
    if !fs::try_exists(right).await? {
        return Ok(false);
    }

    let left_metadata = fs::metadata(left).await?;
    let right_metadata = fs::metadata(right).await?;
    if left_metadata.len() != right_metadata.len() {
        return Ok(false);
    }

    Ok(fs::read(left).await? == fs::read(right).await?)
}

async fn directories_match(left: &Path, right: &Path) -> Result<bool> {
    if !fs::try_exists(right).await? {
        return Ok(false);
    }

    let mut left_entries = directory_file_names(left).await?;
    let mut right_entries = directory_file_names(right).await?;
    left_entries.sort();
    right_entries.sort();
    if left_entries != right_entries {
        return Ok(false);
    }

    for entry in left_entries {
        if !files_match(&left.join(&entry), &right.join(&entry)).await? {
            return Ok(false);
        }
    }

    Ok(true)
}

async fn directory_file_names(path: &Path) -> Result<Vec<std::ffi::OsString>> {
    let mut file_names = Vec::new();
    let mut entries = fs::read_dir(path).await?;
    while let Some(entry) = entries.next_entry().await? {
        if entry.file_type().await?.is_file() {
            file_names.push(entry.file_name());
        }
    }
    Ok(file_names)
}

async fn required_plugin_asset(project_root: &Path, source_relative: &str) -> Result<PathBuf> {
    let source = project_root.join(source_relative);
    if fs::try_exists(&source).await? {
        return Ok(source);
    }

    Err(ClaudixError::ConfigInvalid {
        message: format!("required plugin asset missing: {}", source.display()),
        recovery: RecoveryHint(hints::RESTORE_PLUGIN_ASSETS),
    })
}

fn local_plugin_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("claudix-plugin")
}

async fn make_executable(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        let mut permissions = fs::metadata(path).await?.permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(path, permissions).await?;
    }

    Ok(())
}

async fn ensure_global_config(config_path: &Path) -> Result<bool> {
    if fs::try_exists(config_path).await? {
        return Ok(false);
    }

    if let Some(parent) = config_path.parent() {
        fs::create_dir_all(parent).await?;
    }

    fs::write(config_path, default_global_config()).await?;
    Ok(true)
}

fn default_global_config() -> &'static str {
    "\
# Global claudix configuration — uncomment and edit as needed.
# Project-level overrides go in .claude/claudix.toml (project wins).
# watch = false                  # opt-in file watcher for saved files

[embedding]
# provider = \"bundled\"           # bundled | http
# model = \"bge-small-en-v1.5\"    # only used by bundled provider
# dimensions = 384               # must match the model
# endpoint = \"http://localhost:11434\"  # for http provider (LM Studio / Ollama)

[indexing]
# reindex_after_hours = 24       # auto-reindex threshold on session start

[hooks]
# auto_index_on_session_start = true  # trigger background reindex when stale
# intercept_grep = true               # redirect conceptual Grep/rg to search_code
# auto_reembed_on_edit = true         # re-embed edited files in background
# reindex_debounce_secs = 10          # watch = false: coalesce rapid edits, reindex after N idle secs
# reindex_max_wait_secs = 60          # hard cap so a continuously-edited file still reindexes
# surface_related_on_read = false     # opt-in: surface related code on Read (off by default)

[search]
# top_k = 10                     # default result count for search_code
# cross_repos = [\"/path/to/another/repo\"]  # extra already-indexed repos to search read-only
"
}

fn install_source_root(project_root: &Path) -> Result<PathBuf> {
    if is_claudix_plugin_root(project_root) {
        return Ok(project_root.to_path_buf());
    }

    match std::env::var_os("CLAUDE_PLUGIN_ROOT") {
        Some(path) => Ok(PathBuf::from(path)),
        None => Ok(PathBuf::from(env!("CARGO_MANIFEST_DIR"))),
    }
}

fn plugin_root_from_env(
    project_root: &Path,
    plugin_root_env: Option<std::ffi::OsString>,
) -> Result<PathBuf> {
    if let Some(path) = plugin_root_env {
        let plugin_root = PathBuf::from(path);
        if is_claudix_plugin_root(&plugin_root) {
            return Ok(plugin_root);
        }
    }

    if is_claudix_plugin_root(project_root) {
        return Ok(local_plugin_root());
    }

    Err(ClaudixError::ConfigInvalid {
        message: "CLAUDE_PLUGIN_ROOT is not set".into(),
        recovery: RecoveryHint(hints::INSTALL_FROM_PLUGIN_ROOT),
    })
}

fn is_claudix_plugin_root(path: &Path) -> bool {
    let manifest_path = path.join(".claude-plugin").join("plugin.json");
    let Ok(manifest) = std::fs::read_to_string(manifest_path) else {
        return false;
    };

    manifest.contains("\"name\": \"claudix\"")
}

fn global_config_path() -> Result<PathBuf> {
    dirs::home_dir()
        .map(|home| home.join(".claude").join("claudix.toml"))
        .ok_or_else(|| ClaudixError::ConfigInvalid {
            message: "home directory is not available".into(),
            recovery: RecoveryHint(hints::SET_HOME_FOR_INSTALL),
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    mod fixture {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/common/fixture.rs"
        ));
    }

    use fixture::TestFixture;

    #[tokio::test]
    async fn ensure_global_config_writes_default_once() {
        let temp = tempdir();
        assert!(temp.is_ok());
        let temp = temp.ok().unwrap_or_else(|| unreachable!());
        let config_path = temp.path().join(".claude").join("claudix.toml");

        let wrote_config = ensure_global_config(&config_path).await;
        assert!(wrote_config.is_ok());
        assert!(wrote_config.ok().unwrap_or(false));

        let contents = fs::read_to_string(&config_path).await;
        assert!(contents.is_ok());
        assert!(contents.ok().unwrap_or_default().contains("[embedding]"));

        let wrote_config = ensure_global_config(&config_path).await;
        assert!(wrote_config.is_ok());
        assert!(!wrote_config.ok().unwrap_or(true));
    }

    #[tokio::test]
    async fn install_copies_plugin_assets_into_plugin_root() {
        let fixture = TestFixture::new("small_rust");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());
        let plugin_root = fixture.root().join("plugin-root");

        let result =
            install_plugin_assets(Path::new(env!("CARGO_MANIFEST_DIR")), &plugin_root).await;
        assert!(result.is_ok());
        assert!(result.ok().unwrap_or(false));

        let second_result =
            install_plugin_assets(Path::new(env!("CARGO_MANIFEST_DIR")), &plugin_root).await;
        assert!(second_result.is_ok());
        assert!(!second_result.ok().unwrap_or(true));

        let plugin_manifest =
            fs::read_to_string(plugin_root.join(".claude-plugin").join("plugin.json")).await;
        assert!(plugin_manifest.is_ok());
        let plugin_manifest = plugin_manifest.ok().unwrap_or_default();
        assert!(plugin_manifest.contains("\"name\": \"claudix\""));
        assert!(plugin_manifest.contains("\"mcpServers\""));
        assert!(plugin_manifest.contains("\"command\": \"node\""));
        assert!(plugin_manifest.contains("\"mcp\""));
        assert!(plugin_manifest.contains("claudix-bootstrap.js"));

        let hooks_manifest = fs::read_to_string(plugin_root.join("hooks").join("hooks.json")).await;
        assert!(hooks_manifest.is_ok());
        let hooks_manifest = hooks_manifest.ok().unwrap_or_default();
        assert!(hooks_manifest.contains("bin/claudix-bootstrap.js"));
        assert!(hooks_manifest.contains("hook SessionStart"));

        let bootstrap =
            fs::read_to_string(plugin_root.join("bin").join("claudix-bootstrap.js")).await;
        assert!(bootstrap.is_ok());
        let bootstrap = bootstrap.ok().unwrap_or_default();
        assert!(
            bootstrap.contains("readWantedVersion"),
            "bootstrap source missing"
        );

        let search_command =
            fs::read_to_string(plugin_root.join("commands").join("search.md")).await;
        assert!(search_command.is_ok());
        let search_command = search_command.ok().unwrap_or_default();
        assert!(search_command.contains("!`claudix search"));
        assert!(!search_command.contains("CLAUDE_PLUGIN_ROOT"));

        // scripts/ is no longer an install asset (bash wrappers deleted).
        let scripts_copied = fs::try_exists(plugin_root.join("scripts"))
            .await
            .unwrap_or(false);
        assert!(
            !scripts_copied,
            "scripts/ must not be copied into the plugin root"
        );
    }

    #[test]
    fn plugin_root_uses_claudix_environment_value_outside_local_checkout() {
        let fixture = TestFixture::new("small_rust");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());
        let env_root = fixture.root().join("env-plugin-root");
        assert!(std::fs::create_dir_all(env_root.join(".claude-plugin")).is_ok());
        assert!(
            std::fs::write(
                env_root.join(".claude-plugin").join("plugin.json"),
                "{\"name\": \"claudix\"}",
            )
            .is_ok()
        );
        let project_root = fixture.root().join("project");
        assert!(std::fs::create_dir_all(&project_root).is_ok());

        let result = plugin_root_from_env(&project_root, Some(env_root.clone().into_os_string()));
        assert!(result.is_ok());
        assert_eq!(result.ok().unwrap_or_default(), env_root);
    }

    #[test]
    fn plugin_root_ignores_foreign_environment_value() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let fixture = TestFixture::new("small_rust");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());

        let result = plugin_root_from_env(root, Some(fixture.root().as_os_str().to_os_string()));
        assert!(result.is_ok());
        assert_eq!(
            result.ok().unwrap_or_default(),
            root.join("target").join("claudix-plugin")
        );
    }

    #[test]
    fn plugin_root_falls_back_to_local_manifest() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));

        let result = plugin_root_from_env(root, None);
        assert!(result.is_ok());
        assert_eq!(
            result.ok().unwrap_or_default(),
            root.join("target").join("claudix-plugin")
        );
    }

    #[test]
    fn plugin_root_requires_environment_or_local_manifest() {
        let fixture = TestFixture::new("small_rust");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());

        let result = plugin_root_from_env(fixture.root(), None);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn setup_state_reports_missing_plugin_files() {
        let fixture = TestFixture::new("small_rust");
        assert!(fixture.is_ok());
        let fixture = fixture.ok().unwrap_or_else(|| unreachable!());

        let setup_state = super::setup_state(fixture.root()).await;
        assert!(
            matches!(setup_state, SetupState::Missing(parts) if parts.contains(&"plugin files"))
        );
    }

    #[test]
    fn default_global_config_includes_commented_defaults() {
        let config = default_global_config();
        assert!(config.contains("[embedding]"));
        assert!(config.contains("provider = \"bundled\""));
        assert!(config.contains("reindex_after_hours = 24"));
        assert!(config.contains("[hooks]"));
        assert!(config.contains("intercept_grep = true"));
        assert!(config.contains("auto_reembed_on_edit = true"));
        assert!(config.contains("[search]"));
        assert!(config.contains("top_k = 10"));
    }
}
