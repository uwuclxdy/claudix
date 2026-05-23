mod common {
    pub mod fixture {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/common/fixture.rs"
        ));
    }

    pub mod config_support {
        use claudix;

        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/common/config_support.rs"
        ));
    }
}

use std::sync::Arc;

use claudix::search::SearchQuery;
use claudix::types::{Language, RelativePath};
use claudix::{Claudix, ClaudixError, IndexStats};
use common::config_support::{stub_config, stub_config_with_model};
use common::fixture::TestFixture;

// ── a ──────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn full_index_enumerates_and_persists_chunks() {
    let fixture = TestFixture::new("small_rust");
    assert!(fixture.is_ok(), "fixture setup failed");
    let fixture = fixture.ok().unwrap_or_else(|| unreachable!());

    let claudix = Claudix::new(fixture.root().to_path_buf(), Arc::new(stub_config())).await;
    assert!(claudix.is_ok(), "Claudix::new failed");
    let claudix = claudix.ok().unwrap_or_else(|| unreachable!());

    let stats = claudix.index_full(&mut ()).await;
    assert!(stats.is_ok(), "index_full failed: {stats:?}");
    let stats = stats.ok().unwrap_or_else(|| unreachable!());

    assert_eq!(
        stats,
        IndexStats {
            file_count: 2,
            chunk_count: 3,
        },
        "unexpected index stats: {stats:?}"
    );
}

// ── b ──────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn full_reindex_after_file_edit_reflects_changes() {
    let fixture = TestFixture::new("small_rust");
    assert!(fixture.is_ok(), "fixture setup failed");
    let fixture = fixture.ok().unwrap_or_else(|| unreachable!());

    let claudix = Claudix::new(fixture.root().to_path_buf(), Arc::new(stub_config())).await;
    assert!(claudix.is_ok(), "Claudix::new failed");
    let claudix = claudix.ok().unwrap_or_else(|| unreachable!());

    let first = claudix.index_full(&mut ()).await;
    assert!(first.is_ok(), "initial index_full failed: {first:?}");

    let write = std::fs::write(
        fixture.root().join("src/math.rs"),
        "pub fn square(x: i32) -> i32 { x * x }\n",
    );
    assert!(write.is_ok(), "write math.rs failed");

    let second = claudix.index_full(&mut ()).await;
    assert!(second.is_ok(), "reindex failed: {second:?}");

    let results = claudix
        .search(SearchQuery {
            query: "square".to_owned(),
            top_k: 10,
            language_filter: None,
            path_prefix: None,
            repos: Vec::new(),
        })
        .await;
    assert!(results.is_ok(), "search failed: {results:?}");
    let results = results.ok().unwrap_or_else(|| unreachable!()).results;

    let found = results
        .iter()
        .any(|r| r.chunk.name.as_deref() == Some("square"));
    assert!(
        found,
        "expected chunk named 'square' in results: {results:?}"
    );
}

// ── c ──────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn reindex_file_updates_target_preserves_others() {
    let fixture = TestFixture::new("small_rust");
    assert!(fixture.is_ok(), "fixture setup failed");
    let fixture = fixture.ok().unwrap_or_else(|| unreachable!());

    let claudix = Claudix::new(fixture.root().to_path_buf(), Arc::new(stub_config())).await;
    assert!(claudix.is_ok(), "Claudix::new failed");
    let claudix = claudix.ok().unwrap_or_else(|| unreachable!());

    let first = claudix.index_full(&mut ()).await;
    assert!(first.is_ok(), "initial index_full failed: {first:?}");

    let write = std::fs::write(
        fixture.root().join("src/math.rs"),
        "pub fn square(x: i32) -> i32 { x * x }\n",
    );
    assert!(write.is_ok(), "write math.rs failed");

    let reindex = claudix
        .reindex_file(&fixture.root().join("src/math.rs"))
        .await;
    assert!(reindex.is_ok(), "reindex_file failed: {reindex:?}");

    let greet_results = claudix
        .search(SearchQuery {
            query: "greet".to_owned(),
            top_k: 10,
            language_filter: None,
            path_prefix: None,
            repos: Vec::new(),
        })
        .await;
    assert!(
        greet_results.is_ok(),
        "search(greet) failed: {greet_results:?}"
    );
    let greet_results = greet_results.ok().unwrap_or_else(|| unreachable!()).results;
    assert!(
        !greet_results.is_empty(),
        "expected greet to still be found after reindex_file"
    );

    let square_results = claudix
        .search(SearchQuery {
            query: "square".to_owned(),
            top_k: 10,
            language_filter: None,
            path_prefix: None,
            repos: Vec::new(),
        })
        .await;
    assert!(
        square_results.is_ok(),
        "search(square) failed: {square_results:?}"
    );
    let square_results = square_results
        .ok()
        .unwrap_or_else(|| unreachable!())
        .results;
    assert!(
        square_results
            .iter()
            .any(|r| r.chunk.name.as_deref() == Some("square")),
        "expected 'square' chunk after reindex_file: {square_results:?}"
    );
}

// ── d ──────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn search_language_filter_excludes_other_languages() {
    let fixture = TestFixture::new("small_rust");
    assert!(fixture.is_ok(), "fixture setup failed");
    let fixture = fixture.ok().unwrap_or_else(|| unreachable!());

    let claudix = Claudix::new(fixture.root().to_path_buf(), Arc::new(stub_config())).await;
    assert!(claudix.is_ok(), "Claudix::new failed");
    let claudix = claudix.ok().unwrap_or_else(|| unreachable!());

    let index = claudix.index_full(&mut ()).await;
    assert!(index.is_ok(), "index_full failed: {index:?}");

    let results = claudix
        .search(SearchQuery {
            query: "greet add".to_owned(),
            top_k: 10,
            language_filter: Some(vec![Language::Rust]),
            path_prefix: None,
            repos: Vec::new(),
        })
        .await;
    assert!(
        results.is_ok(),
        "search with language filter failed: {results:?}"
    );
    let results = results.ok().unwrap_or_else(|| unreachable!()).results;

    for result in &results {
        assert_eq!(
            result.chunk.language,
            Language::Rust,
            "non-Rust chunk leaked through language filter: {:?}",
            result.chunk
        );
    }
}

// ── e ──────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn search_path_prefix_restricts_results() {
    let fixture = TestFixture::new("small_rust");
    assert!(fixture.is_ok(), "fixture setup failed");
    let fixture = fixture.ok().unwrap_or_else(|| unreachable!());

    let claudix = Claudix::new(fixture.root().to_path_buf(), Arc::new(stub_config())).await;
    assert!(claudix.is_ok(), "Claudix::new failed");
    let claudix = claudix.ok().unwrap_or_else(|| unreachable!());

    let index = claudix.index_full(&mut ()).await;
    assert!(index.is_ok(), "index_full failed: {index:?}");

    let results = claudix
        .search(SearchQuery {
            query: "add".to_owned(),
            top_k: 10,
            language_filter: None,
            path_prefix: Some(RelativePath::new("src/math")),
            repos: Vec::new(),
        })
        .await;
    assert!(
        results.is_ok(),
        "search with path prefix failed: {results:?}"
    );
    let results = results.ok().unwrap_or_else(|| unreachable!()).results;

    for result in &results {
        assert!(
            result
                .chunk
                .file_path
                .starts_with(&RelativePath::new("src/math")),
            "result outside path prefix: {:?}",
            result.chunk.file_path
        );
    }
}

// ── f ──────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn indexignore_excludes_skip_indexinclude_reinstates_reinclude() {
    let fixture = TestFixture::new("ignore_overrides");
    assert!(fixture.is_ok(), "fixture setup failed");
    let fixture = fixture.ok().unwrap_or_else(|| unreachable!());

    let claudix = Claudix::new(fixture.root().to_path_buf(), Arc::new(stub_config())).await;
    assert!(claudix.is_ok(), "Claudix::new failed");
    let claudix = claudix.ok().unwrap_or_else(|| unreachable!());

    let stats = claudix.index_full(&mut ()).await;
    assert!(stats.is_ok(), "index_full failed: {stats:?}");
    let stats = stats.ok().unwrap_or_else(|| unreachable!());

    assert_eq!(
        stats.file_count, 2,
        "expected 2 files (keep.rs + reinclude.rs), got {}: {stats:?}",
        stats.file_count
    );

    let results = claudix
        .search(SearchQuery {
            query: "reinclude".to_owned(),
            top_k: 10,
            language_filter: None,
            path_prefix: None,
            repos: Vec::new(),
        })
        .await;
    assert!(results.is_ok(), "search(reinclude) failed: {results:?}");
    let results = results.ok().unwrap_or_else(|| unreachable!()).results;

    let from_reinclude = results
        .iter()
        .any(|r| r.chunk.file_path.as_str().contains("reinclude"));
    assert!(
        from_reinclude,
        "expected at least one result from reinclude.rs: {results:?}"
    );
}

// ── g ──────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn schema_model_mismatch_errors_on_open() {
    let fixture = TestFixture::new("small_rust");
    assert!(fixture.is_ok(), "fixture setup failed");
    let fixture = fixture.ok().unwrap_or_else(|| unreachable!());

    let claudix_v1 = Claudix::new(fixture.root().to_path_buf(), Arc::new(stub_config())).await;
    assert!(claudix_v1.is_ok(), "Claudix::new (v1) failed");
    let claudix_v1 = claudix_v1.ok().unwrap_or_else(|| unreachable!());

    let index = claudix_v1.index_full(&mut ()).await;
    assert!(index.is_ok(), "index_full failed: {index:?}");

    drop(claudix_v1);

    let claudix_v2 = Claudix::new(
        fixture.root().to_path_buf(),
        Arc::new(stub_config_with_model("stub-v2")),
    )
    .await;
    assert!(
        claudix_v2.is_err(),
        "expected EmbeddingModelMismatch error, but Claudix::new succeeded"
    );

    let error = claudix_v2.err().unwrap_or_else(|| unreachable!());
    assert!(
        matches!(error, ClaudixError::EmbeddingModelMismatch { .. }),
        "expected EmbeddingModelMismatch, got: {error:?}"
    );
}

// ── h ──────────────────────────────────────────────────────────────────────

#[test]
#[ignore = "spawns the compiled binary"]
fn hook_exits_zero_with_corrupt_manifest() {
    use assert_cmd::cargo::cargo_bin;
    use std::process::{Command, Stdio};

    let fixture = TestFixture::new("small_rust");
    assert!(fixture.is_ok(), "fixture setup failed");
    let fixture = fixture.ok().unwrap_or_else(|| unreachable!());
    let root = fixture.root();

    let config_contents = "[embedding]\nmodel = \"stub-v1\"\ndimensions = 8\n";
    let config_path = root.join("test-config.toml");
    let write_cfg = std::fs::write(&config_path, config_contents);
    assert!(write_cfg.is_ok(), "write test config failed");

    let mkdir = std::fs::create_dir_all(root.join(".claudix/index"));
    assert!(mkdir.is_ok(), "create index dir failed");

    // Manifest lives at <state_dir>/manifest.json (default: .claudix/manifest.json)
    let write_manifest = std::fs::write(root.join(".claudix/manifest.json"), "not valid json {{{{");
    assert!(write_manifest.is_ok(), "write corrupt manifest failed");

    let output = Command::new(cargo_bin("claudix"))
        .current_dir(root)
        .env("CLAUDE_PROJECT_DIR", root)
        .env("CIRRUS_CONFIG", &config_path)
        .args(["hook", "SessionStart"])
        .stdin(Stdio::null())
        .output();
    assert!(output.is_ok(), "failed to spawn claudix binary");
    let output = output.ok().unwrap_or_else(|| unreachable!());

    assert!(
        output.status.success(),
        "hook must exit 0 even with corrupt store, got: {}",
        output.status
    );
}

#[test]
fn index_progress_writes_status_to_stderr() {
    use assert_cmd::cargo::cargo_bin;
    use std::process::{Command, Stdio};

    let fixture = TestFixture::new("small_rust");
    assert!(fixture.is_ok(), "fixture setup failed");
    let fixture = fixture.ok().unwrap_or_else(|| unreachable!());
    let root = fixture.root();

    let config_path = root.join("test-config.toml");
    let write_cfg = std::fs::write(
        &config_path,
        "[embedding]\nmodel = \"stub-v1\"\ndimensions = 8\n",
    );
    assert!(write_cfg.is_ok(), "write test config failed");

    let output = Command::new(cargo_bin("claudix"))
        .current_dir(root)
        .env("CLAUDE_PROJECT_DIR", root)
        .env("CIRRUS_CONFIG", &config_path)
        .args(["index", "--progress"])
        .stdin(Stdio::null())
        .output();
    assert!(output.is_ok(), "failed to spawn claudix binary");
    let output = output.ok().unwrap_or_else(|| unreachable!());

    assert!(
        output.status.success(),
        "index --progress failed: {}",
        output.status
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        stdout.contains("indexed 2 files into 3 chunks"),
        "expected final summary on stdout, got: {stdout}"
    );
    assert!(
        stderr.contains("indexed src/lib.rs") && stderr.contains("indexed src/math.rs"),
        "expected indexed files on stderr, got: {stderr}"
    );
    assert!(
        stderr.contains("skipped test-config.toml: no indexable chunks"),
        "expected skipped reason on stderr, got: {stderr}"
    );
    assert!(
        !stdout.contains("indexed src/lib.rs"),
        "progress status leaked to stdout: {stdout}"
    );
}

#[test]
fn index_progress_reports_verified_files_after_reindex() {
    use assert_cmd::cargo::cargo_bin;
    use std::process::{Command, Stdio};

    let fixture = TestFixture::new("small_rust");
    assert!(fixture.is_ok(), "fixture setup failed");
    let fixture = fixture.ok().unwrap_or_else(|| unreachable!());
    let root = fixture.root();

    let config_path = root.join("test-config.toml");
    let write_cfg = std::fs::write(
        &config_path,
        "[embedding]\nmodel = \"stub-v1\"\ndimensions = 8\n",
    );
    assert!(write_cfg.is_ok(), "write test config failed");

    let first = Command::new(cargo_bin("claudix"))
        .current_dir(root)
        .env("CLAUDE_PROJECT_DIR", root)
        .env("CIRRUS_CONFIG", &config_path)
        .arg("index")
        .stdin(Stdio::null())
        .status();
    assert!(first.is_ok(), "failed to spawn claudix binary");
    assert!(first.ok().unwrap_or_else(|| unreachable!()).success());

    let output = Command::new(cargo_bin("claudix"))
        .current_dir(root)
        .env("CLAUDE_PROJECT_DIR", root)
        .env("CIRRUS_CONFIG", &config_path)
        .args(["index", "--progress"])
        .stdin(Stdio::null())
        .output();
    assert!(output.is_ok(), "failed to spawn claudix binary");
    let output = output.ok().unwrap_or_else(|| unreachable!());

    assert!(output.status.success(), "reindex failed: {}", output.status);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("verified src/lib.rs") && stderr.contains("verified src/math.rs"),
        "expected verified files on stderr, got: {stderr}"
    );
}

#[test]
fn index_without_progress_does_not_write_status() {
    use assert_cmd::cargo::cargo_bin;
    use std::process::{Command, Stdio};

    let fixture = TestFixture::new("small_rust");
    assert!(fixture.is_ok(), "fixture setup failed");
    let fixture = fixture.ok().unwrap_or_else(|| unreachable!());
    let root = fixture.root();

    let config_path = root.join("test-config.toml");
    let write_cfg = std::fs::write(
        &config_path,
        "[embedding]\nmodel = \"stub-v1\"\ndimensions = 8\n",
    );
    assert!(write_cfg.is_ok(), "write test config failed");

    let output = Command::new(cargo_bin("claudix"))
        .current_dir(root)
        .env("CLAUDE_PROJECT_DIR", root)
        .env("CIRRUS_CONFIG", &config_path)
        .arg("index")
        .stdin(Stdio::null())
        .output();
    assert!(output.is_ok(), "failed to spawn claudix binary");
    let output = output.ok().unwrap_or_else(|| unreachable!());

    assert!(output.status.success(), "index failed: {}", output.status);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        stdout.contains("indexed 2 files into 3 chunks"),
        "expected final summary on stdout, got: {stdout}"
    );
    assert!(
        !stderr.contains("indexed src/lib.rs"),
        "unexpected progress status on stderr: {stderr}"
    );
}
