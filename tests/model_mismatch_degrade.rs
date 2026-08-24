#![allow(clippy::unwrap_used, clippy::expect_used)]
#![cfg(feature = "test-stub")]

use std::path::Path;
use std::sync::Arc;

use claudix::cli::{SearchHit, SearchOutput, run_search};
use claudix::prompts::hints;
use claudix::{Claudix, Result};

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

use common::config_support::stub_config;
use common::fixture::TestFixture;

/// Write a fixture config pinning the embedding identity under test, so
/// `config::load` inside `run_search` never falls through to the developer's
/// global config. `[search].cross_repos = []` keeps a global cross-repo list
/// from widening the corpus.
fn write_search_config(root: &Path, model: &str, dimensions: u16) -> std::io::Result<()> {
    let dir = root.join(".claude");
    std::fs::create_dir_all(&dir)?;
    std::fs::write(
        dir.join("claudix.toml"),
        format!(
            "[embedding]\nprovider = \"bundled\"\nmodel = \"{model}\"\ndimensions = {dimensions}\n\n[search]\ncross_repos = []\n"
        ),
    )
}

/// Index the tiny fixture under the default stub identity (`stub-v1`, 8 dims),
/// point the fixture config at the mismatching identity under test, and run the
/// public search path. The fixture is returned so its temp dir outlives the
/// store reads.
async fn search_after_identity_change(
    model: &str,
    dimensions: u16,
) -> (TestFixture, Result<SearchOutput>) {
    let fixture = TestFixture::new("small_rust").expect("fixture setup failed");
    let claudix = Claudix::new(fixture.root().to_path_buf(), Arc::new(stub_config()))
        .await
        .expect("indexed claudix init failed");
    claudix
        .index_full(&mut ())
        .await
        .expect("fixture indexing failed");

    write_search_config(fixture.root(), model, dimensions).expect("config write failed");
    let output = run_search(fixture.root(), "add".to_owned(), Some(10), None, None, None).await;
    (fixture, output)
}

fn flat_hits(output: &SearchOutput) -> Vec<&SearchHit> {
    output
        .groups
        .iter()
        .flat_map(|group| group.hits.iter())
        .collect()
}

/// The task's verify line: a stored index whose embedding model differs from
/// the configured provider degrades to lexical hits plus the reindex hint, no
/// error.
#[tokio::test]
async fn model_mismatch_degrades_to_lexical_plus_reindex_hint() {
    let (_fixture, output) = search_after_identity_change("stub-v2", 8).await;

    assert!(
        output.is_ok(),
        "a model-mismatch search must degrade, not error: {output:?}"
    );
    let output = output.expect("search output");
    let hits = flat_hits(&output);
    assert!(!hits.is_empty(), "lexical fallback must still surface hits");
    assert!(
        hits.iter().any(|hit| hit.name.as_deref() == Some("add")),
        "the 'add' identifier/BM25 hit must survive the degraded ranking"
    );
    assert_eq!(
        output.degraded_hint,
        Some(hints::REINDEX_AFTER_MODEL_CHANGE),
        "a model mismatch must carry the reindex hint"
    );
}

/// Same degradation for a dimension mismatch: same model id, wrong width.
#[tokio::test]
async fn dimension_mismatch_degrades_to_lexical_plus_reindex_hint() {
    let (_fixture, output) = search_after_identity_change("stub-v1", 16).await;

    assert!(
        output.is_ok(),
        "a dimension-mismatch search must degrade, not error: {output:?}"
    );
    let output = output.expect("search output");
    let hits = flat_hits(&output);
    assert!(!hits.is_empty(), "lexical fallback must still surface hits");
    assert!(
        hits.iter().any(|hit| hit.name.as_deref() == Some("add")),
        "the 'add' identifier/BM25 hit must survive the degraded ranking"
    );
    assert_eq!(
        output.degraded_hint,
        Some(hints::REINDEX_AFTER_DIMENSION_CHANGE),
        "a dimension mismatch must carry the reindex hint"
    );
}
