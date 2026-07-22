use std::collections::BTreeMap;
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::error::{ClaudixError, RecoveryHint, Result};
use crate::prompts::hints;
use crate::util::parse_rfc3339;

pub const SCHEMA_VERSION: u32 = 1;
pub(crate) const MANIFEST_FILE_NAME: &str = "manifest.json";

// No `Eq`: `similarity_floor` is an `f32`, which is `PartialEq` but not `Eq`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub schema_version: u32,
    pub embedding_model: String,
    pub dimensions: u16,
    pub last_full_index_at: Option<String>,
    pub last_incremental_at: Option<String>,
    pub chunk_count: u64,
    pub file_count: u64,
    #[serde(default)]
    pub file_hashes: BTreeMap<String, [u8; 16]>,
    /// Corpus-relative cosine floor for related-code hints, computed from this
    /// index's own best-neighbor distribution at full-index time so the cutoff
    /// tracks the embedding pipeline's score scale rather than a fixed constant.
    /// `None` for an index built before this field existed or a corpus too small
    /// to estimate one; consumers fall open to `hooks.related_min_similarity`.
    /// Additive and optional, so no `SCHEMA_VERSION` bump: an old manifest
    /// deserializes with this defaulted to `None`.
    #[serde(default)]
    pub similarity_floor: Option<f32>,
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
            file_hashes: BTreeMap::new(),
            similarity_floor: None,
        }
    }

    /// Build an empty manifest tied to the embedding identity in `config`.
    /// Used as the fallback when `read_manifest` returns `None`.
    pub fn for_config(config: &Config) -> Self {
        Self::new(&config.embedding.model, config.embedding.dimensions)
    }

    /// True when the manifest is older than `reindex_after_hours`, or when
    /// `last_full_index_at` is missing / unparseable. A future timestamp
    /// (clock skew, restored backup) is treated as fresh.
    pub fn is_stale(&self, config: &Config) -> bool {
        let Some(last) = self.last_full_index_at.as_deref() else {
            return true;
        };
        let Ok(last) = parse_rfc3339(last) else {
            return true;
        };
        let Ok(age) = SystemTime::now().duration_since(last) else {
            return false;
        };

        age > Duration::from_secs(config.indexing.reindex_after_hours.saturating_mul(3_600))
    }
}

pub(super) fn validate_manifest_compatibility(
    manifest: Manifest,
    expected_model: &str,
    expected_dimensions: u16,
) -> Result<Manifest> {
    if manifest.schema_version != SCHEMA_VERSION {
        return Err(ClaudixError::SchemaMismatch {
            store: manifest.schema_version,
            binary: SCHEMA_VERSION,
            recovery: RecoveryHint(hints::REINDEX_SCHEMA_VERSION),
        });
    }

    if manifest.embedding_model != expected_model {
        return Err(ClaudixError::EmbeddingModelMismatch {
            store_model: manifest.embedding_model,
            active_model: expected_model.to_owned(),
            recovery: RecoveryHint(hints::REINDEX_AFTER_MODEL_CHANGE),
        });
    }

    if manifest.dimensions != expected_dimensions {
        return Err(ClaudixError::DimensionMismatch {
            store_dim: manifest.dimensions,
            model_dim: expected_dimensions,
            recovery: RecoveryHint(hints::REINDEX_AFTER_DIMENSION_CHANGE),
        });
    }

    Ok(manifest)
}

#[cfg(test)]
mod tests {
    use super::*;

    mod config_support {
        use crate as claudix;

        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/common/config_support.rs"
        ));
    }

    use config_support::stub_config;

    #[test]
    fn stale_index_detection_respects_threshold() {
        let mut config = stub_config();
        config.indexing.reindex_after_hours = 24;
        let mut manifest = Manifest::new("stub-v1", 8);
        manifest.last_full_index_at = Some("2026-04-20T00:00:00Z".to_owned());
        assert!(manifest.is_stale(&config));
    }

    /// An index written before `similarity_floor` existed has no key for it, so
    /// it must deserialize to `None` and fall open rather than fail the read.
    #[test]
    fn manifest_without_similarity_floor_deserializes_to_none() {
        let json = r#"{"schema_version":1,"embedding_model":"m","dimensions":8,
            "last_full_index_at":null,"last_incremental_at":null,
            "chunk_count":0,"file_count":0,"file_hashes":{}}"#;
        let manifest: Manifest = serde_json::from_str(json).expect("legacy manifest parses");
        assert_eq!(manifest.similarity_floor, None);
    }

    #[test]
    fn manifest_round_trips_a_stored_floor() {
        let mut manifest = Manifest::new("m", 8);
        manifest.similarity_floor = Some(0.75);
        let json = serde_json::to_string(&manifest).expect("serializes");
        let back: Manifest = serde_json::from_str(&json).expect("parses");
        assert_eq!(back.similarity_floor, Some(0.75));
    }

    #[test]
    fn fresh_index_detection_allows_recent_manifest() {
        let mut config = stub_config();
        config.indexing.reindex_after_hours = 24 * 365 * 20;
        let mut manifest = Manifest::new("stub-v1", 8);
        manifest.last_full_index_at = Some(crate::util::now_rfc3339());
        assert!(!manifest.is_stale(&config));
    }
}
