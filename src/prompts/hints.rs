//! Error recovery hints.
//!
//! Referenced from error construction sites; surfaced to the agent in MCP error
//! responses via `ClaudixError::recovery_hint`.

// --- path safety --------------------------------------------------------
pub const ENUMERATE_INSIDE_PROJECT_DIR: &str = "Only enumerate files inside $CLAUDE_PROJECT_DIR";
pub const STORE_INSIDE_PROJECT_DIR: &str = "Only use store paths inside $CLAUDE_PROJECT_DIR";
pub const REINDEX_INSIDE_PROJECT_DIR: &str = "Only reindex files inside $CLAUDE_PROJECT_DIR";
pub const MCP_PATH_INSIDE_PROJECT_DIR: &str = "Use a path inside $CLAUDE_PROJECT_DIR";
pub const READ_INSIDE_PROJECT_DIR: &str = "Read a path inside the project root";

// --- mcp tool args ------------------------------------------------------
pub const SEARCH_CODE_ARGS: &str =
    "Pass query plus optional top_k, language_filter, path_prefix, and repos";
pub const QUERY_NON_EMPTY: &str = "Pass a non-empty query string to search_code";
pub const REINDEX_ARGS: &str = "Pass force as an optional boolean";
pub const REINDEX_FILE_PATH_ARG: &str = "Pass path for a file inside $CLAUDE_PROJECT_DIR";
pub const PATH_NON_EMPTY: &str = "Pass a non-empty path to reindex_file";
pub const OVERVIEW_ARGS: &str = "Pass an optional path_prefix string";
pub const FIND_DUPLICATES_ARGS: &str =
    "Pass optional min_similarity (number), limit (integer), repos (array of strings)";

// --- config validation --------------------------------------------------
pub const FIX_GLOBAL_CONFIG: &str = "Fix ~/.claude/claudix.toml";
pub const FIX_PROJECT_CONFIG: &str = "Fix .claude/claudix.toml";
pub const FIX_CIRRUS_CONFIG: &str = "Fix the config file that CIRRUS_CONFIG points to";
pub const PROJECT_RELATIVE_PATH: &str =
    "Set the path to a project-relative value such as .claudix/index";
pub const SET_ENDPOINT: &str = "Set [embedding].endpoint or switch provider to bundled";
pub const SET_MODEL_BUNDLED: &str =
    "Set [embedding].model = \"bge-small-en-v1.5\" for the bundled provider";
pub const BUNDLED_DIMENSIONS_384: &str =
    "Set [embedding].dimensions = 384 for the bundled provider";
pub const SET_DIMENSIONS_POSITIVE: &str = "Set [embedding].dimensions to a positive integer";
pub const SET_BATCH_SIZE: &str = "Set [embedding].batch_size to a positive integer";
pub const SET_TIMEOUT_MS: &str = "Set [embedding].timeout_ms to a positive integer";
pub const SET_MAX_FILE_SIZE_KB: &str = "Set [indexing].max_file_size_kb to a positive integer";
pub const SET_CHUNK_OVERLAP: &str = "Set [indexing].chunk_overlap_lines between 0 and 59";
pub const SET_REINDEX_AFTER_HOURS: &str =
    "Set [indexing].reindex_after_hours to a positive integer";
pub const SET_RELATED_TOP_K: &str = "Set [hooks].related_top_k to a positive integer";
pub const SET_RELATED_MIN_SIMILARITY: &str =
    "Set [hooks].related_min_similarity to a value in [0, 1]";
pub const SET_REINDEX_DEBOUNCE_SECS: &str =
    "Set [hooks].reindex_debounce_secs to a positive integer";
pub const SET_REINDEX_MAX_WAIT_SECS: &str =
    "Set [hooks].reindex_max_wait_secs to a positive integer";
pub const REINDEX_MAX_WAIT_GTE_DEBOUNCE: &str =
    "Set [hooks].reindex_max_wait_secs to at least [hooks].reindex_debounce_secs";
pub const SET_SEARCH_TOP_K: &str = "Set [search].top_k to a positive integer";
pub const SET_SIMILARITY_THRESHOLD: &str = "Set [search].similarity_threshold to a value in [0, 1]";
pub const SET_MIN_SCORE: &str = "Set [search].min_score to a value in [0, 1]";
pub const REMOVE_EMPTY_CROSS_REPOS: &str = "Remove empty strings from [search].cross_repos";
pub const SET_IDENTIFIER_BOOST: &str = "Set [search].identifier_boost to a finite positive number";
pub const SET_HYBRID_WEIGHTS_FINITE: &str =
    "Set all [search].hybrid_weights values to finite numbers";
pub const SET_HYBRID_WEIGHTS_NON_NEGATIVE: &str =
    "Set all [search].hybrid_weights values to non-negative numbers";
pub const SET_HYBRID_WEIGHTS_NONZERO: &str =
    "Set at least one of [search].hybrid_weights.dense, .bm25, .rrf to a positive value";

// --- embedding ----------------------------------------------------------
pub const RUN_DOCTOR: &str =
    "Run /claudix:doctor to check the embedding endpoint or switch to the bundled provider";
pub const EMBEDDING_AUTH: &str = "The embedding endpoint requires auth; disable auth on the server or switch to the bundled provider";
pub const EMBEDDING_TIMEOUT: &str = "The model may still be loading; raise [embedding].timeout_ms or switch to the bundled provider";
pub const SET_ENDPOINT_URL: &str = "Set [embedding].endpoint to the base URL for the HTTP provider";
pub const REBUILD_INDEX_DIMENSIONS: &str =
    "Rebuild the index with the configured embedding dimensions or fix the endpoint model";
pub const DOWNLOAD_BUNDLED_ASSETS: &str = "Run claudix again after restoring network access, or switch to [embedding] provider = \"http\"";
pub const BUNDLED_HIDDEN_STATES_384: &str =
    "Use the bundled bge-small-en-v1.5 export with 384-dimensional hidden states";

// --- store --------------------------------------------------------------
pub const REINDEX_SCHEMA_VERSION: &str =
    "Run reindex with force: true to rebuild the store with the current schema version";
pub const REINDEX_AFTER_MODEL_CHANGE: &str =
    "Run reindex with force: true after changing the configured embedding model";
pub const REINDEX_AFTER_DIMENSION_CHANGE: &str =
    "Run reindex with force: true after changing the configured embedding dimensions";
pub const REINDEX_ALIGN_DIMENSIONS: &str =
    "Run reindex with force: true after aligning embedding dimensions with the active model";

// --- cli ----------------------------------------------------------------
pub const FINITE_MIN_SIMILARITY: &str = "Use a finite min_similarity between 0 and 1";
pub const POSITIVE_LIMIT: &str = "Use a positive limit";
pub const GIT_REPO_REQUIRED: &str = "Run claudix index from inside a git repository";
pub const RUN_REINDEX: &str = "Run the reindex tool (or `claudix index`) to build the index";
pub const VALID_HOOK_EVENTS: &str =
    "Use one of: SessionStart, PostToolUse, PreToolUse, UserPromptSubmit";
pub const SEARCH_QUERY_NON_EMPTY: &str = "Pass a non-empty search query";
pub const POSITIVE_TOP_K: &str = "Pass a positive top_k value";
pub const VALID_LANGUAGE_FILTERS: &str =
    "Use one of: rust, python, javascript, typescript, go, java, c, cpp, unknown";

// --- install ------------------------------------------------------------
pub const RESTORE_PLUGIN_ASSETS: &str =
    "Restore the plugin metadata files before running claudix install";
pub const INSTALL_FROM_PLUGIN_ROOT: &str =
    "Run claudix install from the plugin directory or plugin environment";
pub const SET_HOME_FOR_INSTALL: &str = "Set HOME before running claudix install";

// --- util ---------------------------------------------------------------
pub const UTC_TIMESTAMP_FORMAT: &str = "Use UTC timestamps in the form 2026-04-27T12:00:00Z";

// --- i/o & runtime errors -----------------------------------------------
pub const IO_CHECK_DISK: &str =
    "Check disk space and file permissions, then run /claudix:doctor if the error persists";
pub const STORE_DOCTOR: &str =
    "Run /claudix:doctor to inspect the store, or reindex with force: true to rebuild";
pub const LANCE_DOCTOR: &str =
    "Run /claudix:doctor; if the store is corrupt, reindex with force: true to rebuild";
pub const GIT_ENUM_DOCTOR: &str =
    "Ensure the project is a valid git repository and run /claudix:doctor";
pub const IGNORE_PATTERN: &str =
    "Check .indexignore/.indexinclude for invalid glob patterns, then retry";
pub const HTTP_DOCTOR: &str =
    "Run /claudix:doctor to check network access and the configured embedding endpoint";
pub const EMBEDDING_GENERIC: &str = "Run /claudix:doctor to check the embedding provider; switch to bundled if the endpoint is unavailable";
pub const TREE_SITTER_REINDEX: &str =
    "The file may contain syntax that the parser cannot handle; it will be skipped on next reindex";
