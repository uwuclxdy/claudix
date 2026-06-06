//! Every agent-visible string claudix emits, in one place.
//!
//! - [`mcp`] — MCP tool definitions (descriptions + input schemas).
//! - [`hooks`] — hook `additionalContext` / `systemMessage` builders.
//! - [`hints`] — error recovery hints surfaced via `ClaudixError::recovery_hint`.
//!
//! Edit wording here; callers only supply data. Dependency direction is
//! one-way: `prompts` reaches down into lower-level domain types (`search`)
//! for rendering, and the rest of the tree reaches in here for text — never
//! the reverse.
//!
//! Not here, by necessity: `commands/*.md` (slash-command text, read off disk
//! by Claude Code — the binary never serves it) and `hooks/hooks.json`
//! (matcher config, not prose).

pub mod hints;
pub mod hooks;
pub mod mcp;
