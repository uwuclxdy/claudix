use std::env;
use std::io::{self, Read};
use std::panic;

use anyhow::Result;
use clap::{Parser, Subcommand};
use claudix::{ClaudixError, cli, hooks, mcp};
use serde_json::to_string;

#[derive(Debug, Parser)]
#[command(name = "claudix")]
#[command(about = "Local semantic search for Claude Code")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    #[command(about = "Index or re-index the current repository")]
    Index {
        #[arg(long, help = "Clear the index before re-indexing")]
        force: bool,
        #[arg(long, help = "Show live indexing progress")]
        progress: bool,
    },
    #[command(about = "Search indexed code semantically")]
    Search {
        #[arg(num_args = 1.., help = "Natural-language or identifier query (multi-word, no quoting needed)")]
        query: Vec<String>,
        #[arg(long, help = "Maximum results to return (default: from config)")]
        top_k: Option<usize>,
        #[arg(
            long = "language",
            help = "Filter by language (rust, python, go, …); repeatable"
        )]
        language_filter: Vec<String>,
        #[arg(long, help = "Restrict results to paths starting with this prefix")]
        path_prefix: Option<String>,
        #[arg(
            long = "repo",
            help = "Additional already-indexed repo path to search read-only; repeatable. \
                    The active project is always included."
        )]
        repos: Vec<String>,
    },
    #[command(about = "Show index status (chunk count, model, last indexed)")]
    Status,
    #[command(about = "Re-embed a single file after editing")]
    ReindexFile {
        #[arg(help = "Path to the file, relative or absolute inside the project")]
        path: String,
    },
    #[command(about = "Watch saved files and re-embed changed files when watch = true")]
    Watch,
    #[command(about = "Drop the entire index dataset")]
    Clear,
    #[command(about = "Handle a Claude Code hook event (SessionStart | PostToolUse | PreToolUse)")]
    Hook {
        #[arg(help = "Hook event name from Claude Code")]
        event: String,
    },
    #[command(about = "Diagnose binary, index, and embedding health")]
    Doctor,
    #[command(about = "Bootstrap plugin files and download the bundled embedding model")]
    Install,
    #[command(about = "Show a structural map of indexed files grouped by directory")]
    Overview {
        #[arg(
            long,
            help = "Restrict output to files under this project-relative path prefix"
        )]
        path_prefix: Option<String>,
    },
    #[command(about = "Find near-duplicate code chunks within this repo or across listed repos")]
    FindDuplicates {
        #[arg(
            long,
            help = "Cosine similarity floor (0.0-1.0; default 0.85 — raise to see only very close copies)"
        )]
        min_similarity: Option<f32>,
        #[arg(long, help = "Maximum number of pairs to return (default 50)")]
        limit: Option<usize>,
        #[arg(
            long = "repo",
            help = "Additional already-indexed repo path to scan; repeatable. \
                    When specified, ONLY these paths are used — the active project is NOT auto-added."
        )]
        repos: Vec<String>,
    },
    #[command(about = "Run as an MCP server over stdio (invoked by Claude Code)")]
    Mcp,
}

#[tokio::main]
async fn main() {
    panic::set_hook(Box::new(|panic_info| {
        eprintln!("claudix panic: {panic_info}");
    }));

    // Stderr only: stdout is the MCP JSON-RPC channel. Default `warn` so budget
    // overruns surface without `RUST_LOG`; `try_init` stays quiet if a host
    // already installed a subscriber.
    let _ = tracing_subscriber::fmt()
        .with_writer(io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .try_init();

    if let Err(err) = run().await {
        if let Some(hint) = err
            .downcast_ref::<ClaudixError>()
            .and_then(ClaudixError::recovery_hint)
        {
            eprintln!("error: {err}");
            eprintln!("hint: {hint}");
        } else {
            eprintln!("error: {err}");
        }
        std::process::exit(1);
    }
}

async fn run() -> Result<()> {
    let cli = Cli::parse();
    let project_root = active_project_root()?;

    match cli.command {
        Command::Index { force, progress } => {
            if force {
                cli::run_clear_index(&project_root).await?;
            }
            let output = cli::run_index(&project_root, progress).await?;
            println!(
                "indexed {} files into {} chunks",
                output.file_count, output.chunk_count
            );
        }
        Command::Search {
            query,
            top_k,
            language_filter,
            path_prefix,
            repos,
        } => {
            let output = cli::run_search(
                &project_root,
                query.join(" "),
                top_k,
                if language_filter.is_empty() {
                    None
                } else {
                    Some(language_filter)
                },
                path_prefix,
                if repos.is_empty() { None } else { Some(repos) },
            )
            .await?;

            for err in &output.repo_errors {
                eprintln!("warning: {} — {}", err.repo, err.error);
            }

            // Only label groups with their repo when results span more than one.
            let multi_repo = output
                .groups
                .iter()
                .map(|g| g.repo.as_str())
                .collect::<std::collections::HashSet<_>>()
                .len()
                > 1;

            for group in output.groups {
                if multi_repo {
                    println!("{} :: {}:", group.repo, group.directory);
                } else {
                    println!("{}:", group.directory);
                }
                for hit in group.hits {
                    let stale_warning = stale_warning(hit.stale);
                    match hit.name {
                        Some(name) => println!(
                            "  {}:{}-{} [{}] {} {} {}{}",
                            hit.file_path,
                            hit.line_start,
                            hit.line_end,
                            hit.language,
                            hit.kind,
                            name,
                            hit.score,
                            stale_warning
                        ),
                        None => println!(
                            "  {}:{}-{} [{}] {} {}{}",
                            hit.file_path,
                            hit.line_start,
                            hit.line_end,
                            hit.language,
                            hit.kind,
                            hit.score,
                            stale_warning
                        ),
                    }
                }
            }
        }
        Command::Status => {
            let output = cli::run_status(&project_root).await?;
            print_index_stats(
                output.chunk_count,
                output.file_count,
                output.model.as_deref(),
                output.dimensions,
            );
            if let Some(last_full_index_at) = output.last_full_index_at {
                println!("last_full_index_at: {last_full_index_at}");
            }
            if let Some(last_incremental_at) = output.last_incremental_at {
                println!("last_incremental_at: {last_incremental_at}");
            }
            println!("stale: {}", output.stale);
        }
        Command::ReindexFile { path } => {
            let output = cli::run_reindex_file(&project_root, path).await?;
            println!(
                "indexed {} files into {} chunks",
                output.file_count, output.chunk_count
            );
        }
        Command::Watch => {
            cli::run_watch(&project_root).await?;
        }
        Command::Clear => {
            let output = cli::run_clear_index(&project_root).await?;
            println!("cleared: {}", output.cleared);
        }
        Command::Hook { event } => match cli::parse_hook_event(&event) {
            Ok(event) => run_hook_command(&project_root, event).await,
            Err(_) => {
                eprintln!("claudix: unknown hook event '{event}', ignoring (fail-open)");
            }
        },
        Command::Doctor => {
            let output = cli::run_doctor(&project_root).await?;
            println!("project_root: {}", output.project_root);
            println!("index_present: {}", output.index_present);
            print_index_stats(
                output.chunk_count,
                output.file_count,
                output.model.as_deref(),
                output.dimensions,
            );
            println!("embedding_provider: {}", output.embedding_provider);
            println!("embedding_healthy: {}", output.embedding_healthy);
            if output.embedding_model_mismatch {
                println!("embedding_model_mismatch: true");
            }

            if output.embedding_model_mismatch {
                eprintln!(
                    "\nembedding model mismatch — the index was built with a different model.\n\
                     Fix: run `claudix clear && claudix index` to rebuild with the active model."
                );
            } else if !output.embedding_healthy {
                match &output.embedding_error {
                    Some(reason) => eprintln!("\nembedding check failed: {reason}"),
                    None => eprintln!("\nembedding endpoint unavailable."),
                }
                eprintln!(
                    "Fix: start LM Studio / Ollama (or fix the [embedding].endpoint),\n\
                     or run `claudix install` to switch to the bundled model."
                );
            }
            if !output.index_present {
                eprintln!("\nindex not built — run `claudix index` to index the repository.");
            }
            if let Some(reason) = &output.install_error {
                eprintln!("\nbinary install failed (recorded in install.log):");
                eprintln!("  {reason}");
                if let Some(path) = &output.install_log_path {
                    eprintln!("  see {path}");
                }
                eprintln!(
                    "Fix: check the release/network, then restart Claude Code to retry the download."
                );
            }
        }
        Command::Overview { path_prefix } => {
            let output = cli::run_overview(&project_root, path_prefix).await?;
            println!("files: {}", output.file_count);
            println!("chunks: {}", output.chunk_count);
            println!("directories: {}", output.directories.len());
            for dir in &output.directories {
                let langs: Vec<&str> = dir.languages.iter().map(|l| l.language.as_str()).collect();
                println!(
                    "  {} — {} files, {} chunks [{}]",
                    dir.path,
                    dir.file_count,
                    dir.chunk_count,
                    langs.join(", ")
                );
                if !dir.top_identifiers.is_empty() {
                    println!("    identifiers: {}", dir.top_identifiers.join(", "));
                }
            }
        }
        Command::FindDuplicates {
            min_similarity,
            limit,
            repos,
        } => {
            let repos = if repos.is_empty() { None } else { Some(repos) };
            let output =
                cli::run_find_duplicates(&project_root, min_similarity, limit, repos).await?;

            for err in &output.repo_errors {
                eprintln!("warning: {} — {}", err.repo, err.error);
            }

            if output.pairs.is_empty() {
                println!("no near-duplicate pairs found");
            } else {
                println!("found {} pair(s):", output.pairs.len());
                for (i, pair) in output.pairs.iter().enumerate() {
                    println!("\n[{}] similarity: {:.3}", i + 1, pair.similarity);
                    print_duplicate_chunk("  a", &pair.a);
                    print_duplicate_chunk("  b", &pair.b);
                }
            }
        }
        Command::Install => {
            let output = cli::run_install(&project_root).await?;
            println!("plugin config: {}", output.config_path);
        }
        Command::Mcp => {
            mcp::run(&project_root).await?;
        }
    }

    Ok(())
}

fn print_duplicate_chunk(label: &str, chunk: &cli::DuplicateChunk) {
    match &chunk.name {
        Some(name) => println!(
            "{label}: {} {}:{}-{} ({})",
            chunk.repo, chunk.file_path, chunk.line_start, chunk.line_end, name
        ),
        None => println!(
            "{label}: {} {}:{}-{}",
            chunk.repo, chunk.file_path, chunk.line_start, chunk.line_end
        ),
    }
}

fn stale_warning(stale: bool) -> &'static str {
    if stale {
        " [STALE - file modified since index]"
    } else {
        ""
    }
}

fn print_index_stats(
    chunk_count: usize,
    file_count: usize,
    model: Option<&str>,
    dimensions: Option<u16>,
) {
    println!("chunks: {chunk_count}");
    println!("files: {file_count}");
    if let Some(model) = model {
        println!("model: {model}");
    }
    if let Some(dimensions) = dimensions {
        println!("dimensions: {dimensions}");
    }
}

fn active_project_root() -> Result<std::path::PathBuf> {
    match env::var_os("CLAUDE_PROJECT_DIR") {
        Some(path) => Ok(path.into()),
        None => Ok(env::current_dir()?),
    }
}

/// Outer fail-open budget for a whole hook handler. Generous over PreToolUse's
/// internal 1.5 s search budget so legitimate background-spawn paths finish, but
/// finite so a stalled store/embed/lock can't hang the host on the hook.
///
/// The in-process panic guard lives here: `tokio::spawn` turns a panic in the
/// async hook body into a `JoinError` (the `Ok(Err(_))` arm below), and the sync
/// `read_stdin_payload` is wrapped in `catch_unwind`. Both need `panic =
/// "unwind"`; see the release-profile note in Cargo.toml.
const HOOK_COMMAND_TIMEOUT_MS: u64 = 5_000;

/// Warn threshold for the whole hook handler. Fast paths finish well under it;
/// the PreToolUse intercept legitimately runs up to its 1.5 s search budget, so
/// this catches real stalls (NFS hang, lock contention) before the hard timeout.
const HOOK_BUDGET_WARN_MS: u64 = 3_000;

// The warn must fire before the hard timeout, else it can never surface.
const _: () = assert!(HOOK_BUDGET_WARN_MS < HOOK_COMMAND_TIMEOUT_MS);

async fn run_hook_command(project_root: &std::path::Path, event: hooks::HookEvent) {
    let hook_start = std::time::Instant::now();
    let payload = panic::catch_unwind(read_stdin_payload).unwrap_or_default();
    // Canonicalize so `path.strip_prefix(project_root)` lines up with the
    // canonical paths the watcher and `Store::new` use internally. A raw
    // `CLAUDE_PROJECT_DIR` with symlinks or trailing separators would otherwise
    // make hook-side path arithmetic disagree with the rest of the binary.
    let project_root = project_root
        .canonicalize()
        .unwrap_or_else(|_| project_root.to_path_buf());
    // Hooks must fail open: a stalled handler (slow store on NFS, a slow embed,
    // a poisoned mutex) is as fatal to the session as a panic, because the host
    // waits on the hook. Cap the whole handler with an outer budget consistent
    // with PreToolUse's internal one, but generous enough that the legitimate
    // background-spawn paths (which return promptly) never trip it. On elapse we
    // emit no JSON and exit normally; the session continues.
    //
    // The hook body calls into the same lancedb/moka search pipeline the MCP
    // driver serves. That pipeline is `!Send` since lancedb 0.31's uring-reader
    // cache, so run it on a `LocalSet` like the MCP driver. `spawn_local` still
    // yields a `JoinError` on panic, preserving the fail-open contract below.
    let local = tokio::task::LocalSet::new();
    let outcome = local
        .run_until(async move {
            let handle = tokio::task::spawn_local(async move {
                hooks::run(&project_root, event, &payload).await
            });
            tokio::time::timeout(
                std::time::Duration::from_millis(HOOK_COMMAND_TIMEOUT_MS),
                handle,
            )
            .await
        })
        .await;

    match outcome {
        Ok(Ok(Ok(Some(response)))) => {
            if let Ok(encoded) = to_string(&response) {
                println!("{encoded}");
            }
        }
        Ok(Ok(Ok(None))) => {}
        Ok(Ok(Err(error))) => {
            eprintln!("claudix hook failed open: {error}");
        }
        Ok(Err(_)) => {
            eprintln!("claudix hook panicked and failed open");
        }
        Err(_) => {
            eprintln!("claudix hook timed out and failed open");
        }
    }

    let elapsed = hook_start.elapsed();
    if elapsed.as_millis() as u64 > HOOK_BUDGET_WARN_MS {
        tracing::warn!(
            event = ?event,
            elapsed_ms = elapsed.as_millis(),
            budget_ms = HOOK_BUDGET_WARN_MS,
            "hook handler exceeded budget"
        );
    }
}

fn read_stdin_payload() -> String {
    let mut payload = String::new();
    if io::stdin().read_to_string(&mut payload).is_ok() {
        payload
    } else {
        String::new()
    }
}
