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
    },
    #[command(about = "Show index status (chunk count, model, last indexed)")]
    Status,
    #[command(about = "Re-embed a single file after editing")]
    ReindexFile {
        #[arg(help = "Path to the file, relative or absolute inside the project")]
        path: String,
    },
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
    #[command(about = "Run as an MCP server over stdio (invoked by Claude Code)")]
    Mcp,
}

#[tokio::main]
async fn main() {
    panic::set_hook(Box::new(|panic_info| {
        eprintln!("claudix panic: {panic_info}");
    }));

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
        Command::Index { force } => {
            if force {
                cli::run_clear_index(&project_root).await?;
            }
            let output = cli::run_index(&project_root).await?;
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
            )
            .await?;

            for hit in output.hits {
                let stale_warning = stale_warning(hit.stale);
                match hit.name {
                    Some(name) => println!(
                        "{}:{}-{} [{}] {} {} {}{}",
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
                        "{}:{}-{} [{}] {} {}{}",
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
                eprintln!(
                    "\nembedding server not reachable — start LM Studio / Ollama,\n\
                     or run `claudix install` to switch to the bundled model."
                );
            }
            if !output.index_present {
                eprintln!("\nindex not built — run `claudix index` to index the repository.");
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

async fn run_hook_command(project_root: &std::path::Path, event: hooks::HookEvent) {
    let payload = read_stdin_payload();
    let project_root = project_root.to_path_buf();
    let handle = tokio::spawn(async move { hooks::run(&project_root, event, &payload).await });

    match handle.await {
        Ok(Ok(Some(response))) => {
            if let Ok(encoded) = to_string(&response) {
                println!("{encoded}");
            }
        }
        Ok(Ok(None)) => {}
        Ok(Err(error)) => {
            eprintln!("claudix hook failed open: {error}");
        }
        Err(_) => {
            eprintln!("claudix hook panicked and failed open");
        }
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
