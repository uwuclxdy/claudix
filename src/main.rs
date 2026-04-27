use std::env;
use std::io::{self, Read};
use std::panic;

use anyhow::Result;
use clap::{Parser, Subcommand};
use claudix::{cli, hooks, mcp};
use serde_json::to_string;

#[derive(Debug, Parser)]
#[command(name = "claudix")]
#[command(about = "Local semantic search for Claude Code")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Index,
    Search {
        query: String,
        #[arg(long)]
        top_k: Option<usize>,
        #[arg(long = "language")]
        language_filter: Vec<String>,
        #[arg(long)]
        path_prefix: Option<String>,
    },
    Status,
    ReindexFile {
        path: String,
    },
    Clear,
    Hook {
        event: String,
    },
    Doctor,
    Install,
    Mcp,
}

#[tokio::main]
async fn main() -> Result<()> {
    panic::set_hook(Box::new(|panic_info| {
        eprintln!("claudix panic: {panic_info}");
    }));

    let cli = Cli::parse();
    let project_root = active_project_root()?;

    match cli.command {
        Command::Index => {
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
                query,
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
                match hit.name {
                    Some(name) => println!(
                        "{}:{}-{} [{}] {} {} {}",
                        hit.file_path,
                        hit.line_start,
                        hit.line_end,
                        hit.language,
                        hit.kind,
                        name,
                        hit.score
                    ),
                    None => println!(
                        "{}:{}-{} [{}] {} {}",
                        hit.file_path,
                        hit.line_start,
                        hit.line_end,
                        hit.language,
                        hit.kind,
                        hit.score
                    ),
                }
            }
        }
        Command::Status => {
            let output = cli::run_status(&project_root).await?;
            println!("chunks: {}", output.chunk_count);
            println!("files: {}", output.file_count);
            if let Some(model) = output.model {
                println!("model: {model}");
            }
            if let Some(dimensions) = output.dimensions {
                println!("dimensions: {dimensions}");
            }
            if let Some(last_full_index_at) = output.last_full_index_at {
                println!("last_full_index_at: {last_full_index_at}");
            }
            if let Some(last_incremental_at) = output.last_incremental_at {
                println!("last_incremental_at: {last_incremental_at}");
            }
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
        Command::Hook { event } => {
            let event = cli::parse_hook_event(&event)?;
            run_hook_command(&project_root, event).await;
        }
        Command::Doctor => {
            let output = cli::run_doctor(&project_root).await?;
            println!("project_root: {}", output.project_root);
            println!("index_present: {}", output.index_present);
            println!("chunks: {}", output.chunk_count);
            println!("files: {}", output.file_count);
            if let Some(model) = output.model {
                println!("model: {model}");
            }
            if let Some(dimensions) = output.dimensions {
                println!("dimensions: {dimensions}");
            }
            println!("embedding_provider: {}", output.embedding_provider);
            println!("embedding_healthy: {}", output.embedding_healthy);
        }
        Command::Install => {
            println!("install not implemented");
        }
        Command::Mcp => {
            mcp::run(&project_root).await?;
        }
    }

    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_stdin_payload_returns_empty_without_input() {
        let payload = read_stdin_payload();
        assert!(payload.is_empty());
    }
}
