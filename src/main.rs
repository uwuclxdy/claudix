use std::env;

use anyhow::Result;
use clap::{Parser, Subcommand};
use claudix::{cli, mcp};

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
    Hook {
        event: String,
    },
    Doctor,
    Install,
    Mcp,
}

#[tokio::main]
async fn main() -> Result<()> {
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
        Command::Hook { event } => {
            let event = cli::parse_hook_event(&event)?;
            println!("hook {:?} not implemented", event);
        }
        Command::Doctor => {
            println!("doctor not implemented");
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
