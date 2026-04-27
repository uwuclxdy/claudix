use anyhow::Result;
use clap::Parser;

#[derive(Debug, Parser)]
#[command(name = "claudix")]
#[command(about = "Local semantic search for Claude Code")]
struct Cli {}

fn main() -> Result<()> {
    let _ = Cli::parse();
    Ok(())
}
