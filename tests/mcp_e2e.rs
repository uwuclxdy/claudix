#![cfg(feature = "test-stub")]

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};

use assert_cmd::cargo::cargo_bin;
use serde_json::{Value, json};

mod common {
    pub mod fixture {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/common/fixture.rs"
        ));
    }
}

use common::fixture::TestFixture;

fn write_stub_config(path: &std::path::Path) -> std::io::Result<()> {
    std::fs::write(
        path,
        r#"[embedding]
provider = "bundled"
endpoint = ""
model = "stub-v1"
dimensions = 8
batch_size = 16
timeout_ms = 10

[indexing]
respect_gitignore = true
follow_symlinks = false
max_file_size_kb = 512
chunk_overlap_lines = 5
reindex_after_hours = 24

[search]
top_k = 10
identifier_boost = 1.4
similarity_threshold = 0.0

[search.hybrid_weights]
dense = 0.55
bm25 = 0.30
rrf = 0.15

[hooks]
intercept_grep = true
auto_reembed_on_edit = true

[paths]
index_dir = ".claudix/index"
log_dir = ".claudix/logs"
"#,
    )
}

fn next_response(reader: &mut BufReader<std::process::ChildStdout>) -> std::io::Result<Value> {
    let mut line = String::new();
    let read = reader.read_line(&mut line)?;
    if read == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "mcp server closed stdout",
        ));
    }

    serde_json::from_str(line.trim_end())
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))
}

#[test]
#[ignore = "spawns the compiled binary"]
fn search_code_returns_fixture_hit_over_stdio() -> Result<(), Box<dyn std::error::Error>> {
    let fixture = TestFixture::new("small_rust")?;
    let config_path = fixture.root().join("stub-config.toml");
    write_stub_config(&config_path)?;

    let index = Command::new(cargo_bin("claudix"))
        .current_dir(fixture.root())
        .env("CLAUDE_PROJECT_DIR", fixture.root())
        .env("CIRRUS_CONFIG", &config_path)
        .arg("index")
        .status()?;
    assert!(index.success());

    let mut child = Command::new(cargo_bin("claudix"))
        .current_dir(fixture.root())
        .env("CLAUDE_PROJECT_DIR", fixture.root())
        .env("CIRRUS_CONFIG", &config_path)
        .arg("mcp")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    let mut stdin = child.stdin.take().ok_or("missing stdin")?;
    let stdout = child.stdout.take().ok_or("missing stdout")?;
    let mut reader = BufReader::new(stdout);

    writeln!(
        stdin,
        "{}",
        json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": { "name": "test-client", "version": "0.0.0" }
            }
        })
    )?;
    let initialize = next_response(&mut reader)?;
    assert_eq!(initialize["result"]["serverInfo"]["name"], "claudix");

    writeln!(
        stdin,
        "{}",
        json!({
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        })
    )?;

    writeln!(
        stdin,
        "{}",
        json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list"
        })
    )?;
    let tools = next_response(&mut reader)?;
    let names = tools["result"]["tools"]
        .as_array()
        .ok_or("tools list missing array")?
        .iter()
        .filter_map(|tool| tool["name"].as_str())
        .collect::<Vec<_>>();
    assert!(names.contains(&"search_code"));

    writeln!(
        stdin,
        "{}",
        json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "search_code",
                "arguments": {
                    "query": "add",
                    "top_k": 3,
                    "language_filter": ["rust"]
                }
            }
        })
    )?;
    let response = next_response(&mut reader)?;
    let structured = &response["result"]["structuredContent"];
    let groups = structured["groups"]
        .as_array()
        .ok_or("groups missing array")?;
    assert!(!groups.is_empty());
    let top_hits = groups[0]["hits"]
        .as_array()
        .ok_or("hits missing in first group")?;
    assert!(!top_hits.is_empty());
    assert_eq!(top_hits[0]["file_path"], "src/math.rs");
    assert_eq!(top_hits[0]["name"], "add");

    drop(stdin);
    let status = child.wait()?;
    assert!(status.success());

    Ok(())
}
