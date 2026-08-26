#![cfg(feature = "test-stub")]

use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, Command, Stdio};

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

/// Build the `claudix` binary if it is missing. The gate and CI never link it
/// (check/clippy/test stop at metadata), so the first e2e run after a clean
/// target dir pays the link here. Profile-matched: the gate runs these tests
/// under `--release`, a plain `cargo test` under debug.
fn ensure_binary() -> Result<(), Box<dyn std::error::Error>> {
    if cargo_bin("claudix").exists() {
        return Ok(());
    }
    let mut cmd = Command::new(env!("CARGO"));
    cmd.current_dir(env!("CARGO_MANIFEST_DIR"))
        .arg("build")
        .arg("--bin")
        .arg("claudix");
    if !cfg!(debug_assertions) {
        cmd.arg("--release");
    }
    let out = cmd.output()?;
    if !out.status.success() {
        return Err(format!(
            "cargo build for the e2e binary failed: {}",
            String::from_utf8_lossy(&out.stderr)
        )
        .into());
    }
    Ok(())
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

fn spawn_server(
    fixture: &TestFixture,
    config_path: &Path,
) -> Result<(Child, ChildStdin, BufReader<std::process::ChildStdout>), Box<dyn std::error::Error>> {
    let mut child = Command::new(cargo_bin("claudix"))
        .current_dir(fixture.root())
        .env("CLAUDE_PROJECT_DIR", fixture.root())
        .env("CIRRUS_CONFIG", config_path)
        .arg("mcp")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let stdin = child.stdin.take().ok_or("missing stdin")?;
    let stdout = child.stdout.take().ok_or("missing stdout")?;
    Ok((child, stdin, BufReader::new(stdout)))
}

/// Per-request metadata the 2026-07-28 revision requires on every request.
fn modern_meta() -> Value {
    json!({
        "io.modelcontextprotocol/protocolVersion": "2026-07-28",
        "io.modelcontextprotocol/clientCapabilities": {}
    })
}

/// The legacy flow still works under the dual-era default: initialize handshake,
/// notifications/initialized, then bare tools/list and tools/call.
#[test]
fn legacy_handshake_search_returns_fixture_hit() -> Result<(), Box<dyn std::error::Error>> {
    ensure_binary()?;

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

    let (mut child, mut stdin, mut reader) = spawn_server(&fixture, &config_path)?;

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

/// The modern 2026-07-28 flow: no initialize, per-request `_meta`. This test
/// doubles as the binary-identity guard for the shared CARGO_TARGET_DIR: a
/// stale pre-rmcp-3 binary has no `server/discover` and fails here loudly.
#[test]
fn modern_frames_search_returns_fixture_hit() -> Result<(), Box<dyn std::error::Error>> {
    ensure_binary()?;

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

    let (mut child, mut stdin, mut reader) = spawn_server(&fixture, &config_path)?;

    writeln!(
        stdin,
        "{}",
        json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "server/discover",
            "params": { "_meta": modern_meta() }
        })
    )?;
    let discover = next_response(&mut reader)?;
    assert_eq!(discover["result"]["resultType"], "complete");
    let supported = discover["result"]["supportedVersions"]
        .as_array()
        .ok_or("supportedVersions missing array")?;
    assert!(supported.iter().any(|v| v == "2026-07-28"));
    assert!(discover["result"]["capabilities"].get("tools").is_some());
    assert_eq!(
        discover["result"]["_meta"]["io.modelcontextprotocol/serverInfo"]["name"],
        "claudix"
    );
    assert!(discover["result"].get("ttlMs").is_some());
    assert!(discover["result"].get("cacheScope").is_some());

    writeln!(
        stdin,
        "{}",
        json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": { "_meta": modern_meta() }
        })
    )?;
    let tools = next_response(&mut reader)?;
    assert_eq!(tools["result"]["resultType"], "complete");
    assert_eq!(tools["result"]["cacheScope"], "private");
    let names = tools["result"]["tools"]
        .as_array()
        .ok_or("tools list missing array")?
        .iter()
        .filter_map(|tool| tool["name"].as_str())
        .collect::<Vec<_>>();
    assert_eq!(names, vec!["search_code", "reindex", "find_duplicates"]);

    writeln!(
        stdin,
        "{}",
        json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "_meta": modern_meta(),
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
