//! Warm-embedder side channel between the long-lived MCP server process and
//! short-lived hook processes.
//!
//! The PreToolUse grep interceptor runs in a fresh process per event; building
//! the bundled ONNX provider there costs more than the hook's entire time
//! budget. The MCP server holds a warm provider for the whole session, so it
//! also listens on a loopback port and embeds single queries on behalf of hook
//! processes. Protocol: one JSON line request, one JSON line response, one
//! connection per exchange. The port marker in the store state dir is only a
//! hint — liveness is proven by the connect itself, and every failure path on
//! the client is a `None` (the hook falls back or passes the grep through).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};

use crate::config;
use crate::error::Result;

use super::ProviderCache;

/// Whole-exchange budget on the server side so a stalled peer cannot pin a
/// connection task. Generous: the first exchange may include the provider
/// cold build.
const SERVER_EXCHANGE_TIMEOUT_MS: u64 = 30_000;

/// Marker file name under the store state dir (`.claudix/`).
pub const EMBED_PORT_MARKER_FILE_NAME: &str = "embed-port";

/// Request-line ceiling: far above any real grep query, far below anything an
/// arbitrary local writer could use to balloon the buffer.
const MAX_REQUEST_BYTES: u64 = 64 * 1024;

#[derive(Debug, Serialize, Deserialize)]
pub struct EmbedRequest {
    pub query: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EmbedResponse {
    pub vector: Vec<f32>,
    pub model: String,
    pub dimensions: u16,
}

#[derive(Debug, Serialize, Deserialize)]
struct PortMarker {
    pid: u32,
    port: u16,
}

/// Bind the loopback listener on an ephemeral port and write the marker.
/// Returns the listener; the marker is best-effort (an unwritable state dir
/// only costs hooks the warm path, never the MCP server).
pub async fn bind_and_advertise(marker_path: &Path) -> Result<TcpListener> {
    let listener = TcpListener::bind(("127.0.0.1", 0)).await?;
    let port = listener.local_addr()?.port();
    let marker = PortMarker {
        pid: std::process::id(),
        port,
    };
    if let Ok(payload) = serde_json::to_string(&marker) {
        let _ = std::fs::write(marker_path, payload);
    }
    Ok(listener)
}

/// Remove the marker if this process owns it (pid matches). Best effort.
pub fn withdraw(marker_path: &Path) {
    let owned = read_marker(marker_path).is_some_and(|m| m.pid == std::process::id());
    if owned {
        let _ = std::fs::remove_file(marker_path);
    }
}

fn read_marker(marker_path: &Path) -> Option<PortMarker> {
    let payload = std::fs::read_to_string(marker_path).ok()?;
    serde_json::from_str(&payload).ok()
}

/// Accept loop. Each exchange runs on its own task under a hard timeout; any
/// per-connection error just drops that connection (the client fails open).
pub async fn serve_embed_requests(
    listener: TcpListener,
    project_root: PathBuf,
    cache: Arc<ProviderCache>,
) {
    loop {
        let Ok((stream, _)) = listener.accept().await else {
            // Accept errors are transient (fd pressure) or fatal (listener
            // gone); back off so neither can hot-loop this task.
            tokio::time::sleep(Duration::from_millis(500)).await;
            continue;
        };
        let root = project_root.clone();
        let cache = Arc::clone(&cache);
        tokio::spawn(async move {
            let _ = tokio::time::timeout(
                Duration::from_millis(SERVER_EXCHANGE_TIMEOUT_MS),
                handle_exchange(stream, root, cache),
            )
            .await;
        });
    }
}

async fn handle_exchange(stream: TcpStream, project_root: PathBuf, cache: Arc<ProviderCache>) {
    use tokio::io::AsyncReadExt;

    let (read_half, mut write_half) = stream.into_split();
    let mut line = String::new();
    // Cap the request read: the socket is unauthenticated (loopback), so an
    // arbitrary local writer must not be able to grow the buffer unbounded.
    if BufReader::new(read_half.take(MAX_REQUEST_BYTES))
        .read_line(&mut line)
        .await
        .is_err()
    {
        return;
    }
    let Ok(request) = serde_json::from_str::<EmbedRequest>(&line) else {
        return;
    };
    // Config is reloaded per exchange so a mid-session provider/model change
    // rebuilds through the cache fingerprint instead of serving stale vectors.
    let Ok(config) = config::load(&project_root) else {
        return;
    };
    let Ok(provider) = cache.get_or_build(&config).await else {
        return;
    };
    let Ok(vectors) = provider.embed(&[request.query.as_str()]).await else {
        return;
    };
    let Some(vector) = vectors.into_iter().next() else {
        return;
    };
    let response = EmbedResponse {
        vector,
        model: provider.model_id().to_owned(),
        dimensions: provider.dimensions().0,
    };
    let Ok(mut payload) = serde_json::to_string(&response) else {
        return;
    };
    payload.push('\n');
    let _ = write_half.write_all(payload.as_bytes()).await;
}

/// Client side: one embed exchange against the advertised port. `None` on any
/// failure — missing/stale marker, dead pid, refused connect, timeout, bad
/// payload — the hook decides what a miss means.
pub async fn request_embedding(
    marker_path: &Path,
    query: &str,
    budget: Duration,
) -> Option<EmbedResponse> {
    let marker = read_marker(marker_path)?;
    if !crate::store::marker::process_running(marker.pid) {
        return None;
    }
    let request = EmbedRequest {
        query: query.to_owned(),
    };
    let mut payload = serde_json::to_string(&request).ok()?;
    payload.push('\n');

    let exchange = async {
        let mut stream = TcpStream::connect(("127.0.0.1", marker.port)).await.ok()?;
        stream.write_all(payload.as_bytes()).await.ok()?;
        let mut line = String::new();
        BufReader::new(stream).read_line(&mut line).await.ok()?;
        serde_json::from_str::<EmbedResponse>(&line).ok()
    };
    let response = tokio::time::timeout(budget, exchange).await.ok()??;
    if response.vector.is_empty() {
        return None;
    }
    Some(response)
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use crate::embedding::ProviderCache;

    mod config_support {
        use crate as claudix;

        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/common/config_support.rs"
        ));
    }

    use config_support::stub_config;

    const ONE_SECOND: Duration = Duration::from_secs(1);

    fn temp_marker() -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap_or_else(|_| unreachable!());
        let path = dir.path().join(EMBED_PORT_MARKER_FILE_NAME);
        (dir, path)
    }

    #[tokio::test]
    async fn round_trip_embeds_via_listener() {
        let (dir, marker_path) = temp_marker();
        let config = stub_config();
        let project_root = dir.path().to_path_buf();
        // The listener reloads config from the project root per exchange.
        let claude_dir = project_root.join(".claude");
        assert!(std::fs::create_dir_all(&claude_dir).is_ok());
        let config_text = toml::to_string(&config).unwrap_or_default();
        assert!(std::fs::write(claude_dir.join("claudix.toml"), config_text).is_ok());

        let listener = bind_and_advertise(&marker_path).await;
        assert!(listener.is_ok());
        let listener = listener.ok().unwrap_or_else(|| unreachable!());
        let cache = Arc::new(ProviderCache::new());
        let server = tokio::spawn(serve_embed_requests(listener, project_root, cache));

        let response = request_embedding(&marker_path, "how is config loaded", ONE_SECOND).await;
        server.abort();

        let response = response.unwrap_or_else(|| unreachable!("warm embed must succeed"));
        assert_eq!(response.dimensions, config.embedding.dimensions);
        assert_eq!(response.model, config.embedding.model);
        assert_eq!(
            response.vector.len(),
            usize::from(config.embedding.dimensions)
        );
    }

    #[tokio::test]
    async fn missing_marker_returns_none() {
        let (_dir, marker_path) = temp_marker();
        let response = request_embedding(&marker_path, "anything", ONE_SECOND).await;
        assert!(response.is_none());
    }

    #[tokio::test]
    async fn dead_endpoint_returns_none() {
        let (_dir, marker_path) = temp_marker();
        // Claim a port, advertise it, then free it: live pid, refused connect.
        let listener = bind_and_advertise(&marker_path).await;
        assert!(listener.is_ok());
        drop(listener);

        let response = request_embedding(&marker_path, "anything", ONE_SECOND).await;
        assert!(response.is_none());
    }

    #[tokio::test]
    async fn withdraw_removes_only_own_marker() {
        let (_dir, marker_path) = temp_marker();
        let listener = bind_and_advertise(&marker_path).await;
        assert!(listener.is_ok());
        assert!(marker_path.exists());

        withdraw(&marker_path);
        assert!(!marker_path.exists(), "own marker must be removed");

        // A marker owned by another pid survives withdraw.
        let foreign = serde_json::json!({"pid": u32::MAX, "port": 1});
        assert!(std::fs::write(&marker_path, foreign.to_string()).is_ok());
        withdraw(&marker_path);
        assert!(marker_path.exists(), "foreign marker must survive");
    }
}
