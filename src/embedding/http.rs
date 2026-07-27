use std::time::Duration;

use async_trait::async_trait;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
use serde::Deserialize;

use crate::embedding::Provider;
use crate::error::{ClaudixError, RecoveryHint, Result};
use crate::prompts::hints;
use crate::types::Dimension;

#[derive(Debug, Clone)]
pub struct HttpProvider {
    endpoint: String,
    model_id: String,
    dimensions: Dimension,
    timeout: Duration,
    client: reqwest::Client,
}

impl HttpProvider {
    pub fn new(
        endpoint: impl Into<String>,
        model_id: impl Into<String>,
        dimensions: Dimension,
        timeout: Duration,
        bearer_token: Option<&str>,
    ) -> Result<Self> {
        let endpoint = normalize_endpoint(endpoint.into())?;
        let client = build_client(timeout, bearer_token)?;

        Ok(Self {
            endpoint,
            model_id: model_id.into(),
            dimensions,
            timeout,
            client,
        })
    }

    fn embeddings_url(&self) -> String {
        format!("{}/v1/embeddings", self.endpoint)
    }

    fn transport_error(&self, source: reqwest::Error) -> ClaudixError {
        if source.is_timeout() {
            return ClaudixError::EmbeddingTimedOut {
                endpoint: self.endpoint.clone(),
                timeout_ms: u64::try_from(self.timeout.as_millis()).unwrap_or(u64::MAX),
                recovery: RecoveryHint(hints::EMBEDDING_TIMEOUT),
            };
        }
        ClaudixError::EmbeddingUnreachable {
            endpoint: self.endpoint.clone(),
            source,
            recovery: RecoveryHint(hints::RUN_DOCTOR),
        }
    }

    fn status_error(&self, source: reqwest::Error) -> ClaudixError {
        match source.status().map(|status| status.as_u16()) {
            Some(status @ (401 | 403)) => ClaudixError::EmbeddingAuthRejected {
                endpoint: self.endpoint.clone(),
                status,
                recovery: RecoveryHint(hints::EMBEDDING_AUTH),
            },
            Some(status) => ClaudixError::EmbeddingHttpStatus {
                endpoint: self.endpoint.clone(),
                status,
                recovery: RecoveryHint(hints::RUN_DOCTOR),
            },
            None => self.transport_error(source),
        }
    }
}

#[async_trait]
impl Provider for HttpProvider {
    fn name(&self) -> &str {
        "http"
    }

    fn dimensions(&self) -> Dimension {
        self.dimensions
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn embed(&self, batch: &[&str]) -> Result<Vec<Vec<f32>>> {
        if batch.is_empty() {
            return Ok(Vec::new());
        }

        let mut attempt = 0_u32;
        let mut delay = RETRY_BASE_DELAY;
        let response = loop {
            attempt += 1;
            let send_result = self
                .client
                .post(self.embeddings_url())
                .json(&EmbeddingRequest {
                    model: &self.model_id,
                    input: batch,
                })
                .send()
                .await;

            let response = match send_result {
                Ok(response) => match response.error_for_status() {
                    Ok(response) => response,
                    Err(source)
                        if attempt < MAX_RETRY_ATTEMPTS
                            && source.status().is_some_and(|status| {
                                status.is_server_error() || status.as_u16() == 429
                            }) =>
                    {
                        tokio::time::sleep(delay).await;
                        delay = (delay * 2).min(RETRY_MAX_DELAY);
                        continue;
                    }
                    Err(source) => return Err(self.status_error(source)),
                },
                Err(source) if attempt < MAX_RETRY_ATTEMPTS && is_retryable_transport(&source) => {
                    tokio::time::sleep(delay).await;
                    delay = (delay * 2).min(RETRY_MAX_DELAY);
                    continue;
                }
                Err(source) => return Err(self.transport_error(source)),
            };
            break response;
        };

        // reqwest's Display hides the serde parse detail in `source()`; walk
        // the chain so /claudix:doctor can show "expected value", not
        // "decoding". Drop this if reqwest inlines the parse text into Display.
        let payload: EmbeddingResponse =
            response
                .json()
                .await
                .map_err(|source| ClaudixError::EmbeddingEndpointBadPayload {
                    endpoint: self.endpoint.clone(),
                    reason: std::iter::successors(
                        Some(&source as &dyn std::error::Error),
                        |error| error.source(),
                    )
                    .map(|error| error.to_string())
                    .collect::<Vec<_>>()
                    .join(": "),
                    recovery: RecoveryHint(hints::RUN_DOCTOR),
                })?;
        if payload.data.len() != batch.len() {
            return Err(ClaudixError::EmbeddingEndpointBadPayload {
                endpoint: self.endpoint.clone(),
                reason: format!(
                    "provider returned {} embeddings for {} inputs",
                    payload.data.len(),
                    batch.len()
                ),
                recovery: RecoveryHint(hints::RUN_DOCTOR),
            });
        }
        let mut seen = vec![false; batch.len()];
        let mut items = Vec::with_capacity(payload.data.len());
        for (position, item) in payload.data.into_iter().enumerate() {
            let index = item.index.unwrap_or(position);
            if index >= batch.len() || seen[index] {
                return Err(ClaudixError::EmbeddingEndpointBadPayload {
                    endpoint: self.endpoint.clone(),
                    reason: format!("invalid embedding index {index} for {} inputs", batch.len()),
                    recovery: RecoveryHint(hints::RUN_DOCTOR),
                });
            }
            seen[index] = true;
            items.push((index, item.embedding));
        }
        items.sort_unstable_by_key(|(idx, _)| *idx);
        let vectors: Vec<Vec<f32>> = items.into_iter().map(|(_, embedding)| embedding).collect();

        validate_dimensions(&vectors, self.dimensions, &self.endpoint)?;
        Ok(vectors)
    }

    async fn health_check(&self) -> Result<()> {
        let _ = self.embed(&["health-check"]).await?;
        Ok(())
    }
}

#[derive(serde::Serialize)]
struct EmbeddingRequest<'a> {
    model: &'a str,
    input: &'a [&'a str],
}

#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingItem>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingItem {
    #[serde(default)]
    index: Option<usize>,
    embedding: Vec<f32>,
}

fn build_client(timeout: Duration, bearer_token: Option<&str>) -> Result<reqwest::Client> {
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

    if let Some(token) = bearer_token {
        let value = format!("Bearer {token}");
        let header = HeaderValue::from_str(&value)
            .map_err(|error| ClaudixError::Embedding(error.to_string()))?;
        headers.insert(AUTHORIZATION, header);
    }

    reqwest::Client::builder()
        .timeout(timeout)
        .connect_timeout(timeout.min(Duration::from_secs(10)))
        .tcp_keepalive(Duration::from_secs(30))
        .pool_idle_timeout(Some(Duration::from_secs(20)))
        .default_headers(headers)
        .build()
        .map_err(ClaudixError::from)
}

const MAX_RETRY_ATTEMPTS: u32 = 3;
const RETRY_BASE_DELAY: Duration = Duration::from_millis(200);
const RETRY_MAX_DELAY: Duration = Duration::from_secs(2);

fn is_retryable_transport(error: &reqwest::Error) -> bool {
    error.is_timeout() || error.is_connect() || error.is_request()
}

fn normalize_endpoint(endpoint: String) -> Result<String> {
    let endpoint = endpoint.trim().trim_end_matches('/').to_owned();
    if endpoint.is_empty() {
        return Err(ClaudixError::ConfigInvalid {
            message: "embedding endpoint cannot be empty".to_owned(),
            recovery: RecoveryHint(hints::SET_ENDPOINT_URL),
        });
    }
    Ok(endpoint)
}

fn validate_dimensions(vectors: &[Vec<f32>], dimensions: Dimension, endpoint: &str) -> Result<()> {
    let expected = usize::from(dimensions.0);

    for vector in vectors {
        if vector.len() != expected {
            return Err(ClaudixError::DimensionMismatch {
                store_dim: dimensions.0,
                model_dim: u16::try_from(vector.len()).unwrap_or(u16::MAX),
                recovery: RecoveryHint(hints::REBUILD_INDEX_DIMENSIONS),
            });
        }
        if vector.iter().any(|value| !value.is_finite()) {
            return Err(ClaudixError::EmbeddingEndpointBadPayload {
                endpoint: endpoint.to_owned(),
                reason: "non-finite embedding values".to_owned(),
                recovery: RecoveryHint(hints::RUN_DOCTOR),
            });
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    use tokio::sync::oneshot;

    #[test]
    fn http_provider_rejects_blank_endpoint() {
        let provider = HttpProvider::new(
            "   ",
            "test-model",
            Dimension(2),
            Duration::from_secs(5),
            None,
        );

        assert!(matches!(provider, Err(ClaudixError::ConfigInvalid { .. })));
    }

    #[tokio::test]
    async fn http_provider_embeds_batch() {
        let server = TestServer::spawn(response_with_json(
            r#"{"data":[{"embedding":[0.1,0.2]},{"embedding":[0.3,0.4]}]}"#,
        ))
        .await;

        let provider = HttpProvider::new(
            server.endpoint(),
            "test-model",
            Dimension(2),
            Duration::from_secs(5),
            None,
        );
        assert!(provider.is_ok());
        let provider = provider.ok().unwrap_or_else(|| unreachable!());

        let result = provider.embed(&["alpha", "beta"]).await;
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| unreachable!());

        assert_eq!(result, vec![vec![0.1, 0.2], vec![0.3, 0.4]]);

        let request = server.finish().await;
        assert!(request.contains("POST /v1/embeddings HTTP/1.1"));
        assert!(request.contains("\"model\":\"test-model\""));
        assert!(request.contains("\"input\":[\"alpha\",\"beta\"]"));
    }

    #[tokio::test]
    async fn http_provider_reorders_out_of_order_response() {
        // Response returns item at index 1 first, then index 0.
        let server = TestServer::spawn(response_with_json(
            r#"{"data":[{"index":1,"embedding":[0.3,0.4]},{"index":0,"embedding":[0.1,0.2]}]}"#,
        ))
        .await;

        let provider = HttpProvider::new(
            server.endpoint(),
            "test-model",
            Dimension(2),
            Duration::from_secs(5),
            None,
        );
        assert!(provider.is_ok());
        let provider = provider.ok().unwrap_or_else(|| unreachable!());

        let result = provider.embed(&["alpha", "beta"]).await;
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| unreachable!());

        // Must reorder so that index 0 (alpha → [0.1,0.2]) comes first.
        assert_eq!(result, vec![vec![0.1, 0.2], vec![0.3, 0.4]]);
        server.finish().await;
    }

    #[tokio::test]
    async fn http_provider_reports_duplicate_embedding_index() {
        let server = TestServer::spawn(response_with_json(
            r#"{"data":[{"index":0,"embedding":[0.1,0.2]},{"index":0,"embedding":[0.3,0.4]}]}"#,
        ))
        .await;

        let provider = HttpProvider::new(
            server.endpoint(),
            "test-model",
            Dimension(2),
            Duration::from_secs(5),
            None,
        );
        assert!(provider.is_ok());
        let provider = provider.ok().unwrap_or_else(|| unreachable!());

        let error = provider.embed(&["alpha", "beta"]).await;
        assert!(matches!(
            error,
            Err(ClaudixError::EmbeddingEndpointBadPayload { endpoint, reason, .. })
                if endpoint == server.endpoint() && reason.contains("invalid embedding index")
        ));
        let _ = server.finish().await;
    }

    #[tokio::test]
    async fn http_provider_reports_out_of_range_embedding_index() {
        let server = TestServer::spawn(response_with_json(
            r#"{"data":[{"index":0,"embedding":[0.1,0.2]},{"index":2,"embedding":[0.3,0.4]}]}"#,
        ))
        .await;

        let provider = HttpProvider::new(
            server.endpoint(),
            "test-model",
            Dimension(2),
            Duration::from_secs(5),
            None,
        );
        assert!(provider.is_ok());
        let provider = provider.ok().unwrap_or_else(|| unreachable!());

        let error = provider.embed(&["alpha", "beta"]).await;
        assert!(matches!(
            error,
            Err(ClaudixError::EmbeddingEndpointBadPayload { endpoint, reason, .. })
                if endpoint == server.endpoint() && reason.contains("invalid embedding index")
        ));
        let _ = server.finish().await;
    }

    #[tokio::test]
    async fn http_provider_reports_embedding_count_mismatch() {
        let server =
            TestServer::spawn(response_with_json(r#"{"data":[{"embedding":[0.1,0.2]}]}"#)).await;

        let provider = HttpProvider::new(
            server.endpoint(),
            "test-model",
            Dimension(2),
            Duration::from_secs(5),
            None,
        );
        assert!(provider.is_ok());
        let provider = provider.ok().unwrap_or_else(|| unreachable!());

        let error = provider.embed(&["alpha", "beta"]).await;
        assert!(matches!(
            error,
            Err(ClaudixError::EmbeddingEndpointBadPayload { endpoint, reason, .. })
                if endpoint == server.endpoint()
                    && reason == "provider returned 1 embeddings for 2 inputs"
        ));
        let _ = server.finish().await;
    }

    #[tokio::test]
    async fn http_provider_classifies_non_json_body_as_bad_payload() {
        // LM Studio during warm-up can return 200 + HTML. The decode failure
        // must classify as endpoint-unavailable so FallbackProvider switches to
        // bundled instead of hard-failing the call.
        let server = TestServer::spawn(
            "HTTP/1.1 200 OK\r\ncontent-type: text/html\r\ncontent-length: 5\r\n\r\noops!"
                .to_owned(),
        )
        .await;

        let provider = HttpProvider::new(
            server.endpoint(),
            "test-model",
            Dimension(2),
            Duration::from_secs(5),
            None,
        );
        assert!(provider.is_ok());
        let provider = provider.ok().unwrap_or_else(|| unreachable!());

        let error = provider.embed(&["alpha"]).await;
        assert!(matches!(
            error,
            Err(ClaudixError::EmbeddingEndpointBadPayload { endpoint, reason, .. })
                if endpoint == server.endpoint() && reason.contains("line")
        ));
        let _ = server.finish().await;
    }

    #[test]
    fn validate_dimensions_rejects_non_finite_embedding_values() {
        let error = validate_dimensions(
            &[vec![0.1, f32::INFINITY]],
            Dimension(2),
            "http://test.example",
        );

        assert!(matches!(
            error,
            Err(ClaudixError::EmbeddingEndpointBadPayload { reason, .. })
                if reason.contains("non-finite")
        ));
    }

    #[tokio::test]
    async fn http_provider_reports_dimension_mismatch() {
        let server = TestServer::spawn(response_with_json(
            r#"{"data":[{"embedding":[0.1,0.2,0.3]}]}"#,
        ))
        .await;

        let provider = HttpProvider::new(
            server.endpoint(),
            "test-model",
            Dimension(2),
            Duration::from_secs(5),
            None,
        );
        assert!(provider.is_ok());
        let provider = provider.ok().unwrap_or_else(|| unreachable!());

        let error = provider.embed(&["alpha"]).await;
        assert!(matches!(error, Err(ClaudixError::DimensionMismatch { .. })));
        let _ = server.finish().await;
    }

    #[tokio::test]
    async fn http_provider_health_check_uses_endpoint() {
        let server = TestServer::spawn(response_with_json(
            r#"{"data":[{"embedding":[0.1,0.2,0.3,0.4]}]}"#,
        ))
        .await;

        let provider = HttpProvider::new(
            server.endpoint(),
            "health-model",
            Dimension(4),
            Duration::from_secs(5),
            Some("secret"),
        );
        assert!(provider.is_ok());
        let provider = provider.ok().unwrap_or_else(|| unreachable!());

        assert!(provider.health_check().await.is_ok());

        let request = server.finish().await;
        assert!(request.contains("authorization: Bearer secret"));
        assert!(request.contains("\"input\":[\"health-check\"]"));
    }

    fn response_with_json(body: &str) -> String {
        format!(
            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{}",
            body.len(),
            body,
        )
    }

    #[tokio::test]
    async fn http_provider_returns_empty_for_empty_batch() {
        let provider = HttpProvider::new(
            "http://127.0.0.1:1",
            "test-model",
            Dimension(2),
            Duration::from_millis(50),
            None,
        );
        assert!(provider.is_ok());
        let provider = provider.ok().unwrap_or_else(|| unreachable!());

        let result = provider.embed(&[]).await;
        assert!(result.is_ok());
        assert!(result.ok().unwrap_or_else(|| unreachable!()).is_empty());
    }

    #[tokio::test]
    async fn http_provider_reports_unreachable_endpoint() {
        let listener = TcpListener::bind("127.0.0.1:0").await;
        assert!(listener.is_ok());
        let listener = listener.ok().unwrap_or_else(|| unreachable!());
        let endpoint = format!(
            "http://{}",
            listener.local_addr().ok().unwrap_or_else(|| unreachable!())
        );
        drop(listener);

        // Generous timeout so a refused connection (near-instant on loopback)
        // wins the race and is classified as unreachable, not a timeout — a
        // 50ms budget lost that race on loaded windows runners.
        let provider = HttpProvider::new(
            endpoint.clone(),
            "test-model",
            Dimension(2),
            Duration::from_secs(5),
            None,
        );
        assert!(provider.is_ok());
        let provider = provider.ok().unwrap_or_else(|| unreachable!());

        let error = provider.embed(&["alpha"]).await;
        assert!(matches!(
            error,
            Err(ClaudixError::EmbeddingUnreachable { endpoint: reported, .. }) if reported == endpoint
        ));
    }

    #[tokio::test]
    async fn http_provider_reports_auth_rejection() {
        let server =
            TestServer::spawn("HTTP/1.1 401 Unauthorized\r\ncontent-length: 0\r\n\r\n".to_owned())
                .await;

        let provider = HttpProvider::new(
            server.endpoint(),
            "test-model",
            Dimension(2),
            Duration::from_secs(5),
            None,
        );
        assert!(provider.is_ok());
        let provider = provider.ok().unwrap_or_else(|| unreachable!());

        let error = provider.embed(&["alpha"]).await;
        assert!(matches!(
            error,
            Err(ClaudixError::EmbeddingAuthRejected { status: 401, .. })
        ));
        let _ = server.finish().await;
    }

    #[tokio::test]
    async fn http_provider_retries_5xx_then_succeeds() {
        // A model-loading LM Studio returns 503 until weights are resident;
        // embed() must retry inside the loop and succeed once the server is
        // ready, rather than hard-failing on the first 503.
        let server = MultiResponseServer::spawn(vec![
            "HTTP/1.1 503 Service Unavailable\r\ncontent-length: 0\r\n\r\n".to_owned(),
            "HTTP/1.1 503 Service Unavailable\r\ncontent-length: 0\r\n\r\n".to_owned(),
            response_with_json(r#"{"data":[{"embedding":[0.1,0.2]}]}"#),
        ])
        .await;

        let provider = HttpProvider::new(
            server.endpoint(),
            "test-model",
            Dimension(2),
            Duration::from_secs(5),
            None,
        );
        assert!(provider.is_ok());
        let provider = provider.ok().unwrap_or_else(|| unreachable!());

        let result = provider.embed(&["alpha"]).await;
        assert!(result.is_ok());
        assert_eq!(
            result.ok().unwrap_or_default(),
            vec![vec![0.1_f32, 0.2_f32]]
        );
        // Three requests observed: two 503s + the 200.
        assert_eq!(server.request_count(), 3);
    }

    #[tokio::test]
    async fn http_provider_does_not_retry_4xx() {
        // A 401 must surface immediately as EmbeddingAuthRejected without
        // consuming the retry budget — widening the predicate to retry 4xx
        // would burn the second slot on the canned 200 and silently mask the
        // auth failure.
        let server = MultiResponseServer::spawn(vec![
            "HTTP/1.1 401 Unauthorized\r\ncontent-length: 0\r\n\r\n".to_owned(),
            response_with_json(r#"{"data":[{"embedding":[0.1,0.2]}]}"#),
        ])
        .await;

        let provider = HttpProvider::new(
            server.endpoint(),
            "test-model",
            Dimension(2),
            Duration::from_secs(5),
            None,
        );
        assert!(provider.is_ok());
        let provider = provider.ok().unwrap_or_else(|| unreachable!());

        let error = provider.embed(&["alpha"]).await;
        assert!(matches!(
            error,
            Err(ClaudixError::EmbeddingAuthRejected { status: 401, .. })
        ));
        assert_eq!(server.request_count(), 1);
    }

    #[tokio::test]
    async fn http_provider_reports_timeout_when_response_stalls() {
        let listener = TcpListener::bind("127.0.0.1:0").await;
        assert!(listener.is_ok());
        let listener = listener.ok().unwrap_or_else(|| unreachable!());
        let endpoint = format!(
            "http://{}",
            listener.local_addr().ok().unwrap_or_else(|| unreachable!())
        );

        // Accept connections and read the request, but never write a response,
        // forcing the client's request timeout to fire.
        tokio::spawn(async move {
            while let Ok((mut socket, _)) = listener.accept().await {
                tokio::spawn(async move {
                    let mut buffer = vec![0_u8; 4096];
                    let _ = socket.read(&mut buffer).await;
                    tokio::time::sleep(Duration::from_secs(30)).await;
                });
            }
        });

        let provider = HttpProvider::new(
            endpoint.clone(),
            "test-model",
            Dimension(2),
            Duration::from_millis(50),
            None,
        );
        assert!(provider.is_ok());
        let provider = provider.ok().unwrap_or_else(|| unreachable!());

        let error = provider.embed(&["alpha"]).await;
        assert!(matches!(
            error,
            Err(ClaudixError::EmbeddingTimedOut { endpoint: reported, .. }) if reported == endpoint
        ));
    }

    struct TestServer {
        endpoint: String,
        request_rx: oneshot::Receiver<String>,
        shutdown_tx: oneshot::Sender<()>,
    }

    impl TestServer {
        async fn spawn(response: String) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").await;
            assert!(listener.is_ok());
            let listener = listener.ok().unwrap_or_else(|| unreachable!());
            let endpoint = format!(
                "http://{}",
                listener.local_addr().ok().unwrap_or_else(|| unreachable!())
            );
            let (request_tx, request_rx) = oneshot::channel();
            let (shutdown_tx, shutdown_rx) = oneshot::channel();

            tokio::spawn(async move {
                let accept = listener.accept().await;
                assert!(accept.is_ok());
                let (mut socket, _) = accept.ok().unwrap_or_else(|| unreachable!());

                let mut buffer = vec![0_u8; 4096];
                let read = socket.read(&mut buffer).await;
                assert!(read.is_ok());
                let read = read.ok().unwrap_or_else(|| unreachable!());
                let request = String::from_utf8_lossy(&buffer[..read]).into_owned();
                let _ = request_tx.send(request);

                let _ = socket.write_all(response.as_bytes()).await;
                let _ = socket.shutdown().await;

                let _ = shutdown_rx.await;
            });

            Self {
                endpoint,
                request_rx,
                shutdown_tx,
            }
        }

        fn endpoint(&self) -> &str {
            &self.endpoint
        }

        async fn finish(self) -> String {
            let request = self.request_rx.await.ok().unwrap_or_else(|| unreachable!());
            let _ = self.shutdown_tx.send(());
            request
        }
    }

    /// Loopback server that serves a sequence of canned responses on successive
    /// accepts, with a shared request counter. Used by retry tests that need
    /// the client to observe multiple responses on the same listener.
    struct MultiResponseServer {
        endpoint: String,
        request_count: Arc<AtomicU32>,
        _handle: tokio::task::JoinHandle<()>,
    }

    impl MultiResponseServer {
        async fn spawn(responses: Vec<String>) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").await;
            assert!(listener.is_ok());
            let listener = listener.ok().unwrap_or_else(|| unreachable!());
            let endpoint = format!(
                "http://{}",
                listener.local_addr().ok().unwrap_or_else(|| unreachable!())
            );
            let request_count = Arc::new(AtomicU32::new(0));
            let count_clone = Arc::clone(&request_count);
            let _handle = tokio::spawn(async move {
                for response in responses {
                    let (mut socket, _) = match listener.accept().await {
                        Ok(pair) => pair,
                        Err(_) => return,
                    };
                    let mut buffer = vec![0_u8; 4096];
                    let _ = socket.read(&mut buffer).await;
                    count_clone.fetch_add(1, Ordering::SeqCst);
                    let _ = socket.write_all(response.as_bytes()).await;
                    let _ = socket.shutdown().await;
                }
            });

            Self {
                endpoint,
                request_count,
                _handle,
            }
        }

        fn endpoint(&self) -> &str {
            &self.endpoint
        }

        fn request_count(&self) -> u32 {
            self.request_count.load(Ordering::SeqCst)
        }
    }
}
