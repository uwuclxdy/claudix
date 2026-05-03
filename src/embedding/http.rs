use std::time::Duration;

use async_trait::async_trait;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
use serde::Deserialize;

use crate::embedding::Provider;
use crate::error::{ClaudixError, RecoveryHint, Result};
use crate::types::Dimension;

#[derive(Debug, Clone)]
pub struct HttpProvider {
    endpoint: String,
    model_id: String,
    dimensions: Dimension,
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
        let endpoint = normalize_endpoint(endpoint.into());
        let client = build_client(timeout, bearer_token)?;

        Ok(Self {
            endpoint,
            model_id: model_id.into(),
            dimensions,
            client,
        })
    }

    fn embeddings_url(&self) -> String {
        format!("{}/v1/embeddings", self.endpoint)
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

        let response = self
            .client
            .post(self.embeddings_url())
            .json(&EmbeddingRequest {
                model: &self.model_id,
                input: batch,
            })
            .send()
            .await
            .map_err(|source| ClaudixError::EmbeddingUnreachable {
                endpoint: self.endpoint.clone(),
                source,
                recovery: RecoveryHint(
                    "Run /claudix:doctor to check the embedding endpoint or switch to the bundled provider",
                ),
            })?;

        let response = response.error_for_status().map_err(|source| {
            ClaudixError::EmbeddingUnreachable {
                endpoint: self.endpoint.clone(),
                source,
                recovery: RecoveryHint(
                    "Run /claudix:doctor to check the embedding endpoint or switch to the bundled provider",
                ),
            }
        })?;

        let payload: EmbeddingResponse = response.json().await?;
        if payload.data.len() != batch.len() {
            return Err(ClaudixError::Embedding(format!(
                "provider returned {} embeddings for {} inputs",
                payload.data.len(),
                batch.len()
            )));
        }
        let mut seen = vec![false; batch.len()];
        let mut items = Vec::with_capacity(payload.data.len());
        for (position, item) in payload.data.into_iter().enumerate() {
            let index = item.index.unwrap_or(position);
            if index >= batch.len() || seen[index] {
                return Err(ClaudixError::Embedding(format!(
                    "provider returned invalid embedding index {index} for {} inputs",
                    batch.len()
                )));
            }
            seen[index] = true;
            items.push((index, item.embedding));
        }
        items.sort_unstable_by_key(|(idx, _)| *idx);
        let vectors: Vec<Vec<f32>> = items.into_iter().map(|(_, embedding)| embedding).collect();

        validate_dimensions(&vectors, self.dimensions)?;
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
        .default_headers(headers)
        .build()
        .map_err(ClaudixError::from)
}

fn normalize_endpoint(endpoint: String) -> String {
    endpoint.trim_end_matches('/').to_owned()
}

fn validate_dimensions(vectors: &[Vec<f32>], dimensions: Dimension) -> Result<()> {
    let expected = usize::from(dimensions.0);

    for vector in vectors {
        if vector.len() != expected {
            return Err(ClaudixError::DimensionMismatch {
                store_dim: dimensions.0,
                model_dim: u16::try_from(vector.len()).unwrap_or(u16::MAX),
                recovery: RecoveryHint(
                    "Rebuild the index with the configured embedding dimensions or fix the endpoint model",
                ),
            });
        }
        if vector.iter().any(|value| !value.is_finite()) {
            return Err(ClaudixError::Embedding(
                "provider returned non-finite embedding values".to_owned(),
            ));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    use tokio::sync::oneshot;

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
        )
        .unwrap();

        let result = provider.embed(&["alpha", "beta"]).await.unwrap();

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
        )
        .unwrap();

        let error = provider.embed(&["alpha", "beta"]).await;
        assert!(matches!(error, Err(ClaudixError::Embedding(message)) if message.contains("invalid embedding index 0")));
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
        )
        .unwrap();

        let error = provider.embed(&["alpha", "beta"]).await;
        assert!(matches!(error, Err(ClaudixError::Embedding(message)) if message.contains("invalid embedding index 2")));
        let _ = server.finish().await;
    }

    #[tokio::test]
    async fn http_provider_reports_embedding_count_mismatch() {
        let server = TestServer::spawn(response_with_json(
            r#"{"data":[{"embedding":[0.1,0.2]}]}"#,
        ))
        .await;

        let provider = HttpProvider::new(
            server.endpoint(),
            "test-model",
            Dimension(2),
            Duration::from_secs(5),
            None,
        )
        .unwrap();

        let error = provider.embed(&["alpha", "beta"]).await;
        assert!(matches!(error, Err(ClaudixError::Embedding(message)) if message.contains("1 embeddings for 2 inputs")));
        let _ = server.finish().await;
    }

    #[test]
    fn validate_dimensions_rejects_non_finite_embedding_values() {
        let error = validate_dimensions(&[vec![0.1, f32::INFINITY]], Dimension(2));

        assert!(matches!(error, Err(ClaudixError::Embedding(message)) if message.contains("non-finite embedding")));
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
            Err(ClaudixError::EmbeddingUnreachable { endpoint: reported, .. }) if reported == endpoint
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
}
