use async_trait::async_trait;

use crate::embedding::Provider;
use crate::error::Result;
use crate::types::Dimension;

#[derive(Debug, Clone)]
pub struct StubProvider {
    model_id: String,
    dimensions: Dimension,
}

impl StubProvider {
    pub fn with_model_id(model_id: impl Into<String>, dimensions: Dimension) -> Self {
        Self {
            model_id: model_id.into(),
            dimensions,
        }
    }
}

#[async_trait]
impl Provider for StubProvider {
    fn name(&self) -> &str {
        "stub"
    }

    fn dimensions(&self) -> Dimension {
        self.dimensions
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    async fn embed(&self, batch: &[&str]) -> Result<Vec<Vec<f32>>> {
        Ok(batch
            .iter()
            .map(|input| embed_one(input, self.dimensions))
            .collect())
    }

    async fn health_check(&self) -> Result<()> {
        Ok(())
    }
}

fn embed_one(input: &str, dimensions: Dimension) -> Vec<f32> {
    let dims = usize::from(dimensions.0);
    let mut vector = Vec::with_capacity(dims);

    if dims == 0 {
        return vector;
    }

    let seed = xxhash_rust::xxh3::xxh3_128(input.as_bytes()).to_be_bytes();
    for index in 0..dims {
        let byte = seed[index % seed.len()];
        let scaled = (f32::from(byte) / 255.0) * 2.0 - 1.0;
        vector.push(scaled);
    }

    vector
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn stub_provider_returns_requested_dimensions() {
        let provider = StubProvider::with_model_id("stub-v1", Dimension(8));

        let vectors = provider.embed(&["alpha", "beta"]).await;
        assert!(vectors.is_ok());
        let vectors = vectors.ok().unwrap_or_else(|| unreachable!());

        assert_eq!(vectors.len(), 2);
        assert_eq!(vectors[0].len(), 8);
        assert_eq!(vectors[1].len(), 8);
    }

    #[tokio::test]
    async fn stub_provider_is_deterministic() {
        let provider = StubProvider::with_model_id("stub-v1", Dimension(6));

        let first = provider.embed(&["same input"]).await;
        assert!(first.is_ok());
        let first = first.ok().unwrap_or_else(|| unreachable!());

        let second = provider.embed(&["same input"]).await;
        assert!(second.is_ok());
        let second = second.ok().unwrap_or_else(|| unreachable!());

        assert_eq!(first, second);
    }

    #[tokio::test]
    async fn stub_provider_distinguishes_inputs() {
        let provider = StubProvider::with_model_id("stub-v1", Dimension(4));

        let vectors = provider.embed(&["alpha", "beta"]).await;
        assert!(vectors.is_ok());
        let vectors = vectors.ok().unwrap_or_else(|| unreachable!());

        assert_ne!(vectors[0], vectors[1]);
    }

    #[tokio::test]
    async fn stub_provider_health_check_succeeds() {
        let provider = StubProvider::with_model_id("stub-v1", Dimension(4));

        assert_eq!(provider.name(), "stub");
        assert_eq!(provider.model_id(), "stub-v1");
        assert!(provider.health_check().await.is_ok());
    }
}
