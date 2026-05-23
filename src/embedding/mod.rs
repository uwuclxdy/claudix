pub mod bundled;
pub mod http;
#[cfg(any(test, feature = "test-stub"))]
pub mod stub;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;

use crate::error::Result;
use crate::types::Dimension;

pub use bundled::BundledProvider;
pub use http::HttpProvider;
#[cfg(any(test, feature = "test-stub"))]
pub use stub::StubProvider;

#[async_trait]
pub trait Provider: Send + Sync {
    fn name(&self) -> &str;
    fn dimensions(&self) -> Dimension;
    fn model_id(&self) -> &str;
    async fn embed(&self, batch: &[&str]) -> Result<Vec<Vec<f32>>>;
    async fn health_check(&self) -> Result<()>;
}

pub struct FallbackProvider {
    primary: Arc<dyn Provider>,
    fallback: Arc<dyn Provider>,
    warned: AtomicBool,
}

impl FallbackProvider {
    pub fn new(primary: Arc<dyn Provider>, fallback: Arc<dyn Provider>) -> Self {
        Self {
            primary,
            fallback,
            warned: AtomicBool::new(false),
        }
    }
}

#[async_trait]
impl Provider for FallbackProvider {
    fn name(&self) -> &str {
        self.primary.name()
    }

    fn dimensions(&self) -> Dimension {
        self.primary.dimensions()
    }

    fn model_id(&self) -> &str {
        self.primary.model_id()
    }

    async fn embed(&self, batch: &[&str]) -> Result<Vec<Vec<f32>>> {
        match self.primary.embed(batch).await {
            Ok(vectors) => Ok(vectors),
            Err(error) if error.is_endpoint_unavailable() => {
                if !self.warned.swap(true, Ordering::Relaxed) {
                    eprintln!(
                        "claudix warning: {error}; falling back to bundled embeddings for this session"
                    );
                }
                self.fallback.embed(batch).await
            }
            Err(error) => Err(error),
        }
    }

    async fn health_check(&self) -> Result<()> {
        self.primary.health_check().await.or_else(|error| {
            if error.is_endpoint_unavailable() {
                Ok(())
            } else {
                Err(error)
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ClaudixError;
    use std::time::Duration;

    use tokio::net::TcpListener;

    struct FixedProvider {
        vectors: Vec<Vec<f32>>,
    }

    #[async_trait]
    impl Provider for FixedProvider {
        fn name(&self) -> &str {
            "fixed"
        }

        fn dimensions(&self) -> Dimension {
            Dimension(2)
        }

        fn model_id(&self) -> &str {
            "fixed-model"
        }

        async fn embed(&self, batch: &[&str]) -> Result<Vec<Vec<f32>>> {
            Ok(self.vectors.iter().take(batch.len()).cloned().collect())
        }

        async fn health_check(&self) -> Result<()> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn fallback_provider_uses_fallback_for_unreachable_primary() {
        let listener = TcpListener::bind("127.0.0.1:0").await;
        assert!(listener.is_ok());
        let listener = listener.ok().unwrap_or_else(|| unreachable!());
        let endpoint = format!(
            "http://{}",
            listener.local_addr().ok().unwrap_or_else(|| unreachable!())
        );
        drop(listener);

        let primary = HttpProvider::new(
            endpoint,
            "fixed-model",
            Dimension(2),
            Duration::from_millis(50),
            None,
        );
        assert!(primary.is_ok());
        let fallback = FixedProvider {
            vectors: vec![vec![0.1, 0.2]],
        };
        let provider = FallbackProvider::new(
            Arc::new(primary.ok().unwrap_or_else(|| unreachable!())),
            Arc::new(fallback),
        );

        let result = provider.embed(&["alpha"]).await;
        assert!(result.is_ok());
        assert_eq!(result.ok().unwrap_or_default(), vec![vec![0.1, 0.2]]);
    }

    #[tokio::test]
    async fn fallback_provider_does_not_fallback_for_primary_embedding_errors() {
        struct BadProvider;

        #[async_trait]
        impl Provider for BadProvider {
            fn name(&self) -> &str {
                "bad"
            }

            fn dimensions(&self) -> Dimension {
                Dimension(2)
            }

            fn model_id(&self) -> &str {
                "bad-model"
            }

            async fn embed(&self, _batch: &[&str]) -> Result<Vec<Vec<f32>>> {
                Err(ClaudixError::Embedding("bad payload".to_owned()))
            }

            async fn health_check(&self) -> Result<()> {
                Ok(())
            }
        }

        let provider = FallbackProvider::new(
            Arc::new(BadProvider),
            Arc::new(FixedProvider {
                vectors: vec![vec![0.1, 0.2]],
            }),
        );

        let result = provider.embed(&["alpha"]).await;
        assert!(
            matches!(result, Err(ClaudixError::Embedding(message)) if message == "bad payload")
        );
    }
}
