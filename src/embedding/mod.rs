#[cfg(feature = "bundled-embedder")]
pub mod bundled;
pub mod http;
#[cfg(any(test, feature = "test-stub"))]
pub mod stub;

use async_trait::async_trait;

use crate::error::Result;
use crate::types::Dimension;

#[cfg(feature = "bundled-embedder")]
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
