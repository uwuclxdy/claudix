use async_trait::async_trait;

use crate::error::Result;
use crate::types::Dimension;

#[async_trait]
pub trait Provider: Send + Sync {
    fn name(&self) -> &str;
    fn dimensions(&self) -> Dimension;
    fn model_id(&self) -> &str;
    async fn embed(&self, batch: &[&str]) -> Result<Vec<Vec<f32>>>;
    async fn health_check(&self) -> Result<()>;
}
