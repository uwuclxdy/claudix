#[derive(Debug, Clone, PartialEq)]
pub struct SearchCodeRequest {
    pub query: String,
    pub top_k: Option<u32>,
    pub language_filter: Option<Vec<String>>,
    pub path_prefix: Option<String>,
}
