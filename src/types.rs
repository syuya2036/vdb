use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq, Eq)]
pub struct Metadata {
    pub label: String,
    pub description: Option<String>,
}

impl Default for Metadata {
    fn default() -> Self {
        Self { label: String::new(), description: None }
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct SearchResult {
    pub id: usize,
    pub distance: f32,
    pub metadata: Metadata,
}

#[repr(u8)]
#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq, Eq)]
pub enum Metric {
    /// Cosine similarity metric.
    Cosine = 1,
    /// Euclidean distance metric.
    Euclidean = 2,
    // When adding new variants, assign explicit discriminant values to ensure
    // backward compatibility with existing files.
}
