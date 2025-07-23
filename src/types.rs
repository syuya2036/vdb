use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq, Eq)]
pub struct Metadata {
    pub label: String,
    pub description: Option<String>,
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
    Cosine = 1,
    Euclidean = 2,
}
