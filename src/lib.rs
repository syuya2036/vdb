mod metrics;
mod storage;
mod types;
mod vector_db;

pub use types::{Metadata, Metric, SearchResult};
pub use vector_db::VectorDB;

pub const M: usize = 12;
pub const M0: usize = 24;
