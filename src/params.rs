use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Serialize, Deserialize, Debug)]
pub struct Params {
    pub ef_construction: usize,
    pub ef_search: usize,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            ef_construction: 200,
            ef_search: 50,
        }
    }
}
