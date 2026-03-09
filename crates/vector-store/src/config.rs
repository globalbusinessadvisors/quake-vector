//! HNSW configuration parameters.

use serde::{Deserialize, Serialize};

/// Configuration for the HNSW graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswConfig {
    /// Vector dimensionality.
    pub dimensions: usize,
    /// Max edges per node at layers > 0.
    pub m: usize,
    /// Max edges per node at layer 0.
    pub m_max0: usize,
    /// Candidate list size during construction.
    pub ef_construction: usize,
    /// Default candidate list size during search.
    pub ef_search: usize,
    /// Level generation factor: 1 / ln(m).
    pub ml: f64,
}

impl Default for HnswConfig {
    fn default() -> Self {
        let m = 16;
        Self {
            dimensions: 256,
            m,
            m_max0: 32,
            ef_construction: 200,
            ef_search: 64,
            ml: 1.0 / (m as f64).ln(),
        }
    }
}
