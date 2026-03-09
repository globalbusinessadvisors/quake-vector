//! Graph node representation for the HNSW index.

use serde::{Deserialize, Serialize};
use quake_vector_types::{NodeId, TemporalMeta, Tier};

/// A single node in the HNSW graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: NodeId,
    pub vector: Vec<f32>,
    pub metadata: TemporalMeta,
    /// Edges per layer: edges[layer] contains neighbor NodeIds at that layer.
    pub edges: Vec<Vec<NodeId>>,
    /// Maximum layer this node appears in.
    pub max_layer: usize,
    pub tier: Tier,
    pub inserted_at: u64,
    pub last_accessed: u64,
}

impl GraphNode {
    pub fn new(
        id: NodeId,
        vector: Vec<f32>,
        metadata: TemporalMeta,
        max_layer: usize,
        inserted_at: u64,
    ) -> Self {
        let edges = vec![Vec::new(); max_layer + 1];
        Self {
            id,
            vector,
            metadata,
            edges,
            max_layer,
            tier: Tier::Hot,
            inserted_at,
            last_accessed: inserted_at,
        }
    }
}
