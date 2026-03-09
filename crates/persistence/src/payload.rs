//! Serializable WAL payloads.

use quake_vector_types::{NodeId, TemporalMeta};
use serde::{Deserialize, Serialize};

/// Payload for a node insertion WAL entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InsertNodePayload {
    pub node_id: NodeId,
    pub vector: Vec<f32>,
    pub metadata: TemporalMeta,
}
