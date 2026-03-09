//! Serialize and deserialize HnswGraph state via bincode.

use std::io;
use quake_vector_store::HnswGraph;

/// Serializes and deserializes HNSW graph state.
pub struct GraphSerializer;

impl GraphSerializer {
    /// Serialize the full HNSW graph state to bytes (bincode).
    pub fn serialize(graph: &HnswGraph) -> Vec<u8> {
        bincode::serialize(graph).expect("graph serialization should not fail")
    }

    /// Deserialize an HNSW graph from bytes.
    pub fn deserialize(data: &[u8]) -> io::Result<HnswGraph> {
        bincode::deserialize(data)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }
}
