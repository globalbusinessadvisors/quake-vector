//! VectorStore trait definition.

use quake_vector_types::{NodeId, SearchResult, SeismicEmbedding, TemporalMeta};

/// Trait for a vector store supporting insert and k-nearest-neighbor search.
pub trait VectorStore {
    fn insert(&mut self, embedding: &SeismicEmbedding, meta: &TemporalMeta) -> NodeId;
    fn knn_search(&self, query: &[f32; 256], k: usize, ef: usize) -> SearchResult;
    fn get_meta(&self, id: NodeId) -> Option<&TemporalMeta>;
    fn get_vector(&self, id: NodeId) -> Option<&[f32]>;
    fn node_count(&self) -> u64;
}
