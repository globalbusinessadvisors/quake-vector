//! Adapter wrapping ruvector-core and rvf-index for VectorStore trait.
//!
//! Uses rvf-index's ProgressiveIndex with InMemoryVectorStore for HNSW
//! search, and adapts the results to our domain types.

use std::collections::HashMap;

use quake_vector_types::{NodeId, SearchResult, SeismicEmbedding, TemporalMeta};

use crate::traits::VectorStore;

/// VectorStore implementation backed by rvf-index HnswGraph.
pub struct RuvectorHnswGraph {
    /// Stored vectors for rvf-index's VectorStore trait.
    vectors: Vec<Vec<f32>>,
    /// Metadata keyed by index position.
    meta: HashMap<u64, TemporalMeta>,
    /// HNSW graph from rvf-index.
    graph: rvf_index::HnswGraph,
    /// Next node ID.
    next_id: u64,
}

impl RuvectorHnswGraph {
    pub fn new(m: usize, ef_construction: usize) -> Self {
        let config = rvf_index::HnswConfig {
            m,
            m0: m * 2,
            ef_construction,
        };
        Self {
            vectors: Vec::new(),
            meta: HashMap::new(),
            graph: rvf_index::HnswGraph::new(&config),
            next_id: 0,
        }
    }
}

impl VectorStore for RuvectorHnswGraph {
    fn insert(&mut self, embedding: &SeismicEmbedding, meta: &TemporalMeta) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        self.vectors.push(embedding.vector.to_vec());
        self.meta.insert(id, *meta);

        // Insert into rvf-index HnswGraph using InMemoryVectorStore for distance lookups
        let store = rvf_index::InMemoryVectorStore::new(self.vectors.clone());
        let rng_val = ((id * 7 + 3) % 100) as f64 / 100.0;
        self.graph.insert(id, rng_val, &store, &rvf_index::l2_distance);

        NodeId(id)
    }

    fn knn_search(&self, query: &[f32; 256], k: usize, ef: usize) -> SearchResult {
        if self.vectors.is_empty() {
            return SearchResult {
                neighbors: Vec::new(),
                search_ef: ef as u16,
                nodes_visited: 0,
                latency_us: 0,
            };
        }

        let store = rvf_index::InMemoryVectorStore::new(self.vectors.clone());
        let results = self.graph.search(query, k, ef, &store, &rvf_index::l2_distance);

        SearchResult {
            neighbors: results.iter().map(|&(id, dist)| (NodeId(id), dist)).collect(),
            search_ef: ef as u16,
            nodes_visited: results.len() as u32,
            latency_us: 0,
        }
    }

    fn get_meta(&self, id: NodeId) -> Option<&TemporalMeta> {
        self.meta.get(&id.0)
    }

    fn get_vector(&self, id: NodeId) -> Option<&[f32]> {
        self.vectors.get(id.0 as usize).map(|v| v.as_slice())
    }

    fn node_count(&self) -> u64 {
        self.next_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quake_vector_types::{StationId, WaveType};

    fn make_embedding(seed: u64) -> SeismicEmbedding {
        let mut vector = [0.0f32; 256];
        let mut state = seed ^ 0x12345678;
        for v in vector.iter_mut() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *v = (state as f32 / u64::MAX as f32) * 2.0 - 1.0;
        }
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-8 {
            for v in vector.iter_mut() {
                *v /= norm;
            }
        }
        SeismicEmbedding {
            vector,
            source_window_hash: seed,
            norm,
        }
    }

    fn make_meta(ts: u64) -> TemporalMeta {
        TemporalMeta {
            timestamp_us: ts,
            wave_type: WaveType::P,
            station_id: StationId(1),
            amplitude_rms: 100.0,
            dominant_freq_hz: 5.0,
        }
    }

    #[test]
    fn ruvector_adapter_insert_and_count() {
        let mut graph = RuvectorHnswGraph::new(16, 200);
        for i in 0..10u64 {
            graph.insert(&make_embedding(i), &make_meta(i * 1000));
        }
        assert_eq!(graph.node_count(), 10);
    }

    #[test]
    fn ruvector_adapter_get_meta() {
        let mut graph = RuvectorHnswGraph::new(16, 200);
        let meta = make_meta(42000);
        let id = graph.insert(&make_embedding(42), &meta);
        let got = graph.get_meta(id).unwrap();
        assert_eq!(got.timestamp_us, 42000);
    }
}
