//! Temporal subgraph construction for GNN inference.

use std::collections::HashMap;

use quake_vector_types::{NodeId, TemporalMeta, TimingFeatures};
use quake_vector_store::VectorStore;

/// A node within a temporal subgraph.
#[derive(Debug, Clone)]
pub struct SubgraphNode {
    pub id: NodeId,
    pub embedding: Vec<f32>,
    pub meta: TemporalMeta,
}

/// A directed causal edge between two nodes.
#[derive(Debug, Clone)]
pub struct CausalEdge {
    pub source_idx: usize,
    pub target_idx: usize,
    pub features: TimingFeatures,
}

/// A local temporal subgraph centered on a query node.
#[derive(Debug)]
pub struct TemporalSubgraph {
    pub nodes: Vec<SubgraphNode>,
    pub edges: Vec<CausalEdge>,
    /// For each node index, list of (neighbor_node_index, edge_index) pairs.
    pub adjacency: Vec<Vec<(usize, usize)>>,
    pub center_index: usize,
}

/// Maximum time delta (microseconds) for causal edge creation: 60 seconds.
const MAX_EDGE_DELTA_US: i64 = 60_000_000;

/// Builds temporal subgraphs from vector store neighborhoods and wave history.
pub struct SubgraphConstructionService;

impl SubgraphConstructionService {
    /// Build a temporal subgraph centered on a node.
    ///
    /// Merges KNN neighbors from the store with recent wave history entries,
    /// creates forward-in-time causal edges with timing features.
    pub fn build(
        center_id: NodeId,
        center_embedding: &[f32; 256],
        center_meta: &TemporalMeta,
        store: &dyn VectorStore,
        wave_history: &[(NodeId, TemporalMeta)],
        k: usize,
        history_n: usize,
    ) -> TemporalSubgraph {
        let mut node_map: HashMap<NodeId, usize> = HashMap::new();
        let mut nodes: Vec<SubgraphNode> = Vec::new();

        // Add center node
        let center_index = 0;
        nodes.push(SubgraphNode {
            id: center_id,
            embedding: center_embedding.to_vec(),
            meta: *center_meta,
        });
        node_map.insert(center_id, center_index);

        // KNN search for neighbors
        let search_result = store.knn_search(center_embedding, k, k * 2);
        for (neighbor_id, _dist) in &search_result.neighbors {
            if node_map.contains_key(neighbor_id) {
                continue;
            }
            if let (Some(meta), Some(vec)) = (store.get_meta(*neighbor_id), store.get_vector(*neighbor_id)) {
                let idx = nodes.len();
                nodes.push(SubgraphNode {
                    id: *neighbor_id,
                    embedding: vec.to_vec(),
                    meta: *meta,
                });
                node_map.insert(*neighbor_id, idx);
            }
        }

        // Add history entries
        let history_start = wave_history.len().saturating_sub(history_n);
        for (hist_id, hist_meta) in &wave_history[history_start..] {
            if node_map.contains_key(hist_id) {
                continue;
            }
            if let Some(vec) = store.get_vector(*hist_id) {
                let idx = nodes.len();
                nodes.push(SubgraphNode {
                    id: *hist_id,
                    embedding: vec.to_vec(),
                    meta: *hist_meta,
                });
                node_map.insert(*hist_id, idx);
            }
        }

        // Create causal edges (forward in time, within 60s)
        let mut edges = Vec::new();
        let n = nodes.len();
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let t_i = nodes[i].meta.timestamp_us;
                let t_j = nodes[j].meta.timestamp_us;
                if t_i > t_j {
                    continue; // Only forward in time
                }
                let delta = (t_j as i64) - (t_i as i64);
                if delta > MAX_EDGE_DELTA_US {
                    continue;
                }

                let features = TimingFeatures {
                    delta_us: delta,
                    transition: (nodes[i].meta.wave_type, nodes[j].meta.wave_type),
                    amplitude_ratio: if nodes[i].meta.amplitude_rms > 1e-8 {
                        nodes[j].meta.amplitude_rms / nodes[i].meta.amplitude_rms
                    } else {
                        1.0
                    },
                    freq_shift_hz: nodes[j].meta.dominant_freq_hz - nodes[i].meta.dominant_freq_hz,
                };

                edges.push(CausalEdge {
                    source_idx: i,
                    target_idx: j,
                    features,
                });
            }
        }

        // Build adjacency lists
        let mut adjacency = vec![Vec::new(); n];
        for (edge_idx, edge) in edges.iter().enumerate() {
            adjacency[edge.source_idx].push((edge.target_idx, edge_idx));
            // Also add reverse for undirected message passing in GAT
            adjacency[edge.target_idx].push((edge.source_idx, edge_idx));
        }

        TemporalSubgraph {
            nodes,
            edges,
            adjacency,
            center_index,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quake_vector_types::{SearchResult, SeismicEmbedding, StationId, WaveType};

    /// Minimal VectorStore implementation for testing.
    struct TestStore {
        nodes: Vec<(NodeId, [f32; 256], TemporalMeta)>,
    }

    impl VectorStore for TestStore {
        fn insert(&mut self, embedding: &SeismicEmbedding, meta: &TemporalMeta) -> NodeId {
            let id = NodeId(self.nodes.len() as u64);
            self.nodes.push((id, embedding.vector, *meta));
            id
        }

        fn knn_search(&self, _query: &[f32; 256], k: usize, _ef: usize) -> SearchResult {
            let neighbors: Vec<(NodeId, f32)> = self.nodes.iter()
                .take(k)
                .map(|(id, _, _)| (*id, 0.1))
                .collect();
            SearchResult {
                neighbors,
                search_ef: k as u16,
                nodes_visited: k as u32,
                latency_us: 0,
            }
        }

        fn get_meta(&self, id: NodeId) -> Option<&TemporalMeta> {
            self.nodes.iter().find(|(nid, _, _)| *nid == id).map(|(_, _, m)| m)
        }

        fn get_vector(&self, id: NodeId) -> Option<&[f32]> {
            self.nodes.iter().find(|(nid, _, _)| *nid == id).map(|(_, v, _)| v.as_slice())
        }

        fn node_count(&self) -> u64 {
            self.nodes.len() as u64
        }
    }

    fn make_meta(ts_us: u64, wave: WaveType) -> TemporalMeta {
        TemporalMeta {
            timestamp_us: ts_us,
            wave_type: wave,
            station_id: StationId(1),
            amplitude_rms: 100.0,
            dominant_freq_hz: 5.0,
        }
    }

    fn make_embedding(seed: u64) -> SeismicEmbedding {
        let mut vector = [0.0f32; 256];
        let mut s = seed;
        for v in vector.iter_mut() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            *v = (s >> 33) as f32 / u32::MAX as f32;
        }
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        for v in vector.iter_mut() { *v /= norm; }
        SeismicEmbedding { vector, source_window_hash: seed, norm }
    }

    #[test]
    fn builds_subgraph_with_causal_edges() {
        let mut store = TestStore { nodes: Vec::new() };

        // Insert 50 nodes with increasing timestamps
        for i in 0..50u64 {
            let emb = make_embedding(i);
            let meta = make_meta(i * 1_000_000, WaveType::P); // 1s apart
            store.insert(&emb, &meta);
        }

        let center_emb = make_embedding(25);
        let center_meta = make_meta(25_000_000, WaveType::P);

        let history: Vec<(NodeId, TemporalMeta)> = (0..50)
            .map(|i| (NodeId(i), make_meta(i * 1_000_000, WaveType::P)))
            .collect();

        let subgraph = SubgraphConstructionService::build(
            NodeId(25),
            &center_emb.vector,
            &center_meta,
            &store,
            &history,
            16,
            100,
        );

        // Should have the center node + neighbors
        assert!(!subgraph.nodes.is_empty());
        assert_eq!(subgraph.center_index, 0);
        assert_eq!(subgraph.nodes[0].id, NodeId(25));

        // All causal edges should be forward in time
        for edge in &subgraph.edges {
            let t_src = subgraph.nodes[edge.source_idx].meta.timestamp_us;
            let t_tgt = subgraph.nodes[edge.target_idx].meta.timestamp_us;
            assert!(
                t_src <= t_tgt,
                "causal edge must be forward in time: {} -> {}",
                t_src, t_tgt
            );
            // Delta must be within 60s
            assert!(edge.features.delta_us <= 60_000_000);
            assert!(edge.features.delta_us >= 0);
        }

        // Should have edges
        assert!(!subgraph.edges.is_empty());
    }

    #[test]
    fn no_edges_beyond_60_seconds() {
        let mut store = TestStore { nodes: Vec::new() };

        // Two nodes 90 seconds apart
        let emb0 = make_embedding(0);
        let meta0 = make_meta(0, WaveType::P);
        store.insert(&emb0, &meta0);

        let emb1 = make_embedding(1);
        let meta1 = make_meta(90_000_000, WaveType::S); // 90s later
        store.insert(&emb1, &meta1);

        let subgraph = SubgraphConstructionService::build(
            NodeId(0),
            &emb0.vector,
            &meta0,
            &store,
            &[(NodeId(0), meta0), (NodeId(1), meta1)],
            16,
            100,
        );

        // No edges should exist (> 60s apart)
        assert!(
            subgraph.edges.is_empty(),
            "should have no edges for nodes > 60s apart, got {}",
            subgraph.edges.len()
        );
    }
}
