//! CausalLearner: high-level interface combining subgraph construction,
//! wave history, and the causal model.

use quake_vector_types::{NodeId, SeismicEmbedding, SeismicPrediction, TemporalMeta};
use quake_vector_store::VectorStore;

use crate::history::WaveHistory;
use crate::model::CausalModel;
use crate::subgraph::SubgraphConstructionService;

/// Orchestrates causal learning: maintains wave history, builds subgraphs,
/// and runs inference via the CausalModel.
pub struct CausalLearner {
    model: CausalModel,
    history: WaveHistory,
    k: usize,
    history_n: usize,
}

impl CausalLearner {
    pub fn new() -> Self {
        Self {
            model: CausalModel::new(),
            history: WaveHistory::new(1000),
            k: 16,
            history_n: 100,
        }
    }

    /// Process a new observation: update history, build subgraph, run prediction.
    pub fn process(
        &mut self,
        node_id: NodeId,
        embedding: &SeismicEmbedding,
        meta: &TemporalMeta,
        store: &dyn VectorStore,
    ) -> SeismicPrediction {
        self.history.push(node_id, *meta);

        let subgraph = SubgraphConstructionService::build(
            node_id,
            &embedding.vector,
            meta,
            store,
            self.history.entries(),
            self.k,
            self.history_n,
        );

        self.model.predict(&subgraph)
    }

    /// Access the underlying model.
    pub fn model(&self) -> &CausalModel {
        &self.model
    }

    /// Mutable access to the model (for weight updates).
    pub fn model_mut(&mut self) -> &mut CausalModel {
        &mut self.model
    }

    /// Access wave history.
    pub fn history(&self) -> &WaveHistory {
        &self.history
    }
}

impl Default for CausalLearner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quake_vector_types::{SearchResult, StationId, WaveType};

    struct MockVectorStore {
        nodes: Vec<(NodeId, [f32; 256], TemporalMeta)>,
    }

    impl MockVectorStore {
        fn new_with_nodes(n: usize) -> Self {
            let mut nodes = Vec::new();
            let mut rng = 99u64;
            for i in 0..n {
                let mut vec = [0.0f32; 256];
                for v in vec.iter_mut() {
                    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                    *v = (rng >> 33) as f32 / u32::MAX as f32;
                }
                let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                for v in vec.iter_mut() { *v /= norm; }
                let meta = TemporalMeta {
                    timestamp_us: i as u64 * 1_000_000,
                    wave_type: WaveType::P,
                    station_id: StationId(1),
                    amplitude_rms: 100.0,
                    dominant_freq_hz: 5.0,
                };
                nodes.push((NodeId(i as u64), vec, meta));
            }
            Self { nodes }
        }
    }

    impl VectorStore for MockVectorStore {
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

    #[test]
    fn end_to_end_process() {
        let store = MockVectorStore::new_with_nodes(50);
        let mut learner = CausalLearner::new();

        // Process several observations
        for i in 0..10u64 {
            let embedding = SeismicEmbedding {
                vector: store.nodes[i as usize].1,
                source_window_hash: i,
                norm: 1.0,
            };
            let meta = store.nodes[i as usize].2;

            let prediction = learner.process(NodeId(i), &embedding, &meta, &store);

            assert!(prediction.event_probability >= 0.0 && prediction.event_probability <= 1.0);
            assert!(prediction.estimated_magnitude >= 0.0);
            assert!(prediction.confidence >= 0.0 && prediction.confidence <= 1.0);
        }

        assert_eq!(learner.history().len(), 10);
    }

    #[test]
    fn process_builds_growing_history() {
        let store = MockVectorStore::new_with_nodes(20);
        let mut learner = CausalLearner::new();

        for i in 0..5u64 {
            let embedding = SeismicEmbedding {
                vector: store.nodes[i as usize].1,
                source_window_hash: i,
                norm: 1.0,
            };
            let meta = store.nodes[i as usize].2;
            learner.process(NodeId(i), &embedding, &meta, &store);
        }

        assert_eq!(learner.history().len(), 5);
        let last_3 = learner.history().last_n(3);
        assert_eq!(last_3.len(), 3);
        assert_eq!(last_3[2].0, NodeId(4));
    }
}
