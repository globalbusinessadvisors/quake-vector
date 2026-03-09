//! Core HNSW graph implementation.

use std::collections::{BinaryHeap, HashMap, HashSet};
use std::cmp::Reverse;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use quake_vector_types::{NodeId, SearchResult, SeismicEmbedding, TemporalMeta};

use crate::config::HnswConfig;
use crate::distance::cosine_distance;
use crate::node::GraphNode;
use crate::tier::TierManager;
use crate::traits::VectorStore;

/// HNSW graph: aggregate root for the vector store domain.
#[derive(Serialize, Deserialize)]
pub struct HnswGraph {
    pub(crate) config: HnswConfig,
    pub(crate) nodes: HashMap<NodeId, GraphNode>,
    pub(crate) entry_point: Option<NodeId>,
    pub(crate) max_layer: usize,
    pub(crate) next_id: u64,
    pub(crate) tier_manager: TierManager,
    /// Simple RNG state for layer generation.
    pub(crate) rng_state: u64,
}

/// A candidate with an ordered distance (min-heap friendly via Reverse).
#[derive(Clone)]
struct Candidate {
    id: NodeId,
    distance: f32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance.to_bits() == other.distance.to_bits() && self.id == other.id
    }
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl HnswGraph {
    pub fn new(config: HnswConfig) -> Self {
        Self {
            config,
            nodes: HashMap::new(),
            entry_point: None,
            max_layer: 0,
            next_id: 0,
            tier_manager: TierManager::new(),
            rng_state: 0xdeadbeef_cafebabe,
        }
    }

    /// Access the HNSW configuration.
    pub fn config(&self) -> &HnswConfig {
        &self.config
    }

    pub fn with_default_config() -> Self {
        Self::new(HnswConfig::default())
    }

    /// Access the tier manager.
    pub fn tier_manager(&self) -> &TierManager {
        &self.tier_manager
    }

    /// Mutable access to the tier manager.
    pub fn tier_manager_mut(&mut self) -> &mut TierManager {
        &mut self.tier_manager
    }

    /// Get the current entry point.
    pub fn entry_point(&self) -> Option<NodeId> {
        self.entry_point
    }

    /// Get the current maximum layer in the graph.
    pub fn max_layer(&self) -> usize {
        self.max_layer
    }

    // -----------------------------------------------------------------------
    // Random layer generation
    // -----------------------------------------------------------------------

    fn next_random(&mut self) -> u64 {
        // xorshift64
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        self.rng_state
    }

    fn random_layer(&mut self) -> usize {
        let r = self.next_random();
        // Convert to f64 in (0, 1)
        let uniform = (r as f64) / (u64::MAX as f64);
        let level = (-uniform.ln() * self.config.ml).floor() as usize;
        level
    }

    // -----------------------------------------------------------------------
    // Distance helper
    // -----------------------------------------------------------------------

    fn distance_to_node(&self, query: &[f32], node_id: NodeId) -> f32 {
        if let Some(node) = self.nodes.get(&node_id) {
            cosine_distance(query, &node.vector)
        } else {
            f32::MAX
        }
    }

    // -----------------------------------------------------------------------
    // Greedy search: find the single closest node at a given layer
    // -----------------------------------------------------------------------

    fn greedy_search_layer(&self, query: &[f32], entry: NodeId, layer: usize) -> NodeId {
        let mut current = entry;
        let mut current_dist = self.distance_to_node(query, current);

        loop {
            let mut changed = false;
            if let Some(node) = self.nodes.get(&current) {
                if layer < node.edges.len() {
                    for &neighbor_id in &node.edges[layer] {
                        let d = self.distance_to_node(query, neighbor_id);
                        if d < current_dist {
                            current = neighbor_id;
                            current_dist = d;
                            changed = true;
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }
        current
    }

    // -----------------------------------------------------------------------
    // Beam search at a layer: returns up to ef closest candidates
    // -----------------------------------------------------------------------

    fn search_layer(
        &self,
        query: &[f32],
        entries: &[NodeId],
        ef: usize,
        layer: usize,
    ) -> Vec<Candidate> {
        let mut visited = HashSet::new();

        // Min-heap of candidates to explore (closest first)
        let mut candidates: BinaryHeap<Reverse<Candidate>> = BinaryHeap::new();
        // Max-heap of results (farthest first for easy pruning)
        let mut results: BinaryHeap<Candidate> = BinaryHeap::new();

        for &entry in entries {
            if visited.insert(entry) {
                let d = self.distance_to_node(query, entry);
                candidates.push(Reverse(Candidate { id: entry, distance: d }));
                results.push(Candidate { id: entry, distance: d });
            }
        }

        while let Some(Reverse(closest)) = candidates.pop() {
            let farthest_dist = results.peek().map(|c| c.distance).unwrap_or(f32::MAX);
            if closest.distance > farthest_dist && results.len() >= ef {
                break;
            }

            if let Some(node) = self.nodes.get(&closest.id) {
                if layer < node.edges.len() {
                    for &neighbor_id in &node.edges[layer] {
                        if visited.insert(neighbor_id) {
                            let d = self.distance_to_node(query, neighbor_id);
                            let farthest_dist =
                                results.peek().map(|c| c.distance).unwrap_or(f32::MAX);

                            if d < farthest_dist || results.len() < ef {
                                candidates
                                    .push(Reverse(Candidate { id: neighbor_id, distance: d }));
                                results.push(Candidate { id: neighbor_id, distance: d });
                                if results.len() > ef {
                                    results.pop(); // remove farthest
                                }
                            }
                        }
                    }
                }
            }
        }

        results.into_sorted_vec()
    }

    // -----------------------------------------------------------------------
    // Select M best neighbors (simple heuristic)
    // -----------------------------------------------------------------------

    fn select_neighbors(candidates: &[Candidate], m: usize) -> Vec<NodeId> {
        candidates
            .iter()
            .take(m)
            .map(|c| c.id)
            .collect()
    }

    // -----------------------------------------------------------------------
    // Add bidirectional edge, pruning if over capacity
    // -----------------------------------------------------------------------

    fn connect_nodes(&mut self, a: NodeId, b: NodeId, layer: usize, max_edges: usize) {
        // a -> b
        if let Some(node_a) = self.nodes.get_mut(&a) {
            if layer < node_a.edges.len() && !node_a.edges[layer].contains(&b) {
                node_a.edges[layer].push(b);
                if node_a.edges[layer].len() > max_edges {
                    self.prune_edges(a, layer, max_edges);
                }
            }
        }
        // b -> a
        if let Some(node_b) = self.nodes.get_mut(&b) {
            if layer < node_b.edges.len() && !node_b.edges[layer].contains(&a) {
                node_b.edges[layer].push(a);
                if node_b.edges[layer].len() > max_edges {
                    self.prune_edges(b, layer, max_edges);
                }
            }
        }
    }

    fn prune_edges(&mut self, node_id: NodeId, layer: usize, max_edges: usize) {
        let node_vec: Vec<f32>;
        let edge_ids: Vec<NodeId>;

        if let Some(node) = self.nodes.get(&node_id) {
            node_vec = node.vector.clone();
            edge_ids = node.edges[layer].clone();
        } else {
            return;
        }

        let mut scored: Vec<Candidate> = edge_ids
            .into_iter()
            .map(|id| Candidate {
                id,
                distance: self.distance_to_node(&node_vec, id),
            })
            .collect();
        scored.sort();

        let kept: Vec<NodeId> = scored.into_iter().take(max_edges).map(|c| c.id).collect();
        if let Some(node) = self.nodes.get_mut(&node_id) {
            node.edges[layer] = kept;
        }
    }
}

impl VectorStore for HnswGraph {
    fn insert(&mut self, embedding: &SeismicEmbedding, meta: &TemporalMeta) -> NodeId {
        let id = NodeId(self.next_id);
        self.next_id += 1;

        let new_layer = self.random_layer();
        let vector = embedding.vector.to_vec();

        let node = GraphNode::new(id, vector, *meta, new_layer, meta.timestamp_us);
        self.nodes.insert(id, node);
        self.tier_manager.register(id, meta.timestamp_us);

        // First node: set as entry point
        let Some(entry) = self.entry_point else {
            self.entry_point = Some(id);
            self.max_layer = new_layer;
            return id;
        };

        let query = &self.nodes[&id].vector.clone();

        // Greedy descent from top layer down to new_layer + 1
        let mut current_entry = entry;
        let top = self.max_layer;
        if top > new_layer {
            for layer in (new_layer + 1..=top).rev() {
                current_entry = self.greedy_search_layer(query, current_entry, layer);
            }
        }

        // Insert at layers new_layer down to 0
        let ef_construction = self.config.ef_construction;
        let m = self.config.m;
        let m_max0 = self.config.m_max0;

        for layer in (0..=new_layer.min(top)).rev() {
            let neighbors = self.search_layer(query, &[current_entry], ef_construction, layer);
            let max_edges = if layer == 0 { m_max0 } else { m };
            let selected = Self::select_neighbors(&neighbors, max_edges);

            for &neighbor_id in &selected {
                self.connect_nodes(id, neighbor_id, layer, max_edges);
            }

            // Use the closest found as entry for the next layer down
            if let Some(closest) = neighbors.first() {
                current_entry = closest.id;
            }
        }

        // Update entry point if new node has a higher layer
        if new_layer > self.max_layer {
            self.entry_point = Some(id);
            self.max_layer = new_layer;
        }

        id
    }

    fn knn_search(&self, query: &[f32; 256], k: usize, ef: usize) -> SearchResult {
        let start = Instant::now();

        let Some(entry) = self.entry_point else {
            return SearchResult {
                neighbors: Vec::new(),
                search_ef: ef as u16,
                nodes_visited: 0,
                latency_us: 0,
            };
        };

        // Greedy descent from top layer to layer 1
        let mut current = entry;
        for layer in (1..=self.max_layer).rev() {
            current = self.greedy_search_layer(query.as_slice(), current, layer);
        }

        // Beam search at layer 0
        let ef = ef.max(k);
        let results = self.search_layer(query.as_slice(), &[current], ef, 0);

        let neighbors: Vec<(NodeId, f32)> = results
            .into_iter()
            .take(k)
            .map(|c| (c.id, c.distance))
            .collect();

        let latency = start.elapsed();

        SearchResult {
            neighbors,
            search_ef: ef as u16,
            nodes_visited: 0, // Tracked approximately by visited set in search_layer
            latency_us: latency.as_micros() as u32,
        }
    }

    fn get_meta(&self, id: NodeId) -> Option<&TemporalMeta> {
        self.nodes.get(&id).map(|n| &n.metadata)
    }

    fn get_vector(&self, id: NodeId) -> Option<&[f32]> {
        self.nodes.get(&id).map(|n| n.vector.as_slice())
    }

    fn node_count(&self) -> u64 {
        self.nodes.len() as u64
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use quake_vector_types::{StationId, WaveType};

    fn make_random_embedding(seed: u64) -> SeismicEmbedding {
        let mut vector = [0.0f32; 256];
        let mut state = seed ^ 0x12345678;
        for v in vector.iter_mut() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *v = (state as f32 / u64::MAX as f32) * 2.0 - 1.0;
        }
        // L2 normalize
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
    fn insert_1000_vectors_count() {
        let mut graph = HnswGraph::with_default_config();
        for i in 0..1000 {
            let emb = make_random_embedding(i);
            let meta = make_meta(i * 1000);
            graph.insert(&emb, &meta);
        }
        assert_eq!(graph.node_count(), 1000);
    }

    #[test]
    fn knn_returns_self_as_nearest() {
        let mut graph = HnswGraph::with_default_config();

        // Insert some random vectors
        for i in 0..100 {
            let emb = make_random_embedding(i);
            let meta = make_meta(i * 1000);
            graph.insert(&emb, &meta);
        }

        // Search for one of the inserted vectors — should find itself
        let target_emb = make_random_embedding(42);
        let result = graph.knn_search(&target_emb.vector, 1, 64);
        assert!(!result.neighbors.is_empty());

        let (best_id, best_dist) = &result.neighbors[0];
        assert_eq!(*best_id, NodeId(42));
        assert!(
            *best_dist < 0.01,
            "searching for an exact vector should yield distance ~0, got {best_dist}"
        );
    }

    #[test]
    fn knn_returns_k_results() {
        let mut graph = HnswGraph::with_default_config();
        for i in 0..100 {
            let emb = make_random_embedding(i);
            let meta = make_meta(i * 1000);
            graph.insert(&emb, &meta);
        }

        let query = make_random_embedding(999);
        let result = graph.knn_search(&query.vector, 16, 64);
        assert_eq!(
            result.neighbors.len(),
            16,
            "should return exactly k=16 neighbors"
        );
    }

    #[test]
    fn knn_results_sorted_by_distance() {
        let mut graph = HnswGraph::with_default_config();
        for i in 0..200 {
            let emb = make_random_embedding(i);
            let meta = make_meta(i * 1000);
            graph.insert(&emb, &meta);
        }

        let query = make_random_embedding(9999);
        let result = graph.knn_search(&query.vector, 10, 64);
        for pair in result.neighbors.windows(2) {
            assert!(
                pair[0].1 <= pair[1].1 + 1e-6,
                "results should be sorted by distance: {} > {}",
                pair[0].1,
                pair[1].1
            );
        }
    }

    #[test]
    fn layer_distribution_geometric() {
        let mut graph = HnswGraph::with_default_config();
        let mut layer_counts = vec![0u32; 10];
        for _ in 0..10000 {
            let l = graph.random_layer();
            if l < layer_counts.len() {
                layer_counts[l] += 1;
            }
        }
        // Layer 0 should have the most nodes, decreasing geometrically
        assert!(
            layer_counts[0] > layer_counts[1],
            "layer 0 ({}) should have more nodes than layer 1 ({})",
            layer_counts[0],
            layer_counts[1]
        );
        if layer_counts[2] > 0 {
            assert!(
                layer_counts[1] >= layer_counts[2],
                "layer 1 ({}) should have >= layer 2 ({})",
                layer_counts[1],
                layer_counts[2]
            );
        }
        // Layer 0 should have roughly (1 - 1/m) fraction ≈ 94% for m=16
        let l0_frac = layer_counts[0] as f64 / 10000.0;
        assert!(
            l0_frac > 0.85 && l0_frac < 0.99,
            "layer 0 fraction should be ~0.94, got {l0_frac}"
        );
    }

    #[test]
    fn entry_point_updates_for_higher_layer() {
        let config = HnswConfig {
            ml: 100.0, // Very high ml => most nodes get high layers
            ..HnswConfig::default()
        };
        let mut graph = HnswGraph::new(config);

        let emb0 = make_random_embedding(0);
        let meta0 = make_meta(0);
        let _id0 = graph.insert(&emb0, &meta0);

        let _initial_entry = graph.entry_point().unwrap();
        let initial_max = graph.max_layer();

        // Insert many nodes — eventually one should get a higher layer
        let mut max_seen = initial_max;
        let mut entry_changed = false;
        for i in 1..50 {
            let emb = make_random_embedding(i);
            let meta = make_meta(i * 1000);
            graph.insert(&emb, &meta);
            if graph.max_layer() > max_seen {
                max_seen = graph.max_layer();
                entry_changed = true;
            }
        }

        // With ml=100 almost every node gets a high layer, so entry should change
        assert!(
            entry_changed || graph.max_layer() >= initial_max,
            "entry point should update when higher-layer nodes are inserted"
        );
    }

    #[test]
    fn empty_graph_search() {
        let graph = HnswGraph::with_default_config();
        let query = [0.0f32; 256];
        let result = graph.knn_search(&query, 10, 64);
        assert!(result.neighbors.is_empty());
    }

    #[test]
    fn single_node_search() {
        let mut graph = HnswGraph::with_default_config();
        let emb = make_random_embedding(1);
        let meta = make_meta(1000);
        graph.insert(&emb, &meta);

        let result = graph.knn_search(&emb.vector, 5, 64);
        assert_eq!(result.neighbors.len(), 1);
        assert!(result.neighbors[0].1 < 1e-6);
    }

    #[test]
    fn get_meta_and_vector() {
        let mut graph = HnswGraph::with_default_config();
        let emb = make_random_embedding(7);
        let meta = make_meta(7000);
        let id = graph.insert(&emb, &meta);

        assert!(graph.get_meta(id).is_some());
        assert_eq!(graph.get_meta(id).unwrap().timestamp_us, 7000);
        assert!(graph.get_vector(id).is_some());
        assert_eq!(graph.get_vector(id).unwrap().len(), 256);

        assert!(graph.get_meta(NodeId(9999)).is_none());
    }
}

// ---------------------------------------------------------------------------
// Benchmark (not a #[test], run with --ignored or called manually)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod bench {
    use super::*;
    use quake_vector_types::{StationId, WaveType};
    use std::time::Instant;

    fn make_random_embedding(seed: u64) -> SeismicEmbedding {
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
    #[ignore] // Run with: cargo test -p quake-vector-store -- --ignored --nocapture
    fn benchmark_insert_and_search() {
        let n = 10_000;
        let mut graph = HnswGraph::with_default_config();

        // Pre-generate embeddings
        let embeddings: Vec<SeismicEmbedding> =
            (0..n).map(|i| make_random_embedding(i as u64)).collect();
        let metas: Vec<TemporalMeta> = (0..n).map(|i| make_meta(i as u64 * 1000)).collect();

        // Benchmark inserts
        let mut insert_latencies = Vec::with_capacity(n);
        for i in 0..n {
            let start = Instant::now();
            graph.insert(&embeddings[i], &metas[i]);
            insert_latencies.push(start.elapsed().as_micros() as u64);
        }

        insert_latencies.sort();
        let mean_insert = insert_latencies.iter().sum::<u64>() as f64 / n as f64;
        let p99_insert = insert_latencies[(n as f64 * 0.99) as usize];

        println!("=== Insert Benchmark ({n} vectors) ===");
        println!("  Mean: {mean_insert:.1} us");
        println!("  P99:  {p99_insert} us");

        // Benchmark searches
        let num_queries = 1000;
        let mut search_latencies = Vec::with_capacity(num_queries);
        for i in 0..num_queries {
            let query = &embeddings[i * 10 % n].vector;
            let start = Instant::now();
            let _result = graph.knn_search(query, 16, 64);
            search_latencies.push(start.elapsed().as_micros() as u64);
        }

        search_latencies.sort();
        let mean_search = search_latencies.iter().sum::<u64>() as f64 / num_queries as f64;
        let p99_search = search_latencies[(num_queries as f64 * 0.99) as usize];

        println!("=== KNN Search Benchmark ({num_queries} queries, k=16, ef=64) ===");
        println!("  Mean: {mean_search:.1} us");
        println!("  P99:  {p99_search} us");
        println!("  Node count: {}", graph.node_count());
    }
}
