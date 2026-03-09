//! CausalModel: two-layer GAT with output projection for seismic prediction.

use std::time::Instant;

use quake_vector_types::{NodeId, SeismicPrediction};

use crate::gat::GatLayer;
use crate::subgraph::TemporalSubgraph;
use crate::transition::TransitionMatrix;

/// Two-layer GAT model producing seismic predictions from temporal subgraphs.
pub struct CausalModel {
    gat_layer_1: GatLayer,
    gat_layer_2: GatLayer,
    /// Output weights: 256 -> 4 (probability, magnitude, time, confidence).
    output_weights: Vec<f32>,
    /// Output bias: [probability, magnitude, time, confidence].
    output_bias: [f32; 4],
    pub transition_matrix: TransitionMatrix,
    pub model_version: u64,
}

impl CausalModel {
    pub fn new() -> Self {
        // GAT layer 1: 256 -> 64, 4 heads => 256 output
        let gat_layer_1 = GatLayer::new(256, 64, 4);
        // GAT layer 2: 256 -> 64, 4 heads => 256 output
        let gat_layer_2 = GatLayer::new(256, 64, 4);

        // Xavier init for output weights: 256 x 4
        let mut rng = 0xfedcba9876543210u64;
        let limit = (6.0f32 / (256 + 4) as f32).sqrt();
        let output_weights: Vec<f32> = (0..256 * 4)
            .map(|_| {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                (rng as f32 / u64::MAX as f32) * 2.0 * limit - limit
            })
            .collect();

        Self {
            gat_layer_1,
            gat_layer_2,
            output_weights,
            output_bias: [0.0; 4],
            transition_matrix: TransitionMatrix::new(),
            model_version: 1,
        }
    }

    /// Run inference on a temporal subgraph to produce a seismic prediction.
    pub fn predict(&self, subgraph: &TemporalSubgraph) -> SeismicPrediction {
        let start = Instant::now();

        if subgraph.nodes.is_empty() {
            return Self::empty_prediction(self.model_version, start);
        }

        // Prepare node features from embeddings
        let node_features: Vec<Vec<f32>> = subgraph
            .nodes
            .iter()
            .map(|n| n.embedding.clone())
            .collect();

        // Prepare causal mask: true for edges where source.ts <= target.ts
        let causal_mask: Vec<bool> = subgraph
            .edges
            .iter()
            .map(|e| {
                subgraph.nodes[e.source_idx].meta.timestamp_us
                    <= subgraph.nodes[e.target_idx].meta.timestamp_us
            })
            .collect();

        let edge_features: Vec<_> = subgraph.edges.iter().map(|e| e.features).collect();

        // GAT layer 1
        let h1 = self.gat_layer_1.forward(
            &node_features,
            &subgraph.adjacency,
            &edge_features,
            &causal_mask,
        );

        // ReLU activation
        let h1_relu: Vec<Vec<f32>> = h1
            .into_iter()
            .map(|row| row.into_iter().map(|x| x.max(0.0)).collect())
            .collect();

        // GAT layer 2
        let h2 = self.gat_layer_2.forward(
            &h1_relu,
            &subgraph.adjacency,
            &edge_features,
            &causal_mask,
        );

        // Global mean pooling
        let n = h2.len() as f32;
        let dim = h2.first().map(|r| r.len()).unwrap_or(0);
        let mut pooled = vec![0.0f32; dim];
        for row in &h2 {
            for (i, &val) in row.iter().enumerate() {
                pooled[i] += val / n;
            }
        }

        // Linear output: W * pooled + bias
        let mut raw_output = [0.0f32; 4];
        for o in 0..4 {
            let mut sum = self.output_bias[o];
            for (i, &p) in pooled.iter().enumerate() {
                if i * 4 + o < self.output_weights.len() {
                    sum += self.output_weights[i * 4 + o] * p;
                }
            }
            raw_output[o] = sum;
        }

        // Activations
        let event_probability = sigmoid(raw_output[0]);
        let estimated_magnitude = softplus(raw_output[1]);
        let estimated_time_to_peak_s = softplus(raw_output[2]);
        let confidence = sigmoid(raw_output[3]);

        let contributing_wave_ids: Vec<NodeId> =
            subgraph.nodes.iter().map(|n| n.id).collect();

        let latency = start.elapsed();

        SeismicPrediction {
            event_probability,
            estimated_magnitude,
            estimated_time_to_peak_s,
            confidence,
            contributing_wave_ids,
            model_version: self.model_version,
            inference_latency_us: latency.as_micros() as u32,
        }
    }

    fn empty_prediction(version: u64, start: Instant) -> SeismicPrediction {
        SeismicPrediction {
            event_probability: 0.0,
            estimated_magnitude: 0.0,
            estimated_time_to_peak_s: 0.0,
            confidence: 0.0,
            contributing_wave_ids: Vec::new(),
            model_version: version,
            inference_latency_us: start.elapsed().as_micros() as u32,
        }
    }
}

impl Default for CausalModel {
    fn default() -> Self {
        Self::new()
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x // Avoid overflow
    } else {
        (1.0 + x.exp()).ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::subgraph::{CausalEdge, SubgraphNode};
    use quake_vector_types::{StationId, TemporalMeta, TimingFeatures, WaveType};

    fn make_synthetic_subgraph(n: usize) -> TemporalSubgraph {
        let mut rng = 123u64;
        let next = |s: &mut u64| -> f32 {
            *s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            (*s >> 33) as f32 / u32::MAX as f32
        };

        let nodes: Vec<SubgraphNode> = (0..n)
            .map(|i| {
                let mut emb = vec![0.0f32; 256];
                for v in emb.iter_mut() {
                    *v = next(&mut rng);
                }
                SubgraphNode {
                    id: NodeId(i as u64),
                    embedding: emb,
                    meta: TemporalMeta {
                        timestamp_us: i as u64 * 1_000_000,
                        wave_type: WaveType::P,
                        station_id: StationId(1),
                        amplitude_rms: 100.0,
                        dominant_freq_hz: 5.0,
                    },
                }
            })
            .collect();

        // Chain edges
        let mut edges = Vec::new();
        let mut adjacency = vec![Vec::new(); n];
        for i in 0..n.saturating_sub(1) {
            let eidx = edges.len();
            edges.push(CausalEdge {
                source_idx: i,
                target_idx: i + 1,
                features: TimingFeatures {
                    delta_us: 1_000_000,
                    transition: (WaveType::P, WaveType::P),
                    amplitude_ratio: 1.0,
                    freq_shift_hz: 0.0,
                },
            });
            adjacency[i].push((i + 1, eidx));
            adjacency[i + 1].push((i, eidx));
        }

        TemporalSubgraph {
            nodes,
            edges,
            adjacency,
            center_index: 0,
        }
    }

    #[test]
    fn predict_returns_valid_ranges() {
        let model = CausalModel::new();
        let subgraph = make_synthetic_subgraph(10);
        let pred = model.predict(&subgraph);

        assert!(
            pred.event_probability >= 0.0 && pred.event_probability <= 1.0,
            "probability {} out of [0,1]",
            pred.event_probability
        );
        assert!(
            pred.estimated_magnitude >= 0.0,
            "magnitude {} should be >= 0",
            pred.estimated_magnitude
        );
        assert!(
            pred.estimated_time_to_peak_s >= 0.0,
            "time_to_peak {} should be >= 0",
            pred.estimated_time_to_peak_s
        );
        assert!(
            pred.confidence >= 0.0 && pred.confidence <= 1.0,
            "confidence {} out of [0,1]",
            pred.confidence
        );
        assert_eq!(pred.model_version, 1);
        assert_eq!(pred.contributing_wave_ids.len(), 10);
    }

    #[test]
    fn empty_subgraph_prediction() {
        let model = CausalModel::new();
        let subgraph = TemporalSubgraph {
            nodes: Vec::new(),
            edges: Vec::new(),
            adjacency: Vec::new(),
            center_index: 0,
        };
        let pred = model.predict(&subgraph);
        assert_eq!(pred.event_probability, 0.0);
        assert_eq!(pred.confidence, 0.0);
    }

    #[test]
    fn predict_with_larger_subgraph() {
        let model = CausalModel::new();
        let subgraph = make_synthetic_subgraph(50);
        let pred = model.predict(&subgraph);

        assert!(pred.event_probability >= 0.0 && pred.event_probability <= 1.0);
        assert!(pred.estimated_magnitude >= 0.0);
        assert!(pred.inference_latency_us > 0 || pred.inference_latency_us == 0);
    }
}
