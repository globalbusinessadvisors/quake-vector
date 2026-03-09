//! Graph Attention Network layer with multi-head attention.

use quake_vector_types::TimingFeatures;

/// A single Graph Attention layer with multi-head attention.
pub struct GatLayer {
    in_features: usize,
    out_features: usize,
    num_heads: usize,
    /// Weight matrices: one per head, each in_features x out_features (row-major).
    weights: Vec<Vec<f32>>,
    /// Source attention vectors: one per head, out_features each.
    a_src: Vec<Vec<f32>>,
    /// Target attention vectors: one per head, out_features each.
    a_tgt: Vec<Vec<f32>>,
}

impl GatLayer {
    /// Create a new GAT layer with Xavier/Glorot initialization.
    pub fn new(in_features: usize, out_features: usize, num_heads: usize) -> Self {
        let mut rng_state = 0xabcdef1234567890u64;

        let next_rand = |state: &mut u64| -> f32 {
            *state ^= *state << 13;
            *state ^= *state >> 7;
            *state ^= *state << 17;
            (*state as f32 / u64::MAX as f32) * 2.0 - 1.0
        };

        // Xavier uniform: limit = sqrt(6 / (fan_in + fan_out))
        let limit = (6.0f32 / (in_features + out_features) as f32).sqrt();

        let weights: Vec<Vec<f32>> = (0..num_heads)
            .map(|_| {
                (0..in_features * out_features)
                    .map(|_| next_rand(&mut rng_state) * limit)
                    .collect()
            })
            .collect();

        let attn_limit = (6.0f32 / (out_features + 1) as f32).sqrt();
        let a_src: Vec<Vec<f32>> = (0..num_heads)
            .map(|_| {
                (0..out_features)
                    .map(|_| next_rand(&mut rng_state) * attn_limit)
                    .collect()
            })
            .collect();

        let a_tgt: Vec<Vec<f32>> = (0..num_heads)
            .map(|_| {
                (0..out_features)
                    .map(|_| next_rand(&mut rng_state) * attn_limit)
                    .collect()
            })
            .collect();

        Self {
            in_features,
            out_features,
            num_heads,
            weights,
            a_src,
            a_tgt,
        }
    }

    /// Output dimensionality (out_features * num_heads).
    pub fn output_dim(&self) -> usize {
        self.out_features * self.num_heads
    }

    /// Forward pass through the GAT layer.
    ///
    /// - `node_features`: N x in_features
    /// - `adjacency`: for each node, list of (neighbor_idx, edge_idx)
    /// - `edge_features`: per-edge timing features (unused in basic attention but
    ///   available for future edge-conditioned attention)
    /// - `causal_mask`: per-edge, true = attend, false = mask out
    ///
    /// Returns N x (out_features * num_heads)
    pub fn forward(
        &self,
        node_features: &[Vec<f32>],
        adjacency: &[Vec<(usize, usize)>],
        _edge_features: &[TimingFeatures],
        causal_mask: &[bool],
    ) -> Vec<Vec<f32>> {
        let n = node_features.len();
        let out_dim = self.output_dim();
        let mut output = vec![vec![0.0f32; out_dim]; n];

        if n == 0 {
            return output;
        }

        for head in 0..self.num_heads {
            let w = &self.weights[head];
            let a_s = &self.a_src[head];
            let a_t = &self.a_tgt[head];

            // Project all nodes: h_i = W * x_i
            let projected: Vec<Vec<f32>> = node_features
                .iter()
                .map(|x| self.mat_vec_mul(w, x))
                .collect();

            // Pre-compute a_src . h_i for each node
            let src_scores: Vec<f32> = projected.iter().map(|h| dot(a_s, h)).collect();
            let tgt_scores: Vec<f32> = projected.iter().map(|h| dot(a_t, h)).collect();

            // For each node, compute attention-weighted sum over neighbors
            for i in 0..n {
                if adjacency[i].is_empty() {
                    // No neighbors: use self-projection
                    let offset = head * self.out_features;
                    for (k, &val) in projected[i].iter().enumerate() {
                        output[i][offset + k] = val;
                    }
                    continue;
                }

                // Compute attention scores
                let mut attn_scores: Vec<(usize, f32)> = Vec::new();
                for &(j, edge_idx) in &adjacency[i] {
                    if edge_idx < causal_mask.len() && !causal_mask[edge_idx] {
                        continue; // Masked out
                    }
                    let e = leaky_relu(src_scores[i] + tgt_scores[j], 0.2);
                    attn_scores.push((j, e));
                }

                if attn_scores.is_empty() {
                    let offset = head * self.out_features;
                    for (k, &val) in projected[i].iter().enumerate() {
                        output[i][offset + k] = val;
                    }
                    continue;
                }

                // Softmax
                let max_score = attn_scores
                    .iter()
                    .map(|(_, s)| *s)
                    .fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = attn_scores
                    .iter()
                    .map(|(_, s)| (s - max_score).exp())
                    .sum();

                let offset = head * self.out_features;
                for &(j, score) in &attn_scores {
                    let alpha = (score - max_score).exp() / exp_sum.max(1e-8);
                    for k in 0..self.out_features {
                        output[i][offset + k] += alpha * projected[j][k];
                    }
                }
            }
        }

        output
    }

    /// Multiply weight matrix (in_features x out_features, row-major) by input vector.
    fn mat_vec_mul(&self, w: &[f32], x: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; self.out_features];
        for o in 0..self.out_features {
            let mut sum = 0.0f32;
            for i in 0..self.in_features {
                sum += w[i * self.out_features + o] * x.get(i).copied().unwrap_or(0.0);
            }
            out[o] = sum;
        }
        out
    }

    /// Get attention weights for a specific node (for testing).
    /// Returns attention weights per head for all neighbors.
    pub fn compute_attention_weights(
        &self,
        node_features: &[Vec<f32>],
        adjacency: &[Vec<(usize, usize)>],
        causal_mask: &[bool],
        node_idx: usize,
    ) -> Vec<Vec<f32>> {
        let mut head_weights = Vec::new();

        for head in 0..self.num_heads {
            let w = &self.weights[head];
            let a_s = &self.a_src[head];
            let a_t = &self.a_tgt[head];

            let projected: Vec<Vec<f32>> = node_features
                .iter()
                .map(|x| self.mat_vec_mul(w, x))
                .collect();

            let src_scores: Vec<f32> = projected.iter().map(|h| dot(a_s, h)).collect();
            let tgt_scores: Vec<f32> = projected.iter().map(|h| dot(a_t, h)).collect();

            let mut attn_scores: Vec<f32> = Vec::new();
            for &(j, edge_idx) in &adjacency[node_idx] {
                if edge_idx < causal_mask.len() && !causal_mask[edge_idx] {
                    attn_scores.push(f32::NEG_INFINITY);
                    continue;
                }
                let e = leaky_relu(src_scores[node_idx] + tgt_scores[j], 0.2);
                attn_scores.push(e);
            }

            // Softmax
            if !attn_scores.is_empty() {
                let max_s = attn_scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = attn_scores.iter().map(|s| (s - max_s).exp()).sum();
                let weights: Vec<f32> = attn_scores
                    .iter()
                    .map(|s| (s - max_s).exp() / exp_sum.max(1e-8))
                    .collect();
                head_weights.push(weights);
            } else {
                head_weights.push(Vec::new());
            }
        }

        head_weights
    }
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[inline]
fn leaky_relu(x: f32, negative_slope: f32) -> f32 {
    if x > 0.0 { x } else { negative_slope * x }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quake_vector_types::WaveType;

    fn make_features(n: usize, dim: usize) -> Vec<Vec<f32>> {
        let mut features = Vec::new();
        let mut s = 42u64;
        for _ in 0..n {
            let mut f = Vec::with_capacity(dim);
            for _ in 0..dim {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                f.push((s >> 33) as f32 / u32::MAX as f32);
            }
            features.push(f);
        }
        features
    }

    #[test]
    fn output_shape_correct() {
        let layer = GatLayer::new(256, 64, 4);
        let features = make_features(10, 256);

        // Simple chain adjacency: 0->1->2->...->9
        let mut adjacency = vec![Vec::new(); 10];
        let mut edges = Vec::new();
        let mut causal_mask = Vec::new();
        for i in 0..9 {
            let eidx = edges.len();
            edges.push(TimingFeatures {
                delta_us: 1000,
                transition: (WaveType::P, WaveType::P),
                amplitude_ratio: 1.0,
                freq_shift_hz: 0.0,
            });
            causal_mask.push(true);
            adjacency[i].push((i + 1, eidx));
            adjacency[i + 1].push((i, eidx));
        }

        let output = layer.forward(&features, &adjacency, &edges, &causal_mask);

        assert_eq!(output.len(), 10);
        assert_eq!(output[0].len(), 64 * 4); // out_features * num_heads = 256
    }

    #[test]
    fn attention_weights_sum_to_one() {
        let layer = GatLayer::new(256, 64, 4);
        let features = make_features(5, 256);

        // Fully connected (small graph)
        let mut adjacency = vec![Vec::new(); 5];
        let mut edges = Vec::new();
        let mut causal_mask = Vec::new();
        for i in 0..5 {
            for j in 0..5 {
                if i != j {
                    let eidx = edges.len();
                    edges.push(TimingFeatures {
                        delta_us: 1000,
                        transition: (WaveType::P, WaveType::S),
                        amplitude_ratio: 1.0,
                        freq_shift_hz: 0.0,
                    });
                    causal_mask.push(true);
                    adjacency[i].push((j, eidx));
                }
            }
        }

        // Check attention weights for node 0
        let weights = layer.compute_attention_weights(&features, &adjacency, &causal_mask, 0);
        assert_eq!(weights.len(), 4); // 4 heads

        for (head_idx, head_weights) in weights.iter().enumerate() {
            let sum: f32 = head_weights.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-4,
                "head {head_idx} attention weights sum to {sum}, expected 1.0"
            );
        }
    }

    #[test]
    fn empty_graph_returns_correct_shape() {
        let layer = GatLayer::new(256, 64, 4);
        let features: Vec<Vec<f32>> = Vec::new();
        let adjacency: Vec<Vec<(usize, usize)>> = Vec::new();
        let output = layer.forward(&features, &adjacency, &[], &[]);
        assert!(output.is_empty());
    }

    #[test]
    fn isolated_node_uses_self_projection() {
        let layer = GatLayer::new(256, 64, 4);
        let features = make_features(3, 256);

        // Node 1 has no neighbors
        let adjacency = vec![vec![], vec![], vec![]];
        let output = layer.forward(&features, &adjacency, &[], &[]);

        assert_eq!(output.len(), 3);
        assert_eq!(output[1].len(), 256);
        // Output should be non-zero (self-projection)
        let sum: f32 = output[1].iter().map(|x| x.abs()).sum();
        assert!(sum > 0.0, "isolated node should have non-zero output from self-projection");
    }
}
