//! Low-Rank Adaptation (LoRA) delta computation and application.

use serde::{Deserialize, Serialize};

/// A low-rank weight update: delta = delta_a * delta_b.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraDelta {
    pub rank: usize,
    /// Shape: in_features x rank (row-major, stored as Vec of rows).
    pub delta_a: Vec<Vec<f32>>,
    /// Shape: rank x out_features (row-major, stored as Vec of rows).
    pub delta_b: Vec<Vec<f32>>,
}

impl LoraDelta {
    /// Create a zero-initialized LoRA delta.
    pub fn zeros(in_features: usize, out_features: usize, rank: usize) -> Self {
        Self {
            rank,
            delta_a: vec![vec![0.0; rank]; in_features],
            delta_b: vec![vec![0.0; out_features]; rank],
        }
    }

    /// Apply delta_a * delta_b to weights in-place.
    ///
    /// weights is in_features x out_features (row-major, Vec of rows).
    pub fn apply_to(&self, weights: &mut [Vec<f32>]) {
        let in_features = self.delta_a.len();
        let out_features = if self.delta_b.is_empty() {
            0
        } else {
            self.delta_b[0].len()
        };

        for i in 0..in_features.min(weights.len()) {
            for j in 0..out_features.min(weights[i].len()) {
                let mut sum = 0.0f32;
                for r in 0..self.rank {
                    sum += self.delta_a[i][r] * self.delta_b[r][j];
                }
                weights[i][j] += sum;
            }
        }
    }

    /// Compute a LoRA update from node features and a gradient signal.
    ///
    /// Simplified approach:
    /// 1. Compute mean of node features.
    /// 2. Outer product of gradient_signal x mean_features.
    /// 3. Truncate to first `rank` components (simplified SVD).
    /// 4. Scale by learning_rate.
    pub fn compute_update(
        node_features: &[Vec<f32>],
        gradient_signal: &[f32],
        rank: usize,
        learning_rate: f32,
    ) -> Self {
        if node_features.is_empty() || gradient_signal.is_empty() {
            return Self::zeros(
                gradient_signal.len().max(1),
                node_features.first().map(|f| f.len()).unwrap_or(1),
                rank,
            );
        }

        let feat_dim = node_features[0].len();
        let grad_dim = gradient_signal.len();

        // Mean of node features
        let n = node_features.len() as f32;
        let mut mean_feat = vec![0.0f32; feat_dim];
        for feat in node_features {
            for (i, &v) in feat.iter().enumerate() {
                if i < feat_dim {
                    mean_feat[i] += v / n;
                }
            }
        }

        // delta_a: grad_dim x rank (truncated outer product rows)
        let actual_rank = rank.min(feat_dim).min(grad_dim);
        let delta_a: Vec<Vec<f32>> = (0..grad_dim)
            .map(|i| {
                (0..actual_rank)
                    .map(|r| gradient_signal[i] * learning_rate * if r < feat_dim { mean_feat[r] } else { 0.0 })
                    .collect()
            })
            .collect();

        // delta_b: rank x feat_dim (identity-like scaling)
        let delta_b: Vec<Vec<f32>> = (0..actual_rank)
            .map(|r| {
                (0..feat_dim)
                    .map(|j| if j == r { 1.0 } else { 0.0 })
                    .collect()
            })
            .collect();

        Self {
            rank: actual_rank,
            delta_a,
            delta_b,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_to_modifies_weights() {
        let mut weights = vec![vec![1.0f32; 4]; 3]; // 3x4

        let delta = LoraDelta {
            rank: 2,
            delta_a: vec![
                vec![1.0, 0.0],
                vec![0.0, 1.0],
                vec![0.5, 0.5],
            ],
            delta_b: vec![
                vec![0.1, 0.2, 0.3, 0.4],
                vec![0.4, 0.3, 0.2, 0.1],
            ],
        };

        let original = weights.clone();
        delta.apply_to(&mut weights);

        // Row 0: += [1*0.1+0*0.4, 1*0.2+0*0.3, ...] = [0.1, 0.2, 0.3, 0.4]
        assert!((weights[0][0] - (original[0][0] + 0.1)).abs() < 1e-6);
        assert!((weights[0][3] - (original[0][3] + 0.4)).abs() < 1e-6);

        // Row 1: += [0*0.1+1*0.4, 0*0.2+1*0.3, ...] = [0.4, 0.3, 0.2, 0.1]
        assert!((weights[1][0] - (original[1][0] + 0.4)).abs() < 1e-6);

        // Row 2: += [0.5*0.1+0.5*0.4, ...] = [0.25, ...]
        assert!((weights[2][0] - (original[2][0] + 0.25)).abs() < 1e-6);
    }

    #[test]
    fn zeros_produces_identity_apply() {
        let mut weights = vec![vec![1.0f32; 4]; 3];
        let original = weights.clone();
        let delta = LoraDelta::zeros(3, 4, 2);
        delta.apply_to(&mut weights);
        assert_eq!(weights, original);
    }

    #[test]
    fn compute_update_produces_correct_rank() {
        let features = vec![vec![1.0f32; 8]; 5];
        let gradient = vec![0.1f32; 4];
        let delta = LoraDelta::compute_update(&features, &gradient, 4, 0.01);

        assert_eq!(delta.rank, 4);
        assert_eq!(delta.delta_a.len(), 4); // grad_dim
        assert_eq!(delta.delta_a[0].len(), 4); // rank
        assert_eq!(delta.delta_b.len(), 4); // rank
        assert_eq!(delta.delta_b[0].len(), 8); // feat_dim
    }
}
