//! Micro (fast) LoRA adapter — rank 4, immediate updates.

use serde::{Deserialize, Serialize};
use crate::lora::LoraDelta;

const MICRO_RANK: usize = 4;
const MICRO_LR: f32 = 0.01;

/// Fast micro-adapter: applies rank-4 LoRA updates on every prediction.
#[derive(Serialize, Deserialize)]
pub struct MicroLoraAdapter {
    pub delta: LoraDelta,
    pub update_count: u64,
}

impl MicroLoraAdapter {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self {
            delta: LoraDelta::zeros(in_features, out_features, MICRO_RANK),
            update_count: 0,
        }
    }

    /// Compute and store a new rank-4 delta from subgraph features and prediction error.
    pub fn update(&mut self, subgraph_features: &[Vec<f32>], prediction_error: &[f32]) {
        self.delta =
            LoraDelta::compute_update(subgraph_features, prediction_error, MICRO_RANK, MICRO_LR);
        self.update_count += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn update_produces_rank_4_delta() {
        let mut adapter = MicroLoraAdapter::new(256, 64);
        let features = vec![vec![0.5f32; 256]; 10];
        let error = vec![0.1f32; 4];

        adapter.update(&features, &error);

        assert_eq!(adapter.delta.rank, 4);
        assert_eq!(adapter.update_count, 1);
        assert_eq!(adapter.delta.delta_a.len(), 4); // error dim
    }

    #[test]
    fn multiple_updates_increment_count() {
        let mut adapter = MicroLoraAdapter::new(256, 64);
        let features = vec![vec![0.5f32; 256]; 5];
        let error = vec![0.1f32; 4];

        for _ in 0..10 {
            adapter.update(&features, &error);
        }
        assert_eq!(adapter.update_count, 10);
    }
}
