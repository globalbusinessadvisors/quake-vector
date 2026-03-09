//! Base (medium-speed) LoRA adapter — rank 16, batched updates.

use serde::{Deserialize, Serialize};
use crate::lora::LoraDelta;

const BASE_RANK: usize = 16;
const BASE_LR: f32 = 0.001;
const BATCH_THRESHOLD: usize = 200;

/// Medium-speed base adapter: accumulates observations and produces
/// averaged rank-16 LoRA deltas in batches.
#[derive(Serialize, Deserialize)]
pub struct BaseLoraAdapter {
    pub delta: LoraDelta,
    pub update_count: u64,
    batch_buffer: Vec<(Vec<Vec<f32>>, Vec<f32>)>,
    _in_features: usize,
    _out_features: usize,
}

impl BaseLoraAdapter {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self {
            delta: LoraDelta::zeros(in_features, out_features, BASE_RANK),
            update_count: 0,
            batch_buffer: Vec::new(),
            _in_features: in_features,
            _out_features: out_features,
        }
    }

    /// Accumulate a (features, error) pair into the batch buffer.
    pub fn accumulate(&mut self, features: Vec<Vec<f32>>, error: Vec<f32>) {
        self.batch_buffer.push((features, error));
    }

    /// Number of samples in the current batch buffer.
    pub fn buffer_len(&self) -> usize {
        self.batch_buffer.len()
    }

    /// Produce a batched update if the buffer meets the threshold,
    /// or if forced (non-empty when called by timer).
    ///
    /// Computes an averaged rank-16 LoRA delta, clears the buffer, and returns it.
    pub fn update(&mut self) -> Option<LoraDelta> {
        if self.batch_buffer.is_empty() {
            return None;
        }
        if self.batch_buffer.len() < BATCH_THRESHOLD {
            // Allow forced flush even below threshold
            // (caller decides when to call)
        }

        let n = self.batch_buffer.len() as f32;

        // Average gradient signal
        let error_dim = self.batch_buffer[0].1.len();
        let mut avg_error = vec![0.0f32; error_dim];
        for (_, err) in &self.batch_buffer {
            for (i, &e) in err.iter().enumerate() {
                if i < error_dim {
                    avg_error[i] += e / n;
                }
            }
        }

        // Collect all node features
        let all_features: Vec<Vec<f32>> = self
            .batch_buffer
            .iter()
            .flat_map(|(feats, _)| feats.clone())
            .collect();

        let delta = LoraDelta::compute_update(&all_features, &avg_error, BASE_RANK, BASE_LR);

        self.delta = delta.clone();
        self.update_count += 1;
        self.batch_buffer.clear();

        Some(delta)
    }

    /// Check if the buffer has reached the batch threshold.
    pub fn is_batch_ready(&self) -> bool {
        self.batch_buffer.len() >= BATCH_THRESHOLD
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accumulate_fills_buffer() {
        let mut adapter = BaseLoraAdapter::new(256, 64);
        assert_eq!(adapter.buffer_len(), 0);

        for _ in 0..50 {
            adapter.accumulate(vec![vec![1.0; 256]; 5], vec![0.1; 4]);
        }
        assert_eq!(adapter.buffer_len(), 50);
    }

    #[test]
    fn update_produces_delta_and_clears() {
        let mut adapter = BaseLoraAdapter::new(256, 64);

        for _ in 0..10 {
            adapter.accumulate(vec![vec![1.0; 256]; 5], vec![0.1; 4]);
        }
        assert_eq!(adapter.buffer_len(), 10);

        let delta = adapter.update();
        assert!(delta.is_some());
        let d = delta.unwrap();
        assert_eq!(d.rank, 4); // min(16, 4_error_dim, 256) => 4
        assert_eq!(adapter.buffer_len(), 0);
        assert_eq!(adapter.update_count, 1);
    }

    #[test]
    fn empty_buffer_returns_none() {
        let mut adapter = BaseLoraAdapter::new(256, 64);
        assert!(adapter.update().is_none());
    }

    #[test]
    fn batch_ready_threshold() {
        let mut adapter = BaseLoraAdapter::new(256, 64);
        for _ in 0..199 {
            adapter.accumulate(vec![vec![1.0; 8]; 2], vec![0.1; 4]);
        }
        assert!(!adapter.is_batch_ready());

        adapter.accumulate(vec![vec![1.0; 8]; 2], vec![0.1; 4]);
        assert!(adapter.is_batch_ready());
    }
}
