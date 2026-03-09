//! Adapter wrapping ruvector-sona for the SonaEngine interface.
//!
//! Uses ruvector-sona's SonaEngine with its MicroLoRA and BaseLoRA
//! implementations as an alternative to our custom SONA engine.

use serde::{Deserialize, Serialize};

/// SONA engine adapter backed by ruvector-sona.
#[derive(Serialize, Deserialize)]
pub struct RuvectorSonaEngine {
    /// Inner ruvector-sona engine (not serializable, re-created on deserialize).
    #[serde(skip, default = "create_default_engine")]
    inner: ruvector_sona::SonaEngine,
    /// Track micro update count for compatibility.
    micro_update_count: u64,
}

fn create_default_engine() -> ruvector_sona::SonaEngine {
    ruvector_sona::SonaEngine::new(256)
}

impl RuvectorSonaEngine {
    pub fn new() -> Self {
        Self {
            inner: ruvector_sona::SonaEngine::new(256),
            micro_update_count: 0,
        }
    }

    /// Apply a micro LoRA update using ruvector-sona's implementation.
    pub fn apply_micro_update(&mut self, features: &[Vec<f32>], error: &[f32]) {
        // ruvector-sona expects input and output to both be hidden_dim length (256)
        if let Some(input) = features.first() {
            let mut output = vec![0.0f32; input.len()];
            // Copy error signal into output buffer (padded to hidden_dim)
            let copy_len = error.len().min(output.len());
            output[..copy_len].copy_from_slice(&error[..copy_len]);
            self.inner.apply_micro_lora(input, &mut output);
        }
        self.micro_update_count += 1;
    }

    /// Accumulate base update.
    pub fn accumulate_base_update(&mut self, _features: Vec<Vec<f32>>, _error: Vec<f32>) {
        // ruvector-sona handles base LoRA internally via tick()
    }

    /// Flush base update — delegates to ruvector-sona's tick.
    pub fn flush_base_update(&mut self) {
        self.inner.tick();
    }

    /// Micro update count.
    pub fn micro_update_count(&self) -> u64 {
        self.micro_update_count
    }
}

impl Default for RuvectorSonaEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ruvector_sona_micro_update() {
        let mut engine = RuvectorSonaEngine::new();
        let features = vec![vec![0.1f32; 256]];
        let error = vec![0.01, 0.02, 0.03, 0.04];

        engine.apply_micro_update(&features, &error);
        assert_eq!(engine.micro_update_count(), 1);

        engine.flush_base_update();
    }
}
