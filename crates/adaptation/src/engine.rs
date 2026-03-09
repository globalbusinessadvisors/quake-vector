//! SONA Engine: aggregate root for the three-speed adaptation system.

use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::base::BaseLoraAdapter;
use crate::lora::LoraDelta;
use crate::micro::MicroLoraAdapter;
use crate::probation::{ProbationResult, ProbationState};

/// Aggregate root for SONA three-speed learning.
///
/// Coordinates micro (fast), base (medium), and slow adaptation through
/// LoRA weight deltas and model probation.
#[derive(Serialize, Deserialize)]
pub struct SonaEngine {
    micro_adapter: MicroLoraAdapter,
    base_adapter: BaseLoraAdapter,
    pub current_model_version: u64,
    pub current_score: f32,
    probation: Option<ProbationState>,
}

impl SonaEngine {
    pub fn new() -> Self {
        Self {
            micro_adapter: MicroLoraAdapter::new(256, 64),
            base_adapter: BaseLoraAdapter::new(256, 64),
            current_model_version: 1,
            current_score: 0.0,
            probation: None,
        }
    }

    /// Apply a fast micro-update from subgraph features and prediction error.
    pub fn apply_micro_update(
        &mut self,
        subgraph_features: &[Vec<f32>],
        prediction_error: &[f32],
    ) {
        self.micro_adapter.update(subgraph_features, prediction_error);
        debug!(
            update_count = self.micro_adapter.update_count,
            "micro LoRA update applied"
        );
    }

    /// Accumulate a sample for the base (medium-speed) adapter.
    pub fn accumulate_base_update(&mut self, features: Vec<Vec<f32>>, error: Vec<f32>) {
        self.base_adapter.accumulate(features, error);
    }

    /// Flush the base adapter, producing a batched LoRA delta if the buffer
    /// has accumulated enough samples.
    pub fn flush_base_update(&mut self) -> Option<LoraDelta> {
        let delta = self.base_adapter.update();
        if delta.is_some() {
            debug!(
                update_count = self.base_adapter.update_count,
                buffer_was = "flushed",
                "base LoRA update produced"
            );
        }
        delta
    }

    /// Start a probation period for a candidate model version.
    pub fn start_probation(&mut self, candidate_version: u64, now_us: u64) {
        self.probation = Some(ProbationState::new(candidate_version, now_us));
        debug!(candidate_version, "probation started");
    }

    /// Evaluate the current probation state.
    pub fn evaluate_probation(&mut self, now_us: u64) -> ProbationResult {
        let Some(ref state) = self.probation else {
            return ProbationResult::InProgress;
        };

        if !state.is_expired(now_us) {
            return ProbationResult::InProgress;
        }

        if state.should_promote() {
            let new_version = state.candidate_model_version;
            self.current_model_version = new_version;
            debug!(new_version, "candidate model promoted");
            self.probation = None;
            ProbationResult::Promote
        } else {
            debug!("candidate model rolled back");
            self.probation = None;
            ProbationResult::Rollback
        }
    }

    /// Get the current micro LoRA delta.
    pub fn get_micro_delta(&self) -> &LoraDelta {
        &self.micro_adapter.delta
    }

    /// Get the current base LoRA delta.
    pub fn get_base_delta(&self) -> &LoraDelta {
        &self.base_adapter.delta
    }

    /// Access the probation state (if active).
    pub fn probation(&self) -> Option<&ProbationState> {
        self.probation.as_ref()
    }

    /// Mutable access to the probation state.
    pub fn probation_mut(&mut self) -> Option<&mut ProbationState> {
        self.probation.as_mut()
    }
}

impl Default for SonaEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_micro_base_cycle() {
        let mut engine = SonaEngine::new();

        // Apply micro updates
        let features = vec![vec![1.0f32; 256]; 5];
        let error = vec![0.1, -0.05, 0.02, 0.03];
        engine.apply_micro_update(&features, &error);

        // Verify micro delta was computed
        assert_eq!(engine.get_micro_delta().rank, 4);

        // Accumulate base updates
        for _ in 0..10 {
            engine.accumulate_base_update(
                vec![vec![1.0f32; 256]; 5],
                vec![0.05, -0.02, 0.01, 0.01],
            );
        }

        // Flush base update
        let delta = engine.flush_base_update();
        assert!(delta.is_some());
        assert_eq!(engine.get_base_delta().delta_a.len(), 4); // error dim
    }

    #[test]
    fn probation_lifecycle() {
        let mut engine = SonaEngine::new();
        assert_eq!(engine.current_model_version, 1);

        // Start probation
        engine.start_probation(2, 0);
        assert!(engine.probation().is_some());

        // Not expired yet
        assert_eq!(
            engine.evaluate_probation(1_000_000),
            ProbationResult::InProgress
        );

        // Record results through the probation state
        {
            let p = engine.probation_mut().unwrap();
            use quake_vector_types::NodeId;

            let good_pred = quake_vector_types::SeismicPrediction {
                event_probability: 0.8,
                estimated_magnitude: 3.0,
                estimated_time_to_peak_s: 5.0,
                confidence: 0.8,
                contributing_wave_ids: vec![NodeId(1)],
                model_version: 2,
                inference_latency_us: 100,
            };
            let gt = crate::ground_truth::GroundTruth {
                event_occurred: true,
                actual_magnitude: Some(3.0),
                labeled_at: 0,
            };

            // Candidate gets 10/10 correct
            for _ in 0..10 {
                p.record_candidate(&good_pred, &gt);
            }
            // Current gets 5/10 correct
            let bad_gt = crate::ground_truth::GroundTruth {
                event_occurred: false,
                actual_magnitude: None,
                labeled_at: 0,
            };
            for _ in 0..5 {
                p.record_current(&good_pred, &gt);
            }
            for _ in 0..5 {
                p.record_current(&good_pred, &bad_gt);
            }
        }

        // Expire and promote
        let result = engine.evaluate_probation(4_000_000_000);
        assert_eq!(result, ProbationResult::Promote);
        assert_eq!(engine.current_model_version, 2);
        assert!(engine.probation().is_none());
    }

    #[test]
    fn probation_rollback() {
        let mut engine = SonaEngine::new();
        engine.start_probation(2, 0);

        {
            let p = engine.probation_mut().unwrap();
            use quake_vector_types::NodeId;

            let pred = quake_vector_types::SeismicPrediction {
                event_probability: 0.8,
                estimated_magnitude: 3.0,
                estimated_time_to_peak_s: 5.0,
                confidence: 0.8,
                contributing_wave_ids: vec![NodeId(1)],
                model_version: 2,
                inference_latency_us: 100,
            };
            let gt_true = crate::ground_truth::GroundTruth {
                event_occurred: true,
                actual_magnitude: Some(3.0),
                labeled_at: 0,
            };
            let gt_false = crate::ground_truth::GroundTruth {
                event_occurred: false,
                actual_magnitude: None,
                labeled_at: 0,
            };

            // Candidate worse: 2/10
            for _ in 0..2 {
                p.record_candidate(&pred, &gt_true);
            }
            for _ in 0..8 {
                p.record_candidate(&pred, &gt_false);
            }
            // Current better: 8/10
            for _ in 0..8 {
                p.record_current(&pred, &gt_true);
            }
            for _ in 0..2 {
                p.record_current(&pred, &gt_false);
            }
        }

        let result = engine.evaluate_probation(4_000_000_000);
        assert_eq!(result, ProbationResult::Rollback);
        assert_eq!(engine.current_model_version, 1); // unchanged
    }
}
