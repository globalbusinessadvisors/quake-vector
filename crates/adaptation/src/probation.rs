//! Model probation: A/B testing between candidate and current models.

use serde::{Deserialize, Serialize};
use quake_vector_types::SeismicPrediction;

use crate::ground_truth::GroundTruth;

/// Result of evaluating probation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProbationResult {
    InProgress,
    Promote,
    Rollback,
}

/// Tracks A/B comparison between a candidate model and the current model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbationState {
    pub candidate_model_version: u64,
    pub started_at: u64,
    pub candidate_correct: u32,
    pub candidate_total: u32,
    pub current_correct: u32,
    pub current_total: u32,
    /// Probation duration in microseconds (default: 1 hour).
    pub duration_us: u64,
}

/// Threshold for considering a prediction correct.
const CORRECT_THRESHOLD: f32 = 0.5;

impl ProbationState {
    pub fn new(candidate_version: u64, started_at: u64) -> Self {
        Self {
            candidate_model_version: candidate_version,
            started_at,
            candidate_correct: 0,
            candidate_total: 0,
            current_correct: 0,
            current_total: 0,
            duration_us: 3_600_000_000, // 1 hour
        }
    }

    /// Record a candidate model prediction against ground truth.
    pub fn record_candidate(
        &mut self,
        prediction: &SeismicPrediction,
        ground_truth: &GroundTruth,
    ) {
        self.candidate_total += 1;
        if Self::is_correct(prediction, ground_truth) {
            self.candidate_correct += 1;
        }
    }

    /// Record a current model prediction against ground truth.
    pub fn record_current(
        &mut self,
        prediction: &SeismicPrediction,
        ground_truth: &GroundTruth,
    ) {
        self.current_total += 1;
        if Self::is_correct(prediction, ground_truth) {
            self.current_correct += 1;
        }
    }

    /// Check if the probation period has expired.
    pub fn is_expired(&self, now_us: u64) -> bool {
        now_us.saturating_sub(self.started_at) >= self.duration_us
    }

    /// Determine whether the candidate should be promoted.
    ///
    /// Promotes if candidate accuracy >= current accuracy.
    pub fn should_promote(&self) -> bool {
        self.candidate_accuracy() >= self.current_accuracy()
    }

    fn candidate_accuracy(&self) -> f32 {
        if self.candidate_total == 0 {
            return 0.0;
        }
        self.candidate_correct as f32 / self.candidate_total as f32
    }

    fn current_accuracy(&self) -> f32 {
        if self.current_total == 0 {
            return 0.0;
        }
        self.current_correct as f32 / self.current_total as f32
    }

    fn is_correct(prediction: &SeismicPrediction, ground_truth: &GroundTruth) -> bool {
        let predicted_event = prediction.event_probability >= CORRECT_THRESHOLD;
        predicted_event == ground_truth.event_occurred
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quake_vector_types::NodeId;

    fn pred(prob: f32) -> SeismicPrediction {
        SeismicPrediction {
            event_probability: prob,
            estimated_magnitude: 3.0,
            estimated_time_to_peak_s: 5.0,
            confidence: 0.8,
            contributing_wave_ids: vec![NodeId(1)],
            model_version: 1,
            inference_latency_us: 100,
        }
    }

    fn gt(event: bool) -> GroundTruth {
        GroundTruth {
            event_occurred: event,
            actual_magnitude: if event { Some(3.0) } else { None },
            labeled_at: 0,
        }
    }

    #[test]
    fn promote_when_candidate_better() {
        let mut state = ProbationState::new(2, 0);

        // Candidate: 8/10 correct
        for _ in 0..8 {
            state.record_candidate(&pred(0.8), &gt(true));
        }
        for _ in 0..2 {
            state.record_candidate(&pred(0.8), &gt(false));
        }

        // Current: 5/10 correct
        for _ in 0..5 {
            state.record_current(&pred(0.8), &gt(true));
        }
        for _ in 0..5 {
            state.record_current(&pred(0.8), &gt(false));
        }

        assert!(state.should_promote());
    }

    #[test]
    fn rollback_when_candidate_worse() {
        let mut state = ProbationState::new(2, 0);

        // Candidate: 3/10 correct
        for _ in 0..3 {
            state.record_candidate(&pred(0.8), &gt(true));
        }
        for _ in 0..7 {
            state.record_candidate(&pred(0.8), &gt(false));
        }

        // Current: 7/10 correct
        for _ in 0..7 {
            state.record_current(&pred(0.8), &gt(true));
        }
        for _ in 0..3 {
            state.record_current(&pred(0.8), &gt(false));
        }

        assert!(!state.should_promote());
    }

    #[test]
    fn promote_when_equal_accuracy() {
        let mut state = ProbationState::new(2, 0);

        state.record_candidate(&pred(0.8), &gt(true));
        state.record_current(&pred(0.8), &gt(true));

        assert!(state.should_promote()); // >= means equal is promote
    }

    #[test]
    fn expiry_detection() {
        let state = ProbationState::new(2, 1_000_000);
        assert!(!state.is_expired(2_000_000));
        assert!(!state.is_expired(3_600_000_000)); // exactly at boundary
        assert!(state.is_expired(3_601_000_001));
    }
}
