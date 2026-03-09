//! Alert decision logic: evaluates predictions against thresholds,
//! cool-down, and consensus requirements.

use quake_vector_types::{Alert, AlertLevel, ClaimId, SeismicPrediction, StationId};

use crate::builder::AlertBuilder;
use crate::consensus::ConsensusManager;
use crate::cooldown::CoolDownState;
use crate::thresholds::AlertThresholds;

/// The result of evaluating a prediction for alerting.
#[derive(Debug)]
pub enum AlertDecision {
    /// Probability below threshold.
    NoAlert,
    /// Cool-down is active; alert suppressed.
    Suppressed { reason: String },
    /// Low or Medium alert emitted locally.
    EmitLocal(Alert),
    /// High/Critical alert deferred for multi-station consensus.
    DeferForConsensus(ClaimId),
    /// High/Critical alert capped at Medium (no mesh available).
    EmitCappedAtMedium(Alert),
}

/// Stateless service that makes alert decisions.
pub struct AlertDecisionService;

impl AlertDecisionService {
    /// Evaluate a prediction and decide the appropriate alert action.
    pub fn evaluate(
        prediction: &SeismicPrediction,
        station_id: StationId,
        now_us: u64,
        thresholds: &AlertThresholds,
        cool_down: &mut CoolDownState,
        consensus: &mut ConsensusManager,
        signing_key: &ed25519_dalek::SigningKey,
        mesh_available: bool,
    ) -> AlertDecision {
        // Determine level from probability
        let Some(level) = thresholds.determine_level(prediction.event_probability) else {
            return AlertDecision::NoAlert;
        };

        // Check cool-down
        if cool_down.is_active(now_us, level) {
            if !cool_down.should_override(prediction.estimated_magnitude) {
                return AlertDecision::Suppressed {
                    reason: format!(
                        "cool-down active for level {:?}, magnitude {:.1} not exceeding previous {:.1}",
                        level, prediction.estimated_magnitude, cool_down.last_magnitude
                    ),
                };
            }
        }

        match level {
            AlertLevel::High | AlertLevel::Critical => {
                if mesh_available {
                    let claim_id =
                        consensus.create_claim(prediction, level, now_us);
                    AlertDecision::DeferForConsensus(claim_id)
                } else {
                    // Cap at Medium when no mesh
                    let alert = AlertBuilder::build(
                        prediction,
                        AlertLevel::Medium,
                        station_id,
                        Vec::new(),
                        signing_key,
                        now_us,
                    );
                    cool_down.activate(AlertLevel::Medium, prediction.estimated_magnitude, now_us);
                    AlertDecision::EmitCappedAtMedium(alert)
                }
            }
            _ => {
                let alert = AlertBuilder::build(
                    prediction,
                    level,
                    station_id,
                    Vec::new(),
                    signing_key,
                    now_us,
                );
                cool_down.activate(level, prediction.estimated_magnitude, now_us);
                AlertDecision::EmitLocal(alert)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quake_vector_types::NodeId;
    use quake_vector_persistence::CryptoService;

    fn make_prediction(prob: f32, mag: f32) -> SeismicPrediction {
        SeismicPrediction {
            event_probability: prob,
            estimated_magnitude: mag,
            estimated_time_to_peak_s: 5.0,
            confidence: 0.8,
            contributing_wave_ids: vec![NodeId(1)],
            model_version: 1,
            inference_latency_us: 100,
        }
    }

    fn test_key() -> ed25519_dalek::SigningKey {
        CryptoService::generate_keypair().0
    }

    #[test]
    fn low_prediction_emits_local() {
        let pred = make_prediction(0.35, 2.0);
        let mut cd = CoolDownState::new();
        let mut cm = ConsensusManager::new();
        let key = test_key();

        let decision = AlertDecisionService::evaluate(
            &pred, StationId(1), 1_000_000,
            &AlertThresholds::default(), &mut cd, &mut cm, &key, true,
        );
        assert!(matches!(decision, AlertDecision::EmitLocal(_)));
    }

    #[test]
    fn medium_prediction_emits_local() {
        let pred = make_prediction(0.55, 3.0);
        let mut cd = CoolDownState::new();
        let mut cm = ConsensusManager::new();
        let key = test_key();

        let decision = AlertDecisionService::evaluate(
            &pred, StationId(1), 1_000_000,
            &AlertThresholds::default(), &mut cd, &mut cm, &key, true,
        );
        assert!(matches!(decision, AlertDecision::EmitLocal(_)));
    }

    #[test]
    fn high_with_mesh_defers() {
        let pred = make_prediction(0.75, 4.0);
        let mut cd = CoolDownState::new();
        let mut cm = ConsensusManager::new();
        let key = test_key();

        let decision = AlertDecisionService::evaluate(
            &pred, StationId(1), 1_000_000,
            &AlertThresholds::default(), &mut cd, &mut cm, &key, true,
        );
        assert!(matches!(decision, AlertDecision::DeferForConsensus(_)));
    }

    #[test]
    fn high_without_mesh_caps_at_medium() {
        let pred = make_prediction(0.75, 4.0);
        let mut cd = CoolDownState::new();
        let mut cm = ConsensusManager::new();
        let key = test_key();

        let decision = AlertDecisionService::evaluate(
            &pred, StationId(1), 1_000_000,
            &AlertThresholds::default(), &mut cd, &mut cm, &key, false,
        );
        assert!(matches!(decision, AlertDecision::EmitCappedAtMedium(_)));
        if let AlertDecision::EmitCappedAtMedium(alert) = decision {
            assert_eq!(alert.level, AlertLevel::Medium);
        }
    }

    #[test]
    fn cooldown_suppresses_duplicate() {
        let pred = make_prediction(0.35, 2.0);
        let mut cd = CoolDownState::new();
        let mut cm = ConsensusManager::new();
        let key = test_key();

        // First alert goes through
        let d1 = AlertDecisionService::evaluate(
            &pred, StationId(1), 1_000_000,
            &AlertThresholds::default(), &mut cd, &mut cm, &key, true,
        );
        assert!(matches!(d1, AlertDecision::EmitLocal(_)));

        // Second within cool-down is suppressed
        let d2 = AlertDecisionService::evaluate(
            &pred, StationId(1), 2_000_000,
            &AlertThresholds::default(), &mut cd, &mut cm, &key, true,
        );
        assert!(matches!(d2, AlertDecision::Suppressed { .. }));
    }

    #[test]
    fn below_threshold_no_alert() {
        let pred = make_prediction(0.1, 1.0);
        let mut cd = CoolDownState::new();
        let mut cm = ConsensusManager::new();
        let key = test_key();

        let decision = AlertDecisionService::evaluate(
            &pred, StationId(1), 1_000_000,
            &AlertThresholds::default(), &mut cd, &mut cm, &key, true,
        );
        assert!(matches!(decision, AlertDecision::NoAlert));
    }
}
