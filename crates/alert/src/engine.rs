//! AlertEngine: aggregate root combining all alert subsystems.

use std::path::Path;

use ed25519_dalek::SigningKey;
use tracing::{debug, warn};

use quake_vector_types::{Alert, AlertLevel, ClaimId, SeismicPrediction, StationId};

use crate::builder::AlertBuilder;
use crate::consensus::ConsensusManager;
use crate::cooldown::CoolDownState;
use crate::decision::{AlertDecision, AlertDecisionService};
use crate::log::AlertLog;
use crate::thresholds::AlertThresholds;

/// Aggregate root for the alert domain.
pub struct AlertEngine {
    pub station_id: StationId,
    thresholds: AlertThresholds,
    cool_down: CoolDownState,
    consensus: ConsensusManager,
    log: AlertLog,
    signing_key: SigningKey,
}

impl AlertEngine {
    /// Create a new AlertEngine.
    pub fn new(
        station_id: StationId,
        signing_key: SigningKey,
        log_dir: &Path,
    ) -> std::io::Result<Self> {
        Ok(Self {
            station_id,
            thresholds: AlertThresholds::default(),
            cool_down: CoolDownState::new(),
            consensus: ConsensusManager::new(),
            log: AlertLog::new(log_dir)?,
            signing_key,
        })
    }

    /// Process a prediction and decide whether to emit an alert.
    pub fn process_prediction(
        &mut self,
        prediction: &SeismicPrediction,
        now_us: u64,
        mesh_available: bool,
    ) -> AlertDecision {
        let decision = AlertDecisionService::evaluate(
            prediction,
            self.station_id,
            now_us,
            &self.thresholds,
            &mut self.cool_down,
            &mut self.consensus,
            &self.signing_key,
            mesh_available,
        );

        match &decision {
            AlertDecision::EmitLocal(alert) | AlertDecision::EmitCappedAtMedium(alert) => {
                if let Err(e) = self.log.append(alert) {
                    warn!(error = %e, "failed to log alert");
                }
                debug!(level = ?alert.level, "alert emitted");
            }
            AlertDecision::DeferForConsensus(claim_id) => {
                debug!(?claim_id, "alert deferred for consensus");
            }
            AlertDecision::Suppressed { reason } => {
                debug!(%reason, "alert suppressed");
            }
            AlertDecision::NoAlert => {}
        }

        decision
    }

    /// Process a consensus confirmation from another station.
    ///
    /// If consensus is reached, builds and returns the alert at the original level.
    pub fn process_consensus_confirmation(
        &mut self,
        claim_id: &ClaimId,
        confirming_station: StationId,
    ) -> Option<Alert> {
        let status = self.consensus.add_confirmation(claim_id, confirming_station);

        if status == crate::consensus::ConsensusStatus::Reached {
            let claim = self.consensus.remove_claim(claim_id)?;
            let mut consensus_stations = claim.confirmations.clone();
            consensus_stations.push(self.station_id);

            let alert = AlertBuilder::build(
                &claim.prediction,
                claim.original_level,
                self.station_id,
                consensus_stations,
                &self.signing_key,
                claim.created_at_us,
            );

            self.cool_down.activate(
                claim.original_level,
                claim.prediction.estimated_magnitude,
                claim.created_at_us,
            );

            if let Err(e) = self.log.append(&alert) {
                warn!(error = %e, "failed to log consensus alert");
            }

            debug!(level = ?alert.level, "consensus alert emitted");
            Some(alert)
        } else {
            None
        }
    }

    /// Tick: check for expired claims and emit at MEDIUM for timed-out ones.
    pub fn tick(&mut self, now_us: u64) -> Vec<Alert> {
        let expired = self.consensus.check_expired(now_us);
        let mut alerts = Vec::new();

        for (claim_id, _original_level) in expired {
            // Emit at Medium for expired claims
            // We need to reconstruct a basic prediction — use default values
            let alert = Alert {
                alert_id: quake_vector_types::AlertId(uuid::Uuid::new_v4()),
                station_id: self.station_id,
                timestamp: now_us,
                level: AlertLevel::Medium,
                probability: 0.0,
                magnitude_estimate: 0.0,
                time_to_peak_s: 0.0,
                confidence: 0.0,
                wave_evidence: Vec::new(),
                consensus_stations: Vec::new(),
                signature: Vec::new(),
            };

            if let Err(e) = self.log.append(&alert) {
                warn!(error = %e, "failed to log expired claim alert");
            }

            debug!(?claim_id, "expired claim emitted at MEDIUM");
            alerts.push(alert);
        }

        alerts
    }

    /// Access the thresholds.
    pub fn thresholds(&self) -> &AlertThresholds {
        &self.thresholds
    }

    /// Set custom thresholds.
    pub fn set_thresholds(&mut self, thresholds: AlertThresholds) {
        self.thresholds = thresholds;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quake_vector_types::NodeId;
    use quake_vector_persistence::CryptoService;
    use std::fs;

    fn temp_dir() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "qv_alert_engine_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn make_prediction(prob: f32) -> SeismicPrediction {
        SeismicPrediction {
            event_probability: prob,
            estimated_magnitude: 3.5,
            estimated_time_to_peak_s: 5.0,
            confidence: 0.8,
            contributing_wave_ids: vec![NodeId(1)],
            model_version: 1,
            inference_latency_us: 100,
        }
    }

    #[test]
    fn end_to_end_low_alert() {
        let dir = temp_dir();
        let (sk, vk) = CryptoService::generate_keypair();
        let mut engine = AlertEngine::new(StationId(1), sk, &dir).unwrap();

        let pred = make_prediction(0.4); // Low
        let decision = engine.process_prediction(&pred, 1_000_000, true);

        match decision {
            AlertDecision::EmitLocal(alert) => {
                assert_eq!(alert.level, AlertLevel::Low);
                assert!(AlertBuilder::verify(&alert, &vk));
            }
            other => panic!("expected EmitLocal, got {other:?}"),
        }

        // Verify log file was written
        let log_content = fs::read_to_string(dir.join("alerts.jsonl")).unwrap();
        assert!(!log_content.is_empty());

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn end_to_end_consensus_flow() {
        let dir = temp_dir();
        let (sk, _) = CryptoService::generate_keypair();
        let mut engine = AlertEngine::new(StationId(1), sk, &dir).unwrap();

        let pred = make_prediction(0.8); // High
        let decision = engine.process_prediction(&pred, 1_000_000, true);

        let claim_id = match decision {
            AlertDecision::DeferForConsensus(id) => id,
            other => panic!("expected DeferForConsensus, got {other:?}"),
        };

        // Confirm from another station
        let alert = engine.process_consensus_confirmation(&claim_id, StationId(2));
        assert!(alert.is_some());
        let alert = alert.unwrap();
        assert_eq!(alert.level, AlertLevel::High);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn expired_claims_emit_at_medium() {
        let dir = temp_dir();
        let (sk, _) = CryptoService::generate_keypair();
        let mut engine = AlertEngine::new(StationId(1), sk, &dir).unwrap();

        let pred = make_prediction(0.8);
        engine.process_prediction(&pred, 1_000_000, true);

        // Tick past consensus window
        let alerts = engine.tick(20_000_000);
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].level, AlertLevel::Medium);

        fs::remove_dir_all(&dir).ok();
    }
}
