//! Alert construction with Ed25519 signing.

use ed25519_dalek::SigningKey;
use uuid::Uuid;

use quake_vector_types::{Alert, AlertId, AlertLevel, NodeId, SeismicPrediction, StationId};
use quake_vector_persistence::CryptoService;

/// Builds and signs Alert instances.
pub struct AlertBuilder;

impl AlertBuilder {
    /// Build a signed alert from a prediction.
    ///
    /// All fields except `signature` are serialized via bincode, then signed
    /// with the station's Ed25519 key.
    pub fn build(
        prediction: &SeismicPrediction,
        level: AlertLevel,
        station_id: StationId,
        consensus_stations: Vec<StationId>,
        signing_key: &SigningKey,
        now_us: u64,
    ) -> Alert {
        let alert_id = AlertId(Uuid::new_v4());
        let wave_evidence: Vec<NodeId> = prediction.contributing_wave_ids.clone();

        // Build the alert without signature first
        let mut alert = Alert {
            alert_id,
            station_id,
            timestamp: now_us,
            level,
            probability: prediction.event_probability,
            magnitude_estimate: prediction.estimated_magnitude,
            time_to_peak_s: prediction.estimated_time_to_peak_s,
            confidence: prediction.confidence,
            wave_evidence,
            consensus_stations,
            signature: Vec::new(),
        };

        // Serialize for signing (all fields except signature)
        let sign_data = Self::signable_bytes(&alert);
        alert.signature = CryptoService::sign(&sign_data, signing_key);

        alert
    }

    /// Verify an alert's signature.
    pub fn verify(alert: &Alert, verifying_key: &ed25519_dalek::VerifyingKey) -> bool {
        let sign_data = Self::signable_bytes(alert);
        CryptoService::verify(&sign_data, &alert.signature, verifying_key)
    }

    /// Produce the canonical bytes for signing/verification.
    fn signable_bytes(alert: &Alert) -> Vec<u8> {
        // Use bincode to serialize a tuple of all fields except signature
        let signable = (
            &alert.alert_id,
            &alert.station_id,
            alert.timestamp,
            &alert.level,
            alert.probability,
            alert.magnitude_estimate,
            alert.time_to_peak_s,
            alert.confidence,
            &alert.wave_evidence,
            &alert.consensus_stations,
        );
        bincode::serialize(&signable).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_prediction() -> SeismicPrediction {
        SeismicPrediction {
            event_probability: 0.8,
            estimated_magnitude: 4.5,
            estimated_time_to_peak_s: 7.0,
            confidence: 0.85,
            contributing_wave_ids: vec![NodeId(1), NodeId(2)],
            model_version: 1,
            inference_latency_us: 200,
        }
    }

    #[test]
    fn signature_verifies_correctly() {
        let (sk, vk) = CryptoService::generate_keypair();
        let pred = make_prediction();

        let alert = AlertBuilder::build(
            &pred, AlertLevel::High, StationId(1),
            vec![StationId(2)], &sk, 1_000_000,
        );

        assert!(!alert.signature.is_empty());
        assert!(AlertBuilder::verify(&alert, &vk));
    }

    #[test]
    fn tampered_alert_fails_verification() {
        let (sk, vk) = CryptoService::generate_keypair();
        let pred = make_prediction();

        let mut alert = AlertBuilder::build(
            &pred, AlertLevel::High, StationId(1),
            Vec::new(), &sk, 1_000_000,
        );

        // Tamper with probability
        alert.probability = 0.99;
        assert!(!AlertBuilder::verify(&alert, &vk));
    }

    #[test]
    fn alert_fields_populated() {
        let (sk, _) = CryptoService::generate_keypair();
        let pred = make_prediction();

        let alert = AlertBuilder::build(
            &pred, AlertLevel::Critical, StationId(42),
            vec![StationId(10), StationId(20)], &sk, 5_000_000,
        );

        assert_eq!(alert.station_id, StationId(42));
        assert_eq!(alert.level, AlertLevel::Critical);
        assert_eq!(alert.timestamp, 5_000_000);
        assert_eq!(alert.probability, 0.8);
        assert_eq!(alert.consensus_stations.len(), 2);
        assert_eq!(alert.wave_evidence.len(), 2);
    }
}
