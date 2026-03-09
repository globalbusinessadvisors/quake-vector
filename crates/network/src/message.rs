//! Mesh message types and payloads.

use serde::{Deserialize, Serialize};

use quake_vector_types::{ClaimId, MessageId, SeismicPrediction, StationId};

/// Type of mesh message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MeshMessageType {
    PredictionClaim,
    PredictionConfirm,
    HealthStatus,
}

/// A signed message exchanged between mesh peers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshMessage {
    pub message_id: MessageId,
    pub from: StationId,
    pub message_type: MeshMessageType,
    pub payload: Vec<u8>,
    pub timestamp_us: u64,
    pub signature: Vec<u8>,
}

/// Payload for a prediction claim broadcast.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionClaimPayload {
    pub claim_id: ClaimId,
    pub prediction: SeismicPrediction,
}

/// Payload for a prediction confirmation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionConfirmPayload {
    pub claim_id: ClaimId,
    pub confirming_station: StationId,
}

impl MeshMessage {
    /// Get the bytes used for signing: everything except the signature field.
    pub fn signable_bytes(&self) -> Vec<u8> {
        let signable = (
            &self.message_id,
            &self.from,
            &self.message_type,
            &self.payload,
            self.timestamp_us,
        );
        bincode::serialize(&signable).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quake_vector_types::{NodeId};
    use uuid::Uuid;

    #[test]
    fn round_trip_serialize_claim_message() {
        let claim_payload = PredictionClaimPayload {
            claim_id: ClaimId(Uuid::new_v4()),
            prediction: SeismicPrediction {
                event_probability: 0.8,
                estimated_magnitude: 4.0,
                estimated_time_to_peak_s: 5.0,
                confidence: 0.85,
                contributing_wave_ids: vec![NodeId(1), NodeId(2)],
                model_version: 1,
                inference_latency_us: 200,
            },
        };

        let payload_bytes = bincode::serialize(&claim_payload).unwrap();

        let msg = MeshMessage {
            message_id: MessageId(Uuid::new_v4()),
            from: StationId(42),
            message_type: MeshMessageType::PredictionClaim,
            payload: payload_bytes,
            timestamp_us: 1_000_000,
            signature: vec![0u8; 64],
        };

        let serialized = serde_json::to_string(&msg).unwrap();
        let deserialized: MeshMessage = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.from, StationId(42));
        assert_eq!(deserialized.message_type, MeshMessageType::PredictionClaim);
        assert_eq!(deserialized.timestamp_us, 1_000_000);

        let restored: PredictionClaimPayload =
            bincode::deserialize(&deserialized.payload).unwrap();
        assert_eq!(restored.claim_id, claim_payload.claim_id);
        assert!((restored.prediction.event_probability - 0.8).abs() < 1e-6);
    }

    #[test]
    fn round_trip_confirm_payload() {
        let payload = PredictionConfirmPayload {
            claim_id: ClaimId(Uuid::new_v4()),
            confirming_station: StationId(7),
        };
        let bytes = bincode::serialize(&payload).unwrap();
        let restored: PredictionConfirmPayload = bincode::deserialize(&bytes).unwrap();
        assert_eq!(restored.confirming_station, StationId(7));
    }
}
