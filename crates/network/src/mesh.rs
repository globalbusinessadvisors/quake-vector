//! Mesh manager: peer management, message signing, and event dispatch.

use std::collections::HashMap;

use ed25519_dalek::{SigningKey, VerifyingKey};
use uuid::Uuid;

use quake_vector_types::{ClaimId, MessageId, SeismicPrediction, StationId};
use quake_vector_persistence::CryptoService;

use crate::message::{
    MeshMessage, MeshMessageType, PredictionClaimPayload, PredictionConfirmPayload,
};
use crate::peer::{PeerHealth, PeerState};
use crate::queue::BoundedQueue;

/// Events produced when receiving a mesh message.
#[derive(Debug)]
pub enum MeshEvent {
    ClaimReceived {
        claim_id: ClaimId,
        from: StationId,
        prediction: SeismicPrediction,
    },
    ConfirmReceived {
        claim_id: ClaimId,
        from: StationId,
    },
    HealthReceived {
        from: StationId,
    },
    InvalidSignature {
        from: StationId,
    },
    UnknownPeer {
        from: StationId,
    },
}

/// Manages the mesh network of peer stations.
pub struct MeshManager {
    pub station_id: StationId,
    peers: HashMap<StationId, PeerState>,
    outbound_queue: BoundedQueue<MeshMessage>,
    signing_key: SigningKey,
}

impl MeshManager {
    pub fn new(station_id: StationId, signing_key: SigningKey) -> Self {
        Self {
            station_id,
            peers: HashMap::new(),
            outbound_queue: BoundedQueue::new(1000),
            signing_key,
        }
    }

    /// Register a peer.
    pub fn add_peer(&mut self, station_id: StationId, public_key: Vec<u8>) {
        self.peers
            .insert(station_id, PeerState::new(station_id, public_key, 0));
    }

    /// Broadcast a prediction claim to all peers.
    pub fn broadcast_claim(
        &mut self,
        claim_id: ClaimId,
        prediction: &SeismicPrediction,
        now_us: u64,
    ) -> MeshMessage {
        let payload = PredictionClaimPayload {
            claim_id,
            prediction: prediction.clone(),
        };
        let payload_bytes = bincode::serialize(&payload).unwrap_or_default();

        let msg = self.build_signed_message(
            MeshMessageType::PredictionClaim,
            payload_bytes,
            now_us,
        );
        self.outbound_queue.push(msg.clone());
        msg
    }

    /// Send a prediction confirmation.
    pub fn send_confirm(
        &mut self,
        claim_id: ClaimId,
        _target: StationId,
        now_us: u64,
    ) -> MeshMessage {
        let payload = PredictionConfirmPayload {
            claim_id,
            confirming_station: self.station_id,
        };
        let payload_bytes = bincode::serialize(&payload).unwrap_or_default();

        let msg = self.build_signed_message(
            MeshMessageType::PredictionConfirm,
            payload_bytes,
            now_us,
        );
        self.outbound_queue.push(msg.clone());
        msg
    }

    /// Receive and process an incoming mesh message.
    pub fn receive_message(&mut self, msg: &MeshMessage) -> MeshEvent {
        // Look up peer
        let Some(peer) = self.peers.get_mut(&msg.from) else {
            return MeshEvent::UnknownPeer { from: msg.from };
        };

        // Verify signature
        if peer.public_key.len() != 32 {
            return MeshEvent::InvalidSignature { from: msg.from };
        }
        let key_bytes: [u8; 32] = peer.public_key.as_slice().try_into().unwrap();
        let Ok(verifying_key) = VerifyingKey::from_bytes(&key_bytes) else {
            return MeshEvent::InvalidSignature { from: msg.from };
        };

        let sign_data = msg.signable_bytes();
        if !CryptoService::verify(&sign_data, &msg.signature, &verifying_key) {
            return MeshEvent::InvalidSignature { from: msg.from };
        }

        // Update peer last seen
        peer.mark_seen(msg.timestamp_us);

        // Parse payload
        match msg.message_type {
            MeshMessageType::PredictionClaim => {
                match bincode::deserialize::<PredictionClaimPayload>(&msg.payload) {
                    Ok(p) => MeshEvent::ClaimReceived {
                        claim_id: p.claim_id,
                        from: msg.from,
                        prediction: p.prediction,
                    },
                    Err(_) => MeshEvent::InvalidSignature { from: msg.from },
                }
            }
            MeshMessageType::PredictionConfirm => {
                match bincode::deserialize::<PredictionConfirmPayload>(&msg.payload) {
                    Ok(p) => MeshEvent::ConfirmReceived {
                        claim_id: p.claim_id,
                        from: msg.from,
                    },
                    Err(_) => MeshEvent::InvalidSignature { from: msg.from },
                }
            }
            MeshMessageType::HealthStatus => MeshEvent::HealthReceived { from: msg.from },
        }
    }

    /// Update all peers' health based on current time.
    pub fn update_peer_health(&mut self, now_us: u64) {
        for peer in self.peers.values_mut() {
            peer.update_health(now_us);
        }
    }

    /// Drain the outbound message queue.
    pub fn drain_outbound(&mut self) -> Vec<MeshMessage> {
        self.outbound_queue.drain()
    }

    /// True if at least one peer is Healthy.
    pub fn mesh_available(&self) -> bool {
        self.peers.values().any(|p| p.health == PeerHealth::Healthy)
    }

    /// Get peer state.
    pub fn get_peer(&self, id: &StationId) -> Option<&PeerState> {
        self.peers.get(id)
    }

    /// Number of known peers.
    pub fn peer_count(&self) -> usize {
        self.peers.len()
    }

    /// Get all peer states.
    pub fn peer_states(&self) -> Vec<&PeerState> {
        self.peers.values().collect()
    }

    // -----------------------------------------------------------------------

    fn build_signed_message(
        &self,
        message_type: MeshMessageType,
        payload: Vec<u8>,
        now_us: u64,
    ) -> MeshMessage {
        let mut msg = MeshMessage {
            message_id: MessageId(Uuid::new_v4()),
            from: self.station_id,
            message_type,
            payload,
            timestamp_us: now_us,
            signature: Vec::new(),
        };
        let sign_data = msg.signable_bytes();
        msg.signature = CryptoService::sign(&sign_data, &self.signing_key);
        msg
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quake_vector_types::NodeId;

    fn make_prediction() -> SeismicPrediction {
        SeismicPrediction {
            event_probability: 0.8,
            estimated_magnitude: 4.0,
            estimated_time_to_peak_s: 5.0,
            confidence: 0.85,
            contributing_wave_ids: vec![NodeId(1)],
            model_version: 1,
            inference_latency_us: 200,
        }
    }

    fn setup_two_peers() -> (MeshManager, MeshManager) {
        let (sk1, vk1) = CryptoService::generate_keypair();
        let (sk2, vk2) = CryptoService::generate_keypair();

        let mut mgr1 = MeshManager::new(StationId(1), sk1);
        let mut mgr2 = MeshManager::new(StationId(2), sk2);

        // Register each other as peers
        mgr1.add_peer(StationId(2), vk2.to_bytes().to_vec());
        mgr2.add_peer(StationId(1), vk1.to_bytes().to_vec());

        (mgr1, mgr2)
    }

    #[test]
    fn broadcast_claim_creates_valid_signed_message() {
        let (mut mgr1, _mgr2) = setup_two_peers();
        let pred = make_prediction();
        let claim_id = ClaimId(Uuid::new_v4());

        let msg = mgr1.broadcast_claim(claim_id, &pred, 1_000_000);

        assert_eq!(msg.from, StationId(1));
        assert_eq!(msg.message_type, MeshMessageType::PredictionClaim);
        assert!(!msg.signature.is_empty());
        assert!(!msg.payload.is_empty());

        // Outbound queue should have the message
        let outbound = mgr1.drain_outbound();
        assert_eq!(outbound.len(), 1);
    }

    #[test]
    fn receive_verifies_signature_and_parses() {
        let (mut mgr1, mut mgr2) = setup_two_peers();
        let pred = make_prediction();
        let claim_id = ClaimId(Uuid::new_v4());

        let msg = mgr1.broadcast_claim(claim_id, &pred, 1_000_000);

        // mgr2 receives the message from mgr1
        let event = mgr2.receive_message(&msg);
        match event {
            MeshEvent::ClaimReceived { claim_id: cid, from, prediction } => {
                assert_eq!(cid, claim_id);
                assert_eq!(from, StationId(1));
                assert!((prediction.event_probability - 0.8).abs() < 1e-6);
            }
            other => panic!("expected ClaimReceived, got {other:?}"),
        }
    }

    #[test]
    fn receive_rejects_unknown_peer() {
        let (sk, _) = CryptoService::generate_keypair();
        let mut mgr = MeshManager::new(StationId(1), sk);
        // No peers registered

        let msg = MeshMessage {
            message_id: MessageId(Uuid::new_v4()),
            from: StationId(99),
            message_type: MeshMessageType::HealthStatus,
            payload: Vec::new(),
            timestamp_us: 0,
            signature: vec![0u8; 64],
        };

        let event = mgr.receive_message(&msg);
        assert!(matches!(event, MeshEvent::UnknownPeer { from: StationId(99) }));
    }

    #[test]
    fn receive_rejects_invalid_signature() {
        let (mut mgr1, mut mgr2) = setup_two_peers();
        let pred = make_prediction();
        let claim_id = ClaimId(Uuid::new_v4());

        let mut msg = mgr1.broadcast_claim(claim_id, &pred, 1_000_000);
        // Tamper with payload
        if !msg.payload.is_empty() {
            msg.payload[0] ^= 0xFF;
        }

        let event = mgr2.receive_message(&msg);
        assert!(matches!(event, MeshEvent::InvalidSignature { .. }));
    }

    #[test]
    fn peer_health_transitions() {
        let (sk, _) = CryptoService::generate_keypair();
        let mut mgr = MeshManager::new(StationId(1), sk);
        let (_, vk2) = CryptoService::generate_keypair();
        mgr.add_peer(StationId(2), vk2.to_bytes().to_vec());

        // Initially at time 0 — mark seen
        mgr.peers.get_mut(&StationId(2)).unwrap().mark_seen(0);

        // At 30s: Healthy
        mgr.update_peer_health(30_000_000);
        assert_eq!(mgr.get_peer(&StationId(2)).unwrap().health, PeerHealth::Healthy);

        // At 90s: Stale (60s since last seen)
        mgr.update_peer_health(90_000_000);
        assert_eq!(mgr.get_peer(&StationId(2)).unwrap().health, PeerHealth::Stale);

        // At 350s: Lost (300s since last seen)
        mgr.update_peer_health(350_000_000);
        assert_eq!(mgr.get_peer(&StationId(2)).unwrap().health, PeerHealth::Lost);
    }

    #[test]
    fn mesh_available_reflects_peer_health() {
        let (sk, _) = CryptoService::generate_keypair();
        let mut mgr = MeshManager::new(StationId(1), sk);

        // No peers
        assert!(!mgr.mesh_available());

        // Add peer
        let (_, vk2) = CryptoService::generate_keypair();
        mgr.add_peer(StationId(2), vk2.to_bytes().to_vec());
        mgr.peers.get_mut(&StationId(2)).unwrap().mark_seen(1_000_000);

        // Peer is Healthy
        mgr.update_peer_health(10_000_000);
        assert!(mgr.mesh_available());

        // Peer becomes Lost
        mgr.update_peer_health(400_000_000);
        assert!(!mgr.mesh_available());
    }

    #[test]
    fn confirm_round_trip() {
        let (mut mgr1, mut mgr2) = setup_two_peers();
        let claim_id = ClaimId(Uuid::new_v4());

        let msg = mgr2.send_confirm(claim_id, StationId(1), 2_000_000);
        let event = mgr1.receive_message(&msg);

        match event {
            MeshEvent::ConfirmReceived { claim_id: cid, from } => {
                assert_eq!(cid, claim_id);
                assert_eq!(from, StationId(2));
            }
            other => panic!("expected ConfirmReceived, got {other:?}"),
        }
    }
}
