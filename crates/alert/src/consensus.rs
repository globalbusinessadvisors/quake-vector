//! Multi-station consensus for high-severity alerts.

use std::collections::HashMap;

use quake_vector_types::{AlertLevel, ClaimId, SeismicPrediction, StationId};
use uuid::Uuid;

/// Status of a consensus confirmation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsensusStatus {
    Pending,
    Reached,
    UnknownClaim,
}

/// A pending consensus claim awaiting confirmations.
#[derive(Debug, Clone)]
pub struct PendingClaim {
    pub claim_id: ClaimId,
    pub prediction: SeismicPrediction,
    pub created_at_us: u64,
    pub confirmations: Vec<StationId>,
    pub original_level: AlertLevel,
}

/// Manages multi-station consensus for high-severity alerts.
pub struct ConsensusManager {
    pending_claims: HashMap<ClaimId, PendingClaim>,
    /// Minimum additional confirmations needed (default 1 = 2 total stations).
    pub min_confirmations: u8,
    /// Window in which confirmations must arrive (default 10s).
    pub consensus_window_us: u64,
}

impl ConsensusManager {
    pub fn new() -> Self {
        Self {
            pending_claims: HashMap::new(),
            min_confirmations: 1,
            consensus_window_us: 10_000_000,
        }
    }

    /// Create a new consensus claim, returning its ID.
    pub fn create_claim(
        &mut self,
        prediction: &SeismicPrediction,
        level: AlertLevel,
        now_us: u64,
    ) -> ClaimId {
        let claim_id = ClaimId(Uuid::new_v4());
        self.pending_claims.insert(
            claim_id,
            PendingClaim {
                claim_id,
                prediction: prediction.clone(),
                created_at_us: now_us,
                confirmations: Vec::new(),
                original_level: level,
            },
        );
        claim_id
    }

    /// Add a confirmation from another station.
    pub fn add_confirmation(
        &mut self,
        claim_id: &ClaimId,
        confirming_station: StationId,
    ) -> ConsensusStatus {
        let Some(claim) = self.pending_claims.get_mut(claim_id) else {
            return ConsensusStatus::UnknownClaim;
        };

        if !claim.confirmations.contains(&confirming_station) {
            claim.confirmations.push(confirming_station);
        }

        if claim.confirmations.len() >= self.min_confirmations as usize {
            ConsensusStatus::Reached
        } else {
            ConsensusStatus::Pending
        }
    }

    /// Check if a claim has reached consensus.
    pub fn has_consensus(&self, claim_id: &ClaimId) -> bool {
        self.pending_claims
            .get(claim_id)
            .map(|c| c.confirmations.len() >= self.min_confirmations as usize)
            .unwrap_or(false)
    }

    /// Remove and return expired claims (past the consensus window).
    pub fn check_expired(&mut self, now_us: u64) -> Vec<(ClaimId, AlertLevel)> {
        let expired: Vec<ClaimId> = self
            .pending_claims
            .iter()
            .filter(|(_, claim)| {
                now_us.saturating_sub(claim.created_at_us) >= self.consensus_window_us
            })
            .map(|(id, _)| *id)
            .collect();

        let mut results = Vec::new();
        for id in expired {
            if let Some(claim) = self.pending_claims.remove(&id) {
                results.push((id, claim.original_level));
            }
        }
        results
    }

    /// Get a pending claim by ID.
    pub fn get_claim(&self, claim_id: &ClaimId) -> Option<&PendingClaim> {
        self.pending_claims.get(claim_id)
    }

    /// Remove a claim (after it's been processed).
    pub fn remove_claim(&mut self, claim_id: &ClaimId) -> Option<PendingClaim> {
        self.pending_claims.remove(claim_id)
    }

    /// Number of pending claims.
    pub fn pending_count(&self) -> usize {
        self.pending_claims.len()
    }
}

impl Default for ConsensusManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quake_vector_types::NodeId;

    fn make_prediction(prob: f32) -> SeismicPrediction {
        SeismicPrediction {
            event_probability: prob,
            estimated_magnitude: 4.0,
            estimated_time_to_peak_s: 5.0,
            confidence: 0.8,
            contributing_wave_ids: vec![NodeId(1)],
            model_version: 1,
            inference_latency_us: 100,
        }
    }

    #[test]
    fn claim_lifecycle() {
        let mut cm = ConsensusManager::new();
        let pred = make_prediction(0.8);
        let claim_id = cm.create_claim(&pred, AlertLevel::High, 1_000_000);

        assert!(!cm.has_consensus(&claim_id));
        assert_eq!(cm.pending_count(), 1);

        let status = cm.add_confirmation(&claim_id, StationId(2));
        assert_eq!(status, ConsensusStatus::Reached);
        assert!(cm.has_consensus(&claim_id));
    }

    #[test]
    fn multiple_confirmations_needed() {
        let mut cm = ConsensusManager::new();
        cm.min_confirmations = 2;
        let pred = make_prediction(0.9);
        let claim_id = cm.create_claim(&pred, AlertLevel::Critical, 0);

        assert_eq!(
            cm.add_confirmation(&claim_id, StationId(2)),
            ConsensusStatus::Pending
        );
        assert_eq!(
            cm.add_confirmation(&claim_id, StationId(3)),
            ConsensusStatus::Reached
        );
    }

    #[test]
    fn unknown_claim() {
        let mut cm = ConsensusManager::new();
        let bogus = ClaimId(Uuid::new_v4());
        assert_eq!(
            cm.add_confirmation(&bogus, StationId(1)),
            ConsensusStatus::UnknownClaim
        );
    }

    #[test]
    fn expiration() {
        let mut cm = ConsensusManager::new();
        let pred = make_prediction(0.8);
        let claim_id = cm.create_claim(&pred, AlertLevel::High, 1_000_000);

        // Not expired yet
        assert!(cm.check_expired(5_000_000).is_empty());

        // After 10s window
        let expired = cm.check_expired(12_000_000);
        assert_eq!(expired.len(), 1);
        assert_eq!(expired[0].0, claim_id);
        assert_eq!(expired[0].1, AlertLevel::High);
        assert_eq!(cm.pending_count(), 0);
    }

    #[test]
    fn duplicate_confirmation_ignored() {
        let mut cm = ConsensusManager::new();
        cm.min_confirmations = 2;
        let pred = make_prediction(0.8);
        let claim_id = cm.create_claim(&pred, AlertLevel::High, 0);

        cm.add_confirmation(&claim_id, StationId(2));
        cm.add_confirmation(&claim_id, StationId(2)); // duplicate
        assert!(!cm.has_consensus(&claim_id)); // still need another unique station
    }
}
