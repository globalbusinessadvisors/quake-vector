//! Peer state tracking for mesh nodes.

use quake_vector_types::StationId;

/// Health state of a peer node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PeerHealth {
    /// Seen within last 60 seconds.
    Healthy,
    /// Last seen 60–300 seconds ago.
    Stale,
    /// Last seen > 300 seconds ago.
    Lost,
}

/// Tracked state for a mesh peer.
#[derive(Debug, Clone)]
pub struct PeerState {
    pub station_id: StationId,
    pub public_key: Vec<u8>,
    pub last_seen_us: u64,
    pub health: PeerHealth,
}

/// Thresholds in microseconds.
const STALE_THRESHOLD_US: u64 = 60_000_000;   // 60s
const LOST_THRESHOLD_US: u64 = 300_000_000;    // 300s

impl PeerState {
    pub fn new(station_id: StationId, public_key: Vec<u8>, now_us: u64) -> Self {
        Self {
            station_id,
            public_key,
            last_seen_us: now_us,
            health: PeerHealth::Healthy,
        }
    }

    /// Update health status based on current time.
    pub fn update_health(&mut self, now_us: u64) {
        let elapsed = now_us.saturating_sub(self.last_seen_us);
        self.health = if elapsed <= STALE_THRESHOLD_US {
            PeerHealth::Healthy
        } else if elapsed <= LOST_THRESHOLD_US {
            PeerHealth::Stale
        } else {
            PeerHealth::Lost
        };
    }

    /// Record that we've heard from this peer.
    pub fn mark_seen(&mut self, now_us: u64) {
        self.last_seen_us = now_us;
        self.health = PeerHealth::Healthy;
    }
}
