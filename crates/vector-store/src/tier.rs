//! Tier management for hot/warm/cold data lifecycle.

use serde::{Deserialize, Serialize};
use quake_vector_types::{NodeId, Tier};
use std::collections::HashMap;

/// Manages per-node storage tier assignments.
#[derive(Debug, Serialize, Deserialize)]
pub struct TierManager {
    tiers: HashMap<NodeId, Tier>,
    access_times: HashMap<NodeId, u64>,
}

impl TierManager {
    pub fn new() -> Self {
        Self {
            tiers: HashMap::new(),
            access_times: HashMap::new(),
        }
    }

    /// Register a node as Hot.
    pub fn register(&mut self, id: NodeId, timestamp: u64) {
        self.tiers.insert(id, Tier::Hot);
        self.access_times.insert(id, timestamp);
    }

    /// Record an access, keeping the node Hot.
    pub fn promote(&mut self, id: NodeId, timestamp: u64) {
        self.tiers.insert(id, Tier::Hot);
        self.access_times.insert(id, timestamp);
    }

    /// Demote nodes not accessed within `threshold_us` from Hot to Warm.
    pub fn demote_stale(&mut self, current_time_us: u64, threshold_us: u64) {
        for (id, tier) in self.tiers.iter_mut() {
            if *tier == Tier::Hot {
                if let Some(&last) = self.access_times.get(id) {
                    if current_time_us.saturating_sub(last) > threshold_us {
                        *tier = Tier::Warm;
                    }
                }
            }
        }
    }

    /// Get the current tier for a node.
    pub fn get_tier(&self, id: NodeId) -> Option<Tier> {
        self.tiers.get(&id).copied()
    }
}

impl Default for TierManager {
    fn default() -> Self {
        Self::new()
    }
}
