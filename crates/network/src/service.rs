//! NetworkService: top-level coordinator for mesh and upstream communication.

use ed25519_dalek::SigningKey;

use quake_vector_types::{Alert, StationId};

use crate::mesh::MeshManager;
use crate::upstream::UpstreamRelay;

/// Top-level network coordinator. Delegates to MeshManager and UpstreamRelay.
///
/// Actual UDP/TCP transport will be integrated in the Container domain
/// when threads are wired together.
pub struct NetworkService {
    pub mesh: MeshManager,
    pub upstream: UpstreamRelay,
}

impl NetworkService {
    pub fn new(
        station_id: StationId,
        signing_key: SigningKey,
        upstream_endpoint: Option<String>,
    ) -> Self {
        Self {
            mesh: MeshManager::new(station_id, signing_key),
            upstream: UpstreamRelay::new(upstream_endpoint),
        }
    }

    /// Check if mesh networking is available.
    pub fn mesh_available(&self) -> bool {
        self.mesh.mesh_available()
    }

    /// Relay an alert upstream.
    pub fn relay_alert(&mut self, alert: &Alert) {
        self.upstream.queue_alert(alert);
    }
}
