//! QuakeVector configuration.

use std::path::{Path, PathBuf};

use quake_vector_alert::AlertThresholds;
use quake_vector_types::StationId;

/// Top-level configuration for a QuakeVector node.
#[derive(Debug, Clone)]
pub struct QuakeVectorConfig {
    pub station_id: StationId,
    pub data_dir: PathBuf,
    pub sensor_count: usize,
    pub alert_thresholds: AlertThresholds,
    pub hnsw_m: usize,
    pub hnsw_ef_construction: usize,
    pub hnsw_ef_search: usize,
    pub checkpoint_interval_secs: u64,
    /// (station_id, public_key_bytes)
    pub mesh_peers: Vec<(StationId, Vec<u8>)>,
    pub upstream_endpoint: Option<String>,
    pub enable_watchdog: bool,
}

impl QuakeVectorConfig {
    /// Load configuration from a JSON file.
    pub fn from_file(path: &Path) -> std::io::Result<Self> {
        // Simplified: use defaults for now. Full JSON parsing is a future enhancement.
        let _ = path;
        Ok(Self::default())
    }
}

impl Default for QuakeVectorConfig {
    fn default() -> Self {
        Self {
            station_id: StationId(1),
            data_dir: PathBuf::from("/tmp/quake-vector-data"),
            sensor_count: 1,
            alert_thresholds: AlertThresholds::default(),
            hnsw_m: 16,
            hnsw_ef_construction: 200,
            hnsw_ef_search: 64,
            checkpoint_interval_secs: 900,
            mesh_peers: Vec::new(),
            upstream_endpoint: None,
            enable_watchdog: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_valid() {
        let config = QuakeVectorConfig::default();
        assert_eq!(config.station_id, StationId(1));
        assert_eq!(config.sensor_count, 1);
        assert_eq!(config.hnsw_m, 16);
        assert_eq!(config.hnsw_ef_construction, 200);
        assert_eq!(config.hnsw_ef_search, 64);
        assert_eq!(config.checkpoint_interval_secs, 900);
        assert!(!config.enable_watchdog);
        assert!(config.mesh_peers.is_empty());
        assert!(config.upstream_endpoint.is_none());
    }
}
