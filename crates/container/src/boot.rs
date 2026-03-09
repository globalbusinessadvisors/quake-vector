//! Boot sequencer: initializes all subsystems in order.

use std::time::Instant;

use tracing::{info, warn};

use quake_vector_adaptation::SonaEngine;
use quake_vector_persistence::{CheckpointManager, CryptoService};
use quake_vector_store::{HnswConfig, HnswGraph, VectorStore};

use crate::config::QuakeVectorConfig;

/// Type of boot performed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BootType {
    Cold,
    Warm,
}

/// Report of the boot process.
#[derive(Debug, Clone)]
pub struct BootReport {
    pub boot_type: BootType,
    pub total_duration_ms: u32,
    pub sensors_discovered: usize,
    pub graph_nodes_loaded: u64,
    pub recovery_replayed: u64,
    pub warnings: Vec<String>,
}

/// Outcome of the boot sequencer, carrying initialized subsystems.
pub(crate) struct BootOutcome {
    pub report: BootReport,
    pub hnsw: HnswGraph,
    pub sona: SonaEngine,
    pub persistence: CheckpointManager,
    pub signing_key: ed25519_dalek::SigningKey,
    pub _verifying_key: ed25519_dalek::VerifyingKey,
}

/// Orchestrates the full boot sequence.
pub struct BootSequencer;

impl BootSequencer {
    /// Execute the boot sequence, returning initialized subsystems and a report.
    pub(crate) fn boot(config: &QuakeVectorConfig) -> Result<BootOutcome, std::io::Error> {
        let start = Instant::now();
        let mut warnings = Vec::new();

        // b. Initialize CryptoService
        let key_path = config.data_dir.join("keys").join("ed25519.key");
        let (signing_key, verifying_key) = CryptoService::load_or_create_keypair(&key_path)?;

        // c. Initialize CheckpointManager
        let mut persistence = CheckpointManager::new(&config.data_dir)?;

        // d. Try recovery from checkpoint + WAL
        let (boot_type, hnsw, sona, graph_nodes_loaded, recovery_replayed) =
            Self::init_from_recovery(config, &mut persistence, &verifying_key, &mut warnings);

        let total_duration_ms = start.elapsed().as_millis() as u32;

        let report = BootReport {
            boot_type,
            total_duration_ms,
            sensors_discovered: config.sensor_count,
            graph_nodes_loaded,
            recovery_replayed,
            warnings,
        };

        info!(
            boot_type = ?report.boot_type,
            duration_ms = report.total_duration_ms,
            sensors = report.sensors_discovered,
            nodes = report.graph_nodes_loaded,
            wal_replayed = report.recovery_replayed,
            "boot complete"
        );

        Ok(BootOutcome {
            report,
            hnsw,
            sona,
            persistence,
            signing_key,
            _verifying_key: verifying_key,
        })
    }

    fn init_from_recovery(
        config: &QuakeVectorConfig,
        persistence: &mut CheckpointManager,
        verifying_key: &ed25519_dalek::VerifyingKey,
        warnings: &mut Vec<String>,
    ) -> (BootType, HnswGraph, SonaEngine, u64, u64) {
        // Attempt recovery
        match persistence.recover(verifying_key) {
            Ok(Some(result)) => {
                let node_count = result.graph.node_count();
                let wal_replayed = result.wal_entries_replayed;

                let sona = match result.sona_state.into_engine() {
                    Ok(engine) => engine,
                    Err(e) => {
                        warn!(error = %e, "failed to restore SonaEngine, using fresh");
                        warnings.push(format!("sona restore failed: {e}"));
                        SonaEngine::new()
                    }
                };

                info!(
                    node_count,
                    wal_replayed,
                    "warm boot: recovered from checkpoint"
                );

                (BootType::Warm, result.graph, sona, node_count, wal_replayed)
            }
            Ok(None) => {
                // No checkpoint found — cold boot
                let hnsw = Self::create_empty_graph(config);
                (BootType::Cold, hnsw, SonaEngine::new(), 0, 0)
            }
            Err(e) => {
                warn!(error = %e, "recovery failed, falling back to cold boot");
                warnings.push(format!("recovery failed: {e}"));
                let hnsw = Self::create_empty_graph(config);
                (BootType::Cold, hnsw, SonaEngine::new(), 0, 0)
            }
        }
    }

    fn create_empty_graph(config: &QuakeVectorConfig) -> HnswGraph {
        let hnsw_config = HnswConfig {
            dimensions: 256,
            m: config.hnsw_m,
            m_max0: config.hnsw_m * 2,
            ef_construction: config.hnsw_ef_construction,
            ef_search: config.hnsw_ef_search,
            ml: 1.0 / (config.hnsw_m as f64).ln(),
        };
        HnswGraph::new(hnsw_config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::QuakeVectorConfig;

    fn test_config() -> QuakeVectorConfig {
        let dir = std::env::temp_dir().join(format!(
            "qv_boot_test_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        QuakeVectorConfig {
            data_dir: dir,
            ..QuakeVectorConfig::default()
        }
    }

    #[test]
    fn cold_boot_succeeds() {
        let config = test_config();
        let outcome = BootSequencer::boot(&config).unwrap();

        assert_eq!(outcome.report.boot_type, BootType::Cold);
        assert_eq!(outcome.report.sensors_discovered, 1);
        assert_eq!(outcome.report.graph_nodes_loaded, 0);
        assert_eq!(outcome.report.recovery_replayed, 0);
        assert!(outcome.report.warnings.is_empty());

        std::fs::remove_dir_all(&config.data_dir).ok();
    }
}
