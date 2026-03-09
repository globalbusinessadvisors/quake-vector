//! QuakeVectorRuntime: the main runtime struct owning all domain objects.

use std::time::Instant;

use tracing::{info, warn};

use quake_vector_types::SensorId;
use quake_vector_alert::{AlertDecision, AlertEngine};
use quake_vector_adaptation::SonaEngine;
use quake_vector_embedding::EmbeddingPipeline;
use quake_vector_ingestion::{
    MockSensor, SensorHealthMonitor, SensorStream, WavePattern,
};
use quake_vector_learning::CausalLearner;
use quake_vector_network::{DashboardState, NetworkService, PeerSnapshot};
use quake_vector_persistence::{CheckpointManager, WalOpType, WearMonitor};
use quake_vector_store::{HnswGraph, VectorStore};

use crate::boot::{BootSequencer, BootReport};
use crate::config::QuakeVectorConfig;
use crate::resource::ResourceMonitor;

/// Result of a single processing tick.
#[derive(Debug)]
pub struct TickResult {
    pub windows_processed: usize,
    pub predictions_made: usize,
    pub alerts: Vec<AlertDecision>,
}

/// The main runtime: owns all domain objects and coordinates the processing loop.
pub struct QuakeVectorRuntime {
    pub config: QuakeVectorConfig,
    pub ef_search: usize,
    hnsw: HnswGraph,
    embedding_pipeline: EmbeddingPipeline,
    causal_learner: CausalLearner,
    sona: SonaEngine,
    alert_engine: AlertEngine,
    persistence: CheckpointManager,
    network: NetworkService,
    sensors: Vec<SensorStream>,
    resource_monitor: ResourceMonitor,
    wear_monitor: WearMonitor,
    signing_key: ed25519_dalek::SigningKey,
    next_node_id: u64,
    boot_report: BootReport,
}

impl QuakeVectorRuntime {
    /// Boot the runtime from configuration.
    pub fn boot(config: QuakeVectorConfig) -> Result<Self, std::io::Error> {
        let outcome = BootSequencer::boot(&config)?;

        // e. Create EmbeddingPipeline
        let embedding_pipeline = EmbeddingPipeline::new();

        // f. Create CausalLearner (owns CausalModel)
        let causal_learner = CausalLearner::new();

        // h. Create AlertEngine
        let alert_log_dir = config.data_dir.join("alerts");
        let alert_engine = AlertEngine::new(
            config.station_id,
            outcome.signing_key.clone(),
            &alert_log_dir,
        )?;

        // i. Discover sensors (MockSensors)
        let sensors = Self::create_sensors(&config);

        // j + k. Create NetworkService (MeshManager + UpstreamRelay)
        let mut network = NetworkService::new(
            config.station_id,
            outcome.signing_key.clone(),
            config.upstream_endpoint.clone(),
        );
        for (peer_id, peer_key) in &config.mesh_peers {
            network.mesh.add_peer(*peer_id, peer_key.clone());
        }

        let ef_search = config.hnsw_ef_search;
        let next_node_id = outcome.hnsw.node_count();

        Ok(Self {
            config,
            ef_search,
            hnsw: outcome.hnsw,
            embedding_pipeline,
            causal_learner,
            sona: outcome.sona,
            alert_engine,
            persistence: outcome.persistence,
            network,
            sensors,
            resource_monitor: ResourceMonitor::new(),
            wear_monitor: WearMonitor::new(10 * 1024 * 1024 * 1024), // 10 GB daily budget
            signing_key: outcome.signing_key,
            next_node_id,
            boot_report: outcome.report,
        })
    }

    /// One cycle of the main processing loop.
    pub fn tick(&mut self, now_us: u64) -> TickResult {
        let mut windows_processed = 0;
        let mut predictions_made = 0;
        let mut alerts = Vec::new();

        // a. Poll all sensors
        let mut windows = Vec::new();
        for sensor in &mut self.sensors {
            if let Some(window) = sensor.poll() {
                windows.push(window);
            }
        }

        // b. Process each window
        for window in &windows {
            windows_processed += 1;

            // Compute embedding
            let embedding = self.embedding_pipeline.compute(window);

            // WAL entry before insert
            let wal_payload = self.next_node_id.to_le_bytes().to_vec();
            if let Err(e) = self.persistence.wal().append(WalOpType::InsertNode, &wal_payload) {
                warn!(error = %e, "failed to write WAL entry");
            }

            // Insert into HNSW graph
            let meta = quake_vector_types::TemporalMeta {
                timestamp_us: window.timestamp_us,
                wave_type: window.wave_type,
                station_id: self.config.station_id,
                amplitude_rms: window.rms_amplitude,
                dominant_freq_hz: window.dominant_freq_hz,
            };
            let node_id = self.hnsw.insert(&embedding, &meta);
            self.next_node_id += 1;

            // Causal learning prediction
            let prediction = self.causal_learner.process(
                node_id,
                &embedding,
                &meta,
                &self.hnsw,
            );
            predictions_made += 1;

            // SONA micro update (simplified error signal)
            let error = vec![
                prediction.event_probability - 0.5,
                prediction.estimated_magnitude - 3.0,
                prediction.estimated_time_to_peak_s - 5.0,
                prediction.confidence - 0.5,
            ];
            let features: Vec<Vec<f32>> = vec![embedding.vector.to_vec()];
            self.sona.apply_micro_update(&features, &error);
            self.sona.accumulate_base_update(features, error);

            // Alert processing
            let mesh_available = self.network.mesh_available();
            let decision = self.alert_engine.process_prediction(
                &prediction,
                now_us,
                mesh_available,
            );
            alerts.push(decision);
        }

        // c. Flush base update if ready
        self.sona.flush_base_update();

        // d. Check consensus expirations
        self.alert_engine.tick(now_us);

        TickResult {
            windows_processed,
            predictions_made,
            alerts,
        }
    }

    /// Write a full checkpoint: serialize graph + sona, sign, rotate slot.
    /// Records bytes written to the wear monitor.
    pub fn checkpoint(&mut self) -> Result<(), std::io::Error> {
        let start = Instant::now();

        let slot_info = self.persistence.create_checkpoint(
            &self.hnsw,
            &self.sona,
            &self.signing_key,
        )?;

        // Track checkpoint bytes in wear monitor
        self.wear_monitor.record_write(slot_info.size_bytes);

        let duration_ms = start.elapsed().as_millis();
        info!(
            slot = slot_info.path,
            size_bytes = slot_info.size_bytes,
            sequence = slot_info.sequence_number,
            duration_ms,
            "checkpoint complete"
        );

        Ok(())
    }

    /// Get the boot report.
    pub fn boot_report(&self) -> &BootReport {
        &self.boot_report
    }

    /// Get the HNSW node count.
    pub fn node_count(&self) -> u64 {
        self.hnsw.node_count()
    }

    /// Access the resource monitor.
    pub fn resource_monitor(&self) -> &ResourceMonitor {
        &self.resource_monitor
    }

    /// Access the wear monitor.
    pub fn wear_monitor(&self) -> &WearMonitor {
        &self.wear_monitor
    }

    /// Update dashboard state from current runtime state.
    pub fn update_dashboard(&self, state: &mut DashboardState) {
        state.graph_node_count = self.hnsw.node_count();

        let resource = self.resource_monitor.check();
        state.ram_utilization = resource.ram_utilization;
        state.degradation_level = format!("{:?}", resource.degradation_level).to_lowercase();

        state.sensor_statuses = self
            .sensors
            .iter()
            .map(|s| {
                let sid = s.sensor_id();
                let status = s.health().report_health(sid);
                (sid, status)
            })
            .collect();

        state.peer_statuses = self
            .network
            .mesh
            .peer_states()
            .iter()
            .map(|p| PeerSnapshot::from_peer(p))
            .collect();
    }

    fn create_sensors(config: &QuakeVectorConfig) -> Vec<SensorStream> {
        (0..config.sensor_count)
            .map(|i| {
                let sensor = MockSensor::new(
                    SensorId(i as u64),
                    100, // 100 Hz sample rate
                    WavePattern::FullSequence,
                    1000.0,
                    0.05,
                );
                let health = SensorHealthMonitor::new();
                SensorStream::new(Box::new(sensor), health)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> QuakeVectorConfig {
        let dir = std::env::temp_dir().join(format!(
            "qv_runtime_test_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        QuakeVectorConfig {
            data_dir: dir,
            sensor_count: 2,
            ..QuakeVectorConfig::default()
        }
    }

    #[test]
    fn boot_and_100_ticks() {
        let config = test_config();
        let data_dir = config.data_dir.clone();
        let mut runtime = QuakeVectorRuntime::boot(config).unwrap();

        let mut total_windows = 0;
        let mut total_predictions = 0;

        for i in 0..100 {
            let now_us = i as u64 * 10_000; // 10ms per tick
            let result = runtime.tick(now_us);
            total_windows += result.windows_processed;
            total_predictions += result.predictions_made;
        }

        // With 2 sensors at 100Hz, 64 samples/batch, 256 sample windows:
        // should produce some windows over 100 ticks
        assert!(
            total_windows > 0,
            "expected at least some windows, got {total_windows}"
        );
        assert_eq!(total_predictions, total_windows);
        assert!(runtime.node_count() > 0);

        std::fs::remove_dir_all(&data_dir).ok();
    }

    #[test]
    fn tick_processes_windows_from_sensors() {
        let config = test_config();
        let data_dir = config.data_dir.clone();
        let mut runtime = QuakeVectorRuntime::boot(config).unwrap();

        // Run enough ticks to fill sensor buffers and produce at least one window
        let mut saw_window = false;
        for i in 0..20 {
            let result = runtime.tick(i * 10_000);
            if result.windows_processed > 0 {
                saw_window = true;
                assert_eq!(result.predictions_made, result.windows_processed);
                assert_eq!(result.alerts.len(), result.windows_processed);
            }
        }

        assert!(saw_window, "should have processed at least one window in 20 ticks");

        std::fs::remove_dir_all(&data_dir).ok();
    }
}
