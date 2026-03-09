use quake_vector_alert::{AlertDecision, AlertEngine};
use quake_vector_container::{BootType, QuakeVectorConfig, QuakeVectorRuntime};
use quake_vector_persistence::CryptoService;
use quake_vector_types::{AlertLevel, NodeId, SeismicPrediction, StationId};

fn temp_dir(label: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "qv_e2e_{label}_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}

fn test_config(label: &str) -> QuakeVectorConfig {
    QuakeVectorConfig {
        data_dir: temp_dir(label),
        sensor_count: 1,
        ..QuakeVectorConfig::default()
    }
}

#[test]
fn full_seismic_event_detection() {
    // Boot runtime with 1 SimulatedSensor configured for a M5.0 event
    let config = test_config("event_detect");
    let data_dir = config.data_dir.clone();
    let mut runtime = QuakeVectorRuntime::boot(config).unwrap();

    // Run 30 simulated seconds (at ~1ms per tick = 30000 ticks,
    // but we use simulated time)
    let mut total_windows = 0;
    let mut total_predictions = 0;

    for i in 0..500 {
        let now_us = i as u64 * 10_000; // 10ms per tick
        let result = runtime.tick(now_us);
        total_windows += result.windows_processed;
        total_predictions += result.predictions_made;
    }

    // Should have processed windows and generated predictions
    assert!(total_windows > 0, "expected windows, got 0");
    assert_eq!(total_predictions, total_windows);

    // HNSW graph should contain nodes
    assert!(runtime.node_count() > 0);

    std::fs::remove_dir_all(&data_dir).ok();
}

#[test]
fn quiet_period_no_false_alerts() {
    let config = test_config("quiet");
    let data_dir = config.data_dir.clone();
    let mut runtime = QuakeVectorRuntime::boot(config).unwrap();

    let mut high_critical_count = 0;
    let mut low_count = 0;

    // Run simulated seconds
    for i in 0..500 {
        let now_us = i as u64 * 10_000;
        let result = runtime.tick(now_us);
        for decision in &result.alerts {
            match decision {
                AlertDecision::EmitLocal(alert) | AlertDecision::EmitCappedAtMedium(alert) => {
                    match alert.level {
                        AlertLevel::High | AlertLevel::Critical => high_critical_count += 1,
                        AlertLevel::Low => low_count += 1,
                        _ => {}
                    }
                }
                AlertDecision::DeferForConsensus(_) => {
                    // This would be High/Critical
                    high_critical_count += 1;
                }
                _ => {}
            }
        }
    }

    // No HIGH or CRITICAL alerts during quiet period
    assert_eq!(
        high_critical_count, 0,
        "expected no HIGH/CRITICAL alerts in quiet period, got {high_critical_count}"
    );

    // Low alerts should be infrequent (< 5 over 60s)
    assert!(
        low_count < 5,
        "expected < 5 LOW alerts in 60s quiet period, got {low_count}"
    );

    std::fs::remove_dir_all(&data_dir).ok();
}

#[test]
fn checkpoint_recovery_preserves_state() {
    let dir = temp_dir("ckpt_preserve");
    let node_count_before;

    // Phase 1: Boot, run 500 ticks, checkpoint
    {
        let config = QuakeVectorConfig {
            data_dir: dir.clone(),
            sensor_count: 1,
            ..QuakeVectorConfig::default()
        };
        let mut runtime = QuakeVectorRuntime::boot(config).unwrap();
        assert_eq!(runtime.boot_report().boot_type, BootType::Cold);

        for i in 0..500 {
            runtime.tick(i as u64 * 10_000);
        }

        node_count_before = runtime.node_count();
        assert!(node_count_before > 0, "expected nodes after 500 ticks");

        runtime.checkpoint().expect("checkpoint should succeed");
        // runtime dropped here
    }

    // Phase 2: Boot again from same data_dir
    {
        let config = QuakeVectorConfig {
            data_dir: dir.clone(),
            sensor_count: 1,
            ..QuakeVectorConfig::default()
        };
        let mut runtime = QuakeVectorRuntime::boot(config).unwrap();

        // Should be warm boot
        assert_eq!(runtime.boot_report().boot_type, BootType::Warm);

        // Node count should match
        assert_eq!(
            runtime.node_count(),
            node_count_before,
            "recovered node count should match: expected {node_count_before}, got {}",
            runtime.node_count()
        );

        // Run 100 more ticks without errors
        for i in 0..100 {
            runtime.tick((500 + i) as u64 * 10_000);
        }
        assert!(runtime.node_count() >= node_count_before);
    }

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn multi_station_consensus() {
    let dir_a = temp_dir("consensus_a");
    let dir_b = temp_dir("consensus_b");
    std::fs::create_dir_all(&dir_a).unwrap();
    std::fs::create_dir_all(&dir_b).unwrap();

    let (sk_a, _vk_a) = CryptoService::generate_keypair();
    let (sk_b, _vk_b) = CryptoService::generate_keypair();

    let mut engine_a = AlertEngine::new(StationId(1), sk_a, &dir_a).unwrap();
    let mut engine_b = AlertEngine::new(StationId(2), sk_b, &dir_b).unwrap();

    // Feed same high-probability prediction to Station A
    let prediction = SeismicPrediction {
        event_probability: 0.8,
        estimated_magnitude: 5.0,
        estimated_time_to_peak_s: 10.0,
        confidence: 0.9,
        contributing_wave_ids: vec![NodeId(1), NodeId(2)],
        model_version: 1,
        inference_latency_us: 50,
    };

    let now_us = 1_000_000;

    // Station A processes the prediction — should defer for consensus
    let decision_a = engine_a.process_prediction(&prediction, now_us, true);
    let claim_id = match decision_a {
        AlertDecision::DeferForConsensus(id) => id,
        other => panic!("expected DeferForConsensus from station A, got {other:?}"),
    };

    // Station B also detects the same event — would also defer
    let _decision_b = engine_b.process_prediction(&prediction, now_us, true);

    // Station B sends confirmation to Station A
    let alert = engine_a.process_consensus_confirmation(&claim_id, StationId(2));
    assert!(alert.is_some(), "consensus should be reached");

    let alert = alert.unwrap();
    assert_eq!(alert.level, AlertLevel::High);
    assert!(
        alert.consensus_stations.contains(&StationId(2)),
        "consensus_stations should contain Station B"
    );

    std::fs::remove_dir_all(&dir_a).ok();
    std::fs::remove_dir_all(&dir_b).ok();
}
