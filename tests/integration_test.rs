use quake_vector_container::{QuakeVectorConfig, QuakeVectorRuntime};

fn test_config(label: &str) -> QuakeVectorConfig {
    let dir = std::env::temp_dir().join(format!(
        "qv_integration_{}_{}", label,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    QuakeVectorConfig {
        data_dir: dir,
        sensor_count: 1,
        ..QuakeVectorConfig::default()
    }
}

#[test]
fn boot_and_500_ticks() {
    let config = test_config("500ticks");
    let data_dir = config.data_dir.clone();
    let mut runtime = QuakeVectorRuntime::boot(config).unwrap();

    let mut total_windows = 0usize;
    let mut total_predictions = 0usize;

    for i in 0..500 {
        let now_us = i as u64 * 10_000; // 10ms per tick
        let result = runtime.tick(now_us);
        total_windows += result.windows_processed;
        total_predictions += result.predictions_made;
    }

    // With 1 sensor at 100Hz, 64 samples/batch, 256 sample windows:
    // should produce nodes over 500 ticks
    assert!(
        runtime.node_count() > 0,
        "expected nodes in HNSW graph, got 0"
    );
    assert!(
        total_predictions > 0,
        "expected at least some predictions, got 0"
    );
    assert_eq!(total_predictions, total_windows);

    // Verify runtime drops cleanly (no panic on drop)
    let node_count = runtime.node_count();
    drop(runtime);

    assert!(node_count > 0);

    std::fs::remove_dir_all(&data_dir).ok();
}

#[test]
fn checkpoint_after_ticks() {
    let config = test_config("checkpoint");
    let data_dir = config.data_dir.clone();
    let mut runtime = QuakeVectorRuntime::boot(config).unwrap();

    for i in 0..50 {
        runtime.tick(i * 10_000);
    }

    // Checkpoint should succeed
    runtime.checkpoint().expect("checkpoint should not fail");

    std::fs::remove_dir_all(&data_dir).ok();
}

#[test]
fn boot_report_is_cold() {
    let config = test_config("boot_report");
    let data_dir = config.data_dir.clone();
    let runtime = QuakeVectorRuntime::boot(config).unwrap();

    let report = runtime.boot_report();
    assert_eq!(report.boot_type, quake_vector_container::BootType::Cold);
    assert_eq!(report.sensors_discovered, 1);

    std::fs::remove_dir_all(&data_dir).ok();
}
