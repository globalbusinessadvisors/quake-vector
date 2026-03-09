use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use clap::Parser;
use tracing::{error, info, warn};

use quake_vector_container::{
    DegradationLevel, GracefulDegradation, QuakeVectorConfig, QuakeVectorRuntime,
    WatchdogManager, ThreadHealthMonitor,
};
use quake_vector_network::{DashboardService, DashboardState};

/// QuakeVector seismic detection system.
#[derive(Parser, Debug)]
#[command(name = "quake-vector", about = "QuakeVector seismic detection system")]
struct Cli {
    /// Path to config file (TOML). Optional, uses defaults.
    #[arg(long)]
    config: Option<PathBuf>,

    /// Override data directory.
    #[arg(long, default_value = "/data")]
    data_dir: PathBuf,

    /// Override station ID.
    #[arg(long)]
    station_id: Option<u64>,

    /// Number of mock sensors.
    #[arg(long, default_value_t = 1)]
    sensor_count: usize,

    /// Run for N seconds then exit (0 = run forever).
    #[arg(long, default_value_t = 0)]
    duration: u64,

    /// Enable production mode: watchdog, LinuxGpio, SCHED_FIFO priorities.
    #[arg(long)]
    production: bool,

    /// Enable HTTP monitoring dashboard on 127.0.0.1:8080.
    #[arg(long)]
    dashboard: bool,
}

fn build_config(cli: &Cli) -> QuakeVectorConfig {
    let mut config = if let Some(ref path) = cli.config {
        QuakeVectorConfig::from_file(path).unwrap_or_else(|e| {
            warn!(path = %path.display(), error = %e, "failed to load config, using defaults");
            QuakeVectorConfig::default()
        })
    } else {
        QuakeVectorConfig::default()
    };

    config.data_dir = cli.data_dir.clone();

    if let Some(id) = cli.station_id {
        config.station_id = quake_vector_types::StationId(id);
    }

    config.sensor_count = cli.sensor_count;
    config.enable_watchdog = cli.production;

    config
}

/// Best-effort thread pinning using sched_setaffinity.
fn try_set_thread_affinity(cpu: usize) {
    #[cfg(target_os = "linux")]
    {
        unsafe {
            let mut set: libc::cpu_set_t = std::mem::zeroed();
            libc::CPU_SET(cpu, &mut set);
            let result = libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
            if result != 0 {
                warn!(cpu, "failed to set thread affinity (requires root or CAP_SYS_NICE)");
            } else {
                info!(cpu, "thread pinned to CPU");
            }
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = cpu;
        warn!("thread pinning not supported on this platform");
    }
}

/// Best-effort SCHED_FIFO priority setting.
fn try_set_realtime_priority(priority: i32) {
    #[cfg(target_os = "linux")]
    {
        unsafe {
            let param = libc::sched_param {
                sched_priority: priority,
            };
            let result = libc::sched_setscheduler(0, libc::SCHED_FIFO, &param);
            if result != 0 {
                warn!("failed to set SCHED_FIFO (requires root or CAP_SYS_NICE)");
            } else {
                info!(priority, "SCHED_FIFO enabled");
            }
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = priority;
        warn!("SCHED_FIFO not supported on this platform");
    }
}

fn main() {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    info!(
        sensor_count = cli.sensor_count,
        duration = cli.duration,
        data_dir = %cli.data_dir.display(),
        production = cli.production,
        dashboard = cli.dashboard,
        "QuakeVector starting"
    );

    let config = build_config(&cli);
    let default_checkpoint_interval = Duration::from_secs(config.checkpoint_interval_secs);

    let mut runtime = match QuakeVectorRuntime::boot(config) {
        Ok(rt) => rt,
        Err(e) => {
            error!(error = %e, "failed to boot runtime");
            std::process::exit(1);
        }
    };

    let report = runtime.boot_report();
    info!(
        boot_type = ?report.boot_type,
        duration_ms = report.total_duration_ms,
        sensors = report.sensors_discovered,
        nodes = report.graph_nodes_loaded,
        "boot complete"
    );

    // Dashboard setup
    let dashboard_state = if cli.dashboard {
        let boot_type_str = format!("{:?}", report.boot_type).to_lowercase();
        let state = Arc::new(tokio::sync::RwLock::new(DashboardState::new(
            runtime.config.station_id,
            &boot_type_str,
            report.total_duration_ms,
        )));

        let svc = DashboardService::new("127.0.0.1:8080", Arc::clone(&state));

        // Spawn tokio runtime for the dashboard server
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .expect("failed to create tokio runtime for dashboard");

        let state_clone = Arc::clone(&state);
        std::thread::spawn(move || {
            rt.block_on(async move {
                let _handle = svc.start();
                // Keep the runtime alive
                tokio::signal::ctrl_c().await.ok();
            });
        });

        Some(state_clone)
    } else {
        None
    };

    // Production mode setup
    if cli.production {
        try_set_realtime_priority(50);
        try_set_thread_affinity(0);
    }

    // Watchdog
    let mut watchdog = WatchdogManager::new(if cli.production { 15 } else { 10 })
        .expect("failed to create watchdog");
    if watchdog.is_mock() && cli.production {
        warn!("production mode but watchdog is in mock mode");
    }

    // Thread health monitor
    let mut thread_health = ThreadHealthMonitor::new();
    let main_heartbeat = thread_health.register_thread("main-loop");

    // Signal handling: catch SIGINT and SIGTERM
    let shutdown = Arc::new(AtomicBool::new(false));
    signal_hook::flag::register(signal_hook::consts::SIGINT, shutdown.clone())
        .expect("failed to register SIGINT handler");
    signal_hook::flag::register(signal_hook::consts::SIGTERM, shutdown.clone())
        .expect("failed to register SIGTERM handler");

    let wall_start = Instant::now();
    let mono_start = Instant::now();
    let mut last_log = Instant::now();
    let mut last_checkpoint = Instant::now();
    let mut last_watchdog = Instant::now();
    let mut last_resource_check = Instant::now();
    let mut last_dashboard_update = Instant::now();
    let mut total_windows: u64 = 0;
    let mut total_predictions: u64 = 0;
    let mut total_alerts: u64 = 0;
    let mut period_windows: u64 = 0;
    let mut period_predictions: u64 = 0;

    let mut current_degradation = DegradationLevel::Normal;
    let mut watchdog_paused = false;

    loop {
        if shutdown.load(Ordering::Relaxed) {
            info!("shutdown signal received");
            break;
        }

        // Check duration limit
        if cli.duration > 0 && wall_start.elapsed().as_secs() >= cli.duration {
            info!(elapsed_secs = wall_start.elapsed().as_secs(), "duration limit reached");
            break;
        }

        // Main loop heartbeat
        main_heartbeat.beat();

        let now_us = mono_start.elapsed().as_micros() as u64;
        let result = runtime.tick(now_us);

        total_windows += result.windows_processed as u64;
        total_predictions += result.predictions_made as u64;
        total_alerts += result.alerts.len() as u64;
        period_windows += result.windows_processed as u64;
        period_predictions += result.predictions_made as u64;

        // Watchdog heartbeat every 2 seconds
        if last_watchdog.elapsed().as_secs() >= 2 {
            if !watchdog_paused {
                let stale = thread_health.check_all(10_000);
                if stale.is_empty() {
                    if let Err(e) = watchdog.heartbeat() {
                        warn!(error = %e, "watchdog heartbeat failed");
                    }
                } else {
                    error!(stale_threads = ?stale, "critical thread(s) stale — stopping watchdog heartbeat");
                    watchdog_paused = true;
                }
            }
            last_watchdog = Instant::now();
        }

        // Log summary every 10 seconds
        if last_log.elapsed().as_secs() >= 10 {
            let elapsed = last_log.elapsed().as_secs_f64();
            info!(
                windows_per_sec = format_args!("{:.1}", period_windows as f64 / elapsed),
                predictions_per_sec = format_args!("{:.1}", period_predictions as f64 / elapsed),
                total_alerts,
                node_count = runtime.node_count(),
                "periodic summary"
            );
            period_windows = 0;
            period_predictions = 0;
            last_log = Instant::now();
        }

        // Checkpoint using wear-adjusted interval
        let checkpoint_interval = runtime
            .wear_monitor()
            .recommended_checkpoint_interval(default_checkpoint_interval);
        if last_checkpoint.elapsed() >= checkpoint_interval {
            if let Err(e) = runtime.checkpoint() {
                warn!(error = %e, "checkpoint failed");
            } else {
                info!("checkpoint completed");
            }
            last_checkpoint = Instant::now();
        }

        // Resource monitoring every 30 seconds + graceful degradation
        if last_resource_check.elapsed().as_secs() >= 30 {
            let status = runtime.resource_monitor().check();
            if status.degradation_level != current_degradation {
                GracefulDegradation::apply(status.degradation_level, &mut runtime);
                current_degradation = status.degradation_level;
            }
            last_resource_check = Instant::now();
        }

        // Dashboard state update every 5 seconds
        if let Some(ref state) = dashboard_state {
            if last_dashboard_update.elapsed().as_secs() >= 5 {
                if let Ok(mut s) = state.try_write() {
                    s.uptime_secs = wall_start.elapsed().as_secs();
                    runtime.update_dashboard(&mut s);
                }
                last_dashboard_update = Instant::now();
            }
        }

        // Brief sleep to avoid busy-spinning
        std::thread::sleep(Duration::from_millis(1));
    }

    // Final checkpoint
    info!("running final checkpoint");
    if let Err(e) = runtime.checkpoint() {
        warn!(error = %e, "final checkpoint failed");
    }

    let elapsed = wall_start.elapsed();
    info!(
        elapsed_secs = elapsed.as_secs(),
        total_windows,
        total_predictions,
        total_alerts,
        node_count = runtime.node_count(),
        "QuakeVector shutdown complete"
    );
}
