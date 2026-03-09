//! Sensor health monitoring: tracks sample rate deviation, noise floor, and
//! consecutive read failures per sensor.

use quake_vector_types::{SensorId, SensorStatus};
use std::collections::HashMap;

/// Per-sensor health state.
#[derive(Debug)]
struct SensorHealthState {
    expected_sample_rate: f64,
    /// Exponential moving average of observed sample rate.
    observed_rate_ema: f64,
    /// Exponential moving average of RMS during quiet periods (noise floor).
    noise_floor_ema: f64,
    /// Consecutive read failures.
    consecutive_failures: u32,
    /// Whether EMA has been initialized.
    initialized: bool,
}

/// Monitors health of multiple sensors.
#[derive(Debug)]
pub struct SensorHealthMonitor {
    sensors: HashMap<SensorId, SensorHealthState>,
    /// EMA smoothing factor (alpha). Higher = more responsive.
    ema_alpha: f64,
}

impl SensorHealthMonitor {
    pub fn new() -> Self {
        Self {
            sensors: HashMap::new(),
            ema_alpha: 0.1,
        }
    }

    /// Register a sensor with its expected sample rate.
    pub fn register(&mut self, sensor_id: SensorId, expected_sample_rate: u16) {
        self.sensors.insert(sensor_id, SensorHealthState {
            expected_sample_rate: expected_sample_rate as f64,
            observed_rate_ema: expected_sample_rate as f64,
            noise_floor_ema: 0.0,
            consecutive_failures: 0,
            initialized: false,
        });
    }

    /// Record a successful read with the observed sample rate and RMS amplitude.
    pub fn record_success(&mut self, sensor_id: SensorId, observed_rate: f64, rms: f32) {
        if let Some(state) = self.sensors.get_mut(&sensor_id) {
            state.consecutive_failures = 0;
            if !state.initialized {
                state.observed_rate_ema = observed_rate;
                state.noise_floor_ema = rms as f64;
                state.initialized = true;
            } else {
                state.observed_rate_ema = self.ema_alpha * observed_rate
                    + (1.0 - self.ema_alpha) * state.observed_rate_ema;
                // Update noise floor only during quiet periods (low RMS)
                if (rms as f64) < state.noise_floor_ema * 2.0 || state.noise_floor_ema == 0.0 {
                    state.noise_floor_ema = self.ema_alpha * rms as f64
                        + (1.0 - self.ema_alpha) * state.noise_floor_ema;
                }
            }
        }
    }

    /// Record a read failure for the given sensor.
    pub fn record_failure(&mut self, sensor_id: SensorId) {
        if let Some(state) = self.sensors.get_mut(&sensor_id) {
            state.consecutive_failures += 1;
        }
    }

    /// Report the health status of a sensor.
    ///
    /// - Active: within tolerances
    /// - Degraded: sample rate deviates > 5%
    /// - Failed: > 10 consecutive read failures
    pub fn report_health(&self, sensor_id: SensorId) -> SensorStatus {
        let Some(state) = self.sensors.get(&sensor_id) else {
            return SensorStatus::Failed;
        };

        if state.consecutive_failures > 10 {
            return SensorStatus::Failed;
        }

        if state.expected_sample_rate > 0.0 {
            let deviation = (state.observed_rate_ema - state.expected_sample_rate).abs()
                / state.expected_sample_rate;
            if deviation > 0.05 {
                return SensorStatus::Degraded;
            }
        }

        SensorStatus::Active
    }

    /// Get the current noise floor estimate for a sensor.
    pub fn noise_floor(&self, sensor_id: SensorId) -> f64 {
        self.sensors
            .get(&sensor_id)
            .map(|s| s.noise_floor_ema)
            .unwrap_or(0.0)
    }
}

impl Default for SensorHealthMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn healthy_sensor_is_active() {
        let mut monitor = SensorHealthMonitor::new();
        let sid = SensorId(1);
        monitor.register(sid, 100);
        for _ in 0..20 {
            monitor.record_success(sid, 100.0, 5.0);
        }
        assert_eq!(monitor.report_health(sid), SensorStatus::Active);
    }

    #[test]
    fn deviated_sample_rate_is_degraded() {
        let mut monitor = SensorHealthMonitor::new();
        let sid = SensorId(2);
        monitor.register(sid, 100);
        // Feed a rate that's 10% off — should converge to Degraded
        for _ in 0..100 {
            monitor.record_success(sid, 110.0, 5.0);
        }
        assert_eq!(monitor.report_health(sid), SensorStatus::Degraded);
    }

    #[test]
    fn consecutive_failures_cause_failed() {
        let mut monitor = SensorHealthMonitor::new();
        let sid = SensorId(3);
        monitor.register(sid, 100);
        for _ in 0..11 {
            monitor.record_failure(sid);
        }
        assert_eq!(monitor.report_health(sid), SensorStatus::Failed);
    }

    #[test]
    fn failure_count_resets_on_success() {
        let mut monitor = SensorHealthMonitor::new();
        let sid = SensorId(4);
        monitor.register(sid, 100);
        for _ in 0..5 {
            monitor.record_failure(sid);
        }
        monitor.record_success(sid, 100.0, 5.0);
        for _ in 0..5 {
            monitor.record_failure(sid);
        }
        // Only 5 consecutive failures now, not 10
        assert_ne!(monitor.report_health(sid), SensorStatus::Failed);
    }

    #[test]
    fn unknown_sensor_is_failed() {
        let monitor = SensorHealthMonitor::new();
        assert_eq!(monitor.report_health(SensorId(99)), SensorStatus::Failed);
    }
}
