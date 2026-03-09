//! SensorStream: aggregate root that combines sensor reading, ring buffering,
//! wave classification, and health monitoring into a single poll-based interface.

use quake_vector_types::{ChannelId, WaveType, WaveformWindow};
use tracing::warn;

use crate::classifier::WaveClassifierService;
use crate::health::SensorHealthMonitor;
use crate::ring_buffer::RingBuffer;
use crate::sensor::SensorInterface;

const WINDOW_SIZE: usize = 256;
const WINDOW_OVERLAP: usize = 128;
const BATCH_SIZE: usize = 64;

/// Aggregate root for sensor ingestion. Reads from a sensor, buffers samples,
/// and produces classified WaveformWindows.
pub struct SensorStream {
    sensor: Box<dyn SensorInterface>,
    buffer: RingBuffer<i32>,
    health: SensorHealthMonitor,
    /// Monotonic timestamp counter in microseconds.
    timestamp_us: u64,
}

impl SensorStream {
    pub fn new(sensor: Box<dyn SensorInterface>, health: SensorHealthMonitor) -> Self {
        let sensor_id = sensor.sensor_id();
        let sample_rate = sensor.sample_rate();
        let mut health = health;
        health.register(sensor_id, sample_rate);
        Self {
            sensor,
            // Capacity: enough for several windows
            buffer: RingBuffer::new(WINDOW_SIZE * 4),
            health,
            timestamp_us: 0,
        }
    }

    /// Poll the sensor for new data and return a WaveformWindow if a complete
    /// window (256 samples, 128 overlap) is available.
    pub fn poll(&mut self) -> Option<WaveformWindow> {
        let sensor_id = self.sensor.sensor_id();
        let sample_rate = self.sensor.sample_rate();

        match self.sensor.read_batch(BATCH_SIZE) {
            Ok(samples) => {
                let rms = WaveClassifierService::compute_rms(&samples);
                self.health
                    .record_success(sensor_id, sample_rate as f64, rms);
                self.buffer.push(&samples);
            }
            Err(e) => {
                warn!(sensor = ?sensor_id, error = %e, "sensor read failed");
                self.health.record_failure(sensor_id);
                return None;
            }
        }

        if !self.buffer.has_complete_window(WINDOW_SIZE) {
            return None;
        }

        let raw = self.buffer.extract_window(WINDOW_SIZE, WINDOW_OVERLAP)?;

        let mut window_samples = [0i32; 256];
        window_samples.copy_from_slice(&raw);

        let rms = WaveClassifierService::compute_rms(&window_samples);
        let dom_freq =
            WaveClassifierService::compute_zero_crossing_freq(&window_samples, sample_rate);

        // Advance timestamp by the non-overlapping portion
        let advance_samples = (WINDOW_SIZE - WINDOW_OVERLAP) as u64;
        let us_per_sample = 1_000_000u64 / sample_rate as u64;
        self.timestamp_us += advance_samples * us_per_sample;

        let mut window = WaveformWindow {
            samples: window_samples,
            channel: ChannelId(sensor_id.0),
            timestamp_us: self.timestamp_us,
            wave_type: WaveType::Unknown,
            rms_amplitude: rms,
            dominant_freq_hz: dom_freq,
        };

        window.wave_type = WaveClassifierService::classify(&window);

        Some(window)
    }

    /// Access the health monitor.
    pub fn health(&self) -> &SensorHealthMonitor {
        &self.health
    }

    /// Get the sensor ID.
    pub fn sensor_id(&self) -> quake_vector_types::SensorId {
        self.sensor.sensor_id()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sensor::{MockSensor, WavePattern};
    use quake_vector_types::SensorId;

    #[test]
    fn stream_produces_windows() {
        let sensor = MockSensor::new(SensorId(1), 100, WavePattern::PWave, 1000.0, 0.0);
        let health = SensorHealthMonitor::new();
        let mut stream = SensorStream::new(Box::new(sensor), health);

        // Need enough polls to fill 256 samples (at 64 per batch = 4 polls)
        let mut windows = Vec::new();
        for _ in 0..10 {
            if let Some(w) = stream.poll() {
                windows.push(w);
            }
        }

        assert!(!windows.is_empty(), "should produce at least one window");

        // First window should be classified as P-wave (5Hz signal)
        assert_eq!(windows[0].wave_type, WaveType::P);
        assert!(windows[0].rms_amplitude > 100.0);
        assert!(windows[0].dominant_freq_hz > 3.0);
    }

    #[test]
    fn stream_window_cadence() {
        let sensor = MockSensor::new(SensorId(1), 100, WavePattern::SWave, 1000.0, 0.0);
        let health = SensorHealthMonitor::new();
        let mut stream = SensorStream::new(Box::new(sensor), health);

        let mut window_count = 0;
        // 20 polls × 64 samples = 1280 samples
        // First window at 256, then every 128 after => (1280-256)/128 + 1 = ~9
        for _ in 0..20 {
            if stream.poll().is_some() {
                window_count += 1;
            }
        }

        assert!(
            window_count >= 5,
            "expected at least 5 windows from 1280 samples, got {window_count}"
        );
    }

    #[test]
    fn stream_timestamps_advance() {
        let sensor = MockSensor::new(SensorId(1), 100, WavePattern::PWave, 1000.0, 0.0);
        let health = SensorHealthMonitor::new();
        let mut stream = SensorStream::new(Box::new(sensor), health);

        let mut timestamps = Vec::new();
        for _ in 0..20 {
            if let Some(w) = stream.poll() {
                timestamps.push(w.timestamp_us);
            }
        }

        assert!(timestamps.len() >= 2);
        // Timestamps should be monotonically increasing
        for pair in timestamps.windows(2) {
            assert!(pair[1] > pair[0], "timestamps must increase");
        }
    }
}
