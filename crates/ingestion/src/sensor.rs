//! Sensor interface trait and mock implementation.

use quake_vector_types::SensorId;
use std::f64::consts::PI;

/// Errors that can occur during sensor reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SensorError {
    ReadFailed,
    DeviceDisconnected,
    Timeout,
    CalibrationError,
}

impl std::fmt::Display for SensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ReadFailed => write!(f, "sensor read failed"),
            Self::DeviceDisconnected => write!(f, "sensor device disconnected"),
            Self::Timeout => write!(f, "sensor read timed out"),
            Self::CalibrationError => write!(f, "sensor calibration error"),
        }
    }
}

impl std::error::Error for SensorError {}

/// Trait for reading raw samples from a seismic sensor.
pub trait SensorInterface {
    fn read_batch(&mut self, n: usize) -> Result<Vec<i32>, SensorError>;
    fn sample_rate(&self) -> u16;
    fn channel_count(&self) -> u8;
    fn sensor_id(&self) -> SensorId;
}

/// Wave pattern for mock sensor generation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WavePattern {
    /// P-wave: ~5 Hz
    PWave,
    /// S-wave: ~2 Hz
    SWave,
    /// Surface wave: ~0.3 Hz
    SurfaceWave,
    /// Full seismic sequence: P at t=0, S at t+3s, Surface at t+8s
    FullSequence,
}

/// A mock sensor that generates synthetic seismic waveforms.
pub struct MockSensor {
    sensor_id: SensorId,
    sample_rate: u16,
    pattern: WavePattern,
    amplitude: f64,
    noise_level: f64,
    /// Current sample index (acts as time cursor).
    sample_index: u64,
    /// Simple LCG state for deterministic noise.
    rng_state: u64,
}

impl MockSensor {
    pub fn new(
        sensor_id: SensorId,
        sample_rate: u16,
        pattern: WavePattern,
        amplitude: f64,
        noise_level: f64,
    ) -> Self {
        Self {
            sensor_id,
            sample_rate,
            pattern,
            amplitude,
            noise_level,
            sample_index: 0,
            rng_state: sensor_id.0.wrapping_mul(6364136223846793005).wrapping_add(1),
        }
    }

    fn next_noise(&mut self) -> f64 {
        // Simple LCG for deterministic pseudo-random noise
        self.rng_state = self.rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let val = ((self.rng_state >> 33) as f64) / (u32::MAX as f64);
        (val - 0.5) * 2.0 * self.noise_level
    }

    fn generate_sample(&mut self, t: f64) -> i32 {
        let signal = match self.pattern {
            WavePattern::PWave => self.p_wave(t),
            WavePattern::SWave => self.s_wave(t),
            WavePattern::SurfaceWave => self.surface_wave(t),
            WavePattern::FullSequence => self.full_sequence(t),
        };
        let noise = self.next_noise();
        ((signal + noise) * self.amplitude) as i32
    }

    fn p_wave(&self, t: f64) -> f64 {
        (2.0 * PI * 5.0 * t).sin()
    }

    fn s_wave(&self, t: f64) -> f64 {
        (2.0 * PI * 2.0 * t).sin()
    }

    fn surface_wave(&self, t: f64) -> f64 {
        (2.0 * PI * 0.3 * t).sin()
    }

    fn full_sequence(&self, t: f64) -> f64 {
        // P-wave arrives at t=0, S-wave at t=3s, Surface at t=8s
        // Each phase has an exponential decay envelope
        let mut signal = 0.0;

        // P-wave: starts at t=0
        if t >= 0.0 {
            let envelope = (-0.3 * t).exp();
            signal += envelope * self.p_wave(t);
        }

        // S-wave: starts at t=3s
        if t >= 3.0 {
            let dt = t - 3.0;
            let envelope = (-0.2 * dt).exp();
            signal += 1.5 * envelope * self.s_wave(dt);
        }

        // Surface wave: starts at t=8s
        if t >= 8.0 {
            let dt = t - 8.0;
            let envelope = (-0.1 * dt).exp();
            signal += 2.0 * envelope * self.surface_wave(dt);
        }

        signal
    }
}

impl SensorInterface for MockSensor {
    fn read_batch(&mut self, n: usize) -> Result<Vec<i32>, SensorError> {
        let rate = self.sample_rate as f64;
        let samples: Vec<i32> = (0..n)
            .map(|_| {
                let t = self.sample_index as f64 / rate;
                self.sample_index += 1;
                self.generate_sample(t)
            })
            .collect();
        Ok(samples)
    }

    fn sample_rate(&self) -> u16 {
        self.sample_rate
    }

    fn channel_count(&self) -> u8 {
        1
    }

    fn sensor_id(&self) -> SensorId {
        self.sensor_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_sensor_produces_samples() {
        let mut sensor = MockSensor::new(
            SensorId(1),
            100,
            WavePattern::PWave,
            1000.0,
            0.0,
        );
        let batch = sensor.read_batch(100).unwrap();
        assert_eq!(batch.len(), 100);
        // With a 5Hz sine at 100 samples/s, should see zero crossings
        let sign_changes: usize = batch.windows(2)
            .filter(|w| (w[0] >= 0) != (w[1] >= 0))
            .count();
        // 5Hz sine over 1 second => ~10 zero crossings
        assert!(sign_changes >= 8 && sign_changes <= 12,
            "expected ~10 zero crossings, got {sign_changes}");
    }

    #[test]
    fn full_sequence_timing() {
        let mut sensor = MockSensor::new(
            SensorId(1),
            100,
            WavePattern::FullSequence,
            1000.0,
            0.0,
        );
        // Read 15 seconds of data
        let samples = sensor.read_batch(1500).unwrap();

        // Compute RMS in 1-second windows to detect wave arrivals
        let rms_windows: Vec<f64> = (0..15)
            .map(|sec| {
                let start = sec * 100;
                let end = start + 100;
                let sum_sq: f64 = samples[start..end]
                    .iter()
                    .map(|&s| (s as f64) * (s as f64))
                    .sum();
                (sum_sq / 100.0).sqrt()
            })
            .collect();

        // P-wave should have energy at t=0
        assert!(rms_windows[0] > 100.0, "expected P-wave energy at t=0");
        // S-wave should increase energy around t=3-4s
        assert!(rms_windows[3] > rms_windows[2],
            "expected S-wave arrival increases energy at t=3s");
        // Surface wave should add energy around t=8-9s
        assert!(rms_windows[8] > 100.0,
            "expected Surface wave energy at t=8s");
    }

    #[test]
    fn sensor_interface_methods() {
        let sensor = MockSensor::new(SensorId(42), 200, WavePattern::PWave, 1000.0, 0.0);
        assert_eq!(sensor.sample_rate(), 200);
        assert_eq!(sensor.channel_count(), 1);
        assert_eq!(sensor.sensor_id(), SensorId(42));
    }
}
