//! Seismic event simulator: generates realistic P -> S -> Surface wave sequences.

use std::f64::consts::PI;

use quake_vector_types::SensorId;

use crate::sensor::{SensorError, SensorInterface};

/// Generates a realistic P -> S -> Surface wave sequence.
pub struct SeismicEventSimulator;

impl SeismicEventSimulator {
    /// Generate a full seismic event sequence.
    ///
    /// Returns `(time_seconds, samples)` pairs for the entire sequence.
    /// - `magnitude`: affects amplitudes (Richter-like scale, 1.0-9.0)
    /// - `distance_km`: affects timing gaps and frequency attenuation
    /// - `noise_level`: background noise amplitude (0.0 = none, 1.0 = significant)
    pub fn generate_sequence(
        magnitude: f32,
        distance_km: f32,
        noise_level: f32,
        sample_rate: u16,
    ) -> Vec<(f64, Vec<i32>)> {
        let rate = sample_rate as f64;

        // Amplitude scaling based on magnitude (exponential)
        let base_amp = 10.0f64.powf(magnitude as f64 - 2.0) * 100.0;
        // Distance attenuation (1/sqrt(r))
        let dist_factor = 1.0 / (1.0 + (distance_km as f64 / 10.0).sqrt());
        let amp = base_amp * dist_factor;

        // Timing based on distance (P-wave ~6 km/s, S-wave ~3.5 km/s)
        let p_arrival = 0.0;
        let s_arrival = distance_km as f64 / 3.5 - distance_km as f64 / 6.0;
        let s_arrival = s_arrival.max(2.0).min(10.0); // clamp 2-10s
        let surface_arrival = s_arrival + 3.0 + distance_km as f64 * 0.05;
        let surface_arrival = surface_arrival.min(20.0);

        // Total duration: surface wave + decay time
        let total_duration = surface_arrival + 15.0;
        let total_samples = (total_duration * rate) as usize;

        // Generate in 1-second chunks
        let chunk_samples = sample_rate as usize;
        let num_chunks = (total_samples + chunk_samples - 1) / chunk_samples;

        let mut rng_state: u64 = 0xdeadbeef_12345678;
        let mut result = Vec::with_capacity(num_chunks);

        for chunk_idx in 0..num_chunks {
            let chunk_start = chunk_idx * chunk_samples;
            let chunk_end = (chunk_start + chunk_samples).min(total_samples);
            let t_start = chunk_start as f64 / rate;

            let mut samples = Vec::with_capacity(chunk_end - chunk_start);

            for sample_idx in chunk_start..chunk_end {
                let t = sample_idx as f64 / rate;
                let mut signal = 0.0;

                // P-wave: 5 Hz, sharp onset, moderate amplitude
                if t >= p_arrival {
                    let dt = t - p_arrival;
                    let envelope = (-0.3 * dt).exp();
                    signal += envelope * (2.0 * PI * 5.0 * dt).sin() * amp * 0.4;
                }

                // S-wave: 2 Hz, high amplitude, gradual onset
                if t >= s_arrival {
                    let dt = t - s_arrival;
                    let onset = (dt / 0.5).min(1.0); // 0.5s ramp
                    let envelope = onset * (-0.15 * dt).exp();
                    signal += envelope * (2.0 * PI * 2.0 * dt).sin() * amp * 1.0;
                }

                // Surface wave: 0.3 Hz, highest amplitude, long duration
                if t >= surface_arrival {
                    let dt = t - surface_arrival;
                    let onset = (dt / 1.0).min(1.0); // 1s ramp
                    let envelope = onset * (-0.08 * dt).exp();
                    signal += envelope * (2.0 * PI * 0.3 * dt).sin() * amp * 1.5;
                }

                // Add noise
                rng_state = rng_state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let noise_val =
                    ((rng_state >> 33) as f64 / u32::MAX as f64 - 0.5) * 2.0;
                signal += noise_val * noise_level as f64 * amp * 0.1;

                samples.push(signal as i32);
            }

            result.push((t_start, samples));
        }

        result
    }
}

/// A sensor that replays a pre-generated seismic event sequence.
pub struct SimulatedSensor {
    sensor_id: SensorId,
    sample_rate: u16,
    /// All samples flattened from the sequence.
    samples: Vec<i32>,
    /// Current read position.
    cursor: usize,
}

impl SimulatedSensor {
    /// Create a sensor that will emit the given event sequence.
    pub fn from_event(
        sensor_id: SensorId,
        sample_rate: u16,
        magnitude: f32,
        distance_km: f32,
        noise_level: f32,
    ) -> Self {
        let sequence =
            SeismicEventSimulator::generate_sequence(magnitude, distance_km, noise_level, sample_rate);
        let samples: Vec<i32> = sequence.into_iter().flat_map(|(_, s)| s).collect();
        Self {
            sensor_id,
            sample_rate,
            samples,
            cursor: 0,
        }
    }

    /// Create a noise-only sensor (no seismic event).
    pub fn noise_only(sensor_id: SensorId, sample_rate: u16, noise_amp: f64, duration_secs: f64) -> Self {
        let total = (sample_rate as f64 * duration_secs) as usize;
        let mut rng_state: u64 = sensor_id.0.wrapping_mul(6364136223846793005).wrapping_add(42);
        let samples: Vec<i32> = (0..total)
            .map(|_| {
                rng_state = rng_state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let v = ((rng_state >> 33) as f64 / u32::MAX as f64 - 0.5) * 2.0;
                (v * noise_amp) as i32
            })
            .collect();
        Self {
            sensor_id,
            sample_rate,
            samples,
            cursor: 0,
        }
    }
}

impl SensorInterface for SimulatedSensor {
    fn read_batch(&mut self, n: usize) -> Result<Vec<i32>, SensorError> {
        let end = (self.cursor + n).min(self.samples.len());
        if self.cursor >= self.samples.len() {
            // Past end: return zeros (quiet)
            return Ok(vec![0; n]);
        }
        let batch = self.samples[self.cursor..end].to_vec();
        self.cursor = end;
        // Pad with zeros if we ran out
        if batch.len() < n {
            let mut padded = batch;
            padded.resize(n, 0);
            Ok(padded)
        } else {
            Ok(batch)
        }
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
    fn generate_sequence_has_energy() {
        let seq = SeismicEventSimulator::generate_sequence(5.0, 20.0, 0.01, 100);
        assert!(!seq.is_empty());

        // Total samples should be > 0
        let total_samples: usize = seq.iter().map(|(_, s)| s.len()).sum();
        assert!(total_samples > 1000, "expected >1000 samples, got {total_samples}");

        // Check that early samples (P-wave) have energy
        let first_chunk = &seq[0].1;
        let max_abs: i32 = first_chunk.iter().map(|s| s.abs()).max().unwrap_or(0);
        assert!(max_abs > 10, "P-wave should have amplitude, got max_abs={max_abs}");
    }

    #[test]
    fn simulated_sensor_reads_correctly() {
        let mut sensor = SimulatedSensor::from_event(SensorId(1), 100, 5.0, 20.0, 0.01);
        let batch1 = sensor.read_batch(64).unwrap();
        assert_eq!(batch1.len(), 64);

        let batch2 = sensor.read_batch(64).unwrap();
        assert_eq!(batch2.len(), 64);

        assert_eq!(sensor.sample_rate(), 100);
        assert_eq!(sensor.sensor_id(), SensorId(1));
    }

    #[test]
    fn noise_only_sensor_has_low_energy() {
        let mut sensor = SimulatedSensor::noise_only(SensorId(2), 100, 50.0, 5.0);
        let batch = sensor.read_batch(500).unwrap();
        assert_eq!(batch.len(), 500);

        let max_abs: i32 = batch.iter().map(|s| s.abs()).max().unwrap_or(0);
        assert!(max_abs < 100, "noise should be low, got max_abs={max_abs}");
    }
}
