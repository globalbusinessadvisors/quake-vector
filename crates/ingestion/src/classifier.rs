//! Wave type classification based on zero-crossing frequency analysis.

use quake_vector_types::{WaveType, WaveformWindow};

/// Noise floor threshold — windows below this RMS are classified as Unknown.
const NOISE_FLOOR_RMS: f32 = 10.0;

/// Classifies waveform windows by dominant frequency using zero-crossing rate.
pub struct WaveClassifierService;

impl WaveClassifierService {
    /// Classify a waveform window's wave type based on its dominant frequency.
    ///
    /// Uses zero-crossing rate to estimate dominant frequency:
    /// - P-wave:   > 3 Hz
    /// - S-wave:   1–3 Hz
    /// - Surface:  < 1 Hz
    /// - Unknown:  amplitude below noise floor
    pub fn classify(window: &WaveformWindow) -> WaveType {
        if window.rms_amplitude < NOISE_FLOOR_RMS {
            return WaveType::Unknown;
        }

        let freq = Self::zero_crossing_frequency(&window.samples, window.dominant_freq_hz);
        if freq > 3.0 {
            WaveType::P
        } else if freq >= 1.0 {
            WaveType::S
        } else {
            WaveType::Surface
        }
    }

    /// Estimate dominant frequency from zero-crossing rate.
    ///
    /// If the window already has a computed dominant_freq_hz, use it.
    /// Otherwise, estimate from sample zero crossings using an assumed
    /// sample rate of 100 Hz.
    fn zero_crossing_frequency(samples: &[i32], precomputed_freq: f32) -> f32 {
        if precomputed_freq > 0.0 {
            return precomputed_freq;
        }
        Self::compute_zero_crossing_freq(samples, 100)
    }

    /// Compute zero-crossing frequency from raw samples and sample rate.
    pub fn compute_zero_crossing_freq(samples: &[i32], sample_rate: u16) -> f32 {
        if samples.len() < 2 {
            return 0.0;
        }
        let crossings: usize = samples
            .windows(2)
            .filter(|w| (w[0] >= 0) != (w[1] >= 0))
            .count();
        let duration_s = (samples.len() - 1) as f32 / sample_rate as f32;
        if duration_s <= 0.0 {
            return 0.0;
        }
        // Each full cycle has 2 zero crossings
        crossings as f32 / (2.0 * duration_s)
    }

    /// Compute RMS amplitude for a slice of samples.
    pub fn compute_rms(samples: &[i32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let sum_sq: f64 = samples.iter().map(|&s| (s as f64) * (s as f64)).sum();
        (sum_sq / samples.len() as f64).sqrt() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quake_vector_types::ChannelId;
    use std::f64::consts::PI;

    fn make_sine_window(freq_hz: f64, sample_rate: u16, amplitude: f64) -> WaveformWindow {
        let mut samples = [0i32; 256];
        for (i, sample) in samples.iter_mut().enumerate() {
            let t = i as f64 / sample_rate as f64;
            *sample = (amplitude * (2.0 * PI * freq_hz * t).sin()) as i32;
        }
        let rms = WaveClassifierService::compute_rms(&samples);
        let dom_freq = WaveClassifierService::compute_zero_crossing_freq(&samples, sample_rate);
        WaveformWindow {
            samples,
            channel: ChannelId(1),
            timestamp_us: 0,
            wave_type: WaveType::Unknown,
            rms_amplitude: rms,
            dominant_freq_hz: dom_freq,
        }
    }

    #[test]
    fn classify_p_wave() {
        let window = make_sine_window(5.0, 100, 1000.0);
        assert_eq!(WaveClassifierService::classify(&window), WaveType::P);
    }

    #[test]
    fn classify_s_wave() {
        let window = make_sine_window(2.0, 100, 1000.0);
        assert_eq!(WaveClassifierService::classify(&window), WaveType::S);
    }

    #[test]
    fn classify_surface_wave() {
        let window = make_sine_window(0.3, 100, 1000.0);
        assert_eq!(WaveClassifierService::classify(&window), WaveType::Surface);
    }

    #[test]
    fn classify_noise_floor() {
        let window = make_sine_window(5.0, 100, 1.0);
        assert_eq!(WaveClassifierService::classify(&window), WaveType::Unknown);
    }

    #[test]
    fn zero_crossing_frequency_accuracy() {
        let mut samples = [0i32; 256];
        let sample_rate: u16 = 100;
        let freq = 5.0_f64;
        for (i, sample) in samples.iter_mut().enumerate() {
            let t = i as f64 / sample_rate as f64;
            *sample = (1000.0 * (2.0 * PI * freq * t).sin()) as i32;
        }
        let computed = WaveClassifierService::compute_zero_crossing_freq(&samples, sample_rate);
        assert!((computed - 5.0).abs() < 0.5, "expected ~5Hz, got {computed}");
    }
}
