//! FFT-based spectral feature extraction with Hann windowing.

use rustfft::{num_complex::Complex, FftPlanner};
use std::f32::consts::PI;

/// Extracts 128-dimensional spectral features from a 256-sample window.
pub struct FftFeatureExtractor {
    hann_window: [f32; 256],
    fft: std::sync::Arc<dyn rustfft::Fft<f32>>,
}

impl FftFeatureExtractor {
    pub fn new() -> Self {
        let mut hann_window = [0.0f32; 256];
        for (i, h) in hann_window.iter_mut().enumerate() {
            *h = 0.5 * (1.0 - (2.0 * PI * i as f32 / 255.0).cos());
        }
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(256);
        Self { hann_window, fft }
    }

    /// Extract 128 spectral magnitude features from 256 samples.
    ///
    /// Applies Hann window, computes 256-point FFT, takes magnitude of the
    /// first 128 bins, applies log10(1 + mag) scaling, and normalizes to
    /// zero mean / unit variance.
    pub fn extract(&self, samples: &[i32; 256]) -> [f32; 128] {
        // Apply Hann window and convert to complex
        let mut buffer: Vec<Complex<f32>> = samples
            .iter()
            .enumerate()
            .map(|(i, &s)| Complex {
                re: s as f32 * self.hann_window[i],
                im: 0.0,
            })
            .collect();

        // In-place FFT
        self.fft.process(&mut buffer);

        // Take magnitude of first 128 bins, apply log scaling
        let mut features = [0.0f32; 128];
        for (i, feat) in features.iter_mut().enumerate() {
            let mag = buffer[i].norm();
            *feat = (1.0 + mag).log10();
        }

        // Normalize to zero mean, unit variance
        let mean: f32 = features.iter().sum::<f32>() / 128.0;
        let variance: f32 =
            features.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / 128.0;
        let std_dev = variance.sqrt().max(1e-8);
        for feat in features.iter_mut() {
            *feat = (*feat - mean) / std_dev;
        }

        features
    }
}

impl Default for FftFeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sine_wave_peak_at_correct_bin() {
        let extractor = FftFeatureExtractor::new();
        let freq_hz = 10.0_f32;
        let sample_rate = 256.0_f32; // 256 samples at 256 Hz = 1 second
        let mut samples = [0i32; 256];
        for (i, s) in samples.iter_mut().enumerate() {
            *s = (1000.0 * (2.0 * PI * freq_hz * i as f32 / sample_rate).sin()) as i32;
        }

        let features = extractor.extract(&samples);

        // With 256-point FFT at 256 Hz sample rate, bin spacing = 1 Hz.
        // 10 Hz should peak at bin 10.
        let peak_bin = features
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;

        assert!(
            (peak_bin as i32 - 10).unsigned_abs() <= 1,
            "expected peak near bin 10, got bin {peak_bin}"
        );
    }

    #[test]
    fn output_is_normalized() {
        let extractor = FftFeatureExtractor::new();
        let mut samples = [0i32; 256];
        for (i, s) in samples.iter_mut().enumerate() {
            *s = ((i as f32 * 0.1).sin() * 500.0) as i32;
        }

        let features = extractor.extract(&samples);
        let mean: f32 = features.iter().sum::<f32>() / 128.0;
        let variance: f32 =
            features.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / 128.0;

        assert!(mean.abs() < 0.01, "mean should be ~0, got {mean}");
        assert!(
            (variance - 1.0).abs() < 0.1,
            "variance should be ~1, got {variance}"
        );
    }
}
