//! STA/LTA-based arrival detection features.

/// Extracts 16-dimensional STA/LTA ratio features from a 256-sample window.
pub struct ArrivalDetector;

impl ArrivalDetector {
    pub fn new() -> Self {
        Self
    }

    /// Compute STA/LTA ratios at 16 evenly spaced positions.
    ///
    /// - Short-term window: 10 samples (±5 around position)
    /// - Long-term window: 100 samples (±50 around position)
    /// - Output is normalized across the 16 positions.
    pub fn extract(&self, samples: &[i32; 256]) -> [f32; 16] {
        let abs_samples: Vec<f32> = samples.iter().map(|&s| (s as f32).abs()).collect();
        let n = abs_samples.len();

        let mut ratios = [0.0f32; 16];
        for (i, ratio) in ratios.iter_mut().enumerate() {
            // 16 evenly spaced positions across the 256 samples
            let pos = (i * n) / 16 + n / 32;

            // STA: mean of |samples| in [pos-5, pos+5)
            let sta_start = pos.saturating_sub(5);
            let sta_end = (pos + 5).min(n);
            let sta: f32 = abs_samples[sta_start..sta_end].iter().sum::<f32>()
                / (sta_end - sta_start) as f32;

            // LTA: mean of |samples| in [pos-50, pos+50)
            let lta_start = pos.saturating_sub(50);
            let lta_end = (pos + 50).min(n);
            let lta: f32 = abs_samples[lta_start..lta_end].iter().sum::<f32>()
                / (lta_end - lta_start) as f32;

            *ratio = if lta > 1e-8 { sta / lta } else { 0.0 };
        }

        // Normalize
        let max = ratios.iter().cloned().fold(0.0f32, f32::max);
        if max > 1e-8 {
            for r in ratios.iter_mut() {
                *r /= max;
            }
        }

        ratios
    }
}

impl Default for ArrivalDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sharp_onset_produces_high_sta_lta() {
        // Quiet background, then sharp onset at sample 128
        let mut samples = [1i32; 256];
        for s in samples[128..].iter_mut() {
            *s = 1000;
        }

        let detector = ArrivalDetector::new();
        let ratios = detector.extract(&samples);

        // Positions around the onset (positions 8-9 of 16) should have higher ratios
        // than the early quiet positions (0-3)
        let early_max = ratios[..4].iter().cloned().fold(0.0f32, f32::max);
        let onset_max = ratios[7..10].iter().cloned().fold(0.0f32, f32::max);

        assert!(
            onset_max > early_max,
            "onset region ({onset_max}) should exceed quiet region ({early_max})"
        );
    }

    #[test]
    fn output_dimensions() {
        let samples = [0i32; 256];
        let detector = ArrivalDetector::new();
        let ratios = detector.extract(&samples);
        assert_eq!(ratios.len(), 16);
    }

    #[test]
    fn constant_signal_uniform_ratios() {
        let samples = [500i32; 256];
        let detector = ArrivalDetector::new();
        let ratios = detector.extract(&samples);
        // For a constant signal, STA/LTA ≈ 1.0 everywhere (before normalization)
        // After normalization, all should be close to 1.0
        for &r in ratios.iter() {
            assert!(
                (r - 1.0).abs() < 0.15,
                "constant signal should give uniform ratios, got {r}"
            );
        }
    }
}
