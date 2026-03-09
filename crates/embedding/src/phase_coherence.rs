//! Phase coherence estimation via autocorrelation.

/// Extracts 32-dimensional autocorrelation features as a proxy for phase coherence.
pub struct PhaseCoherenceExtractor;

impl PhaseCoherenceExtractor {
    pub fn new() -> Self {
        Self
    }

    /// Compute normalized autocorrelation at lags 1..=32.
    ///
    /// Each value is in [-1, 1]. High autocorrelation at a lag indicates
    /// periodicity at that interval.
    pub fn extract(&self, samples: &[i32; 256]) -> [f32; 32] {
        let n = samples.len();
        let mean: f64 = samples.iter().map(|&s| s as f64).sum::<f64>() / n as f64;
        let variance: f64 = samples
            .iter()
            .map(|&s| {
                let d = s as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / n as f64;

        let mut output = [0.0f32; 32];

        if variance < 1e-10 {
            return output;
        }

        for (i, out) in output.iter_mut().enumerate() {
            let lag = i + 1;
            let autocorr: f64 = (0..(n - lag))
                .map(|j| (samples[j] as f64 - mean) * (samples[j + lag] as f64 - mean))
                .sum::<f64>()
                / (n as f64 * variance);
            *out = autocorr as f32;
        }

        output
    }
}

impl Default for PhaseCoherenceExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn pure_sine_has_periodic_autocorrelation() {
        let mut samples = [0i32; 256];
        // 8 Hz sine at 256 samples/s => period = 32 samples
        for (i, s) in samples.iter_mut().enumerate() {
            *s = (1000.0 * (2.0 * PI * 8.0 * i as f64 / 256.0).sin()) as i32;
        }

        let extractor = PhaseCoherenceExtractor::new();
        let ac = extractor.extract(&samples);

        // Lag 32 should show high positive autocorrelation (period = 32 samples)
        // Lag 16 should show negative autocorrelation (half period)
        assert!(
            ac[31] > 0.5,
            "autocorrelation at lag 32 should be high for period-32 sine, got {}",
            ac[31]
        );
        assert!(
            ac[15] < -0.3,
            "autocorrelation at lag 16 should be negative for period-32 sine, got {}",
            ac[15]
        );
    }

    #[test]
    fn constant_signal_zero_autocorrelation() {
        let samples = [500i32; 256];
        let extractor = PhaseCoherenceExtractor::new();
        let ac = extractor.extract(&samples);
        // Constant signal has zero variance => all zeros
        for &v in ac.iter() {
            assert!(v.abs() < 1e-6, "constant signal autocorrelation should be 0");
        }
    }

    #[test]
    fn output_bounded() {
        let mut samples = [0i32; 256];
        for (i, s) in samples.iter_mut().enumerate() {
            *s = ((i * 37 + 13) % 2000) as i32 - 1000;
        }
        let extractor = PhaseCoherenceExtractor::new();
        let ac = extractor.extract(&samples);
        for &v in ac.iter() {
            assert!(
                v >= -1.0 - 1e-5 && v <= 1.0 + 1e-5,
                "autocorrelation {v} out of [-1,1]"
            );
        }
    }
}
