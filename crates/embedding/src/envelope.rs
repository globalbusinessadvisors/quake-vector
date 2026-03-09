//! Amplitude envelope extraction at multiple time scales.

/// Extracts 32-dimensional envelope features at 3 scales from a 256-sample window.
pub struct EnvelopeExtractor;

impl EnvelopeExtractor {
    pub fn new() -> Self {
        Self
    }

    /// Extract 32 envelope features: 16 fine + 8 medium + 8 coarse.
    pub fn extract(&self, samples: &[i32; 256]) -> [f32; 32] {
        let rectified: Vec<f32> = samples.iter().map(|&s| (s as f32).abs()).collect();

        let mut output = [0.0f32; 32];

        // Fine scale: 16 evenly-spaced samples from smoothed rectified signal
        let fine = Self::downsample_smooth(&rectified, 16, 8);
        output[..16].copy_from_slice(&fine);

        // Medium scale: downsample 2x first, then 8 samples
        let ds2 = Self::downsample_avg(&rectified, 2);
        let medium = Self::downsample_smooth(&ds2, 8, 4);
        output[16..24].copy_from_slice(&medium);

        // Coarse scale: downsample 4x first, then 8 samples
        let ds4 = Self::downsample_avg(&rectified, 4);
        let coarse = Self::downsample_smooth(&ds4, 8, 4);
        output[24..32].copy_from_slice(&coarse);

        // Normalize each scale independently
        Self::normalize_slice(&mut output[..16]);
        Self::normalize_slice(&mut output[16..24]);
        Self::normalize_slice(&mut output[24..32]);

        output
    }

    /// Downsample by averaging non-overlapping blocks.
    fn downsample_avg(signal: &[f32], factor: usize) -> Vec<f32> {
        signal
            .chunks(factor)
            .map(|chunk| chunk.iter().sum::<f32>() / chunk.len() as f32)
            .collect()
    }

    /// Extract `n` evenly-spaced samples from signal, each smoothed over a
    /// window of `smooth_radius` samples.
    fn downsample_smooth(signal: &[f32], n: usize, smooth_radius: usize) -> Vec<f32> {
        if signal.is_empty() || n == 0 {
            return vec![0.0; n];
        }
        let step = signal.len() as f32 / n as f32;
        (0..n)
            .map(|i| {
                let center = (i as f32 * step + step / 2.0) as usize;
                let center = center.min(signal.len() - 1);
                let start = center.saturating_sub(smooth_radius);
                let end = (center + smooth_radius + 1).min(signal.len());
                let slice = &signal[start..end];
                slice.iter().sum::<f32>() / slice.len() as f32
            })
            .collect()
    }

    fn normalize_slice(slice: &mut [f32]) {
        let max = slice.iter().cloned().fold(0.0f32, f32::max);
        if max > 1e-8 {
            for v in slice.iter_mut() {
                *v /= max;
            }
        }
    }
}

impl Default for EnvelopeExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_function_envelope() {
        // First half zeros, second half high amplitude
        let mut samples = [0i32; 256];
        for s in samples[128..].iter_mut() {
            *s = 1000;
        }

        let extractor = EnvelopeExtractor::new();
        let env = extractor.extract(&samples);

        // Fine scale: first samples should be near 0, last samples near 1
        assert!(env[0] < 0.2, "start of envelope should be low, got {}", env[0]);
        assert!(env[15] > 0.8, "end of envelope should be high, got {}", env[15]);
    }

    #[test]
    fn output_dimensions() {
        let samples = [500i32; 256];
        let extractor = EnvelopeExtractor::new();
        let env = extractor.extract(&samples);
        assert_eq!(env.len(), 32);
    }

    #[test]
    fn normalized_range() {
        let mut samples = [0i32; 256];
        for (i, s) in samples.iter_mut().enumerate() {
            *s = (i as i32) * 10;
        }
        let extractor = EnvelopeExtractor::new();
        let env = extractor.extract(&samples);
        for &v in env.iter() {
            assert!(v >= 0.0 && v <= 1.0 + 1e-6, "value {v} out of [0,1]");
        }
    }
}
