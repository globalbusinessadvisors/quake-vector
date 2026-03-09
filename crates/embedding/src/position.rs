//! Sinusoidal positional encoding of time-of-day.

use std::f32::consts::PI;

const SECONDS_PER_DAY: f32 = 86400.0;
const SECONDS_PER_HOUR: f32 = 3600.0;

/// Encodes time-of-day as 16-dimensional sinusoidal features.
pub struct PositionEncoder;

impl PositionEncoder {
    pub fn new() -> Self {
        Self
    }

    /// Encode a microsecond timestamp into 16 time-of-day features.
    ///
    /// Produces 8 (sin, cos) pairs at frequencies:
    /// 1/day, 2/day, 4/day, 8/day, 1/hour, 2/hour, 4/hour, 8/hour.
    pub fn encode(timestamp_us: u64) -> [f32; 16] {
        let seconds_since_midnight =
            (timestamp_us / 1_000_000) as f32 % SECONDS_PER_DAY;

        let frequencies = [
            1.0 / SECONDS_PER_DAY,
            2.0 / SECONDS_PER_DAY,
            4.0 / SECONDS_PER_DAY,
            8.0 / SECONDS_PER_DAY,
            1.0 / SECONDS_PER_HOUR,
            2.0 / SECONDS_PER_HOUR,
            4.0 / SECONDS_PER_HOUR,
            8.0 / SECONDS_PER_HOUR,
        ];

        let mut output = [0.0f32; 16];
        for (i, &freq) in frequencies.iter().enumerate() {
            let angle = 2.0 * PI * freq * seconds_since_midnight;
            output[i * 2] = angle.sin();
            output[i * 2 + 1] = angle.cos();
        }

        output
    }
}

impl Default for PositionEncoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn midnight_encoding() {
        let enc = PositionEncoder::encode(0);
        // At midnight, all sin components should be ~0, cos components ~1
        for i in 0..8 {
            assert!(enc[i * 2].abs() < 1e-5, "sin at midnight should be 0");
            assert!(
                (enc[i * 2 + 1] - 1.0).abs() < 1e-5,
                "cos at midnight should be 1"
            );
        }
    }

    #[test]
    fn noon_encoding_daily_frequency() {
        // Noon = 43200 seconds = half day
        let noon_us = 43200 * 1_000_000u64;
        let enc = PositionEncoder::encode(noon_us);
        // 1/day frequency at noon: angle = 2π * (1/86400) * 43200 = π
        // sin(π) ≈ 0, cos(π) ≈ -1
        assert!(enc[0].abs() < 0.01, "sin(π) should be ~0, got {}", enc[0]);
        assert!(
            (enc[1] + 1.0).abs() < 0.01,
            "cos(π) should be ~-1, got {}",
            enc[1]
        );
    }

    #[test]
    fn output_bounded() {
        let enc = PositionEncoder::encode(12345_678_900);
        for &v in enc.iter() {
            assert!(v >= -1.0 && v <= 1.0, "value {v} out of [-1,1]");
        }
    }
}
