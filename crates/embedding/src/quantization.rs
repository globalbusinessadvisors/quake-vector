//! Symmetric quantization of f32 vectors to i8 for storage efficiency.

/// Provides f32 <-> i8 symmetric quantization for embedding vectors.
pub struct QuantizationService;

impl QuantizationService {
    /// Quantize a 256-dim f32 vector to i8 with symmetric quantization.
    ///
    /// Returns (quantized_vector, scale, zero_point).
    /// Scale = max(|v|) / 127. Zero point is always 0 for symmetric quantization.
    pub fn quantize_f32_to_i8(vector: &[f32; 256]) -> ([i8; 256], f32, f32) {
        let max_abs = vector
            .iter()
            .map(|v| v.abs())
            .fold(0.0f32, f32::max);

        let scale = if max_abs > 1e-10 {
            max_abs / 127.0
        } else {
            1.0 / 127.0
        };

        let mut quantized = [0i8; 256];
        for (i, &v) in vector.iter().enumerate() {
            let q = (v / scale).round().clamp(-127.0, 127.0) as i8;
            quantized[i] = q;
        }

        (quantized, scale, 0.0)
    }

    /// Dequantize an i8 vector back to f32.
    pub fn dequantize_i8_to_f32(quantized: &[i8; 256], scale: f32) -> [f32; 256] {
        let mut output = [0.0f32; 256];
        for (i, &q) in quantized.iter().enumerate() {
            output[i] = q as f32 * scale;
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_error_under_five_percent() {
        // Create a vector with varied values
        let mut vector = [0.0f32; 256];
        for (i, v) in vector.iter_mut().enumerate() {
            *v = (i as f32 / 256.0) * 2.0 - 1.0; // range [-1, 1]
        }

        let (quantized, scale, _zero) = QuantizationService::quantize_f32_to_i8(&vector);
        let restored = QuantizationService::dequantize_i8_to_f32(&quantized, scale);

        let max_error: f32 = vector
            .iter()
            .zip(restored.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        let max_val = vector.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let relative_error = max_error / max_val;

        assert!(
            relative_error < 0.05,
            "round-trip error {relative_error:.4} exceeds 5%"
        );
    }

    #[test]
    fn zero_vector_round_trip() {
        let vector = [0.0f32; 256];
        let (quantized, scale, _) = QuantizationService::quantize_f32_to_i8(&vector);
        let restored = QuantizationService::dequantize_i8_to_f32(&quantized, scale);
        for &v in restored.iter() {
            assert!(v.abs() < 1e-6);
        }
    }

    #[test]
    fn symmetric_quantization_zero_point_is_zero() {
        let mut vector = [0.0f32; 256];
        vector[0] = 1.0;
        vector[1] = -1.0;
        let (_, _, zero_point) = QuantizationService::quantize_f32_to_i8(&vector);
        assert!((zero_point).abs() < 1e-8);
    }

    #[test]
    fn l2_normalized_vector_round_trip() {
        // L2-normalized vectors are common in our pipeline
        let mut vector = [0.0f32; 256];
        for (i, v) in vector.iter_mut().enumerate() {
            *v = ((i as f32 * 0.1).sin()) * 0.1;
        }
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        for v in vector.iter_mut() {
            *v /= norm;
        }

        let (quantized, scale, _) = QuantizationService::quantize_f32_to_i8(&vector);
        let restored = QuantizationService::dequantize_i8_to_f32(&quantized, scale);

        let mse: f32 = vector
            .iter()
            .zip(restored.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            / 256.0;

        // MSE should be very small for L2-normalized vectors
        assert!(mse < 0.001, "MSE {mse} too high for L2-normalized vector");
    }
}
