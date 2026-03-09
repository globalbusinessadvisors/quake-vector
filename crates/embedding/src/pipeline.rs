//! EmbeddingPipeline: aggregate root that combines all feature extractors
//! into a unified 256-dimensional embedding.

use quake_vector_types::{SeismicEmbedding, WaveType, WaveformWindow};

use crate::arrival::ArrivalDetector;
use crate::envelope::EnvelopeExtractor;
use crate::fft_features::FftFeatureExtractor;
use crate::phase_coherence::PhaseCoherenceExtractor;
use crate::position::PositionEncoder;

/// Combines all feature extractors to produce 256-dim SeismicEmbedding vectors.
///
/// Dimension layout:
/// - [0..128]:   FFT spectral features
/// - [128..160]: Envelope features (3 scales)
/// - [160..176]: STA/LTA arrival features
/// - [176..208]: Phase coherence (autocorrelation)
/// - [208..240]: Geological priors
/// - [240..256]: Positional (time-of-day) encoding
pub struct EmbeddingPipeline {
    fft: FftFeatureExtractor,
    envelope: EnvelopeExtractor,
    arrival: ArrivalDetector,
    phase: PhaseCoherenceExtractor,
    geological_priors: [f32; 32],
    noise_floor: f32,
    noise_ema_alpha: f32,
}

impl EmbeddingPipeline {
    pub fn new() -> Self {
        Self {
            fft: FftFeatureExtractor::new(),
            envelope: EnvelopeExtractor::new(),
            arrival: ArrivalDetector::new(),
            phase: PhaseCoherenceExtractor::new(),
            geological_priors: [0.0; 32],
            noise_floor: 0.0,
            noise_ema_alpha: 0.01,
        }
    }

    /// Compute a 256-dimensional L2-normalized embedding from a waveform window.
    pub fn compute(&mut self, window: &WaveformWindow) -> SeismicEmbedding {
        let mut vector = [0.0f32; 256];

        // FFT features: 128 dims
        let fft_feat = self.fft.extract(&window.samples);
        vector[..128].copy_from_slice(&fft_feat);

        // Envelope features: 32 dims
        let env_feat = self.envelope.extract(&window.samples);
        vector[128..160].copy_from_slice(&env_feat);

        // Arrival features: 16 dims
        let arr_feat = self.arrival.extract(&window.samples);
        vector[160..176].copy_from_slice(&arr_feat);

        // Phase coherence: 32 dims
        let phase_feat = self.phase.extract(&window.samples);
        vector[176..208].copy_from_slice(&phase_feat);

        // Geological priors: 32 dims
        vector[208..240].copy_from_slice(&self.geological_priors);

        // Positional encoding: 16 dims
        let pos_feat = PositionEncoder::encode(window.timestamp_us);
        vector[240..256].copy_from_slice(&pos_feat);

        // L2 normalize
        let norm = l2_norm(&vector);
        if norm > 1e-8 {
            for v in vector.iter_mut() {
                *v /= norm;
            }
        }

        // Update noise floor EMA if this is an Unknown (quiet) window
        if window.wave_type == WaveType::Unknown {
            self.noise_floor = self.noise_ema_alpha * window.rms_amplitude
                + (1.0 - self.noise_ema_alpha) * self.noise_floor;
        }

        // Compute a simple hash of the source window
        let source_window_hash = compute_window_hash(&window.samples);

        SeismicEmbedding {
            vector,
            source_window_hash,
            norm,
        }
    }

    /// Update geological priors (called by SONA slow learner).
    pub fn update_geological_priors(&mut self, priors: [f32; 32]) {
        self.geological_priors = priors;
    }

    /// Get the current adaptive noise floor estimate.
    pub fn noise_floor(&self) -> f32 {
        self.noise_floor
    }
}

impl Default for EmbeddingPipeline {
    fn default() -> Self {
        Self::new()
    }
}

fn l2_norm(v: &[f32; 256]) -> f32 {
    v.iter().map(|&x| x * x).sum::<f32>().sqrt()
}

fn compute_window_hash(samples: &[i32; 256]) -> u64 {
    // Simple FNV-1a-like hash
    let mut hash: u64 = 0xcbf29ce484222325;
    for &s in samples.iter() {
        hash ^= s as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Compute cosine similarity between two 256-dim vectors.
pub fn cosine_similarity(a: &[f32; 256], b: &[f32; 256]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-8 || norm_b < 1e-8 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use quake_vector_types::ChannelId;
    use std::f64::consts::PI;

    fn make_sine_window(freq_hz: f64, amplitude: f64) -> WaveformWindow {
        let sample_rate = 100.0;
        let mut samples = [0i32; 256];
        for (i, s) in samples.iter_mut().enumerate() {
            *s = (amplitude * (2.0 * PI * freq_hz * i as f64 / sample_rate).sin()) as i32;
        }
        let rms = {
            let sum_sq: f64 = samples.iter().map(|&s| (s as f64) * (s as f64)).sum();
            (sum_sq / 256.0).sqrt() as f32
        };
        WaveformWindow {
            samples,
            channel: ChannelId(1),
            timestamp_us: 0,
            wave_type: WaveType::Unknown,
            rms_amplitude: rms,
            dominant_freq_hz: freq_hz as f32,
        }
    }

    #[test]
    fn output_is_l2_normalized() {
        let mut pipeline = EmbeddingPipeline::new();
        let window = make_sine_window(5.0, 1000.0);
        let embedding = pipeline.compute(&window);

        let norm = l2_norm(&embedding.vector);
        assert!(
            (norm - 1.0).abs() < 0.01,
            "embedding should be L2-normalized, norm = {norm}"
        );
    }

    #[test]
    fn p_wave_more_similar_to_p_wave_than_s_wave() {
        let mut pipeline = EmbeddingPipeline::new();

        let p1 = make_sine_window(5.0, 1000.0);
        let p2 = make_sine_window(5.5, 1000.0);
        let s1 = make_sine_window(2.0, 1000.0);

        let emb_p1 = pipeline.compute(&p1);
        let emb_p2 = pipeline.compute(&p2);
        let emb_s1 = pipeline.compute(&s1);

        let sim_pp = cosine_similarity(&emb_p1.vector, &emb_p2.vector);
        let sim_ps = cosine_similarity(&emb_p1.vector, &emb_s1.vector);

        assert!(
            sim_pp > sim_ps,
            "P-P similarity ({sim_pp}) should exceed P-S similarity ({sim_ps})"
        );
    }

    #[test]
    fn different_amplitudes_similar_shape() {
        let mut pipeline = EmbeddingPipeline::new();
        let w1 = make_sine_window(5.0, 1000.0);
        let w2 = make_sine_window(5.0, 2000.0);

        let emb1 = pipeline.compute(&w1);
        let emb2 = pipeline.compute(&w2);

        let sim = cosine_similarity(&emb1.vector, &emb2.vector);
        assert!(
            sim > 0.8,
            "same-frequency signals should be similar regardless of amplitude, got {sim}"
        );
    }

    #[test]
    fn geological_priors_affect_embedding() {
        let mut pipeline = EmbeddingPipeline::new();
        let window = make_sine_window(5.0, 1000.0);

        let emb1 = pipeline.compute(&window);

        pipeline.update_geological_priors([1.0; 32]);
        let emb2 = pipeline.compute(&window);

        // Embeddings should differ after updating priors
        let sim = cosine_similarity(&emb1.vector, &emb2.vector);
        assert!(
            sim < 0.999,
            "geological priors should change the embedding, cosine sim = {sim}"
        );
    }
}
