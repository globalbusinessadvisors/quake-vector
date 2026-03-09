//! Waveform-to-vector pipeline: transforms WaveformWindow data into
//! SeismicEmbedding vectors for similarity search and graph construction.

mod fft_features;
mod envelope;
mod arrival;
mod position;
mod phase_coherence;
mod pipeline;
mod quantization;

pub use fft_features::FftFeatureExtractor;
pub use envelope::EnvelopeExtractor;
pub use arrival::ArrivalDetector;
pub use position::PositionEncoder;
pub use phase_coherence::PhaseCoherenceExtractor;
pub use pipeline::EmbeddingPipeline;
pub use quantization::QuantizationService;
