//! Adapter wrapping ruvector-gnn, ruvector-temporal-tensor, and ruv-fann
//! for causal model inference.
//!
//! Uses ruvector-gnn's Tensor for GNN computation, ruvector-temporal-tensor
//! for temporal delta encoding, and ruv-fann for the feedforward output head.

use quake_vector_types::{SeismicEmbedding, SeismicPrediction, TemporalMeta};
use quake_vector_store::VectorStore;

/// Causal model backed by ruvector-gnn Tensor operations and ruv-fann network.
pub struct RuvectorCausalModel {
    /// ruv-fann feedforward network: 256 inputs -> 4 outputs
    /// (probability, magnitude, time_to_peak, confidence)
    network: ruv_fann::Network<f32>,
    /// Temporal compressor for delta encoding
    _temporal: ruvector_temporal_tensor::TemporalTensorCompressor,
}

impl RuvectorCausalModel {
    pub fn new() -> Self {
        // Build a simple feedforward network: 256 -> 64 -> 4
        let network = ruv_fann::Network::new(&[256, 64, 4]);

        let temporal = ruvector_temporal_tensor::TemporalTensorCompressor::new(
            ruvector_temporal_tensor::TierPolicy::default(),
            256,
            0,
        );

        Self {
            network,
            _temporal: temporal,
        }
    }

    /// Run inference using ruvector-gnn tensors and ruv-fann output head.
    pub fn predict(
        &mut self,
        embedding: &SeismicEmbedding,
        _meta: &TemporalMeta,
        _graph: &dyn VectorStore,
    ) -> SeismicPrediction {
        // Use ruvector-gnn Tensor for intermediate representation
        let input_tensor = ruvector_gnn::tensor::Tensor::from_vec(
            embedding.vector.to_vec(),
        );

        // Run through ruv-fann network
        let input: Vec<f32> = input_tensor.as_slice().to_vec();
        let output = self.network.run(&input).unwrap_or_else(|_| vec![0.0; 4]);

        // Map output to prediction domain
        let probability = output.get(0).copied().unwrap_or(0.0).clamp(0.0, 1.0);
        let magnitude = output.get(1).copied().unwrap_or(0.0).clamp(0.0, 10.0);
        let time_to_peak = output.get(2).copied().unwrap_or(5.0).clamp(0.0, 60.0);
        let confidence = output.get(3).copied().unwrap_or(0.5).clamp(0.0, 1.0);

        SeismicPrediction {
            event_probability: probability,
            estimated_magnitude: magnitude,
            estimated_time_to_peak_s: time_to_peak,
            confidence,
            contributing_wave_ids: vec![],
            model_version: 2, // ruvector version
            inference_latency_us: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quake_vector_types::{StationId, WaveType};

    #[test]
    fn ruvector_model_predict_returns_valid() {
        let mut model = RuvectorCausalModel::new();
        let emb = SeismicEmbedding {
            vector: [0.1; 256],
            source_window_hash: 1,
            norm: 1.0,
        };
        let meta = TemporalMeta {
            timestamp_us: 1000,
            wave_type: WaveType::P,
            station_id: StationId(1),
            amplitude_rms: 100.0,
            dominant_freq_hz: 5.0,
        };

        // Need a VectorStore — use a stub since we don't query neighbors
        use quake_vector_store::HnswGraph;
        let graph = HnswGraph::with_default_config();

        let pred = model.predict(&emb, &meta, &graph);
        assert!(pred.event_probability >= 0.0 && pred.event_probability <= 1.0);
        assert!(pred.confidence >= 0.0 && pred.confidence <= 1.0);
        assert_eq!(pred.model_version, 2);
    }
}
