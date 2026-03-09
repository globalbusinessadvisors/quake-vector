//! Ground truth labeling via heuristic wave sequence analysis.

use quake_vector_types::{SeismicPrediction, TemporalMeta, WaveType};

/// A ground truth label for a prediction.
#[derive(Debug, Clone)]
pub struct GroundTruth {
    pub event_occurred: bool,
    pub actual_magnitude: Option<f32>,
    pub labeled_at: u64,
}

/// Heuristic labeler that determines ground truth from subsequent wave observations.
pub struct GroundTruthLabeler;

impl GroundTruthLabeler {
    /// Label a prediction based on subsequent wave observations.
    ///
    /// Heuristic: if a P-wave prediction (probability > 0.3) is followed by:
    /// - An S-wave within 1–10 seconds, AND
    /// - A Surface wave within 5–30 seconds
    /// then label as event_occurred = true.
    ///
    /// Magnitude is estimated from the maximum RMS amplitude (log scale).
    ///
    /// `subsequent_waves` contains (TemporalMeta, timestamp_us) pairs observed
    /// after the prediction.
    pub fn label(
        prediction: &SeismicPrediction,
        subsequent_waves: &[(TemporalMeta, u64)],
    ) -> Option<GroundTruth> {
        if prediction.event_probability <= 0.3 {
            // Low-probability prediction: label as no event
            return Some(GroundTruth {
                event_occurred: false,
                actual_magnitude: None,
                labeled_at: subsequent_waves
                    .last()
                    .map(|(_, t)| *t)
                    .unwrap_or(0),
            });
        }

        if subsequent_waves.is_empty() {
            return None; // Not enough data to label
        }

        let prediction_time = subsequent_waves
            .first()
            .map(|(_, t)| *t)
            .unwrap_or(0);

        let mut saw_s_wave = false;
        let mut saw_surface_wave = false;
        let mut max_rms: f32 = 0.0;

        for (meta, ts) in subsequent_waves {
            let delta_s = (*ts as f64 - prediction_time as f64) / 1_000_000.0;

            match meta.wave_type {
                WaveType::S if (1.0..=10.0).contains(&delta_s) => {
                    saw_s_wave = true;
                }
                WaveType::Surface if (5.0..=30.0).contains(&delta_s) => {
                    saw_surface_wave = true;
                }
                _ => {}
            }

            if meta.amplitude_rms > max_rms {
                max_rms = meta.amplitude_rms;
            }
        }

        let event_occurred = saw_s_wave && saw_surface_wave;

        let actual_magnitude = if event_occurred && max_rms > 0.0 {
            // Simple log-scale magnitude estimate
            Some((max_rms as f64).log10() as f32)
        } else {
            None
        };

        let labeled_at = subsequent_waves
            .last()
            .map(|(_, t)| *t)
            .unwrap_or(0);

        Some(GroundTruth {
            event_occurred,
            actual_magnitude,
            labeled_at,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quake_vector_types::{NodeId, StationId};

    fn make_prediction(prob: f32) -> SeismicPrediction {
        SeismicPrediction {
            event_probability: prob,
            estimated_magnitude: 3.0,
            estimated_time_to_peak_s: 5.0,
            confidence: 0.8,
            contributing_wave_ids: vec![NodeId(1)],
            model_version: 1,
            inference_latency_us: 100,
        }
    }

    fn make_meta(wave_type: WaveType, rms: f32) -> TemporalMeta {
        TemporalMeta {
            timestamp_us: 0,
            wave_type,
            station_id: StationId(1),
            amplitude_rms: rms,
            dominant_freq_hz: 5.0,
        }
    }

    #[test]
    fn p_s_surface_sequence_labels_as_event() {
        let pred = make_prediction(0.7);
        let base_time: u64 = 1_000_000_000; // 1000s in us

        let waves = vec![
            (make_meta(WaveType::P, 100.0), base_time),
            (make_meta(WaveType::S, 500.0), base_time + 3_000_000),       // +3s
            (make_meta(WaveType::Surface, 800.0), base_time + 10_000_000), // +10s
        ];

        let gt = GroundTruthLabeler::label(&pred, &waves).unwrap();
        assert!(gt.event_occurred);
        assert!(gt.actual_magnitude.is_some());
        assert!(gt.actual_magnitude.unwrap() > 0.0);
    }

    #[test]
    fn isolated_p_wave_no_event() {
        let pred = make_prediction(0.7);
        let base_time: u64 = 1_000_000_000;

        let waves = vec![
            (make_meta(WaveType::P, 100.0), base_time),
            (make_meta(WaveType::P, 90.0), base_time + 2_000_000), // another P, no S/Surface
        ];

        let gt = GroundTruthLabeler::label(&pred, &waves).unwrap();
        assert!(!gt.event_occurred);
        assert!(gt.actual_magnitude.is_none());
    }

    #[test]
    fn low_probability_labels_no_event() {
        let pred = make_prediction(0.1);
        let waves = vec![
            (make_meta(WaveType::S, 500.0), 1_000_000),
        ];

        let gt = GroundTruthLabeler::label(&pred, &waves).unwrap();
        assert!(!gt.event_occurred);
    }

    #[test]
    fn s_wave_outside_timing_window() {
        let pred = make_prediction(0.7);
        let base_time: u64 = 1_000_000_000;

        // S-wave too late (15s instead of 1-10s)
        let waves = vec![
            (make_meta(WaveType::P, 100.0), base_time),
            (make_meta(WaveType::S, 500.0), base_time + 15_000_000),
            (make_meta(WaveType::Surface, 800.0), base_time + 20_000_000),
        ];

        let gt = GroundTruthLabeler::label(&pred, &waves).unwrap();
        assert!(!gt.event_occurred); // S-wave outside window
    }
}
