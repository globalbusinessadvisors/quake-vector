//! Shared domain types for QuakeVector seismic detection system.
//!
//! Defines the ubiquitous language used across all crates: identifiers,
//! waveform structures, embeddings, predictions, alerts, and system enums.

use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Newtype identifiers
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StationId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SensorId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ChannelId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AlertId(pub Uuid);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MessageId(pub Uuid);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ClaimId(pub Uuid);

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WaveType {
    P,
    S,
    Surface,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AlertLevel {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SensorStatus {
    Active,
    Degraded,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Tier {
    Hot,
    Warm,
    Cold,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BootStage {
    ImageVerify,
    SegmentLoad,
    GraphInit,
    ModelLoad,
    SonaInit,
    SensorDiscover,
    ThreadSpawn,
    Operational,
}

// ---------------------------------------------------------------------------
// Domain structs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TemporalMeta {
    pub timestamp_us: u64,
    pub wave_type: WaveType,
    pub station_id: StationId,
    pub amplitude_rms: f32,
    pub dominant_freq_hz: f32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SeismicEmbedding {
    #[serde(with = "BigArray")]
    pub vector: [f32; 256],
    pub source_window_hash: u64,
    pub norm: f32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct WaveformWindow {
    #[serde(with = "BigArray")]
    pub samples: [i32; 256],
    pub channel: ChannelId,
    pub timestamp_us: u64,
    pub wave_type: WaveType,
    pub rms_amplitude: f32,
    pub dominant_freq_hz: f32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TimingFeatures {
    pub delta_us: i64,
    pub transition: (WaveType, WaveType),
    pub amplitude_ratio: f32,
    pub freq_shift_hz: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeismicPrediction {
    pub event_probability: f32,
    pub estimated_magnitude: f32,
    pub estimated_time_to_peak_s: f32,
    pub confidence: f32,
    pub contributing_wave_ids: Vec<NodeId>,
    pub model_version: u64,
    pub inference_latency_us: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub alert_id: AlertId,
    pub station_id: StationId,
    pub timestamp: u64,
    pub level: AlertLevel,
    pub probability: f32,
    pub magnitude_estimate: f32,
    pub time_to_peak_s: f32,
    pub confidence: f32,
    pub wave_evidence: Vec<NodeId>,
    pub consensus_stations: Vec<StationId>,
    pub signature: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub neighbors: Vec<(NodeId, f32)>,
    pub search_ef: u16,
    pub nodes_visited: u32,
    pub latency_us: u32,
}
