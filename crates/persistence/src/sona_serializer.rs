//! Serialize and deserialize SONA engine state via bincode.

use std::io;
use serde::{Deserialize, Serialize};
use quake_vector_adaptation::SonaEngine;

/// Serializable snapshot of SonaEngine state needed for recovery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SonaState {
    /// The serialized SonaEngine bytes (bincode).
    engine_bytes: Vec<u8>,
}

impl SonaState {
    /// Restore a SonaEngine from this state.
    pub fn into_engine(self) -> io::Result<SonaEngine> {
        bincode::deserialize(&self.engine_bytes)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }
}

/// Serializes and deserializes SONA engine state.
pub struct SonaSerializer;

impl SonaSerializer {
    /// Serialize the SonaEngine state to bytes.
    pub fn serialize(sona: &SonaEngine) -> Vec<u8> {
        let engine_bytes = bincode::serialize(sona)
            .expect("sona serialization should not fail");
        let state = SonaState { engine_bytes };
        bincode::serialize(&state).expect("sona state serialization should not fail")
    }

    /// Deserialize SonaState from bytes.
    pub fn deserialize(data: &[u8]) -> io::Result<SonaState> {
        bincode::deserialize(data)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }
}
