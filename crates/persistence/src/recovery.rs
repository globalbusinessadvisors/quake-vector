//! Recovery result types.

use quake_vector_store::HnswGraph;
use crate::sona_serializer::SonaState;

/// Result of a successful crash recovery.
pub struct RecoveryResult {
    pub graph: HnswGraph,
    pub sona_state: SonaState,
    pub wal_entries_replayed: u64,
}
