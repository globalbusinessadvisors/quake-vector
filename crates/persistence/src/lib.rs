//! WAL, checkpoints, and recovery: write-ahead logging and periodic
//! checkpointing for durable state with crash-recovery guarantees.
//!
//! When the `rvf` feature is enabled, an `RvfAvailable` adapter provides
//! access to rvf-wire zero-copy serialization and rvf-crypto hashing.

mod wal_entry;
mod wal_writer;
mod wal_reader;
mod payload;
mod checkpoint;
mod crypto;
mod wear;
mod graph_serializer;
mod sona_serializer;
mod recovery;

#[cfg(feature = "rvf")]
pub mod rvf_adapter;

pub use wal_entry::{WalEntry, WalOpType};
pub use wal_writer::WalWriter;
pub use wal_reader::WalReader;
pub use payload::InsertNodePayload;
pub use checkpoint::{CheckpointManager, CheckpointManifest, CheckpointSlotInfo};
pub use crypto::CryptoService;
pub use wear::WearMonitor;
pub use graph_serializer::GraphSerializer;
pub use sona_serializer::{SonaSerializer, SonaState};
pub use recovery::RecoveryResult;

#[cfg(feature = "rvf")]
pub use rvf_adapter::RvfAvailable;

#[cfg(test)]
mod tests;
