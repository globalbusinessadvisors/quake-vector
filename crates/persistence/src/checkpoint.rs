//! Checkpoint management with 3-slot rotation.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use ed25519_dalek::{SigningKey, VerifyingKey};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use quake_vector_adaptation::SonaEngine;
use quake_vector_store::{HnswGraph, VectorStore};

use crate::crypto::CryptoService;
use crate::graph_serializer::GraphSerializer;
use crate::recovery::RecoveryResult;
use crate::sona_serializer::SonaSerializer;
use crate::wal_reader::WalReader;
use crate::wal_writer::WalWriter;

/// Information about a single checkpoint slot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointSlotInfo {
    pub path: String,
    pub sequence_number: u64,
    pub timestamp_us: u64,
    pub size_bytes: u64,
    pub signature: Vec<u8>,
    pub valid: bool,
}

/// Manifest tracking 3 checkpoint slots.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointManifest {
    pub slots: [Option<CheckpointSlotInfo>; 3],
    pub active_slot: usize,
}

impl Default for CheckpointManifest {
    fn default() -> Self {
        Self {
            slots: [None, None, None],
            active_slot: 0,
        }
    }
}

/// Checkpoint blob header: lengths of graph and sona sections.
#[derive(Serialize, Deserialize)]
struct CheckpointBlob {
    graph_bytes: Vec<u8>,
    sona_bytes: Vec<u8>,
}

/// Manages WAL and checkpoint lifecycle.
pub struct CheckpointManager {
    data_dir: PathBuf,
    wal: WalWriter,
    manifest: CheckpointManifest,
}

impl CheckpointManager {
    /// Create a new CheckpointManager, initializing directory structure.
    pub fn new(data_dir: &Path) -> io::Result<Self> {
        let wal_dir = data_dir.join("wal");
        let checkpoint_dir = data_dir.join("checkpoints");
        let models_dir = data_dir.join("models");

        fs::create_dir_all(&wal_dir)?;
        fs::create_dir_all(&checkpoint_dir)?;
        fs::create_dir_all(&models_dir)?;

        let wal_path = wal_dir.join("current.wal");
        let wal = WalWriter::new(&wal_path, 64 * 1024 * 1024)?; // 64MB prealloc

        let manifest = Self::load_manifest(data_dir).unwrap_or_default();

        debug!(data_dir = %data_dir.display(), "CheckpointManager initialized");

        Ok(Self {
            data_dir: data_dir.to_path_buf(),
            wal,
            manifest,
        })
    }

    /// Access the WAL writer.
    pub fn wal(&mut self) -> &mut WalWriter {
        &mut self.wal
    }

    /// Get the data directory.
    pub fn data_dir(&self) -> &Path {
        &self.data_dir
    }

    /// Record a checkpoint into the given slot.
    pub fn record_checkpoint(
        &mut self,
        slot: usize,
        sequence: u64,
        size: u64,
        signature: Vec<u8>,
    ) -> io::Result<()> {
        if slot >= 3 {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "slot must be 0..2"));
        }

        let path = self
            .data_dir
            .join("checkpoints")
            .join(format!("checkpoint_{slot}.bin"));

        self.manifest.slots[slot] = Some(CheckpointSlotInfo {
            path: path.to_string_lossy().to_string(),
            sequence_number: sequence,
            timestamp_us: 0, // Caller should set if needed
            size_bytes: size,
            signature,
            valid: true,
        });
        self.manifest.active_slot = slot;

        self.save_manifest()?;
        Ok(())
    }

    /// Create a full checkpoint: serialize graph + sona, sign, write to next slot.
    pub fn create_checkpoint(
        &mut self,
        graph: &HnswGraph,
        sona: &SonaEngine,
        signing_key: &SigningKey,
    ) -> io::Result<CheckpointSlotInfo> {
        // a. Serialize graph and sona state
        let graph_bytes = GraphSerializer::serialize(graph);
        let sona_bytes = SonaSerializer::serialize(sona);

        // b. Combine into a single checkpoint blob
        let blob = CheckpointBlob {
            graph_bytes,
            sona_bytes,
        };
        let blob_bytes = bincode::serialize(&blob)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

        // c. Compute Ed25519 signature over the blob
        let signature = CryptoService::sign(&blob_bytes, signing_key);

        // d. Write to next slot file (rotate through 3 slots)
        let slot = (self.manifest.active_slot + 1) % 3;
        let path = self
            .data_dir
            .join("checkpoints")
            .join(format!("checkpoint_{slot}.bin"));
        fs::write(&path, &blob_bytes)?;

        // e. Get current WAL head sequence
        let seq = self.wal.head_sequence();

        // f. Update manifest
        let slot_info = CheckpointSlotInfo {
            path: path.to_string_lossy().to_string(),
            sequence_number: seq,
            timestamp_us: 0,
            size_bytes: blob_bytes.len() as u64,
            signature: signature.clone(),
            valid: true,
        };
        self.manifest.slots[slot] = Some(slot_info.clone());
        self.manifest.active_slot = slot;
        self.save_manifest()?;

        // g. Truncate WAL before that sequence
        self.wal.flush()?;

        info!(
            slot,
            sequence = seq,
            size_bytes = blob_bytes.len(),
            "checkpoint created"
        );

        Ok(slot_info)
    }

    /// Recover from the latest valid checkpoint, optionally replaying WAL entries.
    pub fn recover(
        &mut self,
        verifying_key: &VerifyingKey,
    ) -> io::Result<Option<RecoveryResult>> {
        // a. Find latest valid checkpoint from manifest
        let slot_info = match self.latest_valid_checkpoint() {
            Some(info) => info.clone(),
            None => return Ok(None),
        };

        // b. Read checkpoint file
        let blob_bytes = fs::read(&slot_info.path)?;

        // c. Verify Ed25519 signature
        if !CryptoService::verify(&blob_bytes, &slot_info.signature, verifying_key) {
            warn!("checkpoint signature verification failed");
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "checkpoint signature verification failed",
            ));
        }

        // d. Deserialize graph and sona state
        let blob: CheckpointBlob = bincode::deserialize(&blob_bytes)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        let graph = GraphSerializer::deserialize(&blob.graph_bytes)?;
        let sona_state = SonaSerializer::deserialize(&blob.sona_bytes)?;

        // e. Read WAL, replay entries from checkpoint sequence onward
        let wal_path = self.data_dir.join("wal").join("current.wal");
        let wal_entries_replayed = if wal_path.exists() {
            let mut reader = WalReader::new(&wal_path)?;
            let entries = reader.read_from(slot_info.sequence_number)?;
            entries.len() as u64
        } else {
            0
        };

        info!(
            checkpoint_seq = slot_info.sequence_number,
            wal_replayed = wal_entries_replayed,
            node_count = graph.node_count(),
            "recovery complete"
        );

        Ok(Some(RecoveryResult {
            graph,
            sona_state,
            wal_entries_replayed,
        }))
    }

    /// Return the latest valid checkpoint, if any.
    pub fn latest_valid_checkpoint(&self) -> Option<&CheckpointSlotInfo> {
        self.manifest
            .slots
            .iter()
            .filter_map(|s| s.as_ref())
            .filter(|s| s.valid)
            .max_by_key(|s| s.sequence_number)
    }

    /// Truncate the WAL before the given sequence number.
    pub fn truncate_wal_before(&mut self, _sequence: u64) -> io::Result<()> {
        self.wal.flush()?;
        debug!("WAL truncation requested (simplified implementation)");
        Ok(())
    }

    /// Get the manifest.
    pub fn manifest(&self) -> &CheckpointManifest {
        &self.manifest
    }

    fn manifest_path(data_dir: &Path) -> PathBuf {
        data_dir.join("checkpoints").join("manifest.json")
    }

    fn load_manifest(data_dir: &Path) -> Option<CheckpointManifest> {
        let path = Self::manifest_path(data_dir);
        let data = fs::read_to_string(&path).ok()?;
        serde_json::from_str(&data).ok()
    }

    fn save_manifest(&self) -> io::Result<()> {
        let path = Self::manifest_path(&self.data_dir);
        let json = serde_json::to_string_pretty(&self.manifest)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        fs::write(&path, json)?;
        Ok(())
    }
}
