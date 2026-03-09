//! WAL writer: append-only log with CRC32 integrity.

use std::fs::{File, OpenOptions};
use std::io::{self, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use tracing::debug;

use crate::wal_entry::{WalEntry, WalOpType};
use crate::wal_reader::WalReader;

/// Append-only WAL writer with CRC32 integrity and optional pre-allocation.
pub struct WalWriter {
    file: File,
    path: PathBuf,
    sequence: u64,
}

impl WalWriter {
    /// Open or create a WAL file. Pre-allocates to `preallocate_bytes` on creation.
    /// Seeks to end of valid entries on open.
    pub fn new(path: &Path, preallocate_bytes: u64) -> io::Result<Self> {
        let exists = path.exists();

        let mut file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open(path)?;

        let mut sequence = 0u64;
        let mut write_pos = 0u64;

        if exists {
            // Read existing entries to find the end and current sequence
            let mut reader = WalReader::new(path)?;
            let entries = reader.read_from(0)?;
            if let Some(last) = entries.last() {
                sequence = last.sequence_number + 1;
                write_pos = reader.valid_end_offset();
            }
        } else if preallocate_bytes > 0 {
            // Pre-allocate with zeros
            file.set_len(preallocate_bytes)?;
        }

        file.seek(SeekFrom::Start(write_pos))?;

        debug!(path = %path.display(), sequence, write_pos, "WAL opened");

        Ok(Self {
            file,
            path: path.to_path_buf(),
            sequence,
        })
    }

    /// Append an entry to the WAL. Returns the sequence number of the written entry.
    pub fn append(&mut self, op_type: WalOpType, payload: &[u8]) -> io::Result<u64> {
        let seq = self.sequence;
        let crc = WalEntry::compute_crc32(seq, op_type, payload);
        let payload_len = payload.len() as u32;

        // Write header: seq(8) + op(1) + len(4)
        self.file.write_all(&seq.to_le_bytes())?;
        self.file.write_all(&[op_type as u8])?;
        self.file.write_all(&payload_len.to_le_bytes())?;
        // Write payload
        self.file.write_all(payload)?;
        // Write CRC32
        self.file.write_all(&crc.to_le_bytes())?;

        self.sequence += 1;
        Ok(seq)
    }

    /// Flush the WAL to stable storage.
    pub fn flush(&mut self) -> io::Result<()> {
        self.file.flush()?;
        self.file.sync_data()
    }

    /// Current head sequence number (next to be written).
    pub fn head_sequence(&self) -> u64 {
        self.sequence
    }

    /// Path to the WAL file.
    pub fn path(&self) -> &Path {
        &self.path
    }
}
