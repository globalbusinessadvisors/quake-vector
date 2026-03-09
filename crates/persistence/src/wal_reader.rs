//! WAL reader: reads entries with crash-boundary detection.

use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;

use crate::wal_entry::{WalEntry, WalOpType, WAL_HEADER_SIZE, WAL_TRAILER_SIZE};

/// Reads WAL entries, stopping at the first invalid or truncated entry.
pub struct WalReader {
    file: File,
    file_len: u64,
    /// Byte offset just past the last valid entry read.
    valid_end: u64,
}

impl WalReader {
    pub fn new(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let file_len = file.metadata()?.len();
        Ok(Self {
            file,
            file_len,
            valid_end: 0,
        })
    }

    /// Read all valid entries starting from `start_sequence`.
    ///
    /// Stops at the first entry with an invalid CRC32 (crash boundary),
    /// truncated entry, or zero-filled region.
    pub fn read_from(&mut self, start_sequence: u64) -> io::Result<Vec<WalEntry>> {
        self.file.seek(SeekFrom::Start(0))?;
        let mut entries = Vec::new();
        let mut offset: u64 = 0;

        loop {
            // Check if enough space for a header
            if offset + WAL_HEADER_SIZE as u64 > self.file_len {
                break;
            }

            // Read header
            let mut header = [0u8; WAL_HEADER_SIZE];
            self.file.seek(SeekFrom::Start(offset))?;
            if self.file.read_exact(&mut header).is_err() {
                break;
            }

            let seq = u64::from_le_bytes(header[0..8].try_into().unwrap());
            let op_byte = header[8];
            let payload_len = u32::from_le_bytes(header[9..13].try_into().unwrap());

            // Detect zero-filled pre-allocated region
            if seq == 0 && op_byte == 0 && payload_len == 0 {
                // Could be a valid entry with all zeros, but more likely unused space.
                // Check if the rest is also zero.
                if offset > 0 {
                    break;
                }
                // At offset 0, seq=0 is valid — check op_type
                if WalOpType::from_u8(op_byte).is_none() {
                    break;
                }
            }

            // Validate op_type
            let Some(op_type) = WalOpType::from_u8(op_byte) else {
                break;
            };

            let entry_size = WAL_HEADER_SIZE as u64 + payload_len as u64 + WAL_TRAILER_SIZE as u64;
            if offset + entry_size > self.file_len {
                // Truncated entry
                break;
            }

            // Read payload
            let mut payload = vec![0u8; payload_len as usize];
            if !payload.is_empty() {
                self.file.read_exact(&mut payload)?;
            }

            // Read CRC32
            let mut crc_bytes = [0u8; 4];
            self.file.read_exact(&mut crc_bytes)?;
            let stored_crc = u32::from_le_bytes(crc_bytes);

            // Verify CRC32
            let computed_crc = WalEntry::compute_crc32(seq, op_type, &payload);
            if stored_crc != computed_crc {
                break;
            }

            self.valid_end = offset + entry_size;
            offset = self.valid_end;

            if seq >= start_sequence {
                entries.push(WalEntry {
                    sequence_number: seq,
                    op_type,
                    payload,
                    crc32: stored_crc,
                });
            }
        }

        Ok(entries)
    }

    /// Returns the byte offset just past the last valid entry.
    pub fn valid_end_offset(&self) -> u64 {
        self.valid_end
    }
}
