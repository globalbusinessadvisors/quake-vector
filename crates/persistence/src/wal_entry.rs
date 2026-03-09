//! WAL entry types and wire format.

use serde::{Deserialize, Serialize};

/// Operation types recorded in the WAL.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum WalOpType {
    InsertNode = 0,
    UpdateEdges = 1,
    DeleteNode = 2,
}

impl WalOpType {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::InsertNode),
            1 => Some(Self::UpdateEdges),
            2 => Some(Self::DeleteNode),
            _ => None,
        }
    }
}

/// A single WAL entry.
///
/// Wire format:
/// ```text
/// [seq_num: 8 bytes LE][op_type: 1 byte][payload_len: 4 bytes LE]
/// [payload: N bytes][crc32: 4 bytes LE]
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct WalEntry {
    pub sequence_number: u64,
    pub op_type: WalOpType,
    pub payload: Vec<u8>,
    pub crc32: u32,
}

/// Header size: 8 (seq) + 1 (op) + 4 (len) = 13 bytes.
pub const WAL_HEADER_SIZE: usize = 13;
/// Trailer size: 4 (crc32).
pub const WAL_TRAILER_SIZE: usize = 4;

impl WalEntry {
    /// Total size of this entry on disk.
    pub fn wire_size(&self) -> usize {
        WAL_HEADER_SIZE + self.payload.len() + WAL_TRAILER_SIZE
    }

    /// Compute CRC32 over (seq_num + op_type + payload).
    pub fn compute_crc32(sequence_number: u64, op_type: WalOpType, payload: &[u8]) -> u32 {
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&sequence_number.to_le_bytes());
        hasher.update(&[op_type as u8]);
        hasher.update(payload);
        hasher.finalize()
    }
}
