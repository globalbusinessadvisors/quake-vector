//! Adapter for rvf-wire, rvf-crypto, and rvf-runtime integration.
//!
//! Provides alternative serialization via rvf-wire's zero-copy segment format
//! and cryptographic operations via rvf-crypto.

/// Marker type indicating RVF ecosystem is available.
pub struct RvfAvailable;

impl RvfAvailable {
    /// Check if rvf-types basic types are accessible.
    pub fn check() -> bool {
        // Verify we can access rvf-types enums
        let _seg_type = rvf_types::SegmentType::Vec;
        let _data_type = rvf_types::DataType::F32;
        true
    }

    /// Hash data using rvf-crypto's SHAKE-256.
    pub fn shake256_hash(data: &[u8]) -> Vec<u8> {
        rvf_crypto::shake256_256(data).to_vec()
    }

    /// Create an RvfStore at the given path (if rvf-runtime is available).
    pub fn create_store(
        path: &std::path::Path,
        dimensions: u16,
    ) -> Result<rvf_runtime::RvfStore, String> {
        let opts = rvf_runtime::RvfOptions {
            dimension: dimensions,
            ..Default::default()
        };
        rvf_runtime::RvfStore::create(path, opts).map_err(|e| format!("{e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rvf_types_accessible() {
        assert!(RvfAvailable::check());
    }

    #[test]
    fn rvf_crypto_hash() {
        let hash = RvfAvailable::shake256_hash(b"test data");
        assert_eq!(hash.len(), 32);
        // Should be deterministic
        let hash2 = RvfAvailable::shake256_hash(b"test data");
        assert_eq!(hash, hash2);
    }
}
