//! Ed25519 signing and verification.

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use rand::rngs::OsRng;
use std::fs;
use std::io;
use std::path::Path;

/// Ed25519 cryptographic signing service.
pub struct CryptoService;

impl CryptoService {
    /// Generate a new Ed25519 keypair.
    pub fn generate_keypair() -> (SigningKey, VerifyingKey) {
        let signing_key = SigningKey::generate(&mut OsRng);
        let verifying_key = signing_key.verifying_key();
        (signing_key, verifying_key)
    }

    /// Sign data with the given key.
    pub fn sign(data: &[u8], key: &SigningKey) -> Vec<u8> {
        let sig = key.sign(data);
        sig.to_bytes().to_vec()
    }

    /// Verify a signature against data and a verifying key.
    pub fn verify(data: &[u8], signature: &[u8], key: &VerifyingKey) -> bool {
        if signature.len() != 64 {
            return false;
        }
        let sig_bytes: [u8; 64] = signature.try_into().unwrap();
        let sig = Signature::from_bytes(&sig_bytes);
        key.verify(data, &sig).is_ok()
    }

    /// Load a keypair from disk, or generate and save a new one.
    pub fn load_or_create_keypair(path: &Path) -> io::Result<(SigningKey, VerifyingKey)> {
        if path.exists() {
            let data = fs::read(path)?;
            if data.len() != 32 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "invalid key file size",
                ));
            }
            let key_bytes: [u8; 32] = data.try_into().unwrap();
            let signing_key = SigningKey::from_bytes(&key_bytes);
            let verifying_key = signing_key.verifying_key();
            Ok((signing_key, verifying_key))
        } else {
            let (signing_key, verifying_key) = Self::generate_keypair();
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(path, signing_key.to_bytes())?;
            Ok((signing_key, verifying_key))
        }
    }
}
