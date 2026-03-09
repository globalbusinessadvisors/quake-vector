//! Integration tests for the persistence crate.

#[cfg(test)]
mod wal_tests {
    use crate::wal_entry::{WalOpType, WAL_HEADER_SIZE, WAL_TRAILER_SIZE};
    use crate::wal_reader::WalReader;
    use crate::wal_writer::WalWriter;
    use std::fs::{self, OpenOptions};
    use std::io::{Seek, SeekFrom, Write};

    fn temp_dir() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "qv_wal_test_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn round_trip_100_entries() {
        let dir = temp_dir();
        let wal_path = dir.join("test.wal");

        // Write 100 entries
        {
            let mut writer = WalWriter::new(&wal_path, 0).unwrap();
            for i in 0u64..100 {
                let payload = format!("payload_{i}").into_bytes();
                let seq = writer.append(WalOpType::InsertNode, &payload).unwrap();
                assert_eq!(seq, i);
            }
            writer.flush().unwrap();
            assert_eq!(writer.head_sequence(), 100);
        }

        // Read them back
        {
            let mut reader = WalReader::new(&wal_path).unwrap();
            let entries = reader.read_from(0).unwrap();
            assert_eq!(entries.len(), 100);
            for (i, entry) in entries.iter().enumerate() {
                assert_eq!(entry.sequence_number, i as u64);
                assert_eq!(entry.op_type, WalOpType::InsertNode);
                let expected = format!("payload_{i}").into_bytes();
                assert_eq!(entry.payload, expected);
            }
        }

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn crash_simulation_truncated_entry() {
        let dir = temp_dir();
        let wal_path = dir.join("crash.wal");

        // Write 50 entries
        {
            let mut writer = WalWriter::new(&wal_path, 0).unwrap();
            for i in 0..50 {
                let payload = format!("data_{i}").into_bytes();
                writer.append(WalOpType::InsertNode, &payload).unwrap();
            }
            writer.flush().unwrap();
        }

        // Determine the file size, then append a partial entry (truncated)
        {
            let mut file = OpenOptions::new().append(true).open(&wal_path).unwrap();
            // Write a partial header (only 5 bytes of a 13-byte header)
            file.write_all(&[1, 2, 3, 4, 5]).unwrap();
            file.flush().unwrap();
        }

        // Reader should recover exactly 50 valid entries
        {
            let mut reader = WalReader::new(&wal_path).unwrap();
            let entries = reader.read_from(0).unwrap();
            assert_eq!(
                entries.len(),
                50,
                "should recover 50 entries, got {}",
                entries.len()
            );
            assert_eq!(entries.last().unwrap().sequence_number, 49);
        }

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn crc32_corruption_detection() {
        let dir = temp_dir();
        let wal_path = dir.join("corrupt.wal");

        // Write 10 entries
        {
            let mut writer = WalWriter::new(&wal_path, 0).unwrap();
            for i in 0..10 {
                let payload = format!("entry_{i}").into_bytes();
                writer.append(WalOpType::InsertNode, &payload).unwrap();
            }
            writer.flush().unwrap();
        }

        // Corrupt one byte in the 5th entry's payload
        {
            let mut file = OpenOptions::new().read(true).write(true).open(&wal_path).unwrap();

            // Calculate offset to the 5th entry (index 4)
            // Each entry: header(13) + payload_len + crc(4)
            let mut offset = 0u64;
            for i in 0..4 {
                let payload = format!("entry_{i}").into_bytes();
                offset += (WAL_HEADER_SIZE + payload.len() + WAL_TRAILER_SIZE) as u64;
            }
            // Now corrupt a byte in the 5th entry's payload area
            let corrupt_pos = offset + WAL_HEADER_SIZE as u64 + 1;
            file.seek(SeekFrom::Start(corrupt_pos)).unwrap();
            file.write_all(&[0xFF]).unwrap();
            file.flush().unwrap();
        }

        // Reader should stop at entry 4 (indices 0..3 = 4 entries)
        {
            let mut reader = WalReader::new(&wal_path).unwrap();
            let entries = reader.read_from(0).unwrap();
            assert_eq!(
                entries.len(),
                4,
                "should detect corruption and stop at entry 4, got {}",
                entries.len()
            );
        }

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn preallocated_wal_zero_fill() {
        let dir = temp_dir();
        let wal_path = dir.join("prealloc.wal");

        // Create with 1KB preallocate
        {
            let mut writer = WalWriter::new(&wal_path, 1024).unwrap();
            writer.append(WalOpType::InsertNode, b"hello").unwrap();
            writer.append(WalOpType::UpdateEdges, b"world").unwrap();
            writer.flush().unwrap();
        }

        // File should be >= 1024 bytes
        let meta = fs::metadata(&wal_path).unwrap();
        assert!(meta.len() >= 1024);

        // Should still read only 2 valid entries
        {
            let mut reader = WalReader::new(&wal_path).unwrap();
            let entries = reader.read_from(0).unwrap();
            assert_eq!(entries.len(), 2);
        }

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn read_from_specific_sequence() {
        let dir = temp_dir();
        let wal_path = dir.join("seq.wal");

        {
            let mut writer = WalWriter::new(&wal_path, 0).unwrap();
            for i in 0..20 {
                writer
                    .append(WalOpType::InsertNode, format!("d{i}").as_bytes())
                    .unwrap();
            }
            writer.flush().unwrap();
        }

        {
            let mut reader = WalReader::new(&wal_path).unwrap();
            let entries = reader.read_from(15).unwrap();
            assert_eq!(entries.len(), 5);
            assert_eq!(entries[0].sequence_number, 15);
            assert_eq!(entries[4].sequence_number, 19);
        }

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn reopen_wal_continues_sequence() {
        let dir = temp_dir();
        let wal_path = dir.join("reopen.wal");

        {
            let mut writer = WalWriter::new(&wal_path, 0).unwrap();
            for i in 0..5 {
                writer
                    .append(WalOpType::InsertNode, format!("a{i}").as_bytes())
                    .unwrap();
            }
            writer.flush().unwrap();
        }

        // Reopen and append more
        {
            let mut writer = WalWriter::new(&wal_path, 0).unwrap();
            assert_eq!(writer.head_sequence(), 5);
            for i in 0..5 {
                writer
                    .append(WalOpType::DeleteNode, format!("b{i}").as_bytes())
                    .unwrap();
            }
            writer.flush().unwrap();
        }

        {
            let mut reader = WalReader::new(&wal_path).unwrap();
            let entries = reader.read_from(0).unwrap();
            assert_eq!(entries.len(), 10);
            assert_eq!(entries[4].sequence_number, 4);
            assert_eq!(entries[5].sequence_number, 5);
            assert_eq!(entries[5].op_type, WalOpType::DeleteNode);
        }

        fs::remove_dir_all(&dir).ok();
    }
}

#[cfg(test)]
mod checkpoint_tests {
    use crate::checkpoint::{CheckpointManifest, CheckpointSlotInfo};

    #[test]
    fn manifest_serialize_deserialize() {
        let mut manifest = CheckpointManifest::default();
        manifest.slots[0] = Some(CheckpointSlotInfo {
            path: "/data/checkpoints/checkpoint_0.bin".to_string(),
            sequence_number: 42,
            timestamp_us: 1000000,
            size_bytes: 65536,
            signature: vec![1, 2, 3, 4],
            valid: true,
        });
        manifest.active_slot = 0;

        let json = serde_json::to_string_pretty(&manifest).unwrap();
        let restored: CheckpointManifest = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.active_slot, 0);
        let slot = restored.slots[0].as_ref().unwrap();
        assert_eq!(slot.sequence_number, 42);
        assert_eq!(slot.size_bytes, 65536);
        assert_eq!(slot.signature, vec![1, 2, 3, 4]);
        assert!(slot.valid);
        assert!(restored.slots[1].is_none());
        assert!(restored.slots[2].is_none());
    }

    #[test]
    fn manifest_latest_valid() {
        let mut manifest = CheckpointManifest::default();
        manifest.slots[0] = Some(CheckpointSlotInfo {
            path: "a".into(),
            sequence_number: 10,
            timestamp_us: 0,
            size_bytes: 0,
            signature: vec![],
            valid: true,
        });
        manifest.slots[1] = Some(CheckpointSlotInfo {
            path: "b".into(),
            sequence_number: 20,
            timestamp_us: 0,
            size_bytes: 0,
            signature: vec![],
            valid: false, // invalid
        });
        manifest.slots[2] = Some(CheckpointSlotInfo {
            path: "c".into(),
            sequence_number: 15,
            timestamp_us: 0,
            size_bytes: 0,
            signature: vec![],
            valid: true,
        });

        let latest = manifest
            .slots
            .iter()
            .filter_map(|s| s.as_ref())
            .filter(|s| s.valid)
            .max_by_key(|s| s.sequence_number);

        assert!(latest.is_some());
        assert_eq!(latest.unwrap().sequence_number, 15); // slot 1 is invalid
    }
}

#[cfg(test)]
mod crypto_tests {
    use crate::crypto::CryptoService;
    use std::fs;

    #[test]
    fn sign_and_verify() {
        let (signing_key, verifying_key) = CryptoService::generate_keypair();
        let data = b"seismic event detected at station 42";
        let signature = CryptoService::sign(data, &signing_key);

        assert!(CryptoService::verify(data, &signature, &verifying_key));
    }

    #[test]
    fn tampered_data_fails_verification() {
        let (signing_key, verifying_key) = CryptoService::generate_keypair();
        let data = b"original data";
        let signature = CryptoService::sign(data, &signing_key);

        let tampered = b"tampered data";
        assert!(!CryptoService::verify(tampered, &signature, &verifying_key));
    }

    #[test]
    fn tampered_signature_fails() {
        let (signing_key, verifying_key) = CryptoService::generate_keypair();
        let data = b"important data";
        let mut signature = CryptoService::sign(data, &signing_key);
        signature[0] ^= 0xFF; // flip bits

        assert!(!CryptoService::verify(data, &signature, &verifying_key));
    }

    #[test]
    fn invalid_signature_length_fails() {
        let (_, verifying_key) = CryptoService::generate_keypair();
        assert!(!CryptoService::verify(b"data", &[0u8; 32], &verifying_key));
    }

    #[test]
    fn load_or_create_keypair() {
        let dir = std::env::temp_dir().join(format!(
            "qv_crypto_test_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        let key_path = dir.join("ed25519.key");

        // First call: creates key
        let (sk1, vk1) = CryptoService::load_or_create_keypair(&key_path).unwrap();
        assert!(key_path.exists());

        // Second call: loads same key
        let (sk2, vk2) = CryptoService::load_or_create_keypair(&key_path).unwrap();
        assert_eq!(sk1.to_bytes(), sk2.to_bytes());
        assert_eq!(vk1.to_bytes(), vk2.to_bytes());

        // Verify cross-sign works
        let sig = CryptoService::sign(b"test", &sk1);
        assert!(CryptoService::verify(b"test", &sig, &vk2));

        fs::remove_dir_all(&dir).ok();
    }
}

#[cfg(test)]
mod wear_tests {
    use crate::wear::WearMonitor;

    #[test]
    fn utilization_calculation() {
        let mut monitor = WearMonitor::new(1_000_000);
        assert_eq!(monitor.utilization(), 0.0);

        monitor.record_write(500_000);
        assert!((monitor.utilization() - 0.5).abs() < 0.01);

        monitor.record_write(500_000);
        assert!((monitor.utilization() - 1.0).abs() < 0.01);
    }

    #[test]
    fn threshold_detection() {
        let mut monitor = WearMonitor::new(1000);

        monitor.record_write(800);
        assert!(!monitor.should_reduce_checkpoint_frequency());

        monitor.record_write(1); // 801/1000 = 80.1%
        assert!(monitor.should_reduce_checkpoint_frequency());
        assert!(!monitor.should_emergency_reduce());

        monitor.record_write(150); // 951/1000 = 95.1%
        assert!(monitor.should_emergency_reduce());
    }

    #[test]
    fn daily_reset() {
        let mut monitor = WearMonitor::new(1000);
        monitor.record_write(999);
        assert!(monitor.utilization() > 0.9);

        monitor.reset_daily();
        assert_eq!(monitor.utilization(), 0.0);
        assert_eq!(monitor.bytes_written_today(), 0);
    }

    #[test]
    fn zero_budget() {
        let monitor = WearMonitor::new(0);
        assert_eq!(monitor.utilization(), 1.0);
        assert!(monitor.should_emergency_reduce());
    }
}

#[cfg(test)]
mod checkpoint_manager_tests {
    use crate::checkpoint::CheckpointManager;
    use std::fs;

    #[test]
    fn checkpoint_manager_creates_dirs() {
        let dir = std::env::temp_dir().join(format!(
            "qv_cm_test_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));

        let _mgr = CheckpointManager::new(&dir).unwrap();

        assert!(dir.join("wal").exists());
        assert!(dir.join("checkpoints").exists());
        assert!(dir.join("models").exists());

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn checkpoint_manager_record_and_query() {
        let dir = std::env::temp_dir().join(format!(
            "qv_cm_rq_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));

        let mut mgr = CheckpointManager::new(&dir).unwrap();

        assert!(mgr.latest_valid_checkpoint().is_none());

        mgr.record_checkpoint(0, 100, 4096, vec![1, 2, 3]).unwrap();
        let latest = mgr.latest_valid_checkpoint().unwrap();
        assert_eq!(latest.sequence_number, 100);

        mgr.record_checkpoint(1, 200, 8192, vec![4, 5, 6]).unwrap();
        let latest = mgr.latest_valid_checkpoint().unwrap();
        assert_eq!(latest.sequence_number, 200);

        // Manifest should persist
        assert!(dir.join("checkpoints").join("manifest.json").exists());

        fs::remove_dir_all(&dir).ok();
    }
}

#[cfg(test)]
mod graph_serializer_tests {
    use crate::graph_serializer::GraphSerializer;
    use quake_vector_store::{HnswGraph, VectorStore};
    use quake_vector_types::{SeismicEmbedding, StationId, TemporalMeta, WaveType};

    fn make_embedding(seed: u64) -> SeismicEmbedding {
        let mut vector = [0.0f32; 256];
        let mut state = seed ^ 0x12345678;
        for v in vector.iter_mut() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *v = (state as f32 / u64::MAX as f32) * 2.0 - 1.0;
        }
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-8 {
            for v in vector.iter_mut() {
                *v /= norm;
            }
        }
        SeismicEmbedding {
            vector,
            source_window_hash: seed,
            norm,
        }
    }

    fn make_meta(ts: u64) -> TemporalMeta {
        TemporalMeta {
            timestamp_us: ts,
            wave_type: WaveType::P,
            station_id: StationId(1),
            amplitude_rms: 100.0,
            dominant_freq_hz: 5.0,
        }
    }

    #[test]
    fn round_trip_100_nodes() {
        let mut graph = HnswGraph::with_default_config();
        for i in 0..100 {
            let emb = make_embedding(i);
            let meta = make_meta(i * 1000);
            graph.insert(&emb, &meta);
        }
        assert_eq!(graph.node_count(), 100);

        let bytes = GraphSerializer::serialize(&graph);
        assert!(!bytes.is_empty());

        let restored = GraphSerializer::deserialize(&bytes).unwrap();
        assert_eq!(restored.node_count(), 100);

        // Verify metadata matches for a sample node
        let orig_meta = graph.get_meta(quake_vector_types::NodeId(42)).unwrap();
        let rest_meta = restored.get_meta(quake_vector_types::NodeId(42)).unwrap();
        assert_eq!(orig_meta.timestamp_us, rest_meta.timestamp_us);
        assert_eq!(orig_meta.wave_type, rest_meta.wave_type);

        // Verify vectors match
        let orig_vec = graph.get_vector(quake_vector_types::NodeId(42)).unwrap();
        let rest_vec = restored.get_vector(quake_vector_types::NodeId(42)).unwrap();
        assert_eq!(orig_vec, rest_vec);

        // Verify search still works
        let query = make_embedding(42);
        let result = restored.knn_search(&query.vector, 1, 64);
        assert!(!result.neighbors.is_empty());
        assert_eq!(result.neighbors[0].0, quake_vector_types::NodeId(42));
    }
}

#[cfg(test)]
mod checkpoint_recovery_tests {
    use crate::checkpoint::CheckpointManager;
    use crate::crypto::CryptoService;
    use crate::wal_entry::WalOpType;
    use quake_vector_adaptation::SonaEngine;
    use quake_vector_store::{HnswGraph, VectorStore};
    use quake_vector_types::{SeismicEmbedding, StationId, TemporalMeta, WaveType};
    use std::fs;

    fn temp_dir(label: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "qv_{label}_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        dir
    }

    fn make_embedding(seed: u64) -> SeismicEmbedding {
        let mut vector = [0.0f32; 256];
        let mut state = seed ^ 0x12345678;
        for v in vector.iter_mut() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *v = (state as f32 / u64::MAX as f32) * 2.0 - 1.0;
        }
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-8 {
            for v in vector.iter_mut() {
                *v /= norm;
            }
        }
        SeismicEmbedding {
            vector,
            source_window_hash: seed,
            norm,
        }
    }

    fn make_meta(ts: u64) -> TemporalMeta {
        TemporalMeta {
            timestamp_us: ts,
            wave_type: WaveType::P,
            station_id: StationId(1),
            amplitude_rms: 100.0,
            dominant_freq_hz: 5.0,
        }
    }

    fn populate_graph(graph: &mut HnswGraph, count: u64) {
        for i in 0..count {
            let emb = make_embedding(i);
            let meta = make_meta(i * 1000);
            graph.insert(&emb, &meta);
        }
    }

    #[test]
    fn create_checkpoint_then_recover() {
        let dir = temp_dir("ckpt_recover");
        let (signing_key, verifying_key) = CryptoService::generate_keypair();

        // Create graph with 100 nodes, checkpoint it
        {
            let mut mgr = CheckpointManager::new(&dir).unwrap();
            let mut graph = HnswGraph::with_default_config();
            populate_graph(&mut graph, 100);
            let sona = SonaEngine::new();

            let slot = mgr.create_checkpoint(&graph, &sona, &signing_key).unwrap();
            assert!(slot.size_bytes > 0);
            assert!(!slot.signature.is_empty());
        }

        // Recover and verify
        {
            let mut mgr = CheckpointManager::new(&dir).unwrap();
            let result = mgr.recover(&verifying_key).unwrap();
            assert!(result.is_some());

            let recovery = result.unwrap();
            assert_eq!(recovery.graph.node_count(), 100);

            let sona = recovery.sona_state.into_engine().unwrap();
            assert_eq!(sona.current_model_version, 1);
        }

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn recovery_with_wal_entries() {
        let dir = temp_dir("wal_recovery");
        let (signing_key, verifying_key) = CryptoService::generate_keypair();

        // Create graph, checkpoint, then write 50 WAL entries
        {
            let mut mgr = CheckpointManager::new(&dir).unwrap();
            let mut graph = HnswGraph::with_default_config();
            populate_graph(&mut graph, 100);
            let sona = SonaEngine::new();

            mgr.create_checkpoint(&graph, &sona, &signing_key).unwrap();

            // Simulate 50 more inserts via WAL (without checkpointing)
            for i in 0..50 {
                let payload = i.to_string().into_bytes();
                mgr.wal().append(WalOpType::InsertNode, &payload).unwrap();
            }
            mgr.wal().flush().unwrap();
        }

        // Recover: should find graph + 50 WAL entries replayed
        {
            let mut mgr = CheckpointManager::new(&dir).unwrap();
            let result = mgr.recover(&verifying_key).unwrap().unwrap();

            assert_eq!(result.graph.node_count(), 100);
            assert_eq!(result.wal_entries_replayed, 50);
        }

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn recovery_with_corrupt_wal() {
        use std::io::Write;

        let dir = temp_dir("corrupt_wal_recovery");
        let (signing_key, verifying_key) = CryptoService::generate_keypair();

        {
            let mut mgr = CheckpointManager::new(&dir).unwrap();
            let mut graph = HnswGraph::with_default_config();
            populate_graph(&mut graph, 50);
            let sona = SonaEngine::new();

            mgr.create_checkpoint(&graph, &sona, &signing_key).unwrap();

            // Write 10 WAL entries
            for i in 0..10 {
                let payload = format!("node_{i}").into_bytes();
                mgr.wal().append(WalOpType::InsertNode, &payload).unwrap();
            }
            mgr.wal().flush().unwrap();
        }

        // Corrupt the last WAL entry by appending garbage
        {
            let wal_path = dir.join("wal").join("current.wal");
            let mut file = fs::OpenOptions::new()
                .append(true)
                .open(&wal_path)
                .unwrap();
            // Write partial/corrupt entry
            file.write_all(&[0xFF, 0xFE, 0xFD, 0xFC, 0xFB]).unwrap();
            file.flush().unwrap();
        }

        // Recovery should still succeed with 10 valid WAL entries
        {
            let mut mgr = CheckpointManager::new(&dir).unwrap();
            let result = mgr.recover(&verifying_key).unwrap().unwrap();
            assert_eq!(result.graph.node_count(), 50);
            // WAL reader stops at corruption boundary, should recover 10 entries
            assert_eq!(result.wal_entries_replayed, 10);
        }

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn slot_rotation_overwrites_oldest() {
        let dir = temp_dir("slot_rotation");
        let (signing_key, verifying_key) = CryptoService::generate_keypair();

        let mut mgr = CheckpointManager::new(&dir).unwrap();
        let mut graph = HnswGraph::with_default_config();
        let sona = SonaEngine::new();

        // Create 4 checkpoints — should rotate through slots 0,1,2,0
        for i in 0..4 {
            populate_graph(&mut graph, 10); // 10 more each time
            // Write a WAL entry to advance sequence
            mgr.wal().append(WalOpType::InsertNode, &[i as u8]).unwrap();
            mgr.create_checkpoint(&graph, &sona, &signing_key).unwrap();
        }

        // Manifest should have 3 slots filled; slot 1 should be the oldest remaining
        let manifest = mgr.manifest();
        assert!(manifest.slots[0].is_some()); // overwritten by 4th checkpoint
        assert!(manifest.slots[1].is_some());
        assert!(manifest.slots[2].is_some());

        // The latest should be in slot 1 (0->1->2->0, active=0, next was 1 at 4th)
        // Actually: starts at active_slot=0, first create goes to (0+1)%3=1,
        // second to (1+1)%3=2, third to (2+1)%3=0, fourth to (0+1)%3=1
        // So slot 1 has the latest checkpoint
        let head_seq = mgr.wal().head_sequence();
        let latest = mgr.latest_valid_checkpoint().unwrap();
        assert_eq!(latest.sequence_number, head_seq);

        // Recovery still works
        let result = mgr.recover(&verifying_key).unwrap().unwrap();
        assert_eq!(result.graph.node_count(), 40);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn tampered_checkpoint_rejected() {
        let dir = temp_dir("tamper_ckpt");
        let (signing_key, verifying_key) = CryptoService::generate_keypair();

        {
            let mut mgr = CheckpointManager::new(&dir).unwrap();
            let mut graph = HnswGraph::with_default_config();
            populate_graph(&mut graph, 50);
            let sona = SonaEngine::new();

            mgr.create_checkpoint(&graph, &sona, &signing_key).unwrap();
        }

        // Tamper with the checkpoint file
        {
            let checkpoint_path = dir.join("checkpoints").join("checkpoint_1.bin");
            let mut data = fs::read(&checkpoint_path).unwrap();
            if !data.is_empty() {
                data[0] ^= 0xFF;
            }
            fs::write(&checkpoint_path, &data).unwrap();
        }

        // Recovery should fail with signature verification error
        {
            let mut mgr = CheckpointManager::new(&dir).unwrap();
            let result = mgr.recover(&verifying_key);
            assert!(result.is_err());
        }

        fs::remove_dir_all(&dir).ok();
    }
}
