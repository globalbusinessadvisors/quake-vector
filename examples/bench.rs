//! Latency benchmarks for QuakeVector pipeline components.
//!
//! Run with: cargo run --release --example bench

use std::time::Instant;

use quake_vector_embedding::EmbeddingPipeline;
use quake_vector_learning::CausalLearner;
use quake_vector_store::{HnswGraph, VectorStore};
use quake_vector_types::{
    ChannelId, NodeId, SeismicEmbedding, StationId, TemporalMeta, WaveType, WaveformWindow,
};

fn make_random_window(seed: u64) -> WaveformWindow {
    let mut samples = [0i32; 256];
    let mut state = seed ^ 0xABCDEF01;
    for s in samples.iter_mut() {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        *s = ((state >> 33) as i32).wrapping_sub(i32::MAX / 2) / 1000;
    }
    WaveformWindow {
        samples,
        channel: ChannelId(0),
        timestamp_us: seed * 10_000,
        wave_type: WaveType::P,
        rms_amplitude: 500.0,
        dominant_freq_hz: 5.0,
    }
}

fn make_random_embedding(seed: u64) -> SeismicEmbedding {
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

struct BenchResult {
    name: String,
    iterations: usize,
    mean_us: f64,
    p50_us: u64,
    p99_us: u64,
}

fn run_bench<F>(name: &str, iterations: usize, mut f: F) -> BenchResult
where
    F: FnMut(),
{
    let mut latencies = Vec::with_capacity(iterations);

    // Warmup
    for _ in 0..iterations.min(50) {
        f();
    }

    for _ in 0..iterations {
        let start = Instant::now();
        f();
        latencies.push(start.elapsed().as_micros() as u64);
    }

    latencies.sort();
    let mean = latencies.iter().sum::<u64>() as f64 / iterations as f64;
    let p50 = latencies[iterations / 2];
    let p99 = latencies[(iterations as f64 * 0.99) as usize];

    BenchResult {
        name: name.to_string(),
        iterations,
        mean_us: mean,
        p50_us: p50,
        p99_us: p99,
    }
}

fn bench_embedding_latency() -> BenchResult {
    let mut pipeline = EmbeddingPipeline::new();
    let windows: Vec<WaveformWindow> = (0..1000).map(make_random_window).collect();
    let mut i = 0;

    run_bench("embedding_latency", 1000, || {
        pipeline.compute(&windows[i % 1000]);
        i += 1;
    })
}

fn bench_hnsw_insert() -> BenchResult {
    // Pre-populate with 10,000 nodes
    let mut graph = HnswGraph::with_default_config();
    for i in 0..10_000u64 {
        let emb = make_random_embedding(i);
        let meta = make_meta(i * 1000);
        graph.insert(&emb, &meta);
    }

    let embeddings: Vec<SeismicEmbedding> = (10_000..11_000).map(make_random_embedding).collect();
    let metas: Vec<TemporalMeta> = (10_000..11_000).map(|i| make_meta(i * 1000)).collect();
    let mut idx = 0;

    run_bench("hnsw_insert_latency", 1000, || {
        graph.insert(&embeddings[idx % 1000], &metas[idx % 1000]);
        idx += 1;
    })
}

fn bench_hnsw_search() -> BenchResult {
    let mut graph = HnswGraph::with_default_config();
    for i in 0..10_000u64 {
        let emb = make_random_embedding(i);
        let meta = make_meta(i * 1000);
        graph.insert(&emb, &meta);
    }

    let queries: Vec<[f32; 256]> = (0..1000)
        .map(|i| make_random_embedding(20_000 + i).vector)
        .collect();
    let mut idx = 0;

    run_bench("hnsw_search_latency", 1000, || {
        let _ = graph.knn_search(&queries[idx % 1000], 16, 64);
        idx += 1;
    })
}

fn bench_gnn_inference() -> BenchResult {
    // Build a graph with enough nodes for subgraph construction
    let mut graph = HnswGraph::with_default_config();
    let mut learner = CausalLearner::new();

    for i in 0..200u64 {
        let emb = make_random_embedding(i);
        let meta = make_meta(i * 1000);
        graph.insert(&emb, &meta);
    }

    let embeddings: Vec<SeismicEmbedding> = (200..1200).map(make_random_embedding).collect();
    let metas: Vec<TemporalMeta> = (200..1200).map(|i| make_meta(i * 1000)).collect();
    let mut idx = 0;

    run_bench("gnn_inference_latency", 1000, || {
        let emb = &embeddings[idx % 1000];
        let meta = &metas[idx % 1000];
        let node_id = NodeId(200 + idx as u64);
        let _ = learner.process(node_id, emb, meta, &graph);
        idx += 1;
    })
}

fn bench_full_pipeline() -> BenchResult {
    let mut pipeline = EmbeddingPipeline::new();
    let mut graph = HnswGraph::with_default_config();
    let mut learner = CausalLearner::new();

    // Pre-populate
    for i in 0..100u64 {
        let emb = make_random_embedding(i);
        let meta = make_meta(i * 1000);
        graph.insert(&emb, &meta);
    }

    let windows: Vec<WaveformWindow> = (100..1100).map(make_random_window).collect();
    let mut idx = 0;

    run_bench("full_pipeline_latency", 1000, || {
        let window = &windows[idx % 1000];
        let embedding = pipeline.compute(window);
        let meta = TemporalMeta {
            timestamp_us: window.timestamp_us,
            wave_type: window.wave_type,
            station_id: StationId(1),
            amplitude_rms: window.rms_amplitude,
            dominant_freq_hz: window.dominant_freq_hz,
        };
        let node_id = graph.insert(&embedding, &meta);
        let _ = learner.process(node_id, &embedding, &meta, &graph);
        idx += 1;
    })
}

fn main() {
    println!("QuakeVector Latency Benchmarks");
    println!("==============================\n");
    println!(
        "{:<25} {:>8} {:>10} {:>10} {:>10}",
        "Benchmark", "N", "Mean (us)", "P50 (us)", "P99 (us)"
    );
    println!("{}", "-".repeat(65));

    let benchmarks = [
        bench_embedding_latency(),
        bench_hnsw_insert(),
        bench_hnsw_search(),
        bench_gnn_inference(),
        bench_full_pipeline(),
    ];

    for b in &benchmarks {
        println!(
            "{:<25} {:>8} {:>10.1} {:>10} {:>10}",
            b.name, b.iterations, b.mean_us, b.p50_us, b.p99_us
        );
    }

    println!("\nDone.");
}
