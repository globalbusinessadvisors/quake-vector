//! HNSW graph management: maintains a hierarchical navigable small-world
//! graph for efficient approximate nearest-neighbor search over embeddings.
//!
//! When the `ruvector` feature is enabled, a `RuvectorHnswGraph` adapter
//! wrapping rvf-index's ProgressiveIndex is also available.

mod config;
mod distance;
mod graph;
mod node;
mod tier;
mod traits;

#[cfg(feature = "ruvector")]
pub mod ruvector_adapter;

pub use config::HnswConfig;
pub use distance::cosine_distance;
pub use graph::HnswGraph;
pub use node::GraphNode;
pub use tier::TierManager;
pub use traits::VectorStore;

#[cfg(feature = "ruvector")]
pub use ruvector_adapter::RuvectorHnswGraph;
