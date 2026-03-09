//! GNN causal learning: graph neural network inference over the HNSW graph
//! to detect causal seismic wave propagation patterns and generate predictions.
//!
//! When the `ruvector` feature is enabled, a `RuvectorCausalModel` adapter
//! wrapping ruvector-gnn and ruv-fann is also available.

mod subgraph;
mod gat;
mod transition;
mod model;
mod history;
mod learner;

#[cfg(feature = "ruvector")]
pub mod ruvector_adapter;

pub use subgraph::{TemporalSubgraph, SubgraphNode, CausalEdge, SubgraphConstructionService};
pub use gat::GatLayer;
pub use transition::TransitionMatrix;
pub use model::CausalModel;
pub use history::WaveHistory;
pub use learner::CausalLearner;

#[cfg(feature = "ruvector")]
pub use ruvector_adapter::RuvectorCausalModel;
