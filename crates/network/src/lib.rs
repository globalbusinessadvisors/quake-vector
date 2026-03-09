//! Mesh networking, upstream relay, and dashboard: handles inter-node
//! communication, data relay to upstream servers, and status dashboard serving.

mod message;
mod queue;
mod peer;
mod mesh;
mod upstream;
mod service;
pub mod dashboard;

pub use message::{MeshMessage, MeshMessageType, PredictionClaimPayload, PredictionConfirmPayload};
pub use queue::BoundedQueue;
pub use peer::{PeerState, PeerHealth};
pub use mesh::{MeshManager, MeshEvent};
pub use upstream::UpstreamRelay;
pub use service::NetworkService;
pub use dashboard::{DashboardState, DashboardService, PeerSnapshot};
