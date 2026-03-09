//! SONA three-speed learning: self-organizing network adaptation with fast,
//! medium, and slow learning rates for continuous model refinement.
//!
//! When the `ruvector` feature is enabled, a `RuvectorSonaEngine` adapter
//! wrapping ruvector-sona is also available.

mod lora;
mod micro;
mod base;
mod ground_truth;
mod probation;
mod engine;

#[cfg(feature = "ruvector")]
pub mod ruvector_adapter;

pub use lora::LoraDelta;
pub use micro::MicroLoraAdapter;
pub use base::BaseLoraAdapter;
pub use ground_truth::{GroundTruth, GroundTruthLabeler};
pub use probation::{ProbationState, ProbationResult};
pub use engine::SonaEngine;

#[cfg(feature = "ruvector")]
pub use ruvector_adapter::RuvectorSonaEngine;
