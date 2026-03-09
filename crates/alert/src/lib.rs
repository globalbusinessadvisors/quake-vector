//! Prediction alerting: evaluates SeismicPrediction results against thresholds
//! and emits signed Alert messages with consensus validation.

mod thresholds;
mod cooldown;
mod consensus;
mod log;
mod decision;
mod builder;
mod engine;
pub mod gpio;

pub use thresholds::AlertThresholds;
pub use cooldown::CoolDownState;
pub use consensus::{ConsensusManager, ConsensusStatus, PendingClaim};
pub use log::AlertLog;
pub use decision::{AlertDecision, AlertDecisionService};
pub use builder::AlertBuilder;
pub use engine::AlertEngine;
pub use gpio::{GpioOutput, GpioAlertService, MockGpio, LinuxGpio};
