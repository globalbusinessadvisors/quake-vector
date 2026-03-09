//! RVF boot, watchdog, and degradation: manages the boot sequence through
//! BootStage phases, monitors system health, and handles graceful degradation.

mod config;
mod boot;
mod resource;
mod runtime;
mod degradation;
mod watchdog;
mod thread_health;

pub use config::QuakeVectorConfig;
pub use boot::{BootSequencer, BootReport, BootType};
pub use resource::{ResourceMonitor, ResourceStatus, DegradationLevel};
pub use runtime::{QuakeVectorRuntime, TickResult};
pub use degradation::GracefulDegradation;
pub use watchdog::WatchdogManager;
pub use thread_health::{ThreadHealthMonitor, ThreadHeartbeatHandle};
