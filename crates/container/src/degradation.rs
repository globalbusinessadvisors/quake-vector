//! Graceful degradation: adjusts runtime parameters based on resource pressure.

use tracing::{info, warn};

use crate::resource::DegradationLevel;
use crate::runtime::QuakeVectorRuntime;

/// Applies degradation policies to the runtime based on resource pressure.
pub struct GracefulDegradation;

impl GracefulDegradation {
    /// Adjust runtime parameters based on the degradation level.
    ///
    /// - Normal: restore default ef_search.
    /// - Warning: reduce ef_search to 32.
    /// - Critical: reduce ef_search to 16.
    pub fn apply(level: DegradationLevel, runtime: &mut QuakeVectorRuntime) {
        match level {
            DegradationLevel::Normal => {
                runtime.ef_search = runtime.config.hnsw_ef_search;
                info!("degradation: Normal — restored default ef_search={}", runtime.ef_search);
            }
            DegradationLevel::Warning => {
                runtime.ef_search = 32;
                warn!("degradation: Warning — reduced ef_search to 32");
            }
            DegradationLevel::Critical => {
                runtime.ef_search = 16;
                warn!("degradation: Critical — reduced ef_search to 16");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::QuakeVectorConfig;
    use crate::runtime::QuakeVectorRuntime;

    fn make_runtime() -> QuakeVectorRuntime {
        let mut config = QuakeVectorConfig::default();
        config.data_dir = std::env::temp_dir().join(format!(
            "qv_degrade_test_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        QuakeVectorRuntime::boot(config).unwrap()
    }

    #[test]
    fn ef_search_changes_at_each_level() {
        let mut runtime = make_runtime();
        let default_ef = runtime.config.hnsw_ef_search;

        // Normal
        GracefulDegradation::apply(DegradationLevel::Normal, &mut runtime);
        assert_eq!(runtime.ef_search, default_ef);

        // Warning
        GracefulDegradation::apply(DegradationLevel::Warning, &mut runtime);
        assert_eq!(runtime.ef_search, 32);

        // Critical
        GracefulDegradation::apply(DegradationLevel::Critical, &mut runtime);
        assert_eq!(runtime.ef_search, 16);

        // Back to Normal
        GracefulDegradation::apply(DegradationLevel::Normal, &mut runtime);
        assert_eq!(runtime.ef_search, default_ef);

        std::fs::remove_dir_all(&runtime.config.data_dir).ok();
    }
}
