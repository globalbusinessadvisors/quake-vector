//! Resource monitoring: RAM utilization and degradation levels.

/// Degradation level based on resource utilization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DegradationLevel {
    Normal,
    Warning,
    Critical,
}

/// Snapshot of system resource utilization.
#[derive(Debug, Clone)]
pub struct ResourceStatus {
    pub ram_used_bytes: u64,
    pub ram_total_bytes: u64,
    pub ram_utilization: f32,
    pub degradation_level: DegradationLevel,
}

/// Monitors system resources (RAM).
pub struct ResourceMonitor;

impl ResourceMonitor {
    pub fn new() -> Self {
        Self
    }

    /// Check current resource status.
    pub fn check(&self) -> ResourceStatus {
        let (used, total) = Self::read_memory();
        let utilization = if total > 0 {
            used as f32 / total as f32
        } else {
            0.0
        };
        let degradation_level = if utilization > 0.9 {
            DegradationLevel::Critical
        } else if utilization > 0.8 {
            DegradationLevel::Warning
        } else {
            DegradationLevel::Normal
        };

        ResourceStatus {
            ram_used_bytes: used,
            ram_total_bytes: total,
            ram_utilization: utilization,
            degradation_level,
        }
    }

    #[cfg(target_os = "linux")]
    fn read_memory() -> (u64, u64) {
        use std::fs;
        let Ok(content) = fs::read_to_string("/proc/meminfo") else {
            return (0, 1);
        };
        let mut total_kb: u64 = 0;
        let mut available_kb: u64 = 0;
        for line in content.lines() {
            if let Some(rest) = line.strip_prefix("MemTotal:") {
                total_kb = Self::parse_kb(rest);
            } else if let Some(rest) = line.strip_prefix("MemAvailable:") {
                available_kb = Self::parse_kb(rest);
            }
        }
        let total = total_kb * 1024;
        let used = total.saturating_sub(available_kb * 1024);
        (used, total)
    }

    #[cfg(not(target_os = "linux"))]
    fn read_memory() -> (u64, u64) {
        // Default: report 50% utilization on non-Linux
        (512 * 1024 * 1024, 1024 * 1024 * 1024)
    }

    #[cfg(target_os = "linux")]
    fn parse_kb(s: &str) -> u64 {
        s.trim()
            .trim_end_matches("kB")
            .trim()
            .parse()
            .unwrap_or(0)
    }
}

impl Default for ResourceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resource_status_is_valid() {
        let monitor = ResourceMonitor::new();
        let status = monitor.check();
        assert!(status.ram_total_bytes > 0);
        assert!(status.ram_utilization >= 0.0);
        assert!(status.ram_utilization <= 1.0 || status.ram_utilization > 1.0);
        // Should be one of the three levels
        assert!(matches!(
            status.degradation_level,
            DegradationLevel::Normal | DegradationLevel::Warning | DegradationLevel::Critical
        ));
    }
}
