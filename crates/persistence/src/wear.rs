//! Flash wear monitoring for embedded deployments.

/// Monitors daily write budget to reduce flash/SSD wear.
#[derive(Debug)]
pub struct WearMonitor {
    daily_budget_bytes: u64,
    bytes_written_today: u64,
}

impl WearMonitor {
    pub fn new(daily_budget_bytes: u64) -> Self {
        Self {
            daily_budget_bytes,
            bytes_written_today: 0,
        }
    }

    /// Record bytes written.
    pub fn record_write(&mut self, bytes: u64) {
        self.bytes_written_today += bytes;
    }

    /// Current utilization as a fraction (0.0 to 1.0+).
    pub fn utilization(&self) -> f32 {
        if self.daily_budget_bytes == 0 {
            return 1.0;
        }
        self.bytes_written_today as f32 / self.daily_budget_bytes as f32
    }

    /// Returns true if checkpoint frequency should be reduced (> 80% utilized).
    pub fn should_reduce_checkpoint_frequency(&self) -> bool {
        self.utilization() > 0.80
    }

    /// Returns true for emergency write reduction (> 95% utilized).
    pub fn should_emergency_reduce(&self) -> bool {
        self.utilization() > 0.95
    }

    /// Reset the daily counter (call at midnight or start of day).
    pub fn reset_daily(&mut self) {
        self.bytes_written_today = 0;
    }

    /// Bytes written today.
    pub fn bytes_written_today(&self) -> u64 {
        self.bytes_written_today
    }

    /// Recommended checkpoint interval based on wear utilization.
    ///
    /// - Normal (< 80%): `default_interval`
    /// - High (80-95%): 2x `default_interval`
    /// - Emergency (> 95%): 4x `default_interval`
    pub fn recommended_checkpoint_interval(
        &self,
        default_interval: std::time::Duration,
    ) -> std::time::Duration {
        if self.should_emergency_reduce() {
            default_interval * 4
        } else if self.should_reduce_checkpoint_frequency() {
            default_interval * 2
        } else {
            default_interval
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn recommended_interval_changes_with_utilization() {
        let default = Duration::from_secs(900);

        // Normal: < 80%
        let mut monitor = WearMonitor::new(1_000_000);
        monitor.record_write(500_000); // 50%
        assert_eq!(monitor.recommended_checkpoint_interval(default), default);

        // High: > 80%
        monitor.record_write(350_000); // 85%
        assert_eq!(
            monitor.recommended_checkpoint_interval(default),
            default * 2
        );

        // Emergency: > 95%
        monitor.record_write(120_000); // 97%
        assert_eq!(
            monitor.recommended_checkpoint_interval(default),
            default * 4
        );
    }

    #[test]
    fn reset_daily_restores_normal_interval() {
        let default = Duration::from_secs(900);
        let mut monitor = WearMonitor::new(1_000_000);
        monitor.record_write(960_000); // 96%
        assert_eq!(
            monitor.recommended_checkpoint_interval(default),
            default * 4
        );

        monitor.reset_daily();
        assert_eq!(monitor.recommended_checkpoint_interval(default), default);
    }
}
