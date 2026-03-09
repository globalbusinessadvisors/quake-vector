//! Cool-down state to suppress duplicate alerts.

use quake_vector_types::AlertLevel;

/// Tracks cool-down state to avoid alert spam.
#[derive(Debug, Clone)]
pub struct CoolDownState {
    pub last_alert_level: Option<AlertLevel>,
    pub last_alert_time_us: u64,
    pub last_magnitude: f32,
    pub cool_down_duration_us: u64,
}

impl CoolDownState {
    pub fn new() -> Self {
        Self {
            last_alert_level: None,
            last_alert_time_us: 0,
            last_magnitude: 0.0,
            cool_down_duration_us: 30_000_000, // 30 seconds
        }
    }

    /// Returns true if cool-down is active for the given level at the given time.
    ///
    /// Active if an alert at this level or higher was emitted within the window.
    pub fn is_active(&self, now_us: u64, level: AlertLevel) -> bool {
        let Some(last_level) = self.last_alert_level else {
            return false;
        };
        if now_us.saturating_sub(self.last_alert_time_us) >= self.cool_down_duration_us {
            return false;
        }
        level_rank(last_level) >= level_rank(level)
    }

    /// Returns true if the new magnitude warrants overriding the cool-down.
    ///
    /// Override if new magnitude exceeds previous by >= 0.5.
    pub fn should_override(&self, new_magnitude: f32) -> bool {
        new_magnitude >= self.last_magnitude + 0.5
    }

    /// Activate cool-down with the given alert parameters.
    pub fn activate(&mut self, level: AlertLevel, magnitude: f32, now_us: u64) {
        self.last_alert_level = Some(level);
        self.last_alert_time_us = now_us;
        self.last_magnitude = magnitude;
    }

    /// Reset cool-down state.
    pub fn reset(&mut self) {
        self.last_alert_level = None;
        self.last_alert_time_us = 0;
        self.last_magnitude = 0.0;
    }
}

impl Default for CoolDownState {
    fn default() -> Self {
        Self::new()
    }
}

fn level_rank(level: AlertLevel) -> u8 {
    match level {
        AlertLevel::Low => 0,
        AlertLevel::Medium => 1,
        AlertLevel::High => 2,
        AlertLevel::Critical => 3,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inactive_when_no_prior_alert() {
        let cd = CoolDownState::new();
        assert!(!cd.is_active(1_000_000, AlertLevel::Low));
    }

    #[test]
    fn active_within_window() {
        let mut cd = CoolDownState::new();
        cd.activate(AlertLevel::Medium, 3.0, 1_000_000);
        // Within 30s window
        assert!(cd.is_active(20_000_000, AlertLevel::Low));
        assert!(cd.is_active(20_000_000, AlertLevel::Medium));
        // Higher level is not suppressed
        assert!(!cd.is_active(20_000_000, AlertLevel::High));
    }

    #[test]
    fn inactive_after_window() {
        let mut cd = CoolDownState::new();
        cd.activate(AlertLevel::High, 4.0, 1_000_000);
        // After 30s
        assert!(!cd.is_active(32_000_000, AlertLevel::Low));
    }

    #[test]
    fn magnitude_override() {
        let mut cd = CoolDownState::new();
        cd.activate(AlertLevel::Medium, 3.0, 1_000_000);
        assert!(!cd.should_override(3.3));
        assert!(cd.should_override(3.5));
        assert!(cd.should_override(4.0));
    }

    #[test]
    fn reset_clears_state() {
        let mut cd = CoolDownState::new();
        cd.activate(AlertLevel::Critical, 5.0, 1_000_000);
        cd.reset();
        assert!(!cd.is_active(1_500_000, AlertLevel::Low));
    }
}
