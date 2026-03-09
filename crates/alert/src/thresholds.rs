//! Alert threshold configuration with level determination.

use quake_vector_types::AlertLevel;

/// Configurable probability thresholds for alert levels.
///
/// Invariant: low < medium < high < critical.
#[derive(Debug, Clone)]
pub struct AlertThresholds {
    pub low: f32,
    pub medium: f32,
    pub high: f32,
    pub critical: f32,
}

impl AlertThresholds {
    /// Create thresholds, validating the ordering invariant.
    pub fn new(low: f32, medium: f32, high: f32, critical: f32) -> Result<Self, &'static str> {
        if !(low < medium && medium < high && high < critical) {
            return Err("thresholds must satisfy low < medium < high < critical");
        }
        Ok(Self { low, medium, high, critical })
    }

    /// Determine alert level from event probability.
    pub fn determine_level(&self, probability: f32) -> Option<AlertLevel> {
        if probability >= self.critical {
            Some(AlertLevel::Critical)
        } else if probability >= self.high {
            Some(AlertLevel::High)
        } else if probability >= self.medium {
            Some(AlertLevel::Medium)
        } else if probability >= self.low {
            Some(AlertLevel::Low)
        } else {
            None
        }
    }
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            low: 0.3,
            medium: 0.5,
            high: 0.7,
            critical: 0.9,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn correct_levels_for_probability_ranges() {
        let t = AlertThresholds::default();
        assert_eq!(t.determine_level(0.1), None);
        assert_eq!(t.determine_level(0.29), None);
        assert_eq!(t.determine_level(0.3), Some(AlertLevel::Low));
        assert_eq!(t.determine_level(0.49), Some(AlertLevel::Low));
        assert_eq!(t.determine_level(0.5), Some(AlertLevel::Medium));
        assert_eq!(t.determine_level(0.69), Some(AlertLevel::Medium));
        assert_eq!(t.determine_level(0.7), Some(AlertLevel::High));
        assert_eq!(t.determine_level(0.89), Some(AlertLevel::High));
        assert_eq!(t.determine_level(0.9), Some(AlertLevel::Critical));
        assert_eq!(t.determine_level(1.0), Some(AlertLevel::Critical));
    }

    #[test]
    fn invalid_ordering_rejected() {
        assert!(AlertThresholds::new(0.5, 0.3, 0.7, 0.9).is_err());
        assert!(AlertThresholds::new(0.3, 0.5, 0.5, 0.9).is_err());
    }

    #[test]
    fn valid_custom_thresholds() {
        let t = AlertThresholds::new(0.2, 0.4, 0.6, 0.8).unwrap();
        assert_eq!(t.determine_level(0.5), Some(AlertLevel::Medium));
    }
}
