//! Upstream relay: queues alerts for transmission to upstream servers.

use quake_vector_types::{Alert, AlertLevel};

use crate::queue::BoundedQueue;

/// Queues alerts as JSON strings for upstream transmission.
pub struct UpstreamRelay {
    pub endpoint: Option<String>,
    outbound_queue: BoundedQueue<String>,
}

impl UpstreamRelay {
    pub fn new(endpoint: Option<String>) -> Self {
        Self {
            endpoint,
            outbound_queue: BoundedQueue::new(500),
        }
    }

    /// Queue an alert for upstream relay.
    ///
    /// CRITICAL alerts use priority queuing (pushed to front).
    pub fn queue_alert(&mut self, alert: &Alert) {
        let json = serde_json::to_string(alert).unwrap_or_default();
        if alert.level == AlertLevel::Critical {
            self.outbound_queue.push_priority(json);
        } else {
            self.outbound_queue.push(json);
        }
    }

    /// Drain the outbound queue.
    pub fn drain_outbound(&mut self) -> Vec<String> {
        self.outbound_queue.drain()
    }

    /// Number of queued messages.
    pub fn queue_len(&self) -> usize {
        self.outbound_queue.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quake_vector_types::{AlertId, StationId};
    use uuid::Uuid;

    fn make_alert(level: AlertLevel) -> Alert {
        Alert {
            alert_id: AlertId(Uuid::new_v4()),
            station_id: StationId(1),
            timestamp: 1_000_000,
            level,
            probability: 0.8,
            magnitude_estimate: 4.0,
            time_to_peak_s: 5.0,
            confidence: 0.85,
            wave_evidence: Vec::new(),
            consensus_stations: Vec::new(),
            signature: Vec::new(),
        }
    }

    #[test]
    fn critical_alerts_get_priority() {
        let mut relay = UpstreamRelay::new(None);

        // Queue some low alerts
        relay.queue_alert(&make_alert(AlertLevel::Low));
        relay.queue_alert(&make_alert(AlertLevel::Medium));

        // Queue a critical alert — should go to front
        relay.queue_alert(&make_alert(AlertLevel::Critical));

        let msgs = relay.drain_outbound();
        assert_eq!(msgs.len(), 3);

        // First message should be the critical one
        let first: Alert = serde_json::from_str(&msgs[0]).unwrap();
        assert_eq!(first.level, AlertLevel::Critical);
    }

    #[test]
    fn queue_and_drain() {
        let mut relay = UpstreamRelay::new(Some("http://upstream.example.com".into()));

        relay.queue_alert(&make_alert(AlertLevel::Low));
        relay.queue_alert(&make_alert(AlertLevel::High));
        assert_eq!(relay.queue_len(), 2);

        let msgs = relay.drain_outbound();
        assert_eq!(msgs.len(), 2);
        assert_eq!(relay.queue_len(), 0);
    }
}
