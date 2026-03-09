//! Thread health monitoring via lock-free atomic heartbeats.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// Handle given to a thread to report heartbeats.
pub struct ThreadHeartbeatHandle {
    last_heartbeat_us: Arc<AtomicU64>,
}

impl ThreadHeartbeatHandle {
    /// Record a heartbeat at the current time.
    pub fn beat(&self) {
        let now_us = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;
        self.last_heartbeat_us.store(now_us, Ordering::Relaxed);
    }
}

struct ThreadEntry {
    name: String,
    last_heartbeat_us: Arc<AtomicU64>,
}

/// Monitors health of registered threads via atomic heartbeats.
pub struct ThreadHealthMonitor {
    threads: Vec<ThreadEntry>,
}

impl ThreadHealthMonitor {
    pub fn new() -> Self {
        Self {
            threads: Vec::new(),
        }
    }

    /// Register a thread and return a handle for heartbeating.
    pub fn register_thread(&mut self, name: &str) -> ThreadHeartbeatHandle {
        let ts = Arc::new(AtomicU64::new(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_micros() as u64,
        ));
        self.threads.push(ThreadEntry {
            name: name.to_string(),
            last_heartbeat_us: Arc::clone(&ts),
        });
        ThreadHeartbeatHandle {
            last_heartbeat_us: ts,
        }
    }

    /// Returns names of threads that have not heartbeated within `max_stale_ms`.
    pub fn check_all(&self, max_stale_ms: u64) -> Vec<String> {
        let now_us = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;
        let max_stale_us = max_stale_ms * 1000;

        self.threads
            .iter()
            .filter(|entry| {
                let last = entry.last_heartbeat_us.load(Ordering::Relaxed);
                now_us.saturating_sub(last) > max_stale_us
            })
            .map(|entry| entry.name.clone())
            .collect()
    }
}

impl Default for ThreadHealthMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_stale_thread() {
        let mut monitor = ThreadHealthMonitor::new();
        let handle_a = monitor.register_thread("sensor-poll");
        let _handle_b = monitor.register_thread("embedding");

        // Both just registered, should be fresh
        let stale = monitor.check_all(1000);
        assert!(stale.is_empty(), "no threads should be stale yet: {stale:?}");

        // Beat only handle_a
        handle_a.beat();

        // Simulate staleness by checking with 0ms tolerance
        // Both were registered with "now" timestamps and handle_a just beat,
        // so with 0ms tolerance everything is stale
        let stale = monitor.check_all(0);
        // At 0ms tolerance, both should be stale (can't beat faster than check)
        assert!(!stale.is_empty());

        // With generous tolerance, none should be stale
        let stale = monitor.check_all(60_000);
        assert!(stale.is_empty(), "with 60s tolerance nothing should be stale");
    }

    #[test]
    fn fresh_threads_not_reported() {
        let mut monitor = ThreadHealthMonitor::new();
        let handle = monitor.register_thread("main-loop");
        handle.beat();

        let stale = monitor.check_all(5000);
        assert!(stale.is_empty());
    }
}
