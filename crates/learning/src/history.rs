//! Sliding window of recent wave observations.

use quake_vector_types::{NodeId, TemporalMeta};

/// A fixed-capacity sliding window of (NodeId, TemporalMeta) pairs.
pub struct WaveHistory {
    entries: Vec<(NodeId, TemporalMeta)>,
    capacity: usize,
}

impl WaveHistory {
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity.min(1024)),
            capacity,
        }
    }

    /// Push a new observation, evicting the oldest if at capacity.
    pub fn push(&mut self, id: NodeId, meta: TemporalMeta) {
        if self.entries.len() == self.capacity {
            self.entries.remove(0);
        }
        self.entries.push((id, meta));
    }

    /// Return the last `n` entries (or all if fewer than `n`).
    pub fn last_n(&self, n: usize) -> &[(NodeId, TemporalMeta)] {
        let start = self.entries.len().saturating_sub(n);
        &self.entries[start..]
    }

    /// Return all entries.
    pub fn entries(&self) -> &[(NodeId, TemporalMeta)] {
        &self.entries
    }

    /// Current number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quake_vector_types::{StationId, WaveType};

    fn make_meta(ts: u64) -> TemporalMeta {
        TemporalMeta {
            timestamp_us: ts,
            wave_type: WaveType::P,
            station_id: StationId(1),
            amplitude_rms: 100.0,
            dominant_freq_hz: 5.0,
        }
    }

    #[test]
    fn capacity_enforced() {
        let mut history = WaveHistory::new(5);
        for i in 0..10 {
            history.push(NodeId(i), make_meta(i * 1000));
        }
        assert_eq!(history.len(), 5);
        // Should contain entries 5..10
        assert_eq!(history.entries()[0].0, NodeId(5));
        assert_eq!(history.entries()[4].0, NodeId(9));
    }

    #[test]
    fn last_n_returns_tail() {
        let mut history = WaveHistory::new(100);
        for i in 0..20 {
            history.push(NodeId(i), make_meta(i * 1000));
        }

        let last_5 = history.last_n(5);
        assert_eq!(last_5.len(), 5);
        assert_eq!(last_5[0].0, NodeId(15));
        assert_eq!(last_5[4].0, NodeId(19));
    }

    #[test]
    fn last_n_larger_than_entries() {
        let mut history = WaveHistory::new(100);
        for i in 0..3 {
            history.push(NodeId(i), make_meta(i * 1000));
        }
        let all = history.last_n(100);
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn empty_history() {
        let history = WaveHistory::new(10);
        assert!(history.is_empty());
        assert_eq!(history.last_n(5).len(), 0);
    }

    #[test]
    fn oldest_entries_evicted() {
        let mut history = WaveHistory::new(3);
        history.push(NodeId(0), make_meta(0));
        history.push(NodeId(1), make_meta(1000));
        history.push(NodeId(2), make_meta(2000));
        history.push(NodeId(3), make_meta(3000));

        assert_eq!(history.len(), 3);
        assert_eq!(history.entries()[0].0, NodeId(1));
        assert_eq!(history.entries()[2].0, NodeId(3));
    }
}
