//! Fixed-capacity circular buffer for streaming sensor samples.

use std::collections::VecDeque;

/// A fixed-capacity circular buffer that overwrites oldest data on overflow.
#[derive(Debug)]
pub struct RingBuffer<T> {
    buf: VecDeque<T>,
    capacity: usize,
}

impl<T: Clone> RingBuffer<T> {
    /// Create a new ring buffer with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            buf: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Append items, discarding oldest data if the buffer would exceed capacity.
    pub fn push(&mut self, items: &[T]) {
        for item in items {
            if self.buf.len() == self.capacity {
                self.buf.pop_front();
            }
            self.buf.push_back(item.clone());
        }
    }

    /// Returns true if there are at least `window_size` samples available.
    pub fn has_complete_window(&self, window_size: usize) -> bool {
        self.buf.len() >= window_size
    }

    /// Extract a window of `size` samples. After extraction, advance the read
    /// pointer by `size - overlap` samples. Returns None if insufficient data.
    pub fn extract_window(&mut self, size: usize, overlap: usize) -> Option<Vec<T>> {
        if self.buf.len() < size {
            return None;
        }
        let window: Vec<T> = self.buf.iter().take(size).cloned().collect();
        let advance = size.saturating_sub(overlap);
        for _ in 0..advance {
            self.buf.pop_front();
        }
        Some(window)
    }

    /// Current number of samples in the buffer.
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// Returns true if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_push_and_extract() {
        let mut rb = RingBuffer::new(10);
        rb.push(&[1, 2, 3, 4, 5]);
        assert_eq!(rb.len(), 5);
        assert!(!rb.has_complete_window(6));
        assert!(rb.has_complete_window(5));

        let window = rb.extract_window(5, 0).unwrap();
        assert_eq!(window, vec![1, 2, 3, 4, 5]);
        assert_eq!(rb.len(), 0);
    }

    #[test]
    fn overflow_discards_oldest() {
        let mut rb = RingBuffer::new(4);
        rb.push(&[1, 2, 3, 4]);
        assert_eq!(rb.len(), 4);

        rb.push(&[5, 6]);
        assert_eq!(rb.len(), 4);

        let window = rb.extract_window(4, 0).unwrap();
        assert_eq!(window, vec![3, 4, 5, 6]);
    }

    #[test]
    fn extract_with_overlap() {
        let mut rb = RingBuffer::new(16);
        rb.push(&(1..=10).collect::<Vec<i32>>());

        // Extract window of 4 with overlap 2 => advance by 2
        let w1 = rb.extract_window(4, 2).unwrap();
        assert_eq!(w1, vec![1, 2, 3, 4]);
        assert_eq!(rb.len(), 8); // 10 - 2 advanced

        let w2 = rb.extract_window(4, 2).unwrap();
        assert_eq!(w2, vec![3, 4, 5, 6]);
        assert_eq!(rb.len(), 6);
    }

    #[test]
    fn extract_returns_none_when_insufficient() {
        let mut rb = RingBuffer::new(10);
        rb.push(&[1, 2]);
        assert!(rb.extract_window(5, 0).is_none());
    }

    #[test]
    fn overlap_greater_than_size_clamps_to_zero_advance() {
        let mut rb = RingBuffer::new(10);
        rb.push(&[1, 2, 3, 4, 5]);
        let w = rb.extract_window(3, 5).unwrap();
        assert_eq!(w, vec![1, 2, 3]);
        // advance = 3 - 5 saturates to 0, so nothing consumed
        assert_eq!(rb.len(), 5);
    }

    #[test]
    fn massive_overflow() {
        let mut rb = RingBuffer::new(4);
        rb.push(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        assert_eq!(rb.len(), 4);
        let w = rb.extract_window(4, 0).unwrap();
        assert_eq!(w, vec![7, 8, 9, 10]);
    }

    #[test]
    fn empty_buffer() {
        let rb: RingBuffer<i32> = RingBuffer::new(8);
        assert!(rb.is_empty());
        assert!(!rb.has_complete_window(1));
    }
}
