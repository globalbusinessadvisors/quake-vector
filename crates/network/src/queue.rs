//! Generic bounded FIFO queue with capacity limit and priority push.

use std::collections::VecDeque;

/// A bounded FIFO queue that evicts the oldest item when full.
#[derive(Debug)]
pub struct BoundedQueue<T> {
    buf: VecDeque<T>,
    capacity: usize,
}

impl<T> BoundedQueue<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            buf: VecDeque::with_capacity(capacity.min(1024)),
            capacity,
        }
    }

    /// Push to the back. Returns the evicted item if the queue was full.
    pub fn push(&mut self, item: T) -> Option<T> {
        let evicted = if self.buf.len() >= self.capacity {
            self.buf.pop_front()
        } else {
            None
        };
        self.buf.push_back(item);
        evicted
    }

    /// Pop from the front.
    pub fn pop(&mut self) -> Option<T> {
        self.buf.pop_front()
    }

    /// Push to the front (priority). Evicts from the back if full.
    pub fn push_priority(&mut self, item: T) -> Option<T> {
        let evicted = if self.buf.len() >= self.capacity {
            self.buf.pop_back()
        } else {
            None
        };
        self.buf.push_front(item);
        evicted
    }

    /// Number of items in the queue.
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// Drain all items.
    pub fn drain(&mut self) -> Vec<T> {
        self.buf.drain(..).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_and_pop() {
        let mut q = BoundedQueue::new(3);
        assert!(q.push(1).is_none());
        assert!(q.push(2).is_none());
        assert!(q.push(3).is_none());
        assert_eq!(q.len(), 3);

        assert_eq!(q.pop(), Some(1));
        assert_eq!(q.pop(), Some(2));
        assert_eq!(q.pop(), Some(3));
        assert_eq!(q.pop(), None);
    }

    #[test]
    fn overflow_evicts_oldest() {
        let mut q = BoundedQueue::new(3);
        q.push(1);
        q.push(2);
        q.push(3);

        let evicted = q.push(4);
        assert_eq!(evicted, Some(1));
        assert_eq!(q.len(), 3);
        assert_eq!(q.pop(), Some(2));
    }

    #[test]
    fn priority_push_to_front() {
        let mut q = BoundedQueue::new(3);
        q.push(1);
        q.push(2);
        q.push(3);

        // Priority push evicts from back
        let evicted = q.push_priority(0);
        assert_eq!(evicted, Some(3));
        assert_eq!(q.len(), 3);
        assert_eq!(q.pop(), Some(0)); // priority item is at front
        assert_eq!(q.pop(), Some(1));
        assert_eq!(q.pop(), Some(2));
    }

    #[test]
    fn priority_push_when_not_full() {
        let mut q = BoundedQueue::new(5);
        q.push(2);
        q.push(3);
        let evicted = q.push_priority(1);
        assert!(evicted.is_none());
        assert_eq!(q.pop(), Some(1));
        assert_eq!(q.pop(), Some(2));
    }

    #[test]
    fn drain_empties_queue() {
        let mut q = BoundedQueue::new(5);
        q.push(10);
        q.push(20);
        q.push(30);

        let items = q.drain();
        assert_eq!(items, vec![10, 20, 30]);
        assert!(q.is_empty());
    }

    #[test]
    fn empty_queue() {
        let q: BoundedQueue<i32> = BoundedQueue::new(10);
        assert!(q.is_empty());
        assert_eq!(q.len(), 0);
    }
}
