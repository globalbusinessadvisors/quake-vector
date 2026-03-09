//! Distance functions for vector comparison.

/// Cosine distance between two vectors: 1.0 - dot_product(a, b).
/// Assumes both vectors are L2-normalized, so dot product equals cosine similarity.
#[inline]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    1.0 - dot
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_vectors_zero_distance() {
        let mut v = vec![0.0f32; 256];
        // Make a unit vector
        v[0] = 1.0;
        let d = cosine_distance(&v, &v);
        assert!(d.abs() < 1e-6, "identical vectors should have distance 0, got {d}");
    }

    #[test]
    fn orthogonal_vectors_distance_one() {
        let mut a = vec![0.0f32; 256];
        let mut b = vec![0.0f32; 256];
        a[0] = 1.0;
        b[1] = 1.0;
        let d = cosine_distance(&a, &b);
        assert!(
            (d - 1.0).abs() < 1e-6,
            "orthogonal vectors should have distance 1, got {d}"
        );
    }

    #[test]
    fn opposite_vectors_distance_two() {
        let mut a = vec![0.0f32; 256];
        let mut b = vec![0.0f32; 256];
        a[0] = 1.0;
        b[0] = -1.0;
        let d = cosine_distance(&a, &b);
        assert!(
            (d - 2.0).abs() < 1e-6,
            "opposite vectors should have distance 2, got {d}"
        );
    }
}
