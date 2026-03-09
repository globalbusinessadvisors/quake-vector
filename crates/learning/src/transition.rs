//! Learnable wave type transition matrix.

use quake_vector_types::WaveType;

/// A 3x3 learnable transition matrix indexed by (WaveType, WaveType).
///
/// Rows correspond to source wave types [P, S, Surface], columns to targets.
/// WaveType::Unknown is mapped to the P row/column as a fallback.
#[derive(Debug, Clone)]
pub struct TransitionMatrix {
    /// 3x3 matrix in row-major order.
    data: [[f32; 3]; 3],
}

impl TransitionMatrix {
    /// Create a new transition matrix initialized to uniform 1/3 values.
    pub fn new() -> Self {
        Self {
            data: [[1.0 / 3.0; 3]; 3],
        }
    }

    /// Get the transition bias from one wave type to another.
    pub fn get_bias(&self, from: WaveType, to: WaveType) -> f32 {
        self.data[Self::wave_idx(from)][Self::wave_idx(to)]
    }

    /// Update a transition value and re-normalize the row.
    pub fn update(&mut self, from: WaveType, to: WaveType, delta: f32) {
        let row = Self::wave_idx(from);
        let col = Self::wave_idx(to);
        self.data[row][col] = (self.data[row][col] + delta).max(0.0);
        self.normalize_row(row);
    }

    /// Normalize a row to sum to 1.0.
    fn normalize_row(&mut self, row: usize) {
        let sum: f32 = self.data[row].iter().sum();
        if sum > 1e-8 {
            for v in self.data[row].iter_mut() {
                *v /= sum;
            }
        } else {
            // Reset to uniform if all zeros
            self.data[row] = [1.0 / 3.0; 3];
        }
    }

    /// Map WaveType to matrix index.
    fn wave_idx(w: WaveType) -> usize {
        match w {
            WaveType::P | WaveType::Unknown => 0,
            WaveType::S => 1,
            WaveType::Surface => 2,
        }
    }

    /// Get the full matrix (for inspection/testing).
    pub fn matrix(&self) -> &[[f32; 3]; 3] {
        &self.data
    }
}

impl Default for TransitionMatrix {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_rows_sum_to_one() {
        let tm = TransitionMatrix::new();
        for row in tm.matrix() {
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6, "row sum = {sum}");
        }
    }

    #[test]
    fn rows_sum_to_one_after_update() {
        let mut tm = TransitionMatrix::new();
        tm.update(WaveType::P, WaveType::S, 0.5);
        tm.update(WaveType::S, WaveType::Surface, -0.1);
        tm.update(WaveType::Surface, WaveType::P, 1.0);

        for (i, row) in tm.matrix().iter().enumerate() {
            let sum: f32 = row.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "row {i} sum = {sum} after update"
            );
        }
    }

    #[test]
    fn negative_update_clamps_to_zero() {
        let mut tm = TransitionMatrix::new();
        // Subtract more than current value
        tm.update(WaveType::P, WaveType::P, -1.0);
        assert!(tm.get_bias(WaveType::P, WaveType::P) >= 0.0);
        // Row should still sum to 1
        let sum: f32 = tm.matrix()[0].iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn get_bias_reflects_updates() {
        let mut tm = TransitionMatrix::new();
        let initial = tm.get_bias(WaveType::P, WaveType::S);
        tm.update(WaveType::P, WaveType::S, 0.5);
        let updated = tm.get_bias(WaveType::P, WaveType::S);
        assert!(updated > initial, "bias should increase after positive update");
    }
}
