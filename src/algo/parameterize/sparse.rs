//! Simple sparse matrix and conjugate gradient solver.
//!
//! This module provides a lightweight sparse matrix implementation (CSR format)
//! and a conjugate gradient solver for symmetric positive definite systems.

use nalgebra::DVector;

use crate::error::{MeshError, Result};

/// Compressed Sparse Row (CSR) matrix.
///
/// Stores a sparse matrix in CSR format for efficient matrix-vector multiplication.
#[derive(Debug, Clone)]
pub struct CsrMatrix {
    /// Number of rows.
    rows: usize,
    /// Number of columns.
    cols: usize,
    /// Row pointers: row_ptr[i] is the index in col_idx/values where row i starts.
    /// Length is rows + 1, with row_ptr[rows] = nnz.
    row_ptr: Vec<usize>,
    /// Column indices for each non-zero value.
    col_idx: Vec<usize>,
    /// Non-zero values.
    values: Vec<f64>,
}

impl CsrMatrix {
    /// Create a CSR matrix from triplets (row, col, value).
    ///
    /// Duplicate entries at the same (row, col) are summed.
    pub fn from_triplets(rows: usize, cols: usize, mut triplets: Vec<(usize, usize, f64)>) -> Self {
        if triplets.is_empty() {
            return Self {
                rows,
                cols,
                row_ptr: vec![0; rows + 1],
                col_idx: Vec::new(),
                values: Vec::new(),
            };
        }

        // Sort by (row, col) for CSR construction
        triplets.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

        // Merge duplicates and build CSR
        let mut row_ptr = vec![0usize; rows + 1];
        let mut col_idx = Vec::with_capacity(triplets.len());
        let mut values = Vec::with_capacity(triplets.len());

        let mut prev_row = usize::MAX;
        let mut prev_col = usize::MAX;

        for (row, col, val) in triplets {
            if row == prev_row && col == prev_col {
                // Same position: accumulate value
                *values.last_mut().unwrap() += val;
            } else {
                // New entry
                col_idx.push(col);
                values.push(val);
                // Update row pointers for any skipped rows
                for r in (prev_row.wrapping_add(1))..=row {
                    row_ptr[r] = col_idx.len() - 1;
                }
                prev_row = row;
                prev_col = col;
            }
        }

        // Fill remaining row pointers
        let nnz = col_idx.len();
        for r in (prev_row + 1)..=rows {
            row_ptr[r] = nnz;
        }

        Self {
            rows,
            cols,
            row_ptr,
            col_idx,
            values,
        }
    }

    /// Get the number of rows.
    #[inline]
    pub fn nrows(&self) -> usize {
        self.rows
    }

    /// Get the number of columns.
    #[inline]
    pub fn ncols(&self) -> usize {
        self.cols
    }

    /// Get the number of non-zero entries.
    #[inline]
    #[allow(dead_code)]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Multiply matrix by vector: y = A * x.
    pub fn mul_vec(&self, x: &DVector<f64>) -> DVector<f64> {
        assert_eq!(x.len(), self.cols, "Vector dimension mismatch");

        let mut y = DVector::zeros(self.rows);

        for i in 0..self.rows {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];

            let mut sum = 0.0;
            for k in start..end {
                sum += self.values[k] * x[self.col_idx[k]];
            }
            y[i] = sum;
        }

        y
    }

    /// Multiply matrix by vector, adding to existing vector: y += A * x.
    #[allow(dead_code)]
    pub fn mul_vec_add(&self, x: &DVector<f64>, y: &mut DVector<f64>) {
        assert_eq!(x.len(), self.cols, "Vector dimension mismatch");
        assert_eq!(y.len(), self.rows, "Output dimension mismatch");

        for i in 0..self.rows {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];

            let mut sum = 0.0;
            for k in start..end {
                sum += self.values[k] * x[self.col_idx[k]];
            }
            y[i] += sum;
        }
    }
}

/// Solve A*x = b using the Conjugate Gradient method.
///
/// Requires A to be symmetric positive definite.
///
/// # Arguments
///
/// * `a` - The system matrix (must be symmetric positive definite)
/// * `b` - The right-hand side vector
/// * `x0` - Optional initial guess (zeros if None)
/// * `max_iter` - Maximum number of iterations
/// * `tolerance` - Convergence tolerance (relative residual norm)
///
/// # Returns
///
/// The solution vector x, or an error if convergence fails.
pub fn conjugate_gradient(
    a: &CsrMatrix,
    b: &DVector<f64>,
    x0: Option<&DVector<f64>>,
    max_iter: usize,
    tolerance: f64,
) -> Result<DVector<f64>> {
    let n = b.len();
    assert_eq!(a.nrows(), n, "Matrix-vector dimension mismatch");
    assert_eq!(a.ncols(), n, "Matrix must be square");

    // Initial guess
    let mut x = match x0 {
        Some(x0) => x0.clone(),
        None => DVector::zeros(n),
    };

    // r = b - A*x
    let mut r = b - a.mul_vec(&x);

    // Check if initial guess is already good enough
    let b_norm = b.norm();
    if b_norm < 1e-15 {
        return Ok(x);
    }

    let mut r_norm_sq = r.dot(&r);
    if r_norm_sq.sqrt() / b_norm < tolerance {
        return Ok(x);
    }

    // p = r
    let mut p = r.clone();

    for _iter in 0..max_iter {
        // Ap = A * p
        let ap = a.mul_vec(&p);

        // alpha = (r · r) / (p · Ap)
        let p_ap = p.dot(&ap);
        if p_ap.abs() < 1e-15 {
            // Matrix might be singular or nearly so
            break;
        }
        let alpha = r_norm_sq / p_ap;

        // x = x + alpha * p
        x += alpha * &p;

        // r = r - alpha * Ap
        r -= alpha * &ap;

        // Check convergence
        let new_r_norm_sq = r.dot(&r);
        if new_r_norm_sq.sqrt() / b_norm < tolerance {
            return Ok(x);
        }

        // beta = (r_new · r_new) / (r_old · r_old)
        let beta = new_r_norm_sq / r_norm_sq;

        // p = r + beta * p
        p = &r + beta * &p;

        r_norm_sq = new_r_norm_sq;
    }

    // Did not converge
    Err(MeshError::ConvergenceFailed {
        iterations: max_iter,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csr_from_triplets() {
        // 2x2 matrix:
        // [ 4  1 ]
        // [ 1  3 ]
        let triplets = vec![(0, 0, 4.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 3.0)];
        let a = CsrMatrix::from_triplets(2, 2, triplets);

        assert_eq!(a.nrows(), 2);
        assert_eq!(a.ncols(), 2);
        assert_eq!(a.nnz(), 4);
    }

    #[test]
    fn test_csr_from_triplets_with_duplicates() {
        // Same matrix but with duplicate entries that should be summed
        let triplets = vec![
            (0, 0, 2.0),
            (0, 0, 2.0), // Duplicate: should sum to 4.0
            (0, 1, 1.0),
            (1, 0, 1.0),
            (1, 1, 3.0),
        ];
        let a = CsrMatrix::from_triplets(2, 2, triplets);

        let x = DVector::from_vec(vec![1.0, 0.0]);
        let y = a.mul_vec(&x);

        assert!((y[0] - 4.0).abs() < 1e-10);
        assert!((y[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_csr_mul_vec() {
        // [ 4  1 ]   [ 1 ]   [ 5 ]
        // [ 1  3 ] * [ 1 ] = [ 4 ]
        let triplets = vec![(0, 0, 4.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 3.0)];
        let a = CsrMatrix::from_triplets(2, 2, triplets);

        let x = DVector::from_vec(vec![1.0, 1.0]);
        let y = a.mul_vec(&x);

        assert!((y[0] - 5.0).abs() < 1e-10);
        assert!((y[1] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_cg_simple() {
        // Solve:
        // [ 4  1 ]   [ x ]   [ 1 ]
        // [ 1  3 ] * [ y ] = [ 2 ]
        //
        // Solution: x = 1/11, y = 7/11
        let triplets = vec![(0, 0, 4.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 3.0)];
        let a = CsrMatrix::from_triplets(2, 2, triplets);
        let b = DVector::from_vec(vec![1.0, 2.0]);

        let x = conjugate_gradient(&a, &b, None, 100, 1e-10).unwrap();

        // Verify A*x = b
        let residual = a.mul_vec(&x) - b;
        assert!(residual.norm() < 1e-8);

        // Check solution values
        assert!((x[0] - 1.0 / 11.0).abs() < 1e-8);
        assert!((x[1] - 7.0 / 11.0).abs() < 1e-8);
    }

    #[test]
    fn test_cg_larger_system() {
        // 4x4 symmetric positive definite matrix (diagonally dominant)
        let triplets = vec![
            (0, 0, 10.0),
            (0, 1, 1.0),
            (0, 2, 2.0),
            (1, 0, 1.0),
            (1, 1, 10.0),
            (1, 2, 1.0),
            (2, 0, 2.0),
            (2, 1, 1.0),
            (2, 2, 10.0),
            (2, 3, 1.0),
            (3, 2, 1.0),
            (3, 3, 10.0),
        ];
        let a = CsrMatrix::from_triplets(4, 4, triplets);
        let b = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

        let x = conjugate_gradient(&a, &b, None, 100, 1e-10).unwrap();

        // Verify A*x = b
        let residual = a.mul_vec(&x) - &b;
        assert!(residual.norm() < 1e-8);
    }

    #[test]
    fn test_cg_with_initial_guess() {
        let triplets = vec![(0, 0, 4.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 3.0)];
        let a = CsrMatrix::from_triplets(2, 2, triplets);
        let b = DVector::from_vec(vec![1.0, 2.0]);

        // Start with a good initial guess
        let x0 = DVector::from_vec(vec![0.1, 0.6]);
        let x = conjugate_gradient(&a, &b, Some(&x0), 100, 1e-10).unwrap();

        let residual = a.mul_vec(&x) - b;
        assert!(residual.norm() < 1e-8);
    }
}
