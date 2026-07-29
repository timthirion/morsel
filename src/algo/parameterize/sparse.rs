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
                // Update row pointers for any skipped rows. The range inverts
                // when this entry shares `row` with the previous one, so guard
                // rather than slicing straight away.
                let entry = col_idx.len() - 1;
                let first_skipped = prev_row.wrapping_add(1);
                if first_skipped <= row {
                    row_ptr[first_skipped..=row].fill(entry);
                }
                prev_row = row;
                prev_col = col;
            }
        }

        // Fill remaining row pointers
        let nnz = col_idx.len();
        let first_unset = prev_row.wrapping_add(1);
        if first_unset <= rows {
            row_ptr[first_unset..=rows].fill(nnz);
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

impl CsrMatrix {
    /// Enumerate the stored non-zeros as `(row, col, value)`.
    ///
    /// Useful for re-deriving a reduced system — see [`PinnedReduction`] — without
    /// having to keep the original triplet list alive alongside the matrix.
    pub(crate) fn triplets(&self) -> Vec<(usize, usize, f64)> {
        let mut out = Vec::with_capacity(self.values.len());
        for row in 0..self.rows {
            for k in self.row_ptr[row]..self.row_ptr[row + 1] {
                out.push((row, self.col_idx[k], self.values[k]));
            }
        }
        out
    }

    /// Extract the main diagonal.
    pub fn diagonal(&self) -> DVector<f64> {
        let n = self.rows.min(self.cols);
        let mut d = DVector::zeros(n);
        for i in 0..n {
            for k in self.row_ptr[i]..self.row_ptr[i + 1] {
                if self.col_idx[k] == i {
                    d[i] += self.values[k];
                }
            }
        }
        d
    }
}

/// Solve A*x = b using Jacobi-preconditioned Conjugate Gradient.
///
/// Requires A to be symmetric positive definite.
///
/// The diagonal preconditioner earns its place on the parameterization systems,
/// which are cotangent-weighted Laplacians (or their normal equations). Cotangent
/// weights vary with triangle shape, so the diagonal varies with vertex valence
/// and local element quality; scaling it away leaves CG facing only the mesh's
/// intrinsic conditioning.
///
/// It also rescues systems whose diagonal has been inflated deliberately — a pin
/// imposed by a large penalty, say — but that is a workaround, not a fix. LSCM
/// and ARAP both eliminate their pinned degrees of freedom instead, so no such
/// term reaches this solver.
///
/// # Arguments
///
/// * `a` - The system matrix (must be symmetric positive definite)
/// * `b` - The right-hand side vector
/// * `x0` - Optional initial guess (zeros if None)
/// * `max_iter` - Maximum number of iterations
/// * `tolerance` - Convergence tolerance (relative residual norm)
pub fn preconditioned_conjugate_gradient(
    a: &CsrMatrix,
    b: &DVector<f64>,
    x0: Option<&DVector<f64>>,
    max_iter: usize,
    tolerance: f64,
) -> Result<DVector<f64>> {
    let n = b.len();
    assert_eq!(a.nrows(), n, "Matrix-vector dimension mismatch");
    assert_eq!(a.ncols(), n, "Matrix must be square");

    // Inverse diagonal, falling back to 1 where the diagonal vanishes.
    let diag = a.diagonal();
    let inv_diag: Vec<f64> = diag
        .iter()
        .map(|&d| if d.abs() > 1e-300 { 1.0 / d } else { 1.0 })
        .collect();
    let apply_precond = |v: &DVector<f64>| -> DVector<f64> {
        DVector::from_iterator(n, v.iter().zip(&inv_diag).map(|(&vi, &m)| vi * m))
    };

    let mut x = match x0 {
        Some(x0) => x0.clone(),
        None => DVector::zeros(n),
    };

    let b_norm = b.norm();
    if b_norm < 1e-15 {
        return Ok(x);
    }

    let mut r = b - a.mul_vec(&x);
    if r.norm() / b_norm < tolerance {
        return Ok(x);
    }

    let mut z = apply_precond(&r);
    let mut p = z.clone();
    let mut rz = r.dot(&z);

    for _ in 0..max_iter {
        let ap = a.mul_vec(&p);
        let p_ap = p.dot(&ap);
        if p_ap.abs() < 1e-300 {
            break;
        }

        let alpha = rz / p_ap;
        x.axpy(alpha, &p, 1.0);
        r.axpy(-alpha, &ap, 1.0);

        if r.norm() / b_norm < tolerance {
            return Ok(x);
        }

        z = apply_precond(&r);
        let rz_new = r.dot(&z);
        if rz.abs() < 1e-300 {
            break;
        }
        let beta = rz_new / rz;
        // p = z + beta * p
        p *= beta;
        p += &z;
        rz = rz_new;
    }

    Err(MeshError::ConvergenceFailed {
        iterations: max_iter,
    })
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

/// A symmetric system with one degree of freedom pinned and eliminated.
///
/// The cotangent Laplacian is singular — its kernel is the constants, because only
/// *differences* of the unknown appear. Pinning a single vertex removes exactly
/// that one-dimensional kernel. Eliminating it, rather than adding a penalty to
/// the diagonal, keeps the condition number the mesh's own and leaves the CG
/// tolerance meaning what it says.
///
/// Two callers rely on this, for the same reason:
///
/// - **ARAP's global step**, where the matrix is shared by `u`, by `v`, and across
///   every local/global iteration while only the right-hand side changes.
/// - **The heat method's Poisson solve**, where `φ` is determined only up to a
///   constant. Without pinning, CG cannot drive the relative residual below the
///   null-space contribution and reports `ConvergenceFailed` at any iteration
///   count — which it did, for every mesh larger than eight vertices.
///
/// [`PinnedReduction::reduce_rhs`] folds the pinned column into the right-hand
/// side, so a non-zero pinned value works as well as zero.
pub(crate) struct PinnedReduction {
    /// The Laplacian restricted to free vertices, `L_ff`.
    pub(crate) matrix: CsrMatrix,
    /// Global vertex index to reduced index; `None` for the pinned vertex.
    reduced_index: Vec<Option<usize>>,
    /// The pinned vertex.
    pinned: usize,
    /// Non-zeros of the pinned column over free rows: `(reduced_row, value)`.
    pinned_column: Vec<(usize, f64)>,
}

impl PinnedReduction {
    pub(crate) fn new(triplets: &[(usize, usize, f64)], n_vertices: usize, pinned: usize) -> Self {
        let mut reduced_index = vec![None; n_vertices];
        let mut n_free = 0;
        for (v, slot) in reduced_index.iter_mut().enumerate() {
            if v != pinned {
                *slot = Some(n_free);
                n_free += 1;
            }
        }

        // The pinned column can receive several triplets for the same row, so
        // accumulate before storing.
        let mut column_acc = vec![0.0; n_free];
        let mut reduced_triplets = Vec::with_capacity(triplets.len());

        for &(row, col, value) in triplets {
            let Some(r) = reduced_index[row] else {
                continue;
            };
            match reduced_index[col] {
                Some(c) => reduced_triplets.push((r, c, value)),
                None => column_acc[r] += value,
            }
        }

        let pinned_column = column_acc
            .into_iter()
            .enumerate()
            .filter(|(_, v)| *v != 0.0)
            .collect();

        Self {
            matrix: CsrMatrix::from_triplets(n_free, n_free, reduced_triplets),
            reduced_index,
            pinned,
            pinned_column,
        }
    }

    pub(crate) fn n_free(&self) -> usize {
        self.matrix.nrows()
    }

    /// The vertex held fixed by this reduction.
    #[allow(dead_code)]
    pub(crate) fn pinned(&self) -> usize {
        self.pinned
    }

    /// Restrict a full right-hand side to the free rows, moving the pinned
    /// vertex's known contribution across: `rhs_f − L_fp x_p`.
    pub(crate) fn reduce_rhs(&self, full: &DVector<f64>, pinned_value: f64) -> DVector<f64> {
        let mut rhs = DVector::zeros(self.n_free());
        for (v, maybe_r) in self.reduced_index.iter().enumerate() {
            if let Some(r) = maybe_r {
                rhs[*r] = full[v];
            }
        }
        for &(r, l_fp) in &self.pinned_column {
            rhs[r] -= l_fp * pinned_value;
        }
        rhs
    }

    /// Scatter a reduced solution back over the full vertex range, leaving the
    /// pinned entry untouched.
    pub(crate) fn scatter(&self, solution: &DVector<f64>, out: &mut [f64]) {
        for (v, maybe_r) in self.reduced_index.iter().enumerate() {
            if let Some(r) = maybe_r {
                out[v] = solution[*r];
            }
        }
    }
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
