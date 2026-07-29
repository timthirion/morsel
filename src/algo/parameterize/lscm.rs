//! Least Squares Conformal Maps (LSCM) parameterization.
//!
//! LSCM computes a conformal (angle-preserving) parameterization of a triangle
//! mesh with boundary. The algorithm minimizes the conformal energy, which
//! measures deviation from a conformal (angle-preserving) map.
//!
//! # References
//!
//! - Lévy, B., Petitjean, S., Ray, N., & Maillot, J. (2002). "Least squares
//!   conformal maps for automatic texture atlas generation." ACM SIGGRAPH.

use nalgebra::{DVector, Point2, Point3};

use crate::error::{MeshError, Result};
use crate::mesh::{to_face_vertex, HalfEdgeMesh, MeshIndex};

use super::sparse::{preconditioned_conjugate_gradient, CsrMatrix};
use super::uv::UVMap;

/// Options for LSCM parameterization.
#[derive(Debug, Clone)]
pub struct LSCMOptions {
    /// Strategy for selecting pinned (fixed) vertices.
    pub pin_strategy: PinStrategy,

    /// Maximum iterations for the conjugate gradient solver.
    ///
    /// With the pins eliminated the system is just a mesh Laplacian, so the
    /// iteration count grows with refinement rather than with any penalty.
    /// Reaching `tolerance = 1e-10` on a flat grid took:
    ///
    /// | vertices | iterations |
    /// |----------|------------|
    /// | 25       | 14         |
    /// | 289      | 81         |
    /// | 1089     | 301        |
    /// | 5041     | 1117       |
    ///
    /// The default leaves room for meshes well past that; each iteration is one
    /// sparse mat-vec, so an unused allowance costs nothing. Exceeding it is
    /// reported as [`MeshError::ConvergenceFailed`] rather than passed off as a
    /// result.
    pub max_iterations: usize,

    /// Convergence tolerance for the CG solver, as a *relative* residual
    /// `‖r‖ / ‖b‖`.
    ///
    /// Pins are imposed by eliminating their degrees of freedom (see
    /// [`reduce_pinned_dofs`]), so both the matrix and `‖b‖` are on the scale of
    /// the conformal energy itself and this value means what it appears to mean.
    /// An earlier penalty formulation coupled the two: it inflated `‖b‖` to
    /// `λ · pin_target`, so a `1e-8` relative tolerance left an absolute
    /// residual of ~`1e-2` in the rows carrying the geometry and LSCM returned
    /// wrong maps while reporting success. That coupling is gone.
    pub tolerance: f64,
}

impl Default for LSCMOptions {
    fn default() -> Self {
        Self {
            pin_strategy: PinStrategy::Automatic,
            max_iterations: 20_000,
            // Now that the residual is a meaningful error proxy, the resulting
            // area error tracks this value directly: 1e-10 in gives ~1e-9 of
            // distortion on a case whose exact answer is the identity.
            tolerance: 1e-10,
        }
    }
}

impl LSCMOptions {
    /// Create options with automatic pin selection (farthest boundary vertices).
    pub fn automatic() -> Self {
        Self::default()
    }

    /// Create options with manually specified pinned vertices.
    pub fn with_pins(pin0: PinnedVertex, pin1: PinnedVertex) -> Self {
        Self {
            pin_strategy: PinStrategy::Manual(pin0, pin1),
            ..Default::default()
        }
    }

    /// Set the maximum CG iterations.
    pub fn with_max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iterations = max_iter;
        self
    }

    /// Set the convergence tolerance.
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }
}

/// Strategy for selecting which vertices to pin (fix) during parameterization.
#[derive(Debug, Clone)]
pub enum PinStrategy {
    /// Automatically select the two farthest boundary vertices.
    Automatic,

    /// Use specified vertex indices with their UV coordinates.
    Manual(PinnedVertex, PinnedVertex),
}

/// A vertex pinned to a specific UV coordinate.
#[derive(Debug, Clone, Copy)]
pub struct PinnedVertex {
    /// The vertex index to pin.
    pub vertex: usize,
    /// The fixed U coordinate.
    pub u: f64,
    /// The fixed V coordinate.
    pub v: f64,
}

impl PinnedVertex {
    /// Create a new pinned vertex.
    pub fn new(vertex: usize, u: f64, v: f64) -> Self {
        Self { vertex, u, v }
    }
}

/// Compute LSCM (Least Squares Conformal Maps) parameterization.
///
/// This algorithm computes UV coordinates that minimize angle (conformal)
/// distortion. It requires the mesh to have a boundary (disk topology).
///
/// # Arguments
///
/// * `mesh` - The input mesh (must have boundary)
/// * `options` - Parameterization options
///
/// # Returns
///
/// UV coordinates for each vertex, or an error if parameterization fails.
///
/// # Errors
///
/// Returns an error if:
/// - The mesh has no boundary (closed mesh)
/// - The mesh is empty
/// - The linear system fails to converge
///
/// # Example
///
/// ```no_run
/// use morsel::prelude::*;
/// use morsel::algo::parameterize::{lscm, LSCMOptions};
///
/// let mesh: HalfEdgeMesh = morsel::io::load("disk.obj").unwrap();
/// let uv_map = lscm(&mesh, &LSCMOptions::default()).unwrap();
/// ```
pub fn lscm<I: MeshIndex>(mesh: &HalfEdgeMesh<I>, options: &LSCMOptions) -> Result<UVMap<I>> {
    let n_vertices = mesh.num_vertices();
    if n_vertices == 0 {
        return Err(MeshError::EmptyMesh);
    }

    // Convert to face-vertex representation
    let (vertices, faces) = to_face_vertex(mesh);

    // Find boundary vertices
    let boundary = find_boundary_vertices(&faces, n_vertices);
    if boundary.is_empty() {
        return Err(MeshError::NoBoundary);
    }

    // Determine pinned vertices
    let (pin0, pin1) = match &options.pin_strategy {
        PinStrategy::Automatic => select_farthest_boundary_pair(&vertices, &boundary),
        PinStrategy::Manual(p0, p1) => (*p0, *p1),
    };

    if pin0.vertex >= n_vertices || pin1.vertex >= n_vertices {
        return Err(MeshError::InvalidState(format!(
            "pinned vertex out of range: {} / {} (mesh has {n_vertices} vertices)",
            pin0.vertex, pin1.vertex
        )));
    }
    if pin0.vertex == pin1.vertex {
        // Two distinct vertices are what remove the similarity kernel; pinning
        // one twice leaves the system singular.
        return Err(MeshError::InvalidState(format!(
            "LSCM needs two distinct pinned vertices, both were {}",
            pin0.vertex
        )));
    }

    // Assemble the conformal energy, then impose the pins by elimination.
    let triplets = build_conformal_normal_equations(&vertices, &faces, n_vertices);
    let (matrix, rhs, reduced_index) = reduce_pinned_dofs(&triplets, n_vertices, &pin0, &pin1);

    // Pinned vertices are exact by construction; the rest come from the solve.
    let mut uv_coords = vec![Point2::origin(); n_vertices];
    uv_coords[pin0.vertex] = Point2::new(pin0.u, pin0.v);
    uv_coords[pin1.vertex] = Point2::new(pin1.u, pin1.v);

    // A mesh of only the two pinned vertices has nothing left to solve for.
    if matrix.nrows() > 0 {
        let solution = preconditioned_conjugate_gradient(
            &matrix,
            &rhs,
            None,
            options.max_iterations,
            options.tolerance,
        )?;

        for i in 0..n_vertices {
            if let Some(r) = reduced_index[i] {
                uv_coords[i].x = solution[r];
            }
            if let Some(r) = reduced_index[n_vertices + i] {
                uv_coords[i].y = solution[r];
            }
        }
    }

    let mut uv_map = UVMap::new(uv_coords);
    uv_map.normalize();

    Ok(uv_map)
}

/// Find boundary vertices from face-vertex representation.
fn find_boundary_vertices(faces: &[[usize; 3]], n_vertices: usize) -> Vec<usize> {
    use std::collections::HashMap;

    // Count edge occurrences
    let mut edge_count: HashMap<(usize, usize), usize> = HashMap::new();

    for face in faces {
        for i in 0..3 {
            let v0 = face[i];
            let v1 = face[(i + 1) % 3];
            let edge = if v0 < v1 { (v0, v1) } else { (v1, v0) };
            *edge_count.entry(edge).or_insert(0) += 1;
        }
    }

    // Boundary edges have count 1
    let mut is_boundary = vec![false; n_vertices];
    for ((v0, v1), count) in edge_count {
        if count == 1 {
            is_boundary[v0] = true;
            is_boundary[v1] = true;
        }
    }

    is_boundary
        .iter()
        .enumerate()
        .filter_map(|(i, &b)| if b { Some(i) } else { None })
        .collect()
}

/// Select two boundary vertices for pinning with UV positions based on 3D geometry.
///
/// Instead of always pinning to (0,0) and (1,0), this function computes UV
/// positions that reflect the 2D extent of the boundary, ensuring the
/// parameterization spans the UV space well.
fn select_farthest_boundary_pair(
    vertices: &[Point3<f64>],
    boundary: &[usize],
) -> (PinnedVertex, PinnedVertex) {
    // Find farthest pair - these will be our pins
    let mut max_dist = 0.0;
    let mut best_pair = (boundary[0], boundary[0]);
    for (i, &v0) in boundary.iter().enumerate() {
        for &v1 in boundary.iter().skip(i + 1) {
            let dist = (vertices[v1] - vertices[v0]).norm_squared();
            if dist > max_dist {
                max_dist = dist;
                best_pair = (v0, v1);
            }
        }
    }

    let p0 = &vertices[best_pair.0];
    let p1 = &vertices[best_pair.1];
    let axis = p1 - p0;
    let axis_len = axis.norm();

    if axis_len < 1e-10 {
        return (
            PinnedVertex::new(best_pair.0, 0.0, 0.0),
            PinnedVertex::new(best_pair.1, 1.0, 0.0),
        );
    }

    let axis_dir = axis / axis_len;

    // Find the boundary vertex farthest from the line between p0 and p1
    // This determines the "height" of the boundary in the perpendicular direction
    let mut max_perp_dist = 0.0;
    let mut third_vertex = best_pair.0;
    for &vi in boundary {
        let to_v = vertices[vi] - p0;
        let proj = to_v.dot(&axis_dir);
        let perp = to_v - proj * axis_dir;
        let perp_dist = perp.norm();
        if perp_dist > max_perp_dist {
            max_perp_dist = perp_dist;
            third_vertex = vi;
        }
    }

    // If boundary has significant perpendicular extent, use the third vertex
    // to set up pins that span the 2D space
    if max_perp_dist > 0.1 * axis_len && third_vertex != best_pair.0 && third_vertex != best_pair.1
    {
        // Use third_vertex as pin1 instead, with appropriate UV
        let p_third = &vertices[third_vertex];
        let to_third = p_third - p0;
        let u_third = to_third.dot(&axis_dir) / axis_len;

        // V is the perpendicular distance, normalized
        let v_third = max_perp_dist / axis_len;

        return (
            PinnedVertex::new(best_pair.0, 0.0, 0.0),
            PinnedVertex::new(third_vertex, u_third, v_third),
        );
    }

    // Boundary is nearly collinear - fall back to standard (0,0) and (1,0)
    // but with a small V offset for pin1 to avoid degeneracy
    (
        PinnedVertex::new(best_pair.0, 0.0, 0.0),
        PinnedVertex::new(best_pair.1, 1.0, 0.1),
    )
}

/// Assemble the unconstrained conformal normal equations `A^T A`.
///
/// The LSCM energy is `E = ‖A · [u; v]‖²`, a homogeneous quadratic — it has no
/// linear term, because every constraint is a conformality condition rather than
/// a target value. Its minimizer is only determined up to a similarity of the
/// plane (translation, rotation, scale: a four-dimensional kernel), which is
/// exactly what pinning two vertices removes.
///
/// The pins are *not* applied here. See [`reduce_pinned_dofs`].
///
/// Indexing: DOF `i` is `u` of vertex `i`, DOF `n + i` is `v` of vertex `i`.
fn build_conformal_normal_equations(
    vertices: &[Point3<f64>],
    faces: &[[usize; 3]],
    n_vertices: usize,
) -> Vec<(usize, usize, f64)> {
    let n = n_vertices;

    // Collect triplets for the matrix
    let mut triplets: Vec<(usize, usize, f64)> = Vec::new();

    // For each triangle, add conformal energy terms
    for face in faces {
        let i = face[0];
        let j = face[1];
        let k = face[2];

        let pi = &vertices[i];
        let pj = &vertices[j];
        let pk = &vertices[k];

        // Project triangle to 2D local coordinates
        // Use the triangle's plane as the local frame
        let e1 = pj - pi;
        let e2 = pk - pi;

        // Local 2D coordinates (pi at origin)
        // x-axis along e1
        let e1_len = e1.norm();
        if e1_len < 1e-10 {
            continue; // Degenerate triangle
        }

        let x_axis = e1 / e1_len;
        let normal = e1.cross(&e2);
        let area = normal.norm() * 0.5;
        if area < 1e-10 {
            continue; // Degenerate triangle
        }

        let y_axis = normal.cross(&e1).normalize();

        // Local coordinates:
        // qi = (0, 0)
        // qj = (|e1|, 0)
        // qk = (e2·x_axis, e2·y_axis)
        let qix = 0.0;
        let qiy = 0.0;
        let qjx = e1_len;
        let qjy = 0.0;
        let qkx = e2.dot(&x_axis);
        let qky = e2.dot(&y_axis);

        // The conformal energy for this triangle measures:
        // |(∂u/∂x, ∂u/∂y) - R90 * (∂v/∂x, ∂v/∂y)|²
        //
        // where R90 is 90-degree rotation.
        //
        // For a linear function on the triangle:
        // ∂u/∂x = (1/2A) * Σ u_i * (y_j - y_k) (cyclic)
        // ∂u/∂y = (1/2A) * Σ u_i * (x_k - x_j) (cyclic)
        //
        // Conformal condition: ∂u/∂x = ∂v/∂y and ∂u/∂y = -∂v/∂x

        // Gradient coefficients for each vertex
        // For vertex i: coeff_x = (y_j - y_k) / (2*area), coeff_y = (x_k - x_j) / (2*area)
        let inv_2a = 1.0 / (2.0 * area);

        let ai_x = (qjy - qky) * inv_2a;
        let ai_y = (qkx - qjx) * inv_2a;
        let aj_x = (qky - qiy) * inv_2a;
        let aj_y = (qix - qkx) * inv_2a;
        let ak_x = (qiy - qjy) * inv_2a;
        let ak_y = (qjx - qix) * inv_2a;

        // The conformal energy squared is:
        // (∂u/∂x - ∂v/∂y)² + (∂u/∂y + ∂v/∂x)²
        //
        // Expanded and collecting terms gives a quadratic in (u_i, u_j, u_k, v_i, v_j, v_k)
        //
        // For the normal equations A^T A, we accumulate:
        // For each pair (m, n) of vertices in the triangle, add contributions to
        // the u-u, v-v, and u-v blocks.

        let verts = [(i, ai_x, ai_y), (j, aj_x, aj_y), (k, ak_x, ak_y)];

        // Weight by triangle area for area-weighted energy
        let weight = area;

        for &(vi, ax_i, ay_i) in &verts {
            for &(vj, ax_j, ay_j) in &verts {
                // u-u block: (ax_i * ax_j + ay_i * ay_j) * weight
                // v-v block: (ax_i * ax_j + ay_i * ay_j) * weight
                // u-v block: (ay_i * ax_j - ax_i * ay_j) * weight
                // v-u block: (ax_i * ay_j - ay_i * ax_j) * weight

                let uu = (ax_i * ax_j + ay_i * ay_j) * weight;
                let uv = (ay_i * ax_j - ax_i * ay_j) * weight;

                // Add to u-u block (rows 0..n, cols 0..n)
                triplets.push((vi, vj, uu));

                // Add to v-v block (rows n..2n, cols n..2n)
                triplets.push((n + vi, n + vj, uu));

                // Add to u-v block (rows 0..n, cols n..2n)
                triplets.push((vi, n + vj, uv));

                // Add to v-u block (rows n..2n, cols 0..n)
                triplets.push((n + vi, vj, -uv));
            }
        }
    }

    triplets
}

/// Impose the pin constraints by *eliminating* the pinned degrees of freedom
/// rather than penalizing them.
///
/// Partition the unknowns into free and pinned, `x = [x_f; x_p]`. Minimizing
/// `xᵀ M x` over `x_f` with `x_p` held fixed gives
///
/// ```text
///     M_ff x_f = −M_fp x_p
/// ```
///
/// so the pinned columns move to the right-hand side and the pinned rows drop
/// out. `M_ff` is symmetric positive definite — fixing two vertices removes the
/// similarity kernel exactly — and, crucially, its condition number is the
/// mesh's own. The penalty formulation this replaces instead added `λ` to four
/// diagonal entries, which inflated the condition number by `λ` and inflated
/// `‖b‖` to `λ · pin_target`, making a *relative* residual test meaningless:
/// at `λ = 10⁶` a `1e-8` relative tolerance still permitted an absolute
/// residual of `10⁻²` in the rows carrying the geometry, and LSCM returned
/// visibly wrong maps above ~50 vertices while reporting convergence.
///
/// Returns the reduced matrix, its right-hand side, and a map from each global
/// DOF to its reduced index (`None` for pinned DOFs).
fn reduce_pinned_dofs(
    triplets: &[(usize, usize, f64)],
    n_vertices: usize,
    pin0: &PinnedVertex,
    pin1: &PinnedVertex,
) -> (CsrMatrix, DVector<f64>, Vec<Option<usize>>) {
    let n = n_vertices;
    let n_dofs = 2 * n;

    // Fixed value per pinned DOF.
    let mut pinned_value = vec![None; n_dofs];
    pinned_value[pin0.vertex] = Some(pin0.u);
    pinned_value[n + pin0.vertex] = Some(pin0.v);
    pinned_value[pin1.vertex] = Some(pin1.u);
    pinned_value[n + pin1.vertex] = Some(pin1.v);

    // Compact the free DOFs into a contiguous range.
    let mut reduced_index = vec![None; n_dofs];
    let mut n_free = 0;
    for dof in 0..n_dofs {
        if pinned_value[dof].is_none() {
            reduced_index[dof] = Some(n_free);
            n_free += 1;
        }
    }

    let mut reduced_triplets: Vec<(usize, usize, f64)> = Vec::with_capacity(triplets.len());
    let mut rhs = DVector::zeros(n_free);

    for &(row, col, value) in triplets {
        // A pinned row is not an equation we solve.
        let Some(r) = reduced_index[row] else {
            continue;
        };

        match pinned_value[col] {
            // Known column: fold its contribution into the right-hand side.
            Some(fixed) => rhs[r] -= value * fixed,
            // Free column: keep it in the matrix.
            None => reduced_triplets.push((r, reduced_index[col].unwrap(), value)),
        }
    }

    let matrix = CsrMatrix::from_triplets(n_free, n_free, reduced_triplets);

    (matrix, rhs, reduced_index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::build_from_triangles;

    fn create_single_triangle() -> HalfEdgeMesh {
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.5, 1.0, 0.0),
        ];
        let faces = vec![[0, 1, 2]];
        build_from_triangles(&vertices, &faces).unwrap()
    }

    fn create_disk_mesh() -> HalfEdgeMesh {
        // Simple disk: center vertex + 6 boundary vertices
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0), // center
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.5, 0.866, 0.0),
            Point3::new(-0.5, 0.866, 0.0),
            Point3::new(-1.0, 0.0, 0.0),
            Point3::new(-0.5, -0.866, 0.0),
            Point3::new(0.5, -0.866, 0.0),
        ];
        let faces = vec![
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 4],
            [0, 4, 5],
            [0, 5, 6],
            [0, 6, 1],
        ];
        build_from_triangles(&vertices, &faces).unwrap()
    }

    fn create_grid_mesh(n: usize) -> HalfEdgeMesh {
        let mut vertices = Vec::new();
        let mut faces = Vec::new();

        // Create (n+1)x(n+1) grid of vertices
        for j in 0..=n {
            for i in 0..=n {
                vertices.push(Point3::new(i as f64, j as f64, 0.0));
            }
        }

        // Create 2 triangles per grid cell
        for j in 0..n {
            for i in 0..n {
                let v00 = j * (n + 1) + i;
                let v10 = j * (n + 1) + i + 1;
                let v01 = (j + 1) * (n + 1) + i;
                let v11 = (j + 1) * (n + 1) + i + 1;

                faces.push([v00, v10, v11]);
                faces.push([v00, v11, v01]);
            }
        }

        build_from_triangles(&vertices, &faces).unwrap()
    }

    fn create_tetrahedron() -> HalfEdgeMesh {
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.5, 1.0, 0.0),
            Point3::new(0.5, 0.5, 1.0),
        ];
        let faces = vec![[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]];
        build_from_triangles(&vertices, &faces).unwrap()
    }

    #[test]
    fn test_lscm_single_triangle() {
        let mesh = create_single_triangle();
        let result = lscm(&mesh, &LSCMOptions::default());
        assert!(result.is_ok());

        let uv_map = result.unwrap();
        assert_eq!(uv_map.len(), 3);

        // Check UVs are in valid range after normalization
        for (_, uv) in uv_map.iter() {
            assert!(uv.x >= -0.1 && uv.x <= 1.1);
            assert!(uv.y >= -0.1 && uv.y <= 1.1);
        }
    }

    #[test]
    fn test_lscm_disk() {
        let mesh = create_disk_mesh();
        let result = lscm(&mesh, &LSCMOptions::default());
        assert!(result.is_ok());

        let uv_map = result.unwrap();
        assert_eq!(uv_map.len(), 7);
    }

    #[test]
    fn test_lscm_grid() {
        let mesh = create_grid_mesh(3);
        let result = lscm(&mesh, &LSCMOptions::default());
        assert!(result.is_ok());

        let uv_map = result.unwrap();
        assert_eq!(uv_map.len(), 16); // 4x4 = 16 vertices
    }

    #[test]
    fn test_lscm_closed_mesh_fails() {
        let mesh = create_tetrahedron();
        let result = lscm(&mesh, &LSCMOptions::default());
        assert!(result.is_err());

        match result.unwrap_err() {
            MeshError::NoBoundary => (),
            e => panic!("Expected NoBoundary error, got {:?}", e),
        }
    }

    #[test]
    fn test_lscm_with_manual_pins() {
        let mesh = create_grid_mesh(2);

        let pin0 = PinnedVertex::new(0, 0.0, 0.0);
        let pin1 = PinnedVertex::new(2, 1.0, 0.0);

        let options = LSCMOptions::with_pins(pin0, pin1);
        let result = lscm(&mesh, &options);
        assert!(result.is_ok());

        let uv_map = result.unwrap();

        // Pinned vertices should be at their specified positions (before normalization)
        // After normalization they may shift, but let's just check the map exists
        assert_eq!(uv_map.len(), 9);
    }

    #[test]
    fn test_find_boundary_vertices() {
        // Single triangle has 3 boundary vertices
        let faces = vec![[0, 1, 2]];
        let boundary = find_boundary_vertices(&faces, 3);
        assert_eq!(boundary.len(), 3);

        // Two triangles sharing an edge: 4 boundary vertices
        let faces = vec![[0, 1, 2], [1, 3, 2]];
        let boundary = find_boundary_vertices(&faces, 4);
        assert_eq!(boundary.len(), 4);
    }

    #[test]
    fn test_select_farthest_pair() {
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.5, 1.0, 0.0),
            Point3::new(2.0, 0.0, 0.0), // Farthest from vertex 0
        ];
        let boundary = vec![0, 1, 2, 3];

        let (pin0, pin1) = select_farthest_boundary_pair(&vertices, &boundary);

        // Pin 0 should be vertex 0 (one end of farthest pair)
        assert_eq!(pin0.vertex, 0);
        assert_eq!(pin0.u, 0.0);
        assert_eq!(pin0.v, 0.0);

        // Pin 1 should be vertex 2 (has perpendicular extent from 0-3 axis)
        // because it provides better 2D UV coverage than the collinear vertex 3
        assert_eq!(pin1.vertex, 2);
        // V should be non-zero (perpendicular distance)
        assert!(pin1.v > 0.0);
    }

    #[test]
    fn test_select_farthest_pair_collinear() {
        // Test with collinear boundary (no perpendicular extent)
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
        ];
        let boundary = vec![0, 1, 2];

        let (pin0, pin1) = select_farthest_boundary_pair(&vertices, &boundary);

        // Should select farthest pair with small V offset
        assert_eq!(pin0.vertex, 0);
        assert_eq!(pin1.vertex, 2);
        assert_eq!(pin1.u, 1.0);
        assert!((pin1.v - 0.1).abs() < 1e-10); // Small V offset for collinear
    }
}
