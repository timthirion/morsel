//! As-Rigid-As-Possible (ARAP) parameterization.
//!
//! ARAP parameterization computes UV coordinates that minimize distortion by
//! preserving local rigidity. Unlike LSCM which only preserves angles, ARAP
//! also tries to preserve area, making it better for texture mapping applications.
//!
//! The algorithm alternates between:
//! 1. **Local step**: Find best-fit rotations for each triangle
//! 2. **Global step**: Solve a sparse linear system to update UV coordinates
//!
//! # References
//!
//! - Liu, L., Zhang, L., Xu, Y., Gotsman, C., & Gortler, S. J. (2008).
//!   "A Local/Global Approach to Mesh Parameterization." SGP 2008.

use nalgebra::{DVector, Matrix2, Point2, Point3, Vector2};

use crate::error::{MeshError, Result};
use crate::mesh::{to_face_vertex, HalfEdgeMesh, MeshIndex};

use super::lscm::{lscm, LSCMOptions};
use super::sparse::{conjugate_gradient, CsrMatrix};
use super::uv::UVMap;

/// Options for ARAP parameterization.
#[derive(Debug, Clone)]
pub struct ARAPOptions {
    /// Number of local-global iterations.
    pub iterations: usize,

    /// Maximum iterations for the conjugate gradient solver (per global step).
    pub max_cg_iterations: usize,

    /// Convergence tolerance for the CG solver.
    pub cg_tolerance: f64,

    /// Whether to use LSCM as initial parameterization.
    /// If false, uses Tutte embedding (uniform weights).
    pub use_lscm_init: bool,
}

impl Default for ARAPOptions {
    fn default() -> Self {
        Self {
            iterations: 10,
            max_cg_iterations: 1000,
            cg_tolerance: 1e-8,
            use_lscm_init: true,
        }
    }
}

impl ARAPOptions {
    /// Create options with the specified number of iterations.
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Set whether to use LSCM initialization.
    pub fn with_lscm_init(mut self, use_lscm: bool) -> Self {
        self.use_lscm_init = use_lscm;
        self
    }

    /// Set the maximum CG iterations per global step.
    pub fn with_max_cg_iterations(mut self, max_iter: usize) -> Self {
        self.max_cg_iterations = max_iter;
        self
    }

    /// Set the CG convergence tolerance.
    pub fn with_cg_tolerance(mut self, tol: f64) -> Self {
        self.cg_tolerance = tol;
        self
    }
}

/// Compute ARAP (As-Rigid-As-Possible) parameterization.
///
/// This algorithm computes UV coordinates that minimize local distortion by
/// preserving rigidity. It requires an initial parameterization (typically LSCM)
/// and iteratively refines it.
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
/// - The initial parameterization fails
/// - The linear system fails to converge
///
/// # Example
///
/// ```no_run
/// use morsel::prelude::*;
/// use morsel::algo::parameterize::{arap, ARAPOptions};
///
/// let mesh: HalfEdgeMesh = morsel::io::load("disk.obj").unwrap();
/// let uv_map = arap(&mesh, &ARAPOptions::default()).unwrap();
/// ```
pub fn arap<I: MeshIndex>(mesh: &HalfEdgeMesh<I>, options: &ARAPOptions) -> Result<UVMap<I>> {
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

    // Get initial parameterization
    let mut uv_coords = if options.use_lscm_init {
        let lscm_options = LSCMOptions::default()
            .with_max_iterations(options.max_cg_iterations)
            .with_tolerance(options.cg_tolerance);
        let uv_map = lscm(mesh, &lscm_options)?;
        uv_map.as_slice().to_vec()
    } else {
        // Use Tutte embedding as fallback
        compute_tutte_embedding(&vertices, &faces, n_vertices, &boundary)?
    };

    // Precompute cotangent weights for all edges
    let cot_weights = compute_cotangent_weights(&vertices, &faces);

    // Precompute the global system matrix (Laplacian with cotangent weights)
    // This matrix doesn't change between iterations
    let (system_matrix, pinned_vertex) =
        build_arap_system_matrix(&faces, n_vertices, &cot_weights, &boundary);

    // Store original 2D edge vectors (from flattening 3D triangles)
    let original_edges = compute_original_edges(&vertices, &faces);

    // ARAP iteration
    for _iter in 0..options.iterations {
        // Local step: compute best-fit rotations for each triangle
        let rotations = compute_local_rotations(&uv_coords, &faces, &original_edges);

        // Global step: solve for new UV coordinates
        let rhs = build_arap_rhs(
            &faces,
            n_vertices,
            &cot_weights,
            &original_edges,
            &rotations,
            pinned_vertex,
            &uv_coords,
        );

        // Solve for u coordinates
        let u_solution = conjugate_gradient(
            &system_matrix,
            &rhs.0,
            None,
            options.max_cg_iterations,
            options.cg_tolerance,
        )?;

        // Solve for v coordinates
        let v_solution = conjugate_gradient(
            &system_matrix,
            &rhs.1,
            None,
            options.max_cg_iterations,
            options.cg_tolerance,
        )?;

        // Update UV coordinates
        for i in 0..n_vertices {
            uv_coords[i] = Point2::new(u_solution[i], v_solution[i]);
        }
    }

    let mut uv_map = UVMap::new(uv_coords);
    uv_map.normalize();

    Ok(uv_map)
}

/// Find boundary vertices from face-vertex representation.
fn find_boundary_vertices(faces: &[[usize; 3]], n_vertices: usize) -> Vec<usize> {
    use std::collections::HashMap;

    let mut edge_count: HashMap<(usize, usize), usize> = HashMap::new();

    for face in faces {
        for i in 0..3 {
            let v0 = face[i];
            let v1 = face[(i + 1) % 3];
            let edge = if v0 < v1 { (v0, v1) } else { (v1, v0) };
            *edge_count.entry(edge).or_insert(0) += 1;
        }
    }

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

/// Compute Tutte embedding (convex boundary, uniform weights).
fn compute_tutte_embedding(
    _vertices: &[Point3<f64>],
    faces: &[[usize; 3]],
    n_vertices: usize,
    boundary: &[usize],
) -> Result<Vec<Point2<f64>>> {
    use std::collections::HashSet;
    use std::f64::consts::PI;

    let boundary_set: HashSet<usize> = boundary.iter().copied().collect();

    // Place boundary vertices on a circle
    let mut uv_coords = vec![Point2::origin(); n_vertices];
    for (i, &v) in boundary.iter().enumerate() {
        let angle = 2.0 * PI * (i as f64) / (boundary.len() as f64);
        uv_coords[v] = Point2::new(angle.cos(), angle.sin());
    }

    // Build Laplacian system for interior vertices
    // L * u = 0 with boundary conditions
    let interior: Vec<usize> = (0..n_vertices)
        .filter(|v| !boundary_set.contains(v))
        .collect();

    if interior.is_empty() {
        return Ok(uv_coords);
    }

    // Build neighbor lists
    let mut neighbors: Vec<Vec<usize>> = vec![Vec::new(); n_vertices];
    for face in faces {
        for i in 0..3 {
            let v0 = face[i];
            let v1 = face[(i + 1) % 3];
            if !neighbors[v0].contains(&v1) {
                neighbors[v0].push(v1);
            }
            if !neighbors[v1].contains(&v0) {
                neighbors[v1].push(v0);
            }
        }
    }

    // Create mapping from vertex index to interior index
    let mut interior_map = vec![usize::MAX; n_vertices];
    for (i, &v) in interior.iter().enumerate() {
        interior_map[v] = i;
    }

    let n_interior = interior.len();

    // Build the system matrix and RHS
    let mut triplets: Vec<(usize, usize, f64)> = Vec::new();
    let mut rhs_u = DVector::zeros(n_interior);
    let mut rhs_v = DVector::zeros(n_interior);

    for (i, &v) in interior.iter().enumerate() {
        let degree = neighbors[v].len() as f64;
        triplets.push((i, i, degree));

        for &neighbor in &neighbors[v] {
            if boundary_set.contains(&neighbor) {
                // Boundary neighbor: add to RHS
                rhs_u[i] += uv_coords[neighbor].x;
                rhs_v[i] += uv_coords[neighbor].y;
            } else {
                // Interior neighbor: add to matrix
                let j = interior_map[neighbor];
                triplets.push((i, j, -1.0));
            }
        }
    }

    let matrix = CsrMatrix::from_triplets(n_interior, n_interior, triplets);

    // Solve for interior u and v
    let u_solution = conjugate_gradient(&matrix, &rhs_u, None, 1000, 1e-8)?;
    let v_solution = conjugate_gradient(&matrix, &rhs_v, None, 1000, 1e-8)?;

    for (i, &v) in interior.iter().enumerate() {
        uv_coords[v] = Point2::new(u_solution[i], v_solution[i]);
    }

    Ok(uv_coords)
}

/// Compute cotangent weights for all edges.
/// Returns a map from (v0, v1) with v0 < v1 to weight.
fn compute_cotangent_weights(
    vertices: &[Point3<f64>],
    faces: &[[usize; 3]],
) -> std::collections::HashMap<(usize, usize), f64> {
    use std::collections::HashMap;

    let mut weights: HashMap<(usize, usize), f64> = HashMap::new();

    for face in faces {
        let p0 = &vertices[face[0]];
        let p1 = &vertices[face[1]];
        let p2 = &vertices[face[2]];

        // Cotangent at vertex 0 (opposite to edge 1-2)
        let cot0 = cotangent_angle(p0, p1, p2);
        // Cotangent at vertex 1 (opposite to edge 0-2)
        let cot1 = cotangent_angle(p1, p0, p2);
        // Cotangent at vertex 2 (opposite to edge 0-1)
        let cot2 = cotangent_angle(p2, p0, p1);

        // Edge 0-1: opposite angle is at vertex 2
        let e01 = canonical_edge(face[0], face[1]);
        *weights.entry(e01).or_insert(0.0) += cot2;

        // Edge 1-2: opposite angle is at vertex 0
        let e12 = canonical_edge(face[1], face[2]);
        *weights.entry(e12).or_insert(0.0) += cot0;

        // Edge 0-2: opposite angle is at vertex 1
        let e02 = canonical_edge(face[0], face[2]);
        *weights.entry(e02).or_insert(0.0) += cot1;
    }

    // Multiply by 0.5 and clamp to positive values
    for weight in weights.values_mut() {
        *weight = (*weight * 0.5).max(1e-6);
    }

    weights
}

/// Compute cotangent of angle at vertex a in triangle (a, b, c).
fn cotangent_angle(a: &Point3<f64>, b: &Point3<f64>, c: &Point3<f64>) -> f64 {
    let ab = b - a;
    let ac = c - a;

    let dot = ab.dot(&ac);
    let cross_len = ab.cross(&ac).norm();

    if cross_len < 1e-10 {
        0.0
    } else {
        dot / cross_len
    }
}

/// Get canonical edge representation (smaller index first).
fn canonical_edge(v0: usize, v1: usize) -> (usize, usize) {
    if v0 < v1 {
        (v0, v1)
    } else {
        (v1, v0)
    }
}

/// Compute original 2D edge vectors by flattening each triangle to its plane.
fn compute_original_edges(
    vertices: &[Point3<f64>],
    faces: &[[usize; 3]],
) -> Vec<[Vector2<f64>; 3]> {
    let mut edges = Vec::with_capacity(faces.len());

    for face in faces {
        let p0 = &vertices[face[0]];
        let p1 = &vertices[face[1]];
        let p2 = &vertices[face[2]];

        // Build local 2D frame
        let e1 = p1 - p0;
        let e2 = p2 - p0;

        let e1_len = e1.norm();
        if e1_len < 1e-10 {
            edges.push([Vector2::zeros(); 3]);
            continue;
        }

        let x_axis = e1 / e1_len;
        let normal = e1.cross(&e2);
        if normal.norm() < 1e-10 {
            edges.push([Vector2::zeros(); 3]);
            continue;
        }

        let y_axis = normal.cross(&e1).normalize();

        // Local 2D coordinates
        let q0 = Vector2::new(0.0, 0.0);
        let q1 = Vector2::new(e1_len, 0.0);
        let q2 = Vector2::new(e2.dot(&x_axis), e2.dot(&y_axis));

        // Edge vectors in local frame
        // Edge 0: from vertex 0 to vertex 1
        // Edge 1: from vertex 1 to vertex 2
        // Edge 2: from vertex 2 to vertex 0
        edges.push([q1 - q0, q2 - q1, q0 - q2]);
    }

    edges
}

/// Local step: compute best-fit rotation for each triangle.
fn compute_local_rotations(
    uv_coords: &[Point2<f64>],
    faces: &[[usize; 3]],
    original_edges: &[[Vector2<f64>; 3]],
) -> Vec<Matrix2<f64>> {
    let mut rotations = Vec::with_capacity(faces.len());

    for (fi, face) in faces.iter().enumerate() {
        // Current UV edge vectors
        let u0 = uv_coords[face[0]].coords;
        let u1 = uv_coords[face[1]].coords;
        let u2 = uv_coords[face[2]].coords;

        let current_edges = [u1 - u0, u2 - u1, u0 - u2];

        // Compute covariance matrix: S = Σ (current_edge * original_edge^T)
        let mut s = Matrix2::zeros();
        for i in 0..3 {
            let c = current_edges[i];
            let o = original_edges[fi][i];
            s += c * o.transpose();
        }

        // SVD to find closest rotation
        let rotation = closest_rotation(&s);
        rotations.push(rotation);
    }

    rotations
}

/// Find the closest rotation matrix to a given matrix using SVD.
fn closest_rotation(m: &Matrix2<f64>) -> Matrix2<f64> {
    // For 2x2, we can compute SVD analytically or use a simple approach
    // R = V * U^T where M = U * S * V^T

    let svd = m.svd(true, true);

    let u = svd.u.unwrap();
    let v_t = svd.v_t.unwrap();

    let mut r = u * v_t;

    // Ensure proper rotation (det = 1, not -1)
    if r.determinant() < 0.0 {
        // Flip sign of second column of U
        let mut u_fixed = u;
        u_fixed[(0, 1)] = -u_fixed[(0, 1)];
        u_fixed[(1, 1)] = -u_fixed[(1, 1)];
        r = u_fixed * v_t;
    }

    r
}

/// Build the ARAP system matrix (cotangent Laplacian with one pinned vertex).
fn build_arap_system_matrix(
    faces: &[[usize; 3]],
    n_vertices: usize,
    cot_weights: &std::collections::HashMap<(usize, usize), f64>,
    boundary: &[usize],
) -> (CsrMatrix, usize) {
    let mut triplets: Vec<(usize, usize, f64)> = Vec::new();

    // Build cotangent Laplacian
    for face in faces {
        for i in 0..3 {
            let v0 = face[i];
            let v1 = face[(i + 1) % 3];
            let edge = canonical_edge(v0, v1);
            let w = cot_weights.get(&edge).copied().unwrap_or(1e-6);

            // Diagonal entries
            triplets.push((v0, v0, w));
            triplets.push((v1, v1, w));

            // Off-diagonal entries
            triplets.push((v0, v1, -w));
            triplets.push((v1, v0, -w));
        }
    }

    // Pin one boundary vertex (add large value to diagonal)
    let pinned = boundary[0];
    let penalty = 1e10;
    triplets.push((pinned, pinned, penalty));

    let matrix = CsrMatrix::from_triplets(n_vertices, n_vertices, triplets);
    (matrix, pinned)
}

/// Build the ARAP right-hand side vectors.
fn build_arap_rhs(
    faces: &[[usize; 3]],
    n_vertices: usize,
    cot_weights: &std::collections::HashMap<(usize, usize), f64>,
    original_edges: &[[Vector2<f64>; 3]],
    rotations: &[Matrix2<f64>],
    pinned_vertex: usize,
    uv_coords: &[Point2<f64>],
) -> (DVector<f64>, DVector<f64>) {
    let mut rhs_u = DVector::zeros(n_vertices);
    let mut rhs_v = DVector::zeros(n_vertices);

    for (fi, face) in faces.iter().enumerate() {
        let r = &rotations[fi];

        // For each edge in the triangle
        for i in 0..3 {
            let v0 = face[i];
            let v1 = face[(i + 1) % 3];
            let edge = canonical_edge(v0, v1);
            let w = cot_weights.get(&edge).copied().unwrap_or(1e-6);

            // Original edge vector (from v0 to v1 in local frame)
            let orig = original_edges[fi][i];

            // Rotated edge
            let rotated = r * orig;

            // Add to RHS: w * R * (p_i - p_j)
            // For vertex v0: += w * rotated
            // For vertex v1: -= w * rotated
            rhs_u[v0] += w * rotated.x;
            rhs_v[v0] += w * rotated.y;
            rhs_u[v1] -= w * rotated.x;
            rhs_v[v1] -= w * rotated.y;
        }
    }

    // Pin constraint
    let penalty = 1e10;
    rhs_u[pinned_vertex] += penalty * uv_coords[pinned_vertex].x;
    rhs_v[pinned_vertex] += penalty * uv_coords[pinned_vertex].y;

    (rhs_u, rhs_v)
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
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
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

        for j in 0..=n {
            for i in 0..=n {
                vertices.push(Point3::new(i as f64, j as f64, 0.0));
            }
        }

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
    fn test_arap_single_triangle() {
        let mesh = create_single_triangle();
        let result = arap(&mesh, &ARAPOptions::default());
        assert!(result.is_ok());

        let uv_map = result.unwrap();
        assert_eq!(uv_map.len(), 3);
    }

    #[test]
    fn test_arap_disk() {
        let mesh = create_disk_mesh();
        let result = arap(&mesh, &ARAPOptions::default());
        assert!(result.is_ok());

        let uv_map = result.unwrap();
        assert_eq!(uv_map.len(), 7);
    }

    #[test]
    fn test_arap_grid() {
        let mesh = create_grid_mesh(3);
        let result = arap(&mesh, &ARAPOptions::default());
        assert!(result.is_ok());

        let uv_map = result.unwrap();
        assert_eq!(uv_map.len(), 16);
    }

    #[test]
    fn test_arap_closed_mesh_fails() {
        let mesh = create_tetrahedron();
        let result = arap(&mesh, &ARAPOptions::default());
        assert!(result.is_err());

        match result.unwrap_err() {
            MeshError::NoBoundary => (),
            e => panic!("Expected NoBoundary error, got {:?}", e),
        }
    }

    #[test]
    fn test_arap_with_tutte_init() {
        let mesh = create_grid_mesh(2);
        let options = ARAPOptions::default().with_lscm_init(false);
        let result = arap(&mesh, &options);
        assert!(result.is_ok());

        let uv_map = result.unwrap();
        assert_eq!(uv_map.len(), 9);
    }

    #[test]
    fn test_arap_few_iterations() {
        let mesh = create_disk_mesh();
        let options = ARAPOptions::default().with_iterations(2);
        let result = arap(&mesh, &options);
        assert!(result.is_ok());
    }

    #[test]
    fn test_closest_rotation() {
        // Identity should give identity
        let m = Matrix2::identity();
        let r = closest_rotation(&m);
        assert!((r - Matrix2::identity()).norm() < 1e-10);

        // Scaled rotation should give just rotation
        let angle: f64 = 0.5;
        let rot = Matrix2::new(angle.cos(), -angle.sin(), angle.sin(), angle.cos());
        let scaled = 2.0 * rot;
        let r = closest_rotation(&scaled);
        assert!((r - rot).norm() < 1e-10);
    }

    #[test]
    fn test_tutte_embedding() {
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.5, 1.0, 0.0),
        ];
        let faces = vec![[0, 1, 2]];
        let boundary = vec![0, 1, 2];

        let uv = compute_tutte_embedding(&vertices, &faces, 3, &boundary).unwrap();

        // All vertices are boundary, should be on unit circle
        for i in 0..3 {
            let r = (uv[i].x.powi(2) + uv[i].y.powi(2)).sqrt();
            assert!((r - 1.0).abs() < 1e-10);
        }
    }
}
