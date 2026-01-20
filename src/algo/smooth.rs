//! Mesh smoothing algorithms.
//!
//! This module provides various mesh smoothing algorithms for noise reduction
//! and surface regularization.
//!
//! # Algorithms
//!
//! - [`laplacian_smooth`]: Classic Laplacian smoothing (may cause shrinkage)
//! - [`taubin_smooth`]: Taubin's λ|μ smoothing (reduces shrinkage)
//! - [`cotangent_smooth`]: Cotangent-weighted Laplacian (geometry-aware)
//! - [`bilateral_smooth`]: Feature-preserving smoothing (preserves sharp edges)
//! - [`mean_curvature_flow`]: Curvature-driven flow (area-minimizing)
//!
//! # Example
//!
//! ```
//! use morsel::prelude::*;
//! use morsel::algo::smooth::{laplacian_smooth, SmoothOptions};
//! use nalgebra::Point3;
//!
//! let vertices = vec![
//!     Point3::new(0.0, 0.0, 0.0),
//!     Point3::new(1.0, 0.0, 0.0),
//!     Point3::new(0.5, 1.0, 0.0),
//!     Point3::new(0.5, 0.5, 0.5), // slightly perturbed
//! ];
//! let faces = vec![[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]];
//! let mut mesh: HalfEdgeMesh = build_from_triangles(&vertices, &faces).unwrap();
//!
//! // Smooth with default options
//! laplacian_smooth(&mut mesh, &SmoothOptions::default());
//! ```

use nalgebra::{Point3, Vector3};
use rayon::prelude::*;

use crate::mesh::{HalfEdgeId, HalfEdgeMesh, MeshIndex, VertexId};

use super::Progress;

/// Options for mesh smoothing algorithms.
#[derive(Debug, Clone)]
pub struct SmoothOptions {
    /// Number of smoothing iterations.
    pub iterations: usize,

    /// Smoothing factor (0.0 to 1.0).
    /// Higher values result in more aggressive smoothing.
    pub lambda: f64,

    /// Whether to preserve boundary vertices (don't move them).
    pub preserve_boundary: bool,

    /// Whether to use parallel execution (default: true).
    pub parallel: bool,
}

impl Default for SmoothOptions {
    fn default() -> Self {
        Self {
            iterations: 1,
            lambda: 0.5,
            preserve_boundary: true,
            parallel: true,
        }
    }
}

impl SmoothOptions {
    /// Create options with the specified number of iterations.
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Create options with the specified lambda value.
    pub fn with_lambda(mut self, lambda: f64) -> Self {
        self.lambda = lambda.clamp(0.0, 1.0);
        self
    }

    /// Create options that allow boundary vertices to move.
    pub fn allow_boundary_movement(mut self) -> Self {
        self.preserve_boundary = false;
        self
    }

    /// Set whether to use parallel execution.
    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    /// Create options for single-threaded execution.
    pub fn sequential(mut self) -> Self {
        self.parallel = false;
        self
    }
}

/// Performs Laplacian smoothing on a mesh.
///
/// Laplacian smoothing moves each vertex towards the centroid of its neighbors.
/// This is a simple and fast smoothing method, but it tends to shrink the mesh
/// over multiple iterations. For shrinkage-resistant smoothing, use [`taubin_smooth`].
///
/// # Arguments
///
/// * `mesh` - The mesh to smooth (modified in place)
/// * `options` - Smoothing parameters
///
/// # Algorithm
///
/// For each iteration:
/// 1. For each vertex v, compute the centroid c of its neighbors
/// 2. Move v towards c: `new_pos = old_pos + λ * (c - old_pos)`
///
/// # Example
///
/// ```
/// use morsel::prelude::*;
/// use morsel::algo::smooth::{laplacian_smooth, SmoothOptions};
/// use nalgebra::Point3;
///
/// let vertices = vec![
///     Point3::new(0.0, 0.0, 0.0),
///     Point3::new(1.0, 0.0, 0.0),
///     Point3::new(0.5, 1.0, 0.0),
/// ];
/// let faces = vec![[0, 1, 2]];
/// let mut mesh: HalfEdgeMesh = build_from_triangles(&vertices, &faces).unwrap();
///
/// let options = SmoothOptions::default()
///     .with_iterations(5)
///     .with_lambda(0.3);
/// laplacian_smooth(&mut mesh, &options);
/// ```
pub fn laplacian_smooth<I: MeshIndex + Sync>(mesh: &mut HalfEdgeMesh<I>, options: &SmoothOptions) {
    if options.iterations == 0 || options.lambda == 0.0 {
        return;
    }

    // Identify boundary vertices once if we need to preserve them
    let boundary_vertices: Vec<bool> = if options.preserve_boundary {
        mesh.vertex_ids()
            .map(|v| mesh.is_boundary_vertex(v))
            .collect()
    } else {
        vec![false; mesh.num_vertices()]
    };

    let num_vertices = mesh.num_vertices();

    for _ in 0..options.iterations {
        // Compute new positions for all vertices
        let new_positions: Vec<Point3<f64>> = if options.parallel {
            (0..num_vertices)
                .into_par_iter()
                .map(|i| {
                    let vid = VertexId::new(i);
                    if boundary_vertices[i] {
                        *mesh.position(vid)
                    } else {
                        compute_laplacian_step(mesh, vid, options.lambda)
                    }
                })
                .collect()
        } else {
            (0..num_vertices)
                .map(|i| {
                    let vid = VertexId::new(i);
                    if boundary_vertices[i] {
                        *mesh.position(vid)
                    } else {
                        compute_laplacian_step(mesh, vid, options.lambda)
                    }
                })
                .collect()
        };

        // Apply new positions
        for i in 0..num_vertices {
            let vid = VertexId::new(i);
            mesh.set_position(vid, new_positions[i]);
        }
    }
}

/// Performs Taubin smoothing on a mesh.
///
/// Taubin smoothing alternates between a positive smoothing step (λ) and a
/// negative "inflation" step (μ) to reduce the shrinkage that occurs with
/// standard Laplacian smoothing.
///
/// # Arguments
///
/// * `mesh` - The mesh to smooth (modified in place)
/// * `options` - Smoothing parameters (uses `lambda` for the positive step)
///
/// # Algorithm
///
/// For each iteration:
/// 1. Apply Laplacian step with factor λ (smoothing)
/// 2. Apply Laplacian step with factor μ < 0 (inflation)
///
/// The μ value is computed as: `μ = λ / (k * λ - 1)` where k > 1 (typically k ≈ 0.1)
/// This ensures the surface doesn't drift while reducing noise.
///
/// # Reference
///
/// Taubin, G. (1995). "A signal processing approach to fair surface design."
/// SIGGRAPH '95.
///
/// # Example
///
/// ```
/// use morsel::prelude::*;
/// use morsel::algo::smooth::{taubin_smooth, SmoothOptions};
/// use nalgebra::Point3;
///
/// let vertices = vec![
///     Point3::new(0.0, 0.0, 0.0),
///     Point3::new(1.0, 0.0, 0.0),
///     Point3::new(0.5, 1.0, 0.0),
/// ];
/// let faces = vec![[0, 1, 2]];
/// let mut mesh: HalfEdgeMesh = build_from_triangles(&vertices, &faces).unwrap();
///
/// let options = SmoothOptions::default()
///     .with_iterations(10)
///     .with_lambda(0.5);
/// taubin_smooth(&mut mesh, &options);
/// ```
pub fn taubin_smooth<I: MeshIndex + Sync>(mesh: &mut HalfEdgeMesh<I>, options: &SmoothOptions) {
    if options.iterations == 0 || options.lambda == 0.0 {
        return;
    }

    // Compute mu for the negative step
    // Using k_pb (passband frequency) = 0.1 as recommended
    let k_pb = 0.1_f64;
    let mu = options.lambda / (k_pb * options.lambda - 1.0);

    // Identify boundary vertices once if we need to preserve them
    let boundary_vertices: Vec<bool> = if options.preserve_boundary {
        mesh.vertex_ids()
            .map(|v| mesh.is_boundary_vertex(v))
            .collect()
    } else {
        vec![false; mesh.num_vertices()]
    };

    for _ in 0..options.iterations {
        // Positive smoothing step (λ)
        apply_laplacian_step_impl(mesh, &boundary_vertices, options.lambda, options.parallel);

        // Negative inflation step (μ)
        apply_laplacian_step_impl(mesh, &boundary_vertices, mu, options.parallel);
    }
}

/// Performs cotangent-weighted Laplacian smoothing.
///
/// This variant weights the contribution of each neighbor by the cotangent
/// of the angles opposite to the edge. This produces more geometrically
/// faithful results than uniform Laplacian smoothing.
///
/// # Arguments
///
/// * `mesh` - The mesh to smooth (modified in place)
/// * `options` - Smoothing parameters
///
/// # Note
///
/// Cotangent weights can become negative for obtuse triangles, which may
/// cause instability. Consider using [`laplacian_smooth`] for meshes with
/// poor triangle quality.
pub fn cotangent_smooth<I: MeshIndex + Sync>(mesh: &mut HalfEdgeMesh<I>, options: &SmoothOptions) {
    if options.iterations == 0 || options.lambda == 0.0 {
        return;
    }

    // Identify boundary vertices once if we need to preserve them
    let boundary_vertices: Vec<bool> = if options.preserve_boundary {
        mesh.vertex_ids()
            .map(|v| mesh.is_boundary_vertex(v))
            .collect()
    } else {
        vec![false; mesh.num_vertices()]
    };

    let num_vertices = mesh.num_vertices();

    for _ in 0..options.iterations {
        // Compute new positions for all vertices
        let new_positions: Vec<Point3<f64>> = if options.parallel {
            (0..num_vertices)
                .into_par_iter()
                .map(|i| {
                    let vid = VertexId::new(i);
                    if boundary_vertices[i] {
                        *mesh.position(vid)
                    } else {
                        compute_cotangent_laplacian_step(mesh, vid, options.lambda)
                    }
                })
                .collect()
        } else {
            (0..num_vertices)
                .map(|i| {
                    let vid = VertexId::new(i);
                    if boundary_vertices[i] {
                        *mesh.position(vid)
                    } else {
                        compute_cotangent_laplacian_step(mesh, vid, options.lambda)
                    }
                })
                .collect()
        };

        // Apply new positions
        for i in 0..num_vertices {
            let vid = VertexId::new(i);
            mesh.set_position(vid, new_positions[i]);
        }
    }
}

/// Options for bilateral mesh smoothing.
#[derive(Debug, Clone)]
pub struct BilateralOptions {
    /// Number of smoothing iterations.
    pub iterations: usize,

    /// Spatial weight parameter (controls smoothing based on distance).
    /// Larger values allow more distant neighbors to contribute.
    pub sigma_c: f64,

    /// Normal weight parameter (controls smoothing based on normal similarity).
    /// Larger values allow more normal variation.
    pub sigma_s: f64,

    /// Whether to preserve boundary vertices.
    pub preserve_boundary: bool,

    /// Whether to use parallel execution (default: true).
    pub parallel: bool,
}

impl Default for BilateralOptions {
    fn default() -> Self {
        Self {
            iterations: 1,
            sigma_c: 1.0,
            sigma_s: 0.5,
            preserve_boundary: true,
            parallel: true,
        }
    }
}

impl BilateralOptions {
    /// Create options with the specified number of iterations.
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Set the spatial weight parameter.
    pub fn with_sigma_c(mut self, sigma_c: f64) -> Self {
        self.sigma_c = sigma_c.max(0.01);
        self
    }

    /// Set the normal weight parameter.
    pub fn with_sigma_s(mut self, sigma_s: f64) -> Self {
        self.sigma_s = sigma_s.max(0.01);
        self
    }

    /// Set whether to use parallel execution.
    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    /// Create options for single-threaded execution.
    pub fn sequential(mut self) -> Self {
        self.parallel = false;
        self
    }
}

/// Performs bilateral mesh smoothing.
///
/// Bilateral smoothing is a feature-preserving smoothing algorithm that
/// considers both spatial proximity and surface normal similarity when
/// averaging vertex positions. This allows it to smooth noise while
/// preserving sharp features and edges.
///
/// # Arguments
///
/// * `mesh` - The mesh to smooth (modified in place)
/// * `options` - Bilateral smoothing parameters
///
/// # Algorithm
///
/// For each vertex, the new position is computed as:
/// ```text
/// p_new = p + λ * Σ w_c(||p_j - p||) * w_s(||n - n_j||) * (p_j - p) / Σ weights
/// ```
///
/// Where:
/// - `w_c` is the spatial weight (Gaussian based on distance)
/// - `w_s` is the normal weight (Gaussian based on normal difference)
///
/// # Reference
///
/// Fleishman, S., Drori, I., & Cohen-Or, D. (2003). "Bilateral mesh denoising."
/// ACM SIGGRAPH 2003.
pub fn bilateral_smooth<I: MeshIndex + Sync>(mesh: &mut HalfEdgeMesh<I>, options: &BilateralOptions) {
    if options.iterations == 0 {
        return;
    }

    // Identify boundary vertices
    let boundary_vertices: Vec<bool> = if options.preserve_boundary {
        mesh.vertex_ids()
            .map(|v| mesh.is_boundary_vertex(v))
            .collect()
    } else {
        vec![false; mesh.num_vertices()]
    };

    let sigma_c_sq = options.sigma_c * options.sigma_c;
    let sigma_s_sq = options.sigma_s * options.sigma_s;
    let num_vertices = mesh.num_vertices();

    for _ in 0..options.iterations {
        // Compute vertex normals for this iteration
        let normals = compute_vertex_normals(mesh);

        // Compute new positions
        let new_positions: Vec<Point3<f64>> = if options.parallel {
            (0..num_vertices)
                .into_par_iter()
                .map(|i| {
                    let vid = VertexId::new(i);
                    if boundary_vertices[i] {
                        return *mesh.position(vid);
                    }

                    let pos = mesh.position(vid);
                    let normal = &normals[i];

                    let mut weighted_sum = Vector3::zeros();
                    let mut weight_total = 0.0;

                    for neighbor_id in mesh.vertex_neighbors(vid) {
                        let neighbor_pos = mesh.position(neighbor_id);
                        let neighbor_normal = &normals[neighbor_id.index()];

                        let dist_sq = (neighbor_pos - pos).norm_squared();
                        let w_c = (-dist_sq / (2.0 * sigma_c_sq)).exp();

                        let normal_diff_sq = (neighbor_normal - normal).norm_squared();
                        let w_s = (-normal_diff_sq / (2.0 * sigma_s_sq)).exp();

                        let weight = w_c * w_s;
                        weighted_sum += weight * (neighbor_pos - pos);
                        weight_total += weight;
                    }

                    if weight_total > 1e-10 {
                        Point3::from(pos.coords + weighted_sum / weight_total)
                    } else {
                        *pos
                    }
                })
                .collect()
        } else {
            (0..num_vertices)
                .map(|i| {
                    let vid = VertexId::new(i);
                    if boundary_vertices[i] {
                        return *mesh.position(vid);
                    }

                    let pos = mesh.position(vid);
                    let normal = &normals[i];

                    let mut weighted_sum = Vector3::zeros();
                    let mut weight_total = 0.0;

                    for neighbor_id in mesh.vertex_neighbors(vid) {
                        let neighbor_pos = mesh.position(neighbor_id);
                        let neighbor_normal = &normals[neighbor_id.index()];

                        let dist_sq = (neighbor_pos - pos).norm_squared();
                        let w_c = (-dist_sq / (2.0 * sigma_c_sq)).exp();

                        let normal_diff_sq = (neighbor_normal - normal).norm_squared();
                        let w_s = (-normal_diff_sq / (2.0 * sigma_s_sq)).exp();

                        let weight = w_c * w_s;
                        weighted_sum += weight * (neighbor_pos - pos);
                        weight_total += weight;
                    }

                    if weight_total > 1e-10 {
                        Point3::from(pos.coords + weighted_sum / weight_total)
                    } else {
                        *pos
                    }
                })
                .collect()
        };

        // Apply new positions
        for i in 0..num_vertices {
            let vid = VertexId::new(i);
            mesh.set_position(vid, new_positions[i]);
        }
    }
}

/// Options for mean curvature flow.
#[derive(Debug, Clone)]
pub struct CurvatureFlowOptions {
    /// Number of flow iterations.
    pub iterations: usize,

    /// Time step for integration (smaller = more stable, larger = faster).
    pub time_step: f64,

    /// Whether to preserve boundary vertices.
    pub preserve_boundary: bool,

    /// Whether to use implicit integration (more stable for larger time steps).
    pub implicit: bool,

    /// Whether to use parallel execution (default: true).
    pub parallel: bool,
}

impl Default for CurvatureFlowOptions {
    fn default() -> Self {
        Self {
            iterations: 1,
            time_step: 0.001,
            preserve_boundary: true,
            implicit: false,
            parallel: true,
        }
    }
}

impl CurvatureFlowOptions {
    /// Create options with the specified number of iterations.
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Set the time step.
    pub fn with_time_step(mut self, time_step: f64) -> Self {
        self.time_step = time_step.max(1e-6);
        self
    }

    /// Set whether to use parallel execution.
    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    /// Create options for single-threaded execution.
    pub fn sequential(mut self) -> Self {
        self.parallel = false;
        self
    }
}

/// Performs mean curvature flow on a mesh.
///
/// Mean curvature flow moves each vertex in the direction of mean curvature,
/// which tends to smooth the surface while minimizing surface area. This is
/// a more geometrically motivated smoothing that preserves features better
/// than uniform Laplacian smoothing.
///
/// # Arguments
///
/// * `mesh` - The mesh to smooth (modified in place)
/// * `options` - Flow parameters
///
/// # Algorithm
///
/// The mean curvature at each vertex is computed using the cotangent formula:
/// ```text
/// H * n = (1 / 2A) * Σ (cot α_ij + cot β_ij) * (p_j - p_i)
/// ```
///
/// Where α_ij and β_ij are the angles opposite to edge (i,j) in the two
/// adjacent triangles, and A is the vertex area.
///
/// # Note
///
/// This is an explicit integration scheme. For stability, use small time steps
/// (around 0.001 or smaller). For aggressive smoothing, increase iterations
/// rather than time step.
///
/// # Reference
///
/// Desbrun, M., et al. (1999). "Implicit fairing of irregular meshes using
/// diffusion and curvature flow." SIGGRAPH 99.
pub fn mean_curvature_flow<I: MeshIndex + Sync>(mesh: &mut HalfEdgeMesh<I>, options: &CurvatureFlowOptions) {
    if options.iterations == 0 {
        return;
    }

    // Identify boundary vertices
    let boundary_vertices: Vec<bool> = if options.preserve_boundary {
        mesh.vertex_ids()
            .map(|v| mesh.is_boundary_vertex(v))
            .collect()
    } else {
        vec![false; mesh.num_vertices()]
    };

    let num_vertices = mesh.num_vertices();
    let time_step = options.time_step;

    for _ in 0..options.iterations {
        // Compute new positions
        let new_positions: Vec<Point3<f64>> = if options.parallel {
            (0..num_vertices)
                .into_par_iter()
                .map(|i| {
                    let vid = VertexId::new(i);
                    if boundary_vertices[i] {
                        return *mesh.position(vid);
                    }

                    let pos = *mesh.position(vid);
                    let (curvature_vector, area) = compute_mean_curvature_vector(mesh, vid);

                    if area > 1e-10 {
                        let displacement = time_step * curvature_vector;
                        Point3::from(pos.coords + displacement)
                    } else {
                        pos
                    }
                })
                .collect()
        } else {
            (0..num_vertices)
                .map(|i| {
                    let vid = VertexId::new(i);
                    if boundary_vertices[i] {
                        return *mesh.position(vid);
                    }

                    let pos = *mesh.position(vid);
                    let (curvature_vector, area) = compute_mean_curvature_vector(mesh, vid);

                    if area > 1e-10 {
                        let displacement = time_step * curvature_vector;
                        Point3::from(pos.coords + displacement)
                    } else {
                        pos
                    }
                })
                .collect()
        };

        // Apply new positions
        for i in 0..num_vertices {
            let vid = VertexId::new(i);
            mesh.set_position(vid, new_positions[i]);
        }
    }
}

/// Compute the mean curvature vector at a vertex.
///
/// Returns (curvature_vector, vertex_area).
/// The curvature vector points in the direction of mean curvature normal.
fn compute_mean_curvature_vector<I: MeshIndex>(
    mesh: &HalfEdgeMesh<I>,
    v: VertexId<I>,
) -> (Vector3<f64>, f64) {
    let pos = *mesh.position(v);

    let mut curvature_sum = Vector3::zeros();
    let mut area = 0.0;

    for he in mesh.vertex_halfedges(v) {
        let neighbor = mesh.dest(he);
        let neighbor_pos = mesh.position(neighbor);

        // Get cotangent weight for this edge
        let weight = compute_edge_cotangent_weight(mesh, he);

        // Accumulate weighted edge vector
        curvature_sum += weight * (neighbor_pos.coords - pos.coords);

        // Accumulate area (using barycentric/Voronoi area)
        if !mesh.is_boundary_halfedge(he) {
            let face_area = compute_triangle_area(
                &pos,
                mesh.position(mesh.dest(he)),
                mesh.position(mesh.dest(mesh.next(he))),
            );
            area += face_area / 3.0; // Barycentric contribution
        }
    }

    // The mean curvature vector is Hn = (1/2A) * Σ w_ij * (p_j - p_i)
    if area > 1e-10 {
        (curvature_sum / (2.0 * area), area)
    } else {
        (Vector3::zeros(), 0.0)
    }
}

/// Compute the area of a triangle.
fn compute_triangle_area(p0: &Point3<f64>, p1: &Point3<f64>, p2: &Point3<f64>) -> f64 {
    let e1 = p1 - p0;
    let e2 = p2 - p0;
    e1.cross(&e2).norm() * 0.5
}

/// Compute vertex normals for all vertices.
fn compute_vertex_normals<I: MeshIndex>(mesh: &HalfEdgeMesh<I>) -> Vec<Vector3<f64>> {
    let mut normals = vec![Vector3::zeros(); mesh.num_vertices()];

    // Accumulate area-weighted face normals
    for fid in mesh.face_ids() {
        let [v0, v1, v2] = mesh.face_triangle(fid);
        let p0 = mesh.position(v0);
        let p1 = mesh.position(v1);
        let p2 = mesh.position(v2);

        let e1 = p1 - p0;
        let e2 = p2 - p0;
        let face_normal = e1.cross(&e2); // Area-weighted

        normals[v0.index()] += face_normal;
        normals[v1.index()] += face_normal;
        normals[v2.index()] += face_normal;
    }

    // Normalize
    for n in &mut normals {
        let len = n.norm();
        if len > 1e-10 {
            *n /= len;
        }
    }

    normals
}

/// Compute one Laplacian smoothing step for a vertex using uniform weights.
fn compute_laplacian_step<I: MeshIndex>(
    mesh: &HalfEdgeMesh<I>,
    v: VertexId<I>,
    lambda: f64,
) -> Point3<f64> {
    let pos = mesh.position(v);

    // Compute centroid of neighbors
    let mut centroid = Vector3::zeros();
    let mut count = 0;

    for neighbor in mesh.vertex_neighbors(v) {
        centroid += mesh.position(neighbor).coords;
        count += 1;
    }

    if count == 0 {
        return *pos;
    }

    centroid /= count as f64;

    // Move towards centroid
    let displacement = centroid - pos.coords;
    Point3::from(pos.coords + lambda * displacement)
}

/// Compute one Laplacian smoothing step for a vertex using cotangent weights.
fn compute_cotangent_laplacian_step<I: MeshIndex>(
    mesh: &HalfEdgeMesh<I>,
    v: VertexId<I>,
    lambda: f64,
) -> Point3<f64> {
    let pos = *mesh.position(v);

    // Compute weighted Laplacian
    let mut weighted_sum = Vector3::zeros();
    let mut weight_total = 0.0;

    for he in mesh.vertex_halfedges(v) {
        let neighbor = mesh.dest(he);
        let neighbor_pos = mesh.position(neighbor);

        // Get the two triangles adjacent to this edge
        let weight = compute_edge_cotangent_weight(mesh, he);

        if weight > 0.0 {
            weighted_sum += weight * (neighbor_pos.coords - pos.coords);
            weight_total += weight;
        }
    }

    if weight_total <= 0.0 {
        return pos;
    }

    // Apply weighted displacement
    Point3::from(pos.coords + lambda * weighted_sum / weight_total)
}

/// Compute the cotangent weight for an edge.
///
/// The weight is (cot(α) + cot(β)) / 2 where α and β are the angles
/// opposite to the edge in the two adjacent triangles.
fn compute_edge_cotangent_weight<I: MeshIndex>(mesh: &HalfEdgeMesh<I>, he: HalfEdgeId<I>) -> f64 {
    let mut weight = 0.0;

    // Get the edge endpoints
    let v0 = mesh.origin(he);
    let v1 = mesh.dest(he);
    let p0 = mesh.position(v0);
    let p1 = mesh.position(v1);

    // Process the face on one side (if not boundary)
    if !mesh.is_boundary_halfedge(he) {
        // The opposite vertex in this face
        let v_opp = mesh.dest(mesh.next(he));
        let p_opp = mesh.position(v_opp);

        weight += cotangent_angle(p_opp, p0, p1);
    }

    // Process the face on the other side (via twin)
    let twin = mesh.twin(he);
    if !mesh.is_boundary_halfedge(twin) {
        let v_opp = mesh.dest(mesh.next(twin));
        let p_opp = mesh.position(v_opp);

        weight += cotangent_angle(p_opp, p1, p0);
    }

    weight * 0.5
}

/// Compute the cotangent of the angle at vertex `a` in triangle (a, b, c).
fn cotangent_angle(a: &Point3<f64>, b: &Point3<f64>, c: &Point3<f64>) -> f64 {
    let ab = b - a;
    let ac = c - a;

    let dot = ab.dot(&ac);
    let cross_norm = ab.cross(&ac).norm();

    if cross_norm < 1e-10 {
        return 0.0; // Degenerate triangle
    }

    dot / cross_norm
}

/// Apply a single Laplacian step to all vertices.
fn apply_laplacian_step_impl<I: MeshIndex + Sync>(
    mesh: &mut HalfEdgeMesh<I>,
    boundary_vertices: &[bool],
    lambda: f64,
    parallel: bool,
) {
    let num_vertices = mesh.num_vertices();

    // Compute new positions
    let new_positions: Vec<Point3<f64>> = if parallel {
        (0..num_vertices)
            .into_par_iter()
            .map(|i| {
                let vid = VertexId::new(i);
                if boundary_vertices[i] {
                    *mesh.position(vid)
                } else {
                    compute_laplacian_step(mesh, vid, lambda)
                }
            })
            .collect()
    } else {
        (0..num_vertices)
            .map(|i| {
                let vid = VertexId::new(i);
                if boundary_vertices[i] {
                    *mesh.position(vid)
                } else {
                    compute_laplacian_step(mesh, vid, lambda)
                }
            })
            .collect()
    };

    // Apply new positions
    for i in 0..num_vertices {
        let vid = VertexId::new(i);
        mesh.set_position(vid, new_positions[i]);
    }
}

// ============================================================================
// Progress-enabled variants
// ============================================================================

/// Laplacian smoothing with progress reporting.
pub fn laplacian_smooth_with_progress<I: MeshIndex + Sync>(
    mesh: &mut HalfEdgeMesh<I>,
    options: &SmoothOptions,
    progress: &Progress,
) {
    if options.iterations == 0 || options.lambda == 0.0 {
        return;
    }

    let boundary_vertices: Vec<bool> = if options.preserve_boundary {
        mesh.vertex_ids()
            .map(|v| mesh.is_boundary_vertex(v))
            .collect()
    } else {
        vec![false; mesh.num_vertices()]
    };

    let num_vertices = mesh.num_vertices();

    for iter in 0..options.iterations {
        progress.report(iter, options.iterations, "Laplacian smoothing");

        // Compute new positions
        let new_positions: Vec<Point3<f64>> = if options.parallel {
            (0..num_vertices)
                .into_par_iter()
                .map(|i| {
                    let vid = VertexId::new(i);
                    if boundary_vertices[i] {
                        *mesh.position(vid)
                    } else {
                        compute_laplacian_step(mesh, vid, options.lambda)
                    }
                })
                .collect()
        } else {
            (0..num_vertices)
                .map(|i| {
                    let vid = VertexId::new(i);
                    if boundary_vertices[i] {
                        *mesh.position(vid)
                    } else {
                        compute_laplacian_step(mesh, vid, options.lambda)
                    }
                })
                .collect()
        };

        for i in 0..num_vertices {
            let vid = VertexId::new(i);
            mesh.set_position(vid, new_positions[i]);
        }
    }
    progress.report(options.iterations, options.iterations, "Laplacian smoothing");
}

/// Taubin smoothing with progress reporting.
pub fn taubin_smooth_with_progress<I: MeshIndex + Sync>(
    mesh: &mut HalfEdgeMesh<I>,
    options: &SmoothOptions,
    progress: &Progress,
) {
    if options.iterations == 0 || options.lambda == 0.0 {
        return;
    }

    let k_pb = 0.1_f64;
    let mu = options.lambda / (k_pb * options.lambda - 1.0);

    let boundary_vertices: Vec<bool> = if options.preserve_boundary {
        mesh.vertex_ids()
            .map(|v| mesh.is_boundary_vertex(v))
            .collect()
    } else {
        vec![false; mesh.num_vertices()]
    };

    for iter in 0..options.iterations {
        progress.report(iter, options.iterations, "Taubin smoothing");

        // Positive step (smoothing)
        apply_laplacian_step_impl(mesh, &boundary_vertices, options.lambda, options.parallel);
        // Negative step (inflation)
        apply_laplacian_step_impl(mesh, &boundary_vertices, mu, options.parallel);
    }
    progress.report(options.iterations, options.iterations, "Taubin smoothing");
}

/// Cotangent smoothing with progress reporting.
pub fn cotangent_smooth_with_progress<I: MeshIndex + Sync>(
    mesh: &mut HalfEdgeMesh<I>,
    options: &SmoothOptions,
    progress: &Progress,
) {
    if options.iterations == 0 || options.lambda == 0.0 {
        return;
    }

    let boundary_vertices: Vec<bool> = if options.preserve_boundary {
        mesh.vertex_ids()
            .map(|v| mesh.is_boundary_vertex(v))
            .collect()
    } else {
        vec![false; mesh.num_vertices()]
    };

    let num_vertices = mesh.num_vertices();

    for iter in 0..options.iterations {
        progress.report(iter, options.iterations, "Cotangent smoothing");

        // Compute new positions
        let new_positions: Vec<Point3<f64>> = if options.parallel {
            (0..num_vertices)
                .into_par_iter()
                .map(|i| {
                    let vid = VertexId::new(i);
                    if boundary_vertices[i] {
                        *mesh.position(vid)
                    } else {
                        compute_cotangent_laplacian_step(mesh, vid, options.lambda)
                    }
                })
                .collect()
        } else {
            (0..num_vertices)
                .map(|i| {
                    let vid = VertexId::new(i);
                    if boundary_vertices[i] {
                        *mesh.position(vid)
                    } else {
                        compute_cotangent_laplacian_step(mesh, vid, options.lambda)
                    }
                })
                .collect()
        };

        for i in 0..num_vertices {
            let vid = VertexId::new(i);
            mesh.set_position(vid, new_positions[i]);
        }
    }
    progress.report(options.iterations, options.iterations, "Cotangent smoothing");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::build_from_triangles;

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

    fn create_single_triangle() -> HalfEdgeMesh {
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.5, 1.0, 0.0),
        ];
        let faces = vec![[0, 1, 2]];
        build_from_triangles(&vertices, &faces).unwrap()
    }

    #[test]
    fn test_laplacian_smooth_preserves_boundary() {
        let mut mesh = create_single_triangle();

        // Store original positions
        let original: Vec<Point3<f64>> = mesh.vertex_ids().map(|v| *mesh.position(v)).collect();

        // Smooth with boundary preservation (default)
        let options = SmoothOptions::default().with_iterations(5);
        laplacian_smooth(&mut mesh, &options);

        // All vertices are boundary, so positions should be unchanged
        for (vid, orig) in mesh.vertex_ids().zip(original.iter()) {
            let pos = mesh.position(vid);
            assert!(
                (pos - orig).norm() < 1e-10,
                "Boundary vertex moved: {:?} -> {:?}",
                orig,
                pos
            );
        }
    }

    #[test]
    fn test_laplacian_smooth_closed_mesh() {
        let mut mesh = create_tetrahedron();

        // Store original centroid
        let original_centroid: Vector3<f64> = mesh
            .vertex_ids()
            .map(|v| mesh.position(v).coords)
            .sum::<Vector3<f64>>()
            / mesh.num_vertices() as f64;

        // Smooth the mesh
        let options = SmoothOptions::default().with_iterations(10).with_lambda(0.5);
        laplacian_smooth(&mut mesh, &options);

        // Centroid should be approximately preserved
        let new_centroid: Vector3<f64> = mesh
            .vertex_ids()
            .map(|v| mesh.position(v).coords)
            .sum::<Vector3<f64>>()
            / mesh.num_vertices() as f64;

        assert!(
            (new_centroid - original_centroid).norm() < 0.1,
            "Centroid drifted too much: {:?} -> {:?}",
            original_centroid,
            new_centroid
        );
    }

    #[test]
    fn test_taubin_smooth_reduces_shrinkage() {
        let mut mesh_laplacian = create_tetrahedron();
        let mut mesh_taubin = create_tetrahedron();

        // Compute original surface area
        let original_area = mesh_laplacian.surface_area();

        // Apply many iterations of both methods
        let options = SmoothOptions::default().with_iterations(20).with_lambda(0.5);

        laplacian_smooth(&mut mesh_laplacian, &options);
        taubin_smooth(&mut mesh_taubin, &options);

        let area_laplacian = mesh_laplacian.surface_area();
        let area_taubin = mesh_taubin.surface_area();

        // Taubin should preserve area better than Laplacian
        let laplacian_shrinkage = (original_area - area_laplacian) / original_area;
        let taubin_shrinkage = (original_area - area_taubin) / original_area;

        assert!(
            taubin_shrinkage.abs() < laplacian_shrinkage.abs(),
            "Taubin should shrink less: Laplacian={:.2}%, Taubin={:.2}%",
            laplacian_shrinkage * 100.0,
            taubin_shrinkage * 100.0
        );
    }

    #[test]
    fn test_zero_iterations_no_change() {
        let mut mesh = create_tetrahedron();
        let original: Vec<Point3<f64>> = mesh.vertex_ids().map(|v| *mesh.position(v)).collect();

        let options = SmoothOptions::default().with_iterations(0);
        laplacian_smooth(&mut mesh, &options);

        for (vid, orig) in mesh.vertex_ids().zip(original.iter()) {
            assert_eq!(mesh.position(vid), orig);
        }
    }

    #[test]
    fn test_zero_lambda_no_change() {
        let mut mesh = create_tetrahedron();
        let original: Vec<Point3<f64>> = mesh.vertex_ids().map(|v| *mesh.position(v)).collect();

        let options = SmoothOptions::default()
            .with_iterations(10)
            .with_lambda(0.0);
        laplacian_smooth(&mut mesh, &options);

        for (vid, orig) in mesh.vertex_ids().zip(original.iter()) {
            assert_eq!(mesh.position(vid), orig);
        }
    }

    #[test]
    fn test_bilateral_smooth_preserves_boundary() {
        let mut mesh = create_single_triangle();

        let original: Vec<Point3<f64>> = mesh.vertex_ids().map(|v| *mesh.position(v)).collect();

        let options = BilateralOptions::default().with_iterations(5);
        bilateral_smooth(&mut mesh, &options);

        // All vertices are boundary, so positions should be unchanged
        for (vid, orig) in mesh.vertex_ids().zip(original.iter()) {
            let pos = mesh.position(vid);
            assert!(
                (pos - orig).norm() < 1e-10,
                "Boundary vertex moved"
            );
        }
    }

    #[test]
    fn test_bilateral_smooth_closed_mesh() {
        let mut mesh = create_tetrahedron();

        // Just verify the mesh stays valid after smoothing
        let options = BilateralOptions::default()
            .with_iterations(3)
            .with_sigma_c(0.5)
            .with_sigma_s(0.3);
        bilateral_smooth(&mut mesh, &options);

        // Mesh should still be valid
        assert!(mesh.is_valid(), "Mesh should be valid after bilateral smooth");

        // All vertices should still exist and have reasonable positions
        for vid in mesh.vertex_ids() {
            let pos = mesh.position(vid);
            assert!(pos.x.is_finite() && pos.y.is_finite() && pos.z.is_finite(),
                    "Vertex position should be finite");
        }
    }

    #[test]
    fn test_mean_curvature_flow_preserves_boundary() {
        let mut mesh = create_single_triangle();

        let original: Vec<Point3<f64>> = mesh.vertex_ids().map(|v| *mesh.position(v)).collect();

        let options = CurvatureFlowOptions::default().with_iterations(10);
        mean_curvature_flow(&mut mesh, &options);

        // All vertices are boundary, so positions should be unchanged
        for (vid, orig) in mesh.vertex_ids().zip(original.iter()) {
            let pos = mesh.position(vid);
            assert!(
                (pos - orig).norm() < 1e-10,
                "Boundary vertex moved"
            );
        }
    }

    #[test]
    fn test_mean_curvature_flow_shrinks_surface() {
        let mut mesh = create_tetrahedron();

        let original_area = mesh.surface_area();

        // Mean curvature flow should reduce surface area (area-minimizing)
        let options = CurvatureFlowOptions::default()
            .with_iterations(100)
            .with_time_step(0.01);
        mean_curvature_flow(&mut mesh, &options);

        let new_area = mesh.surface_area();

        // Surface area should decrease
        assert!(
            new_area < original_area,
            "Mean curvature flow should reduce area: {} -> {}",
            original_area,
            new_area
        );
        assert!(mesh.is_valid(), "Mesh should be valid after flow");
    }

    #[test]
    fn test_cotangent_smooth_closed_mesh() {
        let mut mesh = create_tetrahedron();

        let original_centroid: Vector3<f64> = mesh
            .vertex_ids()
            .map(|v| mesh.position(v).coords)
            .sum::<Vector3<f64>>()
            / mesh.num_vertices() as f64;

        let options = SmoothOptions::default().with_iterations(5).with_lambda(0.3);
        cotangent_smooth(&mut mesh, &options);

        // Centroid should be approximately preserved
        let new_centroid: Vector3<f64> = mesh
            .vertex_ids()
            .map(|v| mesh.position(v).coords)
            .sum::<Vector3<f64>>()
            / mesh.num_vertices() as f64;

        assert!(
            (new_centroid - original_centroid).norm() < 0.2,
            "Centroid drifted too much"
        );
        assert!(mesh.is_valid(), "Mesh should be valid after cotangent smooth");
    }
}
