//! Mesh remeshing algorithms.
//!
//! This module provides algorithms for remeshing triangle meshes:
//!
//! - **Isotropic remeshing**: Uniform edge lengths throughout the mesh
//! - **Anisotropic remeshing**: Edge lengths adapted to local curvature
//!
//! # Isotropic Remeshing
//!
//! The isotropic remeshing algorithm (Botsch & Kobbelt, 2004) iteratively applies:
//!
//! 1. **Split** edges longer than 4/3 × target_length
//! 2. **Collapse** edges shorter than 4/5 × target_length
//! 3. **Flip** edges to improve vertex valence
//! 4. **Tangential smoothing** to regularize vertex positions
//!
//! # Anisotropic Remeshing
//!
//! Anisotropic remeshing adapts edge lengths based on local curvature:
//! - High curvature regions get shorter edges (more detail)
//! - Low curvature regions get longer edges (fewer triangles)
//!
//! # Example
//!
//! ```no_run
//! use morsel::prelude::*;
//! use morsel::algo::remesh::{isotropic_remesh, RemeshOptions};
//!
//! let mut mesh: HalfEdgeMesh = morsel::io::load("input.obj").unwrap();
//!
//! let options = RemeshOptions::with_target_length(0.1)
//!     .with_iterations(5);
//! isotropic_remesh(&mut mesh, &options);
//!
//! morsel::io::save(&mesh, "output.obj").unwrap();
//! ```
//!
//! # References
//!
//! - Botsch, M., & Kobbelt, L. (2004). "A remeshing approach to multiresolution modeling."
//!   Symposium on Geometry Processing.
//! - Dunyach, M., et al. (2013). "Adaptive remeshing for real-time mesh deformation."
//!   Eurographics.

use std::collections::HashSet;

use nalgebra::{Point3, Vector3};

use crate::mesh::{build_from_triangles, to_face_vertex, HalfEdgeMesh, MeshIndex};

/// Options for isotropic remeshing.
#[derive(Debug, Clone)]
pub struct RemeshOptions {
    /// Target edge length for the remeshed surface.
    pub target_length: f64,

    /// Number of remeshing iterations.
    pub iterations: usize,

    /// Whether to preserve boundary edges (don't collapse/flip them).
    pub preserve_boundary: bool,

    /// Number of tangential smoothing iterations per remeshing iteration.
    pub smoothing_iterations: usize,

    /// Smoothing factor for tangential relaxation.
    pub smoothing_lambda: f64,
}

impl RemeshOptions {
    /// Create options with the specified target edge length.
    pub fn with_target_length(target_length: f64) -> Self {
        Self {
            target_length,
            iterations: 5,
            preserve_boundary: true,
            smoothing_iterations: 3,
            smoothing_lambda: 0.5,
        }
    }

    /// Set the number of remeshing iterations.
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Set whether to preserve boundary edges.
    pub fn with_preserve_boundary(mut self, preserve: bool) -> Self {
        self.preserve_boundary = preserve;
        self
    }

    /// Set the number of smoothing iterations per remeshing iteration.
    pub fn with_smoothing_iterations(mut self, iterations: usize) -> Self {
        self.smoothing_iterations = iterations;
        self
    }
}

/// Performs isotropic remeshing on a triangle mesh.
///
/// This algorithm produces a mesh with uniform, near-equilateral triangles
/// with edge lengths close to the specified target length.
///
/// # Arguments
///
/// * `mesh` - The mesh to remesh (modified in place)
/// * `options` - Remeshing parameters
///
/// # Algorithm Steps (per iteration)
///
/// 1. **Edge splitting**: Split edges longer than 4/3 × target_length
/// 2. **Edge collapsing**: Collapse edges shorter than 4/5 × target_length
/// 3. **Edge flipping**: Flip edges to equalize vertex valence
/// 4. **Tangential smoothing**: Smooth while preserving surface features
///
/// # Note
///
/// This implementation rebuilds the mesh connectivity after each major
/// operation for simplicity. For very large meshes, a more incremental
/// approach would be more efficient.
pub fn isotropic_remesh<I: MeshIndex>(mesh: &mut HalfEdgeMesh<I>, options: &RemeshOptions) {
    if options.iterations == 0 || options.target_length <= 0.0 {
        return;
    }

    let high = options.target_length * 4.0 / 3.0;
    let low = options.target_length * 4.0 / 5.0;

    for _iter in 0..options.iterations {
        // Step 1: Split long edges
        split_long_edges(mesh, high, options.preserve_boundary);

        // Step 2: Collapse short edges
        collapse_short_edges(mesh, low, high, options.preserve_boundary);

        // Step 3: Flip edges to improve valence
        flip_edges_to_improve_valence(mesh, options.preserve_boundary);

        // Step 4: Tangential smoothing
        for _ in 0..options.smoothing_iterations {
            tangential_smooth(mesh, options.smoothing_lambda, options.preserve_boundary);
        }
    }
}

// ============================================================================
// Anisotropic Remeshing
// ============================================================================

/// Options for anisotropic remeshing.
///
/// Anisotropic remeshing adapts edge lengths based on local curvature,
/// using shorter edges in high-curvature regions and longer edges in
/// flat regions.
#[derive(Debug, Clone)]
pub struct AnisotropicOptions {
    /// Minimum allowed edge length.
    pub min_length: f64,

    /// Maximum allowed edge length.
    pub max_length: f64,

    /// Number of remeshing iterations.
    pub iterations: usize,

    /// Whether to preserve boundary edges.
    pub preserve_boundary: bool,

    /// Number of tangential smoothing iterations per remeshing iteration.
    pub smoothing_iterations: usize,

    /// Smoothing factor for tangential relaxation.
    pub smoothing_lambda: f64,

    /// Curvature adaptation strength (0.0 = uniform, 1.0 = fully adaptive).
    /// Higher values create more size variation based on curvature.
    pub adaptation: f64,
}

impl AnisotropicOptions {
    /// Create options with the specified edge length bounds.
    ///
    /// # Arguments
    /// * `min_length` - Minimum edge length (used in high curvature regions)
    /// * `max_length` - Maximum edge length (used in flat regions)
    pub fn new(min_length: f64, max_length: f64) -> Self {
        Self {
            min_length,
            max_length,
            iterations: 5,
            preserve_boundary: true,
            smoothing_iterations: 3,
            smoothing_lambda: 0.5,
            adaptation: 1.0,
        }
    }

    /// Set the number of remeshing iterations.
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Set whether to preserve boundary edges.
    pub fn with_preserve_boundary(mut self, preserve: bool) -> Self {
        self.preserve_boundary = preserve;
        self
    }

    /// Set the curvature adaptation strength.
    pub fn with_adaptation(mut self, adaptation: f64) -> Self {
        self.adaptation = adaptation.clamp(0.0, 1.0);
        self
    }

    /// Set the number of smoothing iterations per remeshing iteration.
    pub fn with_smoothing_iterations(mut self, iterations: usize) -> Self {
        self.smoothing_iterations = iterations;
        self
    }
}

/// Performs anisotropic remeshing on a triangle mesh.
///
/// This algorithm produces a mesh with edge lengths adapted to local curvature:
/// shorter edges in high-curvature regions, longer edges in flat regions.
///
/// # Arguments
///
/// * `mesh` - The mesh to remesh (modified in place)
/// * `options` - Remeshing parameters
///
/// # Algorithm Steps (per iteration)
///
/// 1. Compute sizing field from curvature
/// 2. Split edges longer than 4/3 × local_target
/// 3. Collapse edges shorter than 4/5 × local_target
/// 4. Flip edges to equalize vertex valence
/// 5. Tangential smoothing
pub fn anisotropic_remesh<I: MeshIndex>(mesh: &mut HalfEdgeMesh<I>, options: &AnisotropicOptions) {
    if options.iterations == 0 || options.min_length <= 0.0 || options.max_length <= options.min_length {
        return;
    }

    for _iter in 0..options.iterations {
        // Step 1: Compute sizing field from curvature
        let sizing = compute_sizing_field(mesh, options);

        // Step 2: Split long edges (using local target)
        split_long_edges_anisotropic(mesh, &sizing, options.preserve_boundary);

        // Recompute sizing after topology change
        let sizing = compute_sizing_field(mesh, options);

        // Step 3: Collapse short edges (using local target)
        collapse_short_edges_anisotropic(mesh, &sizing, options.preserve_boundary);

        // Recompute sizing after topology change
        let sizing = compute_sizing_field(mesh, options);

        // Step 4: Flip edges to improve valence
        flip_edges_to_improve_valence(mesh, options.preserve_boundary);

        // Step 5: Tangential smoothing
        for _ in 0..options.smoothing_iterations {
            tangential_smooth(mesh, options.smoothing_lambda, options.preserve_boundary);
        }

        // Note: sizing field is recomputed each iteration to adapt to new geometry
        let _ = sizing; // suppress unused warning
    }
}

/// Per-vertex sizing field (target edge length at each vertex).
#[derive(Debug, Clone)]
pub struct SizingField {
    /// Target edge length for each vertex.
    pub vertex_sizes: Vec<f64>,
}

impl SizingField {
    /// Get the target edge length at a vertex.
    pub fn get(&self, vertex_idx: usize) -> f64 {
        self.vertex_sizes.get(vertex_idx).copied().unwrap_or(1.0)
    }

    /// Get the target edge length for an edge (average of endpoint sizes).
    pub fn edge_target(&self, v0: usize, v1: usize) -> f64 {
        (self.get(v0) + self.get(v1)) * 0.5
    }
}

/// Compute the sizing field based on local curvature.
///
/// The sizing field maps curvature to target edge length:
/// - High curvature → min_length
/// - Low curvature → max_length
pub fn compute_sizing_field<I: MeshIndex>(
    mesh: &HalfEdgeMesh<I>,
    options: &AnisotropicOptions,
) -> SizingField {
    let (vertices, faces) = to_face_vertex(mesh);

    // Compute mean curvature magnitude at each vertex
    let curvatures = compute_mean_curvature_magnitudes(&vertices, &faces);

    // Find curvature range for normalization
    let max_curv = curvatures
        .iter()
        .copied()
        .filter(|c| c.is_finite())
        .fold(0.0_f64, f64::max);

    let min_curv = 0.0; // Flat surfaces have zero curvature

    // Map curvature to sizing
    let vertex_sizes: Vec<f64> = curvatures
        .iter()
        .map(|&curv| {
            if !curv.is_finite() || max_curv <= min_curv {
                // Default to midpoint if curvature is invalid or uniform
                (options.min_length + options.max_length) * 0.5
            } else {
                // Normalize curvature to [0, 1]
                let t = ((curv - min_curv) / (max_curv - min_curv)).clamp(0.0, 1.0);

                // Apply adaptation strength
                let t = t * options.adaptation;

                // Interpolate: high curvature (t=1) → min_length, low curvature (t=0) → max_length
                options.max_length * (1.0 - t) + options.min_length * t
            }
        })
        .collect();

    SizingField { vertex_sizes }
}

/// Compute mean curvature magnitude at each vertex using the cotangent Laplacian.
fn compute_mean_curvature_magnitudes(vertices: &[Point3<f64>], faces: &[[usize; 3]]) -> Vec<f64> {
    let n = vertices.len();
    let mut curvatures = vec![0.0; n];
    let mut areas = vec![0.0; n];

    // Build one-ring information
    for face in faces {
        let i0 = face[0];
        let i1 = face[1];
        let i2 = face[2];

        let p0 = &vertices[i0];
        let p1 = &vertices[i1];
        let p2 = &vertices[i2];

        // Edge vectors
        let e0 = p1 - p0;
        let e1 = p2 - p1;
        let e2 = p0 - p2;

        // Triangle area
        let area = e0.cross(&(-e2)).norm() * 0.5;
        if area < 1e-12 {
            continue;
        }

        // Cotangent weights for each edge
        // cot(angle at vertex i) for edge opposite to i
        let cot0 = cotangent_weight(&e1, &e2);
        let cot1 = cotangent_weight(&e2, &e0);
        let cot2 = cotangent_weight(&e0, &e1);

        // Accumulate mean curvature vector (Laplacian)
        // For vertex i0, edges are (i0,i1) and (i0,i2)
        // Edge (i0,i1) has cotangent weight from angle at i2
        // Edge (i0,i2) has cotangent weight from angle at i1

        // Contribution to i0 from edge i0-i1
        curvatures[i0] += cot2 * (p1 - p0).norm();
        // Contribution to i0 from edge i0-i2
        curvatures[i0] += cot1 * (p2 - p0).norm();

        // Contribution to i1 from edge i1-i0
        curvatures[i1] += cot2 * (p0 - p1).norm();
        // Contribution to i1 from edge i1-i2
        curvatures[i1] += cot0 * (p2 - p1).norm();

        // Contribution to i2 from edge i2-i0
        curvatures[i2] += cot1 * (p0 - p2).norm();
        // Contribution to i2 from edge i2-i1
        curvatures[i2] += cot0 * (p1 - p2).norm();

        // Voronoi area contribution (simplified: use barycentric area)
        let area_third = area / 3.0;
        areas[i0] += area_third;
        areas[i1] += area_third;
        areas[i2] += area_third;
    }

    // Normalize by area to get mean curvature
    for i in 0..n {
        if areas[i] > 1e-12 {
            curvatures[i] /= 4.0 * areas[i];
        }
    }

    curvatures
}

/// Compute cotangent of angle between two edge vectors.
fn cotangent_weight(e1: &Vector3<f64>, e2: &Vector3<f64>) -> f64 {
    // e1 and e2 share a vertex, angle is between them
    // cot(theta) = cos(theta) / sin(theta) = (e1 · e2) / |e1 × e2|
    let dot = -e1.dot(e2); // Negative because edges point in opposite directions at the vertex
    let cross_norm = e1.cross(e2).norm();

    if cross_norm < 1e-12 {
        0.0 // Degenerate, return 0
    } else {
        (dot / cross_norm).clamp(-100.0, 100.0) // Clamp to avoid extreme weights
    }
}

/// Split edges longer than their local target (anisotropic version).
fn split_long_edges_anisotropic<I: MeshIndex>(
    mesh: &mut HalfEdgeMesh<I>,
    sizing: &SizingField,
    preserve_boundary: bool,
) {
    let (mut vertices, mut faces) = to_face_vertex(mesh);
    let mut vertex_sizes = sizing.vertex_sizes.clone();

    let mut changed = true;
    while changed {
        changed = false;

        let mut edges_to_split: Vec<(usize, usize)> = Vec::new();
        let mut seen_edges: HashSet<(usize, usize)> = HashSet::new();

        for face in &faces {
            for i in 0..3 {
                let v0 = face[i];
                let v1 = face[(i + 1) % 3];
                let edge = if v0 < v1 { (v0, v1) } else { (v1, v0) };

                if seen_edges.contains(&edge) {
                    continue;
                }
                seen_edges.insert(edge);

                let p0 = &vertices[v0];
                let p1 = &vertices[v1];
                let length = (p1 - p0).norm();

                // Local target for this edge
                let target = (vertex_sizes[v0] + vertex_sizes[v1]) * 0.5;
                let threshold = target * 4.0 / 3.0;

                if length > threshold {
                    if preserve_boundary && is_boundary_edge_in_faces(&faces, v0, v1) {
                        continue;
                    }
                    edges_to_split.push((v0, v1));
                }
            }
        }

        if edges_to_split.is_empty() {
            break;
        }

        for (v0, v1) in edges_to_split {
            // Compute size for new vertex (interpolated)
            let new_size = (vertex_sizes[v0] + vertex_sizes[v1]) * 0.5;

            split_edge(&mut vertices, &mut faces, v0, v1);
            vertex_sizes.push(new_size);
            changed = true;
        }
    }

    if let Ok(new_mesh) = build_from_triangles::<I>(&vertices, &faces) {
        *mesh = new_mesh;
    }
}

/// Collapse edges shorter than their local target (anisotropic version).
fn collapse_short_edges_anisotropic<I: MeshIndex>(
    mesh: &mut HalfEdgeMesh<I>,
    sizing: &SizingField,
    preserve_boundary: bool,
) {
    let (mut vertices, mut faces) = to_face_vertex(mesh);
    let mut vertex_sizes = sizing.vertex_sizes.clone();

    let mut changed = true;
    while changed {
        changed = false;

        let mut edge_to_collapse: Option<(usize, usize)> = None;
        let mut seen_edges: HashSet<(usize, usize)> = HashSet::new();

        for face in &faces {
            for i in 0..3 {
                let v0 = face[i];
                let v1 = face[(i + 1) % 3];
                let edge = if v0 < v1 { (v0, v1) } else { (v1, v0) };

                if seen_edges.contains(&edge) {
                    continue;
                }
                seen_edges.insert(edge);

                let p0 = &vertices[v0];
                let p1 = &vertices[v1];
                let length = (p1 - p0).norm();

                // Local target for this edge
                let target = (vertex_sizes.get(v0).copied().unwrap_or(1.0)
                    + vertex_sizes.get(v1).copied().unwrap_or(1.0))
                    * 0.5;
                let low_threshold = target * 4.0 / 5.0;

                if length < low_threshold {
                    if can_collapse_edge_anisotropic(
                        &vertices,
                        &faces,
                        &vertex_sizes,
                        v0,
                        v1,
                        preserve_boundary,
                    ) {
                        edge_to_collapse = Some((v0, v1));
                        break;
                    }
                }
            }
            if edge_to_collapse.is_some() {
                break;
            }
        }

        if let Some((v0, v1)) = edge_to_collapse {
            // Update sizing for merged vertex
            let new_size = (vertex_sizes.get(v0).copied().unwrap_or(1.0)
                + vertex_sizes.get(v1).copied().unwrap_or(1.0))
                * 0.5;

            collapse_edge(&mut vertices, &mut faces, v0, v1);
            if v0 < vertex_sizes.len() {
                vertex_sizes[v0] = new_size;
            }
            changed = true;
        }
    }

    let (clean_vertices, clean_faces) = cleanup_mesh(&vertices, &faces);

    if !clean_faces.is_empty() {
        if let Ok(new_mesh) = build_from_triangles::<I>(&clean_vertices, &clean_faces) {
            *mesh = new_mesh;
        }
    }
}

/// Check if an edge can be safely collapsed (anisotropic version).
fn can_collapse_edge_anisotropic(
    vertices: &[Point3<f64>],
    faces: &[[usize; 3]],
    vertex_sizes: &[f64],
    v0: usize,
    v1: usize,
    preserve_boundary: bool,
) -> bool {
    // Don't collapse boundary edges if preserving boundary
    if preserve_boundary && is_boundary_edge_in_faces(faces, v0, v1) {
        return false;
    }

    // Don't collapse if both vertices are on boundary
    if preserve_boundary {
        let v0_boundary = is_boundary_vertex_in_faces(faces, v0);
        let v1_boundary = is_boundary_vertex_in_faces(faces, v1);
        if v0_boundary && v1_boundary && !is_boundary_edge_in_faces(faces, v0, v1) {
            return false;
        }
    }

    // Compute midpoint
    let midpoint = (vertices[v0].coords + vertices[v1].coords) * 0.5;

    // Check if collapse would create edges that are too long
    let neighbors: HashSet<usize> = get_vertex_neighbors(faces, v0)
        .union(&get_vertex_neighbors(faces, v1))
        .copied()
        .filter(|&v| v != v0 && v != v1)
        .collect();

    // New vertex size will be average
    let new_size = (vertex_sizes.get(v0).copied().unwrap_or(1.0)
        + vertex_sizes.get(v1).copied().unwrap_or(1.0))
        * 0.5;

    for &neighbor in &neighbors {
        let neighbor_size = vertex_sizes.get(neighbor).copied().unwrap_or(1.0);
        let edge_target = (new_size + neighbor_size) * 0.5;
        let high_threshold = edge_target * 4.0 / 3.0;

        let new_length = (vertices[neighbor].coords - midpoint).norm();
        if new_length > high_threshold {
            return false;
        }
    }

    // Check link condition
    let neighbors_v0 = get_vertex_neighbors(faces, v0);
    let neighbors_v1 = get_vertex_neighbors(faces, v1);
    let common: HashSet<_> = neighbors_v0.intersection(&neighbors_v1).collect();

    if !is_boundary_edge_in_faces(faces, v0, v1) && common.len() != 2 {
        return false;
    }

    true
}

/// Compute the average edge length of a mesh.
///
/// This is useful for determining an appropriate target edge length
/// for remeshing.
pub fn average_edge_length<I: MeshIndex>(mesh: &HalfEdgeMesh<I>) -> f64 {
    let mut total_length = 0.0;
    let mut edge_count = 0;

    // Only count each edge once (skip if twin has lower index)
    for he_id in mesh.halfedge_ids() {
        let twin_id = mesh.twin(he_id);
        if he_id.index() < twin_id.index() {
            total_length += mesh.edge_length(he_id);
            edge_count += 1;
        }
    }

    if edge_count == 0 {
        0.0
    } else {
        total_length / edge_count as f64
    }
}

// ============================================================================
// Edge Split
// ============================================================================

/// Split all edges longer than the threshold.
fn split_long_edges<I: MeshIndex>(mesh: &mut HalfEdgeMesh<I>, threshold: f64, preserve_boundary: bool) {
    // We need to rebuild the mesh after splits, so collect the geometry first
    let (mut vertices, mut faces) = to_face_vertex(mesh);

    let mut changed = true;
    while changed {
        changed = false;

        // Find edges to split
        let mut edges_to_split: Vec<(usize, usize)> = Vec::new();

        // Use a set to avoid duplicate edges
        let mut seen_edges: HashSet<(usize, usize)> = HashSet::new();

        for face in &faces {
            for i in 0..3 {
                let v0 = face[i];
                let v1 = face[(i + 1) % 3];
                let edge = if v0 < v1 { (v0, v1) } else { (v1, v0) };

                if seen_edges.contains(&edge) {
                    continue;
                }
                seen_edges.insert(edge);

                let p0 = &vertices[v0];
                let p1 = &vertices[v1];
                let length = (p1 - p0).norm();

                if length > threshold {
                    // Check if this is a boundary edge (only in one face)
                    if preserve_boundary && is_boundary_edge_in_faces(&faces, v0, v1) {
                        continue;
                    }
                    edges_to_split.push((v0, v1));
                }
            }
        }

        if edges_to_split.is_empty() {
            break;
        }

        // Split each edge
        for (v0, v1) in edges_to_split {
            split_edge(&mut vertices, &mut faces, v0, v1);
            changed = true;
        }
    }

    // Rebuild the mesh
    if let Ok(new_mesh) = build_from_triangles::<I>(&vertices, &faces) {
        *mesh = new_mesh;
    }
}

/// Check if an edge exists in the face list.
fn edge_exists_in_faces(faces: &[[usize; 3]], v0: usize, v1: usize) -> bool {
    for face in faces {
        for i in 0..3 {
            let a = face[i];
            let b = face[(i + 1) % 3];
            if (a == v0 && b == v1) || (a == v1 && b == v0) {
                return true;
            }
        }
    }
    false
}

/// Check if an edge is on the boundary (appears in only one face).
fn is_boundary_edge_in_faces(faces: &[[usize; 3]], v0: usize, v1: usize) -> bool {
    let mut count = 0;
    for face in faces {
        for i in 0..3 {
            let a = face[i];
            let b = face[(i + 1) % 3];
            if (a == v0 && b == v1) || (a == v1 && b == v0) {
                count += 1;
            }
        }
    }
    count == 1
}

/// Split an edge by inserting a vertex at its midpoint.
fn split_edge(vertices: &mut Vec<Point3<f64>>, faces: &mut Vec<[usize; 3]>, v0: usize, v1: usize) {
    // Add midpoint vertex
    let midpoint = Point3::from((vertices[v0].coords + vertices[v1].coords) * 0.5);
    let mid_idx = vertices.len();
    vertices.push(midpoint);

    // Find and replace faces containing this edge
    let mut new_faces: Vec<[usize; 3]> = Vec::new();
    let mut i = 0;
    while i < faces.len() {
        let face = faces[i];

        // Check if this face contains the edge
        let mut edge_idx = None;
        for j in 0..3 {
            let a = face[j];
            let b = face[(j + 1) % 3];
            if (a == v0 && b == v1) || (a == v1 && b == v0) {
                edge_idx = Some(j);
                break;
            }
        }

        if let Some(j) = edge_idx {
            // Split this face into two
            let a = face[j];
            let b = face[(j + 1) % 3];
            let c = face[(j + 2) % 3];

            // Remove old face
            faces.swap_remove(i);

            // Add two new faces
            new_faces.push([a, mid_idx, c]);
            new_faces.push([mid_idx, b, c]);
        } else {
            i += 1;
        }
    }

    faces.extend(new_faces);
}

// ============================================================================
// Edge Collapse
// ============================================================================

/// Collapse all edges shorter than the threshold.
fn collapse_short_edges<I: MeshIndex>(
    mesh: &mut HalfEdgeMesh<I>,
    low_threshold: f64,
    high_threshold: f64,
    preserve_boundary: bool,
) {
    let (mut vertices, mut faces) = to_face_vertex(mesh);

    let mut changed = true;
    while changed {
        changed = false;

        // Find an edge to collapse
        let mut edge_to_collapse: Option<(usize, usize)> = None;
        let mut seen_edges: HashSet<(usize, usize)> = HashSet::new();

        for face in &faces {
            for i in 0..3 {
                let v0 = face[i];
                let v1 = face[(i + 1) % 3];
                let edge = if v0 < v1 { (v0, v1) } else { (v1, v0) };

                if seen_edges.contains(&edge) {
                    continue;
                }
                seen_edges.insert(edge);

                let p0 = &vertices[v0];
                let p1 = &vertices[v1];
                let length = (p1 - p0).norm();

                if length < low_threshold {
                    // Check if collapse is safe
                    if can_collapse_edge(&vertices, &faces, v0, v1, high_threshold, preserve_boundary) {
                        edge_to_collapse = Some((v0, v1));
                        break;
                    }
                }
            }
            if edge_to_collapse.is_some() {
                break;
            }
        }

        if let Some((v0, v1)) = edge_to_collapse {
            collapse_edge(&mut vertices, &mut faces, v0, v1);
            changed = true;
        }
    }

    // Remove unused vertices and rebuild
    let (clean_vertices, clean_faces) = cleanup_mesh(&vertices, &faces);

    if !clean_faces.is_empty() {
        if let Ok(new_mesh) = build_from_triangles::<I>(&clean_vertices, &clean_faces) {
            *mesh = new_mesh;
        }
    }
}

/// Check if an edge can be safely collapsed.
fn can_collapse_edge(
    vertices: &[Point3<f64>],
    faces: &[[usize; 3]],
    v0: usize,
    v1: usize,
    high_threshold: f64,
    preserve_boundary: bool,
) -> bool {
    // Don't collapse boundary edges if preserving boundary
    if preserve_boundary && is_boundary_edge_in_faces(faces, v0, v1) {
        return false;
    }

    // Don't collapse if both vertices are on boundary (would create non-manifold)
    if preserve_boundary {
        let v0_boundary = is_boundary_vertex_in_faces(faces, v0);
        let v1_boundary = is_boundary_vertex_in_faces(faces, v1);
        if v0_boundary && v1_boundary && !is_boundary_edge_in_faces(faces, v0, v1) {
            return false;
        }
    }

    // Compute midpoint
    let midpoint = (vertices[v0].coords + vertices[v1].coords) * 0.5;

    // Check if collapse would create edges that are too long
    let neighbors: HashSet<usize> = get_vertex_neighbors(faces, v0)
        .union(&get_vertex_neighbors(faces, v1))
        .copied()
        .filter(|&v| v != v0 && v != v1)
        .collect();

    for &neighbor in &neighbors {
        let new_length = (vertices[neighbor].coords - midpoint).norm();
        if new_length > high_threshold {
            return false;
        }
    }

    // Check link condition (simplified): ensure we don't create non-manifold edges
    let neighbors_v0 = get_vertex_neighbors(faces, v0);
    let neighbors_v1 = get_vertex_neighbors(faces, v1);
    let common: HashSet<_> = neighbors_v0.intersection(&neighbors_v1).collect();

    // For a valid collapse, there should be exactly 2 common neighbors (the opposite vertices)
    // unless it's a boundary edge
    if !is_boundary_edge_in_faces(faces, v0, v1) && common.len() != 2 {
        return false;
    }

    true
}

/// Check if a vertex is on the boundary.
fn is_boundary_vertex_in_faces(faces: &[[usize; 3]], v: usize) -> bool {
    // A vertex is on boundary if any of its edges is a boundary edge
    for &neighbor in &get_vertex_neighbors(faces, v) {
        if is_boundary_edge_in_faces(faces, v, neighbor) {
            return true;
        }
    }
    false
}

/// Get all vertex neighbors from the face list.
fn get_vertex_neighbors(faces: &[[usize; 3]], v: usize) -> HashSet<usize> {
    let mut neighbors = HashSet::new();
    for face in faces {
        for i in 0..3 {
            if face[i] == v {
                neighbors.insert(face[(i + 1) % 3]);
                neighbors.insert(face[(i + 2) % 3]);
            }
        }
    }
    neighbors
}

/// Collapse an edge by merging v1 into v0.
fn collapse_edge(vertices: &mut Vec<Point3<f64>>, faces: &mut Vec<[usize; 3]>, v0: usize, v1: usize) {
    // Move v0 to midpoint
    vertices[v0] = Point3::from((vertices[v0].coords + vertices[v1].coords) * 0.5);

    // Replace all references to v1 with v0
    for face in faces.iter_mut() {
        for v in face.iter_mut() {
            if *v == v1 {
                *v = v0;
            }
        }
    }

    // Remove degenerate faces (faces with duplicate vertices)
    faces.retain(|face| {
        face[0] != face[1] && face[1] != face[2] && face[0] != face[2]
    });
}

/// Remove unused vertices and reindex faces.
fn cleanup_mesh(vertices: &[Point3<f64>], faces: &[[usize; 3]]) -> (Vec<Point3<f64>>, Vec<[usize; 3]>) {
    // Find used vertices
    let mut used: HashSet<usize> = HashSet::new();
    for face in faces {
        used.insert(face[0]);
        used.insert(face[1]);
        used.insert(face[2]);
    }

    // Create mapping from old to new indices
    let mut old_to_new: Vec<Option<usize>> = vec![None; vertices.len()];
    let mut new_vertices: Vec<Point3<f64>> = Vec::new();

    for old_idx in 0..vertices.len() {
        if used.contains(&old_idx) {
            old_to_new[old_idx] = Some(new_vertices.len());
            new_vertices.push(vertices[old_idx]);
        }
    }

    // Reindex faces
    let new_faces: Vec<[usize; 3]> = faces
        .iter()
        .map(|face| {
            [
                old_to_new[face[0]].unwrap(),
                old_to_new[face[1]].unwrap(),
                old_to_new[face[2]].unwrap(),
            ]
        })
        .collect();

    (new_vertices, new_faces)
}

// ============================================================================
// Edge Flip
// ============================================================================

/// Flip edges to improve vertex valence (operates on HalfEdgeMesh).
fn flip_edges_to_improve_valence<I: MeshIndex>(mesh: &mut HalfEdgeMesh<I>, preserve_boundary: bool) {
    let (vertices, mut faces) = to_face_vertex(mesh);

    flip_edges_for_valence_faces(&vertices, &mut faces, preserve_boundary);

    // Validate the result before rebuilding
    if !validate_face_list(&vertices, &faces) {
        // Faces are invalid, don't rebuild
        return;
    }

    // Rebuild the mesh
    if let Ok(new_mesh) = build_from_triangles::<I>(&vertices, &faces) {
        *mesh = new_mesh;
    }
}

/// Validate a face list for manifold properties.
fn validate_face_list(vertices: &[Point3<f64>], faces: &[[usize; 3]]) -> bool {
    // Check vertex indices
    for face in faces {
        for &vi in face {
            if vi >= vertices.len() {
                return false;
            }
        }
    }

    // Check for non-manifold edges (>2 faces sharing an edge)
    let mut edge_counts: std::collections::HashMap<(usize, usize), usize> = std::collections::HashMap::new();
    for face in faces {
        for i in 0..3 {
            let v0 = face[i];
            let v1 = face[(i + 1) % 3];
            let edge = if v0 < v1 { (v0, v1) } else { (v1, v0) };
            *edge_counts.entry(edge).or_insert(0) += 1;
        }
    }
    if edge_counts.values().any(|&c| c > 2) {
        return false;
    }

    // Check for duplicate directed edges (inconsistent winding)
    let mut directed_counts: std::collections::HashMap<(usize, usize), usize> = std::collections::HashMap::new();
    for face in faces {
        for i in 0..3 {
            let v0 = face[i];
            let v1 = face[(i + 1) % 3];
            *directed_counts.entry((v0, v1)).or_insert(0) += 1;
        }
    }
    if directed_counts.values().any(|&c| c > 1) {
        return false;
    }

    true
}

/// Flip edges to improve vertex valence (operates on face list).
fn flip_edges_for_valence_faces(vertices: &[Point3<f64>], faces: &mut Vec<[usize; 3]>, preserve_boundary: bool) {
    let max_iterations = faces.len() * 3; // Prevent infinite loops
    let mut failed_edges: HashSet<(usize, usize)> = HashSet::new();

    for _iteration in 0..max_iterations {
        // Find the first edge that should be flipped (excluding failed ones)
        let edge_to_flip = find_edge_to_flip_excluding(vertices, faces, preserve_boundary, &failed_edges);

        match edge_to_flip {
            Some((v0, v1)) => {
                // Save state before flip
                let saved_faces = faces.clone();

                flip_edge(faces, v0, v1);

                // Check if flip created invalid state
                if !validate_face_list(vertices, faces) {
                    // Undo the flip
                    *faces = saved_faces;
                    // Mark this edge as failed
                    let edge = if v0 < v1 { (v0, v1) } else { (v1, v0) };
                    failed_edges.insert(edge);
                }
            }
            None => {
                // No more edges to flip
                break;
            }
        }
    }
}

/// Find the first edge that should be flipped, excluding specified edges.
fn find_edge_to_flip_excluding(
    vertices: &[Point3<f64>],
    faces: &[[usize; 3]],
    preserve_boundary: bool,
    exclude: &HashSet<(usize, usize)>,
) -> Option<(usize, usize)> {
    let mut seen_edges: HashSet<(usize, usize)> = HashSet::new();

    for face in faces {
        for i in 0..3 {
            let v0 = face[i];
            let v1 = face[(i + 1) % 3];
            let edge = if v0 < v1 { (v0, v1) } else { (v1, v0) };

            if seen_edges.contains(&edge) || exclude.contains(&edge) {
                continue;
            }
            seen_edges.insert(edge);

            // Skip boundary edges
            if preserve_boundary && is_boundary_edge_in_faces(faces, v0, v1) {
                continue;
            }

            // Check if flip improves valence
            if should_flip_edge(vertices, faces, v0, v1) {
                return Some((v0, v1));
            }
        }
    }

    None
}

/// Compute target valence for a vertex.
fn target_valence(faces: &[[usize; 3]], v: usize) -> usize {
    if is_boundary_vertex_in_faces(faces, v) {
        4 // Boundary vertices should have valence 4
    } else {
        6 // Interior vertices should have valence 6
    }
}

/// Compute current valence of a vertex.
fn vertex_valence(faces: &[[usize; 3]], v: usize) -> usize {
    get_vertex_neighbors(faces, v).len()
}

/// Check if flipping an edge would improve vertex valences.
fn should_flip_edge(vertices: &[Point3<f64>], faces: &[[usize; 3]], v0: usize, v1: usize) -> bool {
    // Find the two triangles sharing this edge
    let mut adjacent_faces: Vec<usize> = Vec::new();
    for (idx, face) in faces.iter().enumerate() {
        for i in 0..3 {
            let a = face[i];
            let b = face[(i + 1) % 3];
            if (a == v0 && b == v1) || (a == v1 && b == v0) {
                adjacent_faces.push(idx);
            }
        }
    }

    if adjacent_faces.len() != 2 {
        return false; // Not an interior edge
    }

    // Find the opposite vertices
    let mut v2 = None;
    let mut v3 = None;

    for &face_idx in &adjacent_faces {
        let face = faces[face_idx];
        for &v in &face {
            if v != v0 && v != v1 {
                if v2.is_none() {
                    v2 = Some(v);
                } else {
                    v3 = Some(v);
                }
            }
        }
    }

    let (v2, v3) = match (v2, v3) {
        (Some(a), Some(b)) => (a, b),
        _ => return false,
    };

    // Check if the new edge (v2-v3) already exists in the mesh
    // If it does, flipping would create a duplicate edge
    if edge_exists_in_faces(faces, v2, v3) {
        return false;
    }

    // Compute valence deviation before flip
    let dev_before = (vertex_valence(faces, v0) as i32 - target_valence(faces, v0) as i32).abs()
        + (vertex_valence(faces, v1) as i32 - target_valence(faces, v1) as i32).abs()
        + (vertex_valence(faces, v2) as i32 - target_valence(faces, v2) as i32).abs()
        + (vertex_valence(faces, v3) as i32 - target_valence(faces, v3) as i32).abs();

    // After flip: v0 and v1 lose one neighbor, v2 and v3 gain one
    let dev_after = (vertex_valence(faces, v0) as i32 - 1 - target_valence(faces, v0) as i32).abs()
        + (vertex_valence(faces, v1) as i32 - 1 - target_valence(faces, v1) as i32).abs()
        + (vertex_valence(faces, v2) as i32 + 1 - target_valence(faces, v2) as i32).abs()
        + (vertex_valence(faces, v3) as i32 + 1 - target_valence(faces, v3) as i32).abs();

    // Also check that flip doesn't create degenerate geometry
    // (the new edge should be inside the quad)
    let p0 = &vertices[v0];
    let p1 = &vertices[v1];
    let p2 = &vertices[v2];
    let p3 = &vertices[v3];

    // Check if the quad is convex (required for valid flip)
    if !is_convex_quad(p0, p2, p1, p3) {
        return false;
    }

    dev_after < dev_before
}

/// Check if a quad is convex.
fn is_convex_quad(p0: &Point3<f64>, p1: &Point3<f64>, p2: &Point3<f64>, p3: &Point3<f64>) -> bool {
    // A quad is convex if all cross products have the same sign
    let v01 = p1 - p0;
    let v12 = p2 - p1;
    let v23 = p3 - p2;
    let v30 = p0 - p3;

    let n0 = v01.cross(&(-v30));
    let n1 = v12.cross(&(-v01));
    let n2 = v23.cross(&(-v12));
    let n3 = v30.cross(&(-v23));

    // Check if all normals point in the same general direction
    let d01 = n0.dot(&n1);
    let d12 = n1.dot(&n2);
    let d23 = n2.dot(&n3);

    d01 > 0.0 && d12 > 0.0 && d23 > 0.0
}

/// Flip an edge in the face list.
fn flip_edge(faces: &mut Vec<[usize; 3]>, v0: usize, v1: usize) -> bool {
    // Find the two faces sharing this edge
    let mut face_info: Vec<(usize, usize)> = Vec::new(); // (face_idx, edge_start_idx)

    for (idx, face) in faces.iter().enumerate() {
        for i in 0..3 {
            let a = face[i];
            let b = face[(i + 1) % 3];
            if (a == v0 && b == v1) || (a == v1 && b == v0) {
                face_info.push((idx, i));
                break; // Only count once per face
            }
        }
    }

    if face_info.len() != 2 {
        return false;
    }

    // Get the two faces and their opposite vertices
    let (idx0, edge_idx0) = face_info[0];
    let (idx1, edge_idx1) = face_info[1];

    let face0 = faces[idx0];
    let face1 = faces[idx1];

    let opp0 = face0[(edge_idx0 + 2) % 3]; // Opposite vertex in face 0
    let opp1 = face1[(edge_idx1 + 2) % 3]; // Opposite vertex in face 1

    // Determine the edge orientation in each face
    let a0 = face0[edge_idx0];

    // Create new faces maintaining consistent winding
    // If face0 has edge a0->b0 (which is either v0->v1 or v1->v0),
    // the new diagonal should connect opp0 to opp1
    if a0 == v0 {
        // face0: v0 -> v1 -> opp0 (CCW around the face)
        // face1: v1 -> v0 -> opp1 (CCW around the face)
        // After flip:
        // new face A: opp0 -> opp1 -> v0
        // new face B: opp1 -> opp0 -> v1
        faces[idx0] = [opp0, opp1, v0];
        faces[idx1] = [opp1, opp0, v1];
    } else {
        // face0: v1 -> v0 -> opp0
        // face1: v0 -> v1 -> opp1
        // After flip:
        // new face A: opp0 -> opp1 -> v1
        // new face B: opp1 -> opp0 -> v0
        faces[idx0] = [opp0, opp1, v1];
        faces[idx1] = [opp1, opp0, v0];
    }

    true
}

// ============================================================================
// Tangential Smoothing
// ============================================================================

/// Apply tangential smoothing to regularize vertex positions.
/// Works on face-vertex representation to avoid half-edge traversal issues.
fn tangential_smooth<I: MeshIndex>(mesh: &mut HalfEdgeMesh<I>, lambda: f64, preserve_boundary: bool) {
    let (vertices, faces) = to_face_vertex(mesh);

    // Compute vertex normals from face normals
    let normals = compute_vertex_normals_from_faces(&vertices, &faces);

    // Identify boundary vertices
    let boundary = compute_boundary_vertices(&faces, vertices.len());

    // Build adjacency from faces
    let neighbors = build_vertex_neighbors(&faces, vertices.len());

    // Compute new positions
    let mut new_positions: Vec<Point3<f64>> = Vec::with_capacity(vertices.len());

    for (idx, pos) in vertices.iter().enumerate() {
        if preserve_boundary && boundary[idx] {
            new_positions.push(*pos);
            continue;
        }

        let vertex_neighbors = &neighbors[idx];
        if vertex_neighbors.is_empty() {
            new_positions.push(*pos);
            continue;
        }

        // Compute centroid of neighbors (Laplacian)
        let mut centroid = Vector3::zeros();
        for &neighbor in vertex_neighbors {
            centroid += vertices[neighbor].coords;
        }
        centroid /= vertex_neighbors.len() as f64;

        // Compute displacement
        let displacement = centroid - pos.coords;

        // Project onto tangent plane (remove normal component)
        let normal = &normals[idx];
        let tangent_displacement = displacement - normal.dot(&displacement) * normal;

        // Apply smoothing
        new_positions.push(Point3::from(pos.coords + lambda * tangent_displacement));
    }

    // Rebuild mesh with new positions
    if let Ok(new_mesh) = build_from_triangles::<I>(&new_positions, &faces) {
        *mesh = new_mesh;
    }
}

/// Compute vertex normals from face data.
fn compute_vertex_normals_from_faces(vertices: &[Point3<f64>], faces: &[[usize; 3]]) -> Vec<Vector3<f64>> {
    let mut normals: Vec<Vector3<f64>> = vec![Vector3::zeros(); vertices.len()];

    for face in faces {
        let p0 = &vertices[face[0]];
        let p1 = &vertices[face[1]];
        let p2 = &vertices[face[2]];

        let e1 = p1 - p0;
        let e2 = p2 - p0;
        let face_normal = e1.cross(&e2); // Area-weighted

        // Add to each vertex
        normals[face[0]] += face_normal;
        normals[face[1]] += face_normal;
        normals[face[2]] += face_normal;
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

/// Compute which vertices are on the boundary.
fn compute_boundary_vertices(faces: &[[usize; 3]], num_vertices: usize) -> Vec<bool> {
    let mut boundary = vec![false; num_vertices];

    for v in 0..num_vertices {
        for &neighbor in &get_vertex_neighbors(faces, v) {
            if is_boundary_edge_in_faces(faces, v, neighbor) {
                boundary[v] = true;
                break;
            }
        }
    }

    boundary
}

/// Build adjacency list from faces.
fn build_vertex_neighbors(faces: &[[usize; 3]], num_vertices: usize) -> Vec<Vec<usize>> {
    let mut neighbors: Vec<HashSet<usize>> = vec![HashSet::new(); num_vertices];

    for face in faces {
        neighbors[face[0]].insert(face[1]);
        neighbors[face[0]].insert(face[2]);
        neighbors[face[1]].insert(face[0]);
        neighbors[face[1]].insert(face[2]);
        neighbors[face[2]].insert(face[0]);
        neighbors[face[2]].insert(face[1]);
    }

    neighbors.into_iter().map(|s| s.into_iter().collect()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn create_grid_mesh(n: usize) -> HalfEdgeMesh {
        let mut vertices = Vec::new();
        let mut faces = Vec::new();

        // Create grid vertices
        for j in 0..=n {
            for i in 0..=n {
                vertices.push(Point3::new(i as f64, j as f64, 0.0));
            }
        }

        // Create triangles
        for j in 0..n {
            for i in 0..n {
                let v00 = j * (n + 1) + i;
                let v10 = v00 + 1;
                let v01 = v00 + (n + 1);
                let v11 = v01 + 1;

                faces.push([v00, v10, v11]);
                faces.push([v00, v11, v01]);
            }
        }

        build_from_triangles(&vertices, &faces).unwrap()
    }

    #[test]
    fn test_average_edge_length() {
        let mesh = create_tetrahedron();
        let avg = average_edge_length(&mesh);
        assert!(avg > 0.0, "Average edge length should be positive");
        assert!(avg < 2.0, "Average edge length should be reasonable");
    }

    #[test]
    fn test_isotropic_remesh_preserves_topology() {
        let mut mesh = create_tetrahedron();
        let original_euler = mesh.num_vertices() as i32 - (mesh.num_halfedges() / 2) as i32 + mesh.num_faces() as i32;

        let options = RemeshOptions::with_target_length(0.5).with_iterations(2);
        isotropic_remesh(&mut mesh, &options);

        // Mesh should still be valid
        assert!(mesh.is_valid(), "Mesh should be valid after remeshing");

        // For a closed mesh, Euler characteristic should be preserved
        let new_euler = mesh.num_vertices() as i32 - (mesh.num_halfedges() / 2) as i32 + mesh.num_faces() as i32;
        assert_eq!(original_euler, new_euler, "Euler characteristic should be preserved");
    }

    #[test]
    fn test_isotropic_remesh_changes_edge_lengths() {
        let mut mesh = create_grid_mesh(3);
        let original_avg = average_edge_length(&mesh);

        // Target a different edge length
        let target = original_avg * 0.5;
        let options = RemeshOptions::with_target_length(target).with_iterations(3);
        isotropic_remesh(&mut mesh, &options);

        let new_avg = average_edge_length(&mesh);

        // New average should be closer to target
        let original_diff = (original_avg - target).abs();
        let new_diff = (new_avg - target).abs();

        assert!(
            new_diff < original_diff,
            "Edge lengths should move toward target: original_avg={}, new_avg={}, target={}",
            original_avg,
            new_avg,
            target
        );
    }

    #[test]
    fn test_zero_iterations_no_change() {
        let mut mesh = create_tetrahedron();
        let original_vertices: Vec<Point3<f64>> = mesh.vertex_ids().map(|v| *mesh.position(v)).collect();
        let original_face_count = mesh.num_faces();

        let options = RemeshOptions::with_target_length(0.5).with_iterations(0);
        isotropic_remesh(&mut mesh, &options);

        assert_eq!(mesh.num_faces(), original_face_count);
        for (vid, orig) in mesh.vertex_ids().zip(original_vertices.iter()) {
            assert_eq!(mesh.position(vid), orig);
        }
    }

    #[test]
    fn test_edge_split() {
        let mut vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
            Point3::new(1.0, 2.0, 0.0),
        ];
        let mut faces = vec![[0, 1, 2]];

        // Split edge 0-1
        split_edge(&mut vertices, &mut faces, 0, 1);

        assert_eq!(vertices.len(), 4);
        assert_eq!(faces.len(), 2);

        // Check midpoint
        let mid = &vertices[3];
        assert!((mid.x - 1.0).abs() < 1e-10);
        assert!((mid.y - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_cleanup_mesh() {
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0), // unused
            Point3::new(2.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
        ];
        let faces = vec![[0, 2, 3]]; // vertex 1 is not used

        let (clean_verts, clean_faces) = cleanup_mesh(&vertices, &faces);

        assert_eq!(clean_verts.len(), 3);
        assert_eq!(clean_faces.len(), 1);
    }

    #[test]
    fn test_edge_flip_two_triangles() {
        // Two triangles sharing an edge:
        //     2
        //    /|\
        //   / | \
        //  0--+--1  (edge 0-1 is shared)
        //   \ | /
        //    \|/
        //     3
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(1.0, -1.0, 0.0),
        ];
        // Face 0: 0->1->2, Face 1: 1->0->3
        let mut faces = vec![[0, 1, 2], [1, 0, 3]];

        // Should be able to flip edge 0-1 to edge 2-3
        let flipped = flip_edge(&mut faces, 0, 1);
        assert!(flipped, "Edge should be flippable");

        // After flip, faces should contain edge 2-3 instead of 0-1
        assert_eq!(faces.len(), 2);

        // Check that no edges are duplicated
        let mut seen_edges: HashSet<(usize, usize)> = HashSet::new();
        for face in &faces {
            for i in 0..3 {
                let v0 = face[i];
                let v1 = face[(i + 1) % 3];
                let edge = if v0 < v1 { (v0, v1) } else { (v1, v0) };
                seen_edges.insert(edge);
            }
        }

        // Should have 5 unique edges (original 5, minus 0-1, plus 2-3)
        assert_eq!(seen_edges.len(), 5);
        // Edge 0-1 should be gone
        assert!(!seen_edges.contains(&(0, 1)));
        // Edge 2-3 should be present
        assert!(seen_edges.contains(&(2, 3)));

        // Rebuild mesh should succeed
        let result = build_from_triangles::<u32>(&vertices, &faces);
        assert!(result.is_ok(), "Mesh should be buildable after flip");
        let mesh = result.unwrap();
        assert!(mesh.is_valid(), "Mesh should be valid after flip");
    }

    #[test]
    fn test_edge_flip_valence_check() {
        // Grid mesh has vertices with valence 6 (interior), which is optimal
        // So no edges should need flipping
        let mesh = create_grid_mesh(2);
        let (vertices, mut faces) = to_face_vertex(&mesh);

        // Try to flip edges
        flip_edges_for_valence_faces(&vertices, &mut faces, true);

        // For a well-structured grid, few or no flips should happen
        // At minimum, the mesh should still be valid
        let result = build_from_triangles::<u32>(&vertices, &faces);
        assert!(result.is_ok(), "Mesh should be buildable after flip attempts");
    }

    #[test]
    fn test_remesh_steps_individually() {
        // Test each remeshing step individually to verify correctness
        let mut mesh = create_tetrahedron();
        let target = 0.5;
        let high = target * 4.0 / 3.0;
        let low = target * 4.0 / 5.0;

        // Step 1: Split long edges
        split_long_edges(&mut mesh, high, true);
        assert!(mesh.is_valid(), "Mesh should be valid after split");

        // Step 2: Collapse short edges
        collapse_short_edges(&mut mesh, low, high, true);
        assert!(mesh.is_valid(), "Mesh should be valid after collapse");

        // Step 3: Flip edges
        flip_edges_to_improve_valence(&mut mesh, true);
        assert!(mesh.is_valid(), "Mesh should be valid after flip");

        // Step 4: Tangential smooth
        tangential_smooth(&mut mesh, 0.5, true);
        assert!(mesh.is_valid(), "Mesh should be valid after smooth");
    }

    // ========================================================================
    // Anisotropic Remeshing Tests
    // ========================================================================

    /// Create a mesh with varying curvature: a bumpy surface.
    fn create_bumpy_mesh() -> HalfEdgeMesh {
        let mut vertices = Vec::new();
        let mut faces = Vec::new();

        // Create a 5x5 grid with a bump in the center
        let n = 5;
        for j in 0..=n {
            for i in 0..=n {
                let x = i as f64;
                let y = j as f64;
                // Gaussian bump in the center
                let cx = n as f64 / 2.0;
                let cy = n as f64 / 2.0;
                let dist2 = (x - cx).powi(2) + (y - cy).powi(2);
                let z = 2.0 * (-dist2 / 2.0).exp(); // Bump height
                vertices.push(Point3::new(x, y, z));
            }
        }

        // Create triangles
        for j in 0..n {
            for i in 0..n {
                let v00 = j * (n + 1) + i;
                let v10 = v00 + 1;
                let v01 = v00 + (n + 1);
                let v11 = v01 + 1;

                faces.push([v00, v10, v11]);
                faces.push([v00, v11, v01]);
            }
        }

        build_from_triangles(&vertices, &faces).unwrap()
    }

    #[test]
    fn test_compute_sizing_field() {
        let mesh = create_bumpy_mesh();
        let options = AnisotropicOptions::new(0.3, 1.5);

        let sizing = compute_sizing_field(&mesh, &options);

        // All sizes should be within bounds
        for &size in &sizing.vertex_sizes {
            assert!(
                size >= options.min_length && size <= options.max_length,
                "Sizing {} should be in [{}, {}]",
                size,
                options.min_length,
                options.max_length
            );
        }

        // The bump center should have higher curvature, thus smaller sizes
        // Center vertex is at index (5/2) * 6 + 5/2 = 2*6 + 2 = 14
        let center_idx = 14;
        let corner_idx = 0; // Corner should be flatter

        // We expect center to have smaller (or equal) size than corner
        // Due to curvature normalization, this should hold
        assert!(
            sizing.vertex_sizes[center_idx] <= sizing.vertex_sizes[corner_idx] + 0.5,
            "Center ({}) should have smaller or similar size than corner ({})",
            sizing.vertex_sizes[center_idx],
            sizing.vertex_sizes[corner_idx]
        );
    }

    #[test]
    fn test_anisotropic_remesh_preserves_validity() {
        // Use the simpler tetrahedron for speed
        let mut mesh = create_tetrahedron();

        let options = AnisotropicOptions::new(0.3, 1.0).with_iterations(1);
        anisotropic_remesh(&mut mesh, &options);

        assert!(mesh.is_valid(), "Mesh should be valid after anisotropic remeshing");
        assert!(mesh.num_faces() > 0, "Mesh should have faces");
        assert!(mesh.num_vertices() > 0, "Mesh should have vertices");
    }

    #[test]
    fn test_anisotropic_remesh_preserves_euler() {
        let mut mesh = create_tetrahedron();
        let original_euler =
            mesh.num_vertices() as i32 - (mesh.num_halfedges() / 2) as i32 + mesh.num_faces() as i32;

        let options = AnisotropicOptions::new(0.3, 0.8).with_iterations(2);
        anisotropic_remesh(&mut mesh, &options);

        let new_euler =
            mesh.num_vertices() as i32 - (mesh.num_halfedges() / 2) as i32 + mesh.num_faces() as i32;

        assert_eq!(
            original_euler, new_euler,
            "Euler characteristic should be preserved"
        );
    }

    #[test]
    fn test_anisotropic_zero_iterations_no_change() {
        let mut mesh = create_tetrahedron();
        let original_vertices: Vec<Point3<f64>> =
            mesh.vertex_ids().map(|v| *mesh.position(v)).collect();
        let original_face_count = mesh.num_faces();

        let options = AnisotropicOptions::new(0.3, 1.0).with_iterations(0);
        anisotropic_remesh(&mut mesh, &options);

        assert_eq!(mesh.num_faces(), original_face_count);
        for (vid, orig) in mesh.vertex_ids().zip(original_vertices.iter()) {
            assert_eq!(mesh.position(vid), orig);
        }
    }

    #[test]
    fn test_anisotropic_adaptation_strength() {
        let mesh = create_bumpy_mesh();

        // Full adaptation
        let options_full = AnisotropicOptions::new(0.3, 1.5).with_adaptation(1.0);
        let sizing_full = compute_sizing_field(&mesh, &options_full);

        // No adaptation (uniform) - all sizes become max_length
        let options_none = AnisotropicOptions::new(0.3, 1.5).with_adaptation(0.0);
        let sizing_none = compute_sizing_field(&mesh, &options_none);

        // With no adaptation, all sizes should be at max_length
        // (curvature contribution is zeroed out, so t=0 → max_length)
        for &size in &sizing_none.vertex_sizes {
            assert!(
                (size - 1.5).abs() < 0.01,
                "With zero adaptation, size {} should equal max_length 1.5",
                size
            );
        }

        // With full adaptation, there should be size variation
        let min_full = sizing_full.vertex_sizes.iter().copied().fold(f64::INFINITY, f64::min);
        let max_full = sizing_full.vertex_sizes.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let range_full = max_full - min_full;

        assert!(
            range_full > 0.1,
            "Full adaptation should produce size variation, got range {}",
            range_full
        );
    }

    #[test]
    fn test_cotangent_weight_computation() {
        // Right angle triangle: cot(90°) = 0
        let e1 = Vector3::new(1.0, 0.0, 0.0);
        let e2 = Vector3::new(0.0, 1.0, 0.0);
        let cot = cotangent_weight(&e1, &e2);
        assert!(cot.abs() < 0.01, "cot(90°) should be ~0, got {}", cot);

        // 45 degree angle: cot(45°) = 1
        let e1 = Vector3::new(1.0, 0.0, 0.0);
        let e2 = Vector3::new(-1.0, 1.0, 0.0); // 45 degrees from e1
        let cot = cotangent_weight(&e1, &e2);
        assert!((cot - 1.0).abs() < 0.01, "cot(45°) should be ~1, got {}", cot);
    }
}
