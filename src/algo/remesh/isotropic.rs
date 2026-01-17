//! Isotropic remeshing algorithm.

use std::collections::HashSet;

use nalgebra::Point3;

use crate::algo::Progress;
use crate::mesh::{build_from_triangles, to_face_vertex, HalfEdgeMesh, MeshIndex};

use super::{
    cleanup_mesh, collapse_edge, flip_edges_for_valence_faces, get_vertex_neighbors,
    is_boundary_edge_in_faces, is_boundary_vertex_in_faces, split_edge, tangential_smooth,
    validate_face_list,
};

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
pub fn isotropic_remesh<I: MeshIndex>(mesh: &mut HalfEdgeMesh<I>, options: &RemeshOptions) {
    isotropic_remesh_internal(mesh, options, None);
}

/// Performs isotropic remeshing with progress reporting.
///
/// See [`isotropic_remesh`] for algorithm details.
pub fn isotropic_remesh_with_progress<I: MeshIndex>(
    mesh: &mut HalfEdgeMesh<I>,
    options: &RemeshOptions,
    progress: &Progress,
) {
    isotropic_remesh_internal(mesh, options, Some(progress));
}

fn isotropic_remesh_internal<I: MeshIndex>(
    mesh: &mut HalfEdgeMesh<I>,
    options: &RemeshOptions,
    progress: Option<&Progress>,
) {
    if options.iterations == 0 || options.target_length <= 0.0 {
        return;
    }

    let high = options.target_length * 4.0 / 3.0;
    let low = options.target_length * 4.0 / 5.0;

    // 4 sub-steps per iteration for more granular progress
    let total_steps = options.iterations * 4;

    for iter in 0..options.iterations {
        let base_step = iter * 4;

        // Step 1: Split long edges (with sub-progress)
        split_long_edges_with_progress(mesh, high, options.preserve_boundary, progress, base_step, total_steps);

        // Step 2: Collapse short edges (with sub-progress)
        collapse_short_edges_with_progress(mesh, low, high, options.preserve_boundary, progress, base_step + 1, total_steps);

        // Step 3: Flip edges to improve valence
        if let Some(p) = progress {
            p.report(base_step + 2, total_steps, "Flipping edges");
        }
        flip_edges_to_improve_valence(mesh, options.preserve_boundary);

        // Step 4: Tangential smoothing
        if let Some(p) = progress {
            p.report(base_step + 3, total_steps, "Smoothing");
        }
        for _ in 0..options.smoothing_iterations {
            tangential_smooth(mesh, options.smoothing_lambda, options.preserve_boundary);
        }
    }

    // Report completion
    if let Some(p) = progress {
        p.report(total_steps, total_steps, "Isotropic remeshing complete");
    }
}

/// Compute the average edge length of a mesh.
///
/// This is useful for determining an appropriate target edge length
/// for remeshing.
pub fn average_edge_length<I: MeshIndex>(mesh: &HalfEdgeMesh<I>) -> f64 {
    let mut total_length = 0.0;
    let mut edge_count = 0;

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

/// Split all edges longer than the threshold (with progress reporting).
fn split_long_edges_with_progress<I: MeshIndex>(
    mesh: &mut HalfEdgeMesh<I>,
    threshold: f64,
    preserve_boundary: bool,
    progress: Option<&Progress>,
    step: usize,
    total_steps: usize,
) {
    let (mut vertices, mut faces) = to_face_vertex(mesh);

    // First pass: count how many edges need splitting to estimate total work
    let initial_long_edges = count_long_edges(&vertices, &faces, threshold, preserve_boundary);
    let mut total_splits = 0usize;
    let estimated_total = initial_long_edges.max(1);

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

        for (v0, v1) in &edges_to_split {
            split_edge(&mut vertices, &mut faces, *v0, *v1);
            total_splits += 1;
            changed = true;

            // Report progress periodically (every 50 splits or so)
            if let Some(p) = progress {
                if total_splits % 50 == 0 || total_splits == estimated_total {
                    // Use estimated_total * 2 as denominator since splitting can create new long edges
                    let effective_total = (estimated_total * 2).max(total_splits);
                    p.report_sub(total_splits, effective_total, step, total_steps, "Splitting edges");
                }
            }
        }
    }

    // Final progress update for this step
    if let Some(p) = progress {
        p.report_sub(1, 1, step, total_steps, "Splitting edges");
    }

    if let Ok(new_mesh) = build_from_triangles::<I>(&vertices, &faces) {
        *mesh = new_mesh;
    }
}

/// Count edges longer than threshold (for progress estimation).
fn count_long_edges(
    vertices: &[Point3<f64>],
    faces: &[[usize; 3]],
    threshold: f64,
    preserve_boundary: bool,
) -> usize {
    let mut count = 0;
    let mut seen_edges: HashSet<(usize, usize)> = HashSet::new();

    for face in faces {
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
                if preserve_boundary && is_boundary_edge_in_faces(faces, v0, v1) {
                    continue;
                }
                count += 1;
            }
        }
    }
    count
}

/// Collapse all edges shorter than the threshold (with progress reporting).
fn collapse_short_edges_with_progress<I: MeshIndex>(
    mesh: &mut HalfEdgeMesh<I>,
    low_threshold: f64,
    high_threshold: f64,
    preserve_boundary: bool,
    progress: Option<&Progress>,
    step: usize,
    total_steps: usize,
) {
    let (mut vertices, mut faces) = to_face_vertex(mesh);

    // Count short edges for progress estimation
    let initial_short_edges = count_short_edges(&vertices, &faces, low_threshold, high_threshold, preserve_boundary);
    let estimated_total = initial_short_edges.max(1);
    let mut total_collapses = 0usize;

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

                if length < low_threshold {
                    if can_collapse_edge(
                        &vertices,
                        &faces,
                        v0,
                        v1,
                        high_threshold,
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
            collapse_edge(&mut vertices, &mut faces, v0, v1);
            total_collapses += 1;
            changed = true;

            // Report progress periodically
            if let Some(p) = progress {
                if total_collapses % 20 == 0 {
                    p.report_sub(total_collapses, estimated_total, step, total_steps, "Collapsing edges");
                }
            }
        }
    }

    // Final progress update for this step
    if let Some(p) = progress {
        p.report_sub(1, 1, step, total_steps, "Collapsing edges");
    }

    let (clean_vertices, clean_faces) = cleanup_mesh(&vertices, &faces);

    if !clean_faces.is_empty() {
        if let Ok(new_mesh) = build_from_triangles::<I>(&clean_vertices, &clean_faces) {
            *mesh = new_mesh;
        }
    }
}

/// Count edges shorter than threshold (for progress estimation).
fn count_short_edges(
    vertices: &[Point3<f64>],
    faces: &[[usize; 3]],
    low_threshold: f64,
    high_threshold: f64,
    preserve_boundary: bool,
) -> usize {
    let mut count = 0;
    let mut seen_edges: HashSet<(usize, usize)> = HashSet::new();

    for face in faces {
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
                if can_collapse_edge(vertices, faces, v0, v1, high_threshold, preserve_boundary) {
                    count += 1;
                }
            }
        }
    }
    count
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
    if preserve_boundary && is_boundary_edge_in_faces(faces, v0, v1) {
        return false;
    }

    if preserve_boundary {
        let v0_boundary = is_boundary_vertex_in_faces(faces, v0);
        let v1_boundary = is_boundary_vertex_in_faces(faces, v1);
        if v0_boundary && v1_boundary && !is_boundary_edge_in_faces(faces, v0, v1) {
            return false;
        }
    }

    let midpoint = (vertices[v0].coords + vertices[v1].coords) * 0.5;

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

    let neighbors_v0 = get_vertex_neighbors(faces, v0);
    let neighbors_v1 = get_vertex_neighbors(faces, v1);
    let common: HashSet<_> = neighbors_v0.intersection(&neighbors_v1).collect();

    if !is_boundary_edge_in_faces(faces, v0, v1) && common.len() != 2 {
        return false;
    }

    true
}

/// Flip edges to improve vertex valence.
fn flip_edges_to_improve_valence<I: MeshIndex>(mesh: &mut HalfEdgeMesh<I>, preserve_boundary: bool) {
    let (vertices, mut faces) = to_face_vertex(mesh);

    flip_edges_for_valence_faces(&vertices, &mut faces, preserve_boundary);

    if !validate_face_list(&vertices, &faces) {
        return;
    }

    if let Ok(new_mesh) = build_from_triangles::<I>(&vertices, &faces) {
        *mesh = new_mesh;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algo::remesh::tests::{create_grid_mesh, create_tetrahedron};

    #[test]
    fn test_average_edge_length() {
        let mesh = create_tetrahedron();
        let avg = average_edge_length(&mesh);
        assert!(avg > 0.0);
        assert!(avg < 2.0);
    }

    #[test]
    fn test_isotropic_remesh_preserves_topology() {
        let mut mesh = create_tetrahedron();
        let original_euler =
            mesh.num_vertices() as i32 - (mesh.num_halfedges() / 2) as i32 + mesh.num_faces() as i32;

        let options = RemeshOptions::with_target_length(0.5).with_iterations(2);
        isotropic_remesh(&mut mesh, &options);

        assert!(mesh.is_valid());

        let new_euler =
            mesh.num_vertices() as i32 - (mesh.num_halfedges() / 2) as i32 + mesh.num_faces() as i32;
        assert_eq!(original_euler, new_euler);
    }

    #[test]
    fn test_isotropic_remesh_changes_edge_lengths() {
        let mut mesh = create_grid_mesh(3);
        let original_avg = average_edge_length(&mesh);

        let target = original_avg * 0.5;
        let options = RemeshOptions::with_target_length(target).with_iterations(3);
        isotropic_remesh(&mut mesh, &options);

        let new_avg = average_edge_length(&mesh);

        let original_diff = (original_avg - target).abs();
        let new_diff = (new_avg - target).abs();

        assert!(new_diff < original_diff);
    }

    #[test]
    fn test_zero_iterations_no_change() {
        let mut mesh = create_tetrahedron();
        let original_vertices: Vec<Point3<f64>> =
            mesh.vertex_ids().map(|v| *mesh.position(v)).collect();
        let original_face_count = mesh.num_faces();

        let options = RemeshOptions::with_target_length(0.5).with_iterations(0);
        isotropic_remesh(&mut mesh, &options);

        assert_eq!(mesh.num_faces(), original_face_count);
        for (vid, orig) in mesh.vertex_ids().zip(original_vertices.iter()) {
            assert_eq!(mesh.position(vid), orig);
        }
    }

    #[test]
    fn test_remesh_steps_individually() {
        let mut mesh = create_tetrahedron();
        let target = 0.5;
        let high = target * 4.0 / 3.0;
        let low = target * 4.0 / 5.0;

        split_long_edges_with_progress(&mut mesh, high, true, None, 0, 4);
        assert!(mesh.is_valid());

        collapse_short_edges_with_progress(&mut mesh, low, high, true, None, 1, 4);
        assert!(mesh.is_valid());

        flip_edges_to_improve_valence(&mut mesh, true);
        assert!(mesh.is_valid());

        tangential_smooth(&mut mesh, 0.5, true);
        assert!(mesh.is_valid());
    }
}
