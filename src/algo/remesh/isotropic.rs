//! Isotropic remeshing algorithm.

use std::collections::HashSet;

use nalgebra::Point3;

use crate::algo::Progress;
use crate::mesh::{build_from_triangles, to_face_vertex, HalfEdgeMesh, MeshIndex};

use super::{
    cleanup_mesh, collapse_edge, flip_edges_for_valence_faces, get_vertex_neighbors,
    is_boundary_edge_in_faces, is_boundary_vertex_in_faces, tangential_smooth,
    validate_face_list, MeshTopology,
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

    /// Whether to use parallel execution (default: true).
    pub parallel: bool,
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
            parallel: true,
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

        #[cfg(debug_assertions)]
        {
            if !mesh.is_valid() {
                eprintln!("WARNING: Mesh invalid after collapse!");
            }
        }

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
            tangential_smooth(mesh, options.smoothing_lambda, options.preserve_boundary, options.parallel);
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
///
/// Uses batch processing: collect all long edges, split them simultaneously.
fn split_long_edges_with_progress<I: MeshIndex>(
    mesh: &mut HalfEdgeMesh<I>,
    threshold: f64,
    preserve_boundary: bool,
    progress: Option<&Progress>,
    step: usize,
    total_steps: usize,
) {
    let (mut vertices, mut faces) = to_face_vertex(mesh);
    let threshold_sq = threshold * threshold;

    // Build initial boundary edge set (boundary edges are preserved)
    let mut boundary_edges: HashSet<(usize, usize)> = HashSet::new();
    if preserve_boundary {
        let mut edge_count: std::collections::HashMap<(usize, usize), usize> =
            std::collections::HashMap::new();
        for face in &faces {
            for i in 0..3 {
                let v0 = face[i];
                let v1 = face[(i + 1) % 3];
                let edge = if v0 < v1 { (v0, v1) } else { (v1, v0) };
                *edge_count.entry(edge).or_insert(0) += 1;
            }
        }
        for (edge, count) in edge_count {
            if count == 1 {
                boundary_edges.insert(edge);
            }
        }
    }

    // Limit iterations to prevent infinite loops on degenerate geometry
    let max_iterations = 20;

    for iteration in 0..max_iterations {
        // Collect all long edges with their face indices
        let mut edge_to_faces: std::collections::HashMap<(usize, usize), Vec<usize>> =
            std::collections::HashMap::new();
        for (fi, face) in faces.iter().enumerate() {
            for i in 0..3 {
                let v0 = face[i];
                let v1 = face[(i + 1) % 3];
                let edge = if v0 < v1 { (v0, v1) } else { (v1, v0) };
                edge_to_faces.entry(edge).or_default().push(fi);
            }
        }

        // Find long edges (excluding boundary if preserving)
        let mut long_edges: Vec<((usize, usize), Vec<usize>)> = edge_to_faces
            .into_iter()
            .filter(|((v0, v1), _)| {
                let dx = vertices[*v1].x - vertices[*v0].x;
                let dy = vertices[*v1].y - vertices[*v0].y;
                let dz = vertices[*v1].z - vertices[*v0].z;
                let len_sq = dx * dx + dy * dy + dz * dz;
                if len_sq <= threshold_sq {
                    return false;
                }
                if preserve_boundary && boundary_edges.contains(&(*v0, *v1)) {
                    return false;
                }
                true
            })
            .collect();

        if long_edges.is_empty() {
            break;
        }

        // Report progress
        if let Some(p) = progress {
            p.report_sub(iteration + 1, max_iterations, step, total_steps, "Splitting edges");
        }

        // Sort by length descending
        long_edges.sort_by(|a, b| {
            let (v0a, v1a) = a.0;
            let (v0b, v1b) = b.0;
            let len_a = (&vertices[v1a] - &vertices[v0a]).norm_squared();
            let len_b = (&vertices[v1b] - &vertices[v0b]).norm_squared();
            len_b.partial_cmp(&len_a).unwrap()
        });

        // Create a map from edge to midpoint vertex index
        let mut edge_midpoints: std::collections::HashMap<(usize, usize), usize> =
            std::collections::HashMap::new();

        // Pre-allocate midpoints for all long edges
        for ((v0, v1), _) in &long_edges {
            let mid = Point3::from((vertices[*v0].coords + vertices[*v1].coords) * 0.5);
            let mid_idx = vertices.len();
            vertices.push(mid);
            edge_midpoints.insert((*v0, *v1), mid_idx);
        }

        // Now split ALL faces that contain long edges
        // Each face may be split multiple times if it has multiple long edges
        let mut new_faces: Vec<[usize; 3]> = Vec::new();
        let faces_to_process: HashSet<usize> = long_edges
            .iter()
            .flat_map(|(_, fis)| fis.iter().copied())
            .collect();

        for fi in 0..faces.len() {
            if !faces_to_process.contains(&fi) {
                new_faces.push(faces[fi]);
                continue;
            }

            // This face has at least one long edge - need to subdivide
            let face = faces[fi];
            let v0 = face[0];
            let v1 = face[1];
            let v2 = face[2];

            // Check which edges have midpoints
            let e01 = if v0 < v1 { (v0, v1) } else { (v1, v0) };
            let e12 = if v1 < v2 { (v1, v2) } else { (v2, v1) };
            let e20 = if v2 < v0 { (v2, v0) } else { (v0, v2) };

            let m01 = edge_midpoints.get(&e01).copied();
            let m12 = edge_midpoints.get(&e12).copied();
            let m20 = edge_midpoints.get(&e20).copied();

            match (m01, m12, m20) {
                (None, None, None) => {
                    // No splits needed (shouldn't happen)
                    new_faces.push(face);
                }
                (Some(m), None, None) => {
                    // Split only edge 0-1
                    new_faces.push([v0, m, v2]);
                    new_faces.push([m, v1, v2]);
                }
                (None, Some(m), None) => {
                    // Split only edge 1-2
                    new_faces.push([v0, v1, m]);
                    new_faces.push([v0, m, v2]);
                }
                (None, None, Some(m)) => {
                    // Split only edge 2-0
                    new_faces.push([v0, v1, m]);
                    new_faces.push([m, v1, v2]);
                }
                (Some(m01), Some(m12), None) => {
                    // Split edges 0-1 and 1-2
                    new_faces.push([v0, m01, v2]);
                    new_faces.push([m01, v1, m12]);
                    new_faces.push([m01, m12, v2]);
                }
                (None, Some(m12), Some(m20)) => {
                    // Split edges 1-2 and 2-0
                    new_faces.push([v0, v1, m12]);
                    new_faces.push([v0, m12, m20]);
                    new_faces.push([m12, v2, m20]);
                }
                (Some(m01), None, Some(m20)) => {
                    // Split edges 0-1 and 2-0
                    new_faces.push([v0, m01, m20]);
                    new_faces.push([m01, v1, v2]);
                    new_faces.push([m01, v2, m20]);
                }
                (Some(m01), Some(m12), Some(m20)) => {
                    // Split all three edges - creates 4 triangles
                    new_faces.push([v0, m01, m20]);
                    new_faces.push([m01, v1, m12]);
                    new_faces.push([m20, m12, v2]);
                    new_faces.push([m01, m12, m20]);
                }
            }
        }

        faces = new_faces;
    }

    // Final progress update
    if let Some(p) = progress {
        p.report_sub(1, 1, step, total_steps, "Splitting edges");
    }

    #[cfg(debug_assertions)]
    eprintln!("Split phase done: {} faces, {} vertices. Building mesh...", faces.len(), vertices.len());

    if let Ok(new_mesh) = build_from_triangles::<I>(&vertices, &faces) {
        *mesh = new_mesh;
    }

    #[cfg(debug_assertions)]
    eprintln!("Split phase: mesh built");
}


/// Collapse all edges shorter than the threshold (with progress reporting).
///
/// Uses batch processing: finds all collapsible edges, selects independent ones
/// (no shared vertices), and collapses them all at once before rebuilding topology.
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

    let max_iterations = 30; // Batches, not individual collapses
    let mut _total_collapses = 0usize;

    for iteration in 0..max_iterations {
        #[cfg(debug_assertions)]
        eprintln!("Collapse iter {}: {} faces, {} vertices", iteration, faces.len(), vertices.len());

        // Build topology once per batch
        let topology = MeshTopology::from_faces(&faces, vertices.len());

        // Find ALL collapsible short edges
        let mut candidate_edges: Vec<(usize, usize, f64)> = Vec::new();

        for &(v0, v1) in topology.edge_faces.keys() {
            let length = (&vertices[v1] - &vertices[v0]).norm();

            if length < low_threshold {
                if can_collapse_edge_fast(
                    &vertices,
                    &topology,
                    v0,
                    v1,
                    high_threshold,
                    preserve_boundary,
                ) {
                    candidate_edges.push((v0, v1, length));
                }
            }
        }

        if candidate_edges.is_empty() {
            break;
        }

        // Sort by length (shortest first) to prioritize removing very short edges
        candidate_edges.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

        // Select independent edges (just check edge vertices, not neighbors)
        let mut used_vertices: HashSet<usize> = HashSet::new();
        let mut edges_to_collapse: Vec<(usize, usize)> = Vec::new();

        for (v0, v1, _len) in candidate_edges {
            if !used_vertices.contains(&v0) && !used_vertices.contains(&v1) {
                edges_to_collapse.push((v0, v1));
                used_vertices.insert(v0);
                used_vertices.insert(v1);
            }
        }

        if edges_to_collapse.is_empty() {
            break;
        }

        // Batch collapse all selected edges
        for (v0, v1) in &edges_to_collapse {
            collapse_edge(&mut vertices, &mut faces, *v0, *v1);
        }
        _total_collapses += edges_to_collapse.len();

        // Report progress
        if let Some(p) = progress {
            p.report_sub(iteration + 1, max_iterations, step, total_steps, "Collapsing edges");
        }
    }

    // Final progress update for this step
    if let Some(p) = progress {
        p.report_sub(1, 1, step, total_steps, "Collapsing edges");
    }

    let (clean_vertices, clean_faces) = cleanup_mesh(&vertices, &faces);

    #[cfg(debug_assertions)]
    {
        eprintln!("Collapse done. Building mesh from {} faces, {} vertices", clean_faces.len(), clean_vertices.len());
        // Check for degenerate faces
        let mut degenerate = 0;
        for face in &clean_faces {
            if face[0] == face[1] || face[1] == face[2] || face[0] == face[2] {
                degenerate += 1;
            }
        }
        if degenerate > 0 {
            eprintln!("WARNING: {} degenerate faces!", degenerate);
        }

        // Check that face list is valid (manifold)
        if !validate_face_list(&clean_vertices, &clean_faces) {
            eprintln!("WARNING: Face list failed validation after collapse!");
        }
    }

    if !clean_faces.is_empty() {
        match build_from_triangles::<I>(&clean_vertices, &clean_faces) {
            Ok(new_mesh) => {
                #[cfg(debug_assertions)]
                eprintln!("Collapse: mesh built successfully with {} halfedges", new_mesh.num_halfedges());
                *mesh = new_mesh;
            }
            Err(_e) => {
                #[cfg(debug_assertions)]
                eprintln!("Collapse: ERROR building mesh: {:?}", _e);
            }
        }
    }
}


/// Check if an edge can be safely collapsed (uses O(n) scans - for reference/testing).
#[allow(dead_code)]
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

/// Check if an edge can be safely collapsed (O(1) using pre-computed topology).
fn can_collapse_edge_fast(
    vertices: &[Point3<f64>],
    topology: &MeshTopology,
    v0: usize,
    v1: usize,
    high_threshold: f64,
    preserve_boundary: bool,
) -> bool {
    let is_boundary = topology.is_boundary_edge(v0, v1);

    if preserve_boundary && is_boundary {
        return false;
    }

    if preserve_boundary {
        let v0_boundary = topology.is_boundary_vertex(v0);
        let v1_boundary = topology.is_boundary_vertex(v1);
        // Don't collapse an interior edge between two boundary vertices
        if v0_boundary && v1_boundary && !is_boundary {
            return false;
        }
    }

    let midpoint = (vertices[v0].coords + vertices[v1].coords) * 0.5;

    // Check that collapsing won't create edges longer than high_threshold
    let neighbors_v0 = topology.neighbors(v0);
    let neighbors_v1 = topology.neighbors(v1);

    for &neighbor in neighbors_v0.iter().chain(neighbors_v1.iter()) {
        if neighbor == v0 || neighbor == v1 {
            continue;
        }
        let new_length = (vertices[neighbor].coords - midpoint).norm();
        if new_length > high_threshold {
            return false;
        }
    }

    // Check link condition: for interior edges, exactly 2 common neighbors
    let common_count = neighbors_v0.intersection(neighbors_v1).count();

    if !is_boundary && common_count != 2 {
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

        tangential_smooth(&mut mesh, 0.5, true, true);
        assert!(mesh.is_valid());
    }
}
