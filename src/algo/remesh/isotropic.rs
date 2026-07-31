//! Isotropic remeshing algorithm.

use std::collections::HashSet;

use nalgebra::Point3;

use crate::algo::remesh::RemeshReport;
use crate::algo::Progress;
use crate::mesh::{build_from_triangles, to_face_vertex, HalfEdgeMesh, MeshIndex};

use super::{
    cleanup_mesh, collapse_edge, flip_edges_for_valence_faces, get_vertex_neighbors,
    is_boundary_edge_in_faces, is_boundary_vertex_in_faces, tangential_smooth, validate_face_list,
    MeshTopology,
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
pub fn isotropic_remesh<I: MeshIndex>(
    mesh: &mut HalfEdgeMesh<I>,
    options: &RemeshOptions,
) -> RemeshReport {
    isotropic_remesh_internal(mesh, options, None)
}

/// Performs isotropic remeshing with progress reporting.
///
/// See [`isotropic_remesh`] for algorithm details.
pub fn isotropic_remesh_with_progress<I: MeshIndex>(
    mesh: &mut HalfEdgeMesh<I>,
    options: &RemeshOptions,
    progress: &Progress,
) -> RemeshReport {
    isotropic_remesh_internal(mesh, options, Some(progress))
}

fn isotropic_remesh_internal<I: MeshIndex>(
    mesh: &mut HalfEdgeMesh<I>,
    options: &RemeshOptions,
    progress: Option<&Progress>,
) -> RemeshReport {
    let faces_before = mesh.num_faces();
    if options.iterations == 0 || options.target_length <= 0.0 {
        return RemeshReport {
            iterations_run: 0,
            converged: false,
            faces_before,
            faces_after: faces_before,
        };
    }

    let high = options.target_length * 4.0 / 3.0;
    let low = options.target_length * 4.0 / 5.0;

    // 4 sub-steps per iteration for more granular progress
    let total_steps = options.iterations * 4;

    let mut converged = true;
    let mut iterations_run = 0;
    for iter in 0..options.iterations {
        let base_step = iter * 4;
        iterations_run += 1;

        // Step 1: Split long edges (with sub-progress)
        converged &= split_long_edges_with_progress(
            mesh,
            high,
            options.preserve_boundary,
            progress,
            base_step,
            total_steps,
        );

        // Step 2: Collapse short edges (with sub-progress)
        converged &= collapse_short_edges_with_progress(
            mesh,
            low,
            high,
            options.preserve_boundary,
            progress,
            base_step + 1,
            total_steps,
        );

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
        converged &= flip_edges_to_improve_valence(mesh, options.preserve_boundary);

        // Step 4: Tangential smoothing
        if let Some(p) = progress {
            p.report(base_step + 3, total_steps, "Smoothing");
        }
        for _ in 0..options.smoothing_iterations {
            converged &= tangential_smooth(
                mesh,
                options.smoothing_lambda,
                options.preserve_boundary,
                options.parallel,
            );
        }

        // Deliberately *not* stopping here. A rejected rebuild means this pass left the
        // mesh alone; the mesh is still valid and later passes can still improve it.
        // Breaking out measurably hurt: on the torus, stopping at the first rejected pass
        // took the worst angle to 24.4° where running all five reaches 29.6°. Anisotropic
        // does break, because its non-convergence is a pass bound being hit — evidence
        // that iteration is diverging, not that one step was unlucky.
    }

    // Report completion
    if let Some(p) = progress {
        p.report(total_steps, total_steps, "Isotropic remeshing complete");
    }

    RemeshReport {
        iterations_run,
        converged,
        faces_before,
        faces_after: mesh.num_faces(),
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

/// Smallest interior angle a split or collapse is allowed to create, in radians (1°).
///
/// A floor rather than a ratio to the parent's angle, because a ratio still permits
/// unbounded decay over twenty passes — halving something twenty times is what produced
/// the 5.5e-8° triangle in the first place. The cost is that a mesh with features
/// genuinely thinner than 1° will not be refined or coarsened across them, which is a
/// better failure than manufacturing slivers: the input's own worst triangle is preserved
/// rather than made worse.
const MIN_ANGLE_FLOOR: f64 = 0.017_453_292_519_943_295; // 1° in radians

/// Smallest interior angle of the triangle `p q r`, in radians.
///
/// `atan2` of the cross and dot products rather than `acos` of a normalised dot, which
/// loses all precision on exactly the thin triangles this exists to detect.
fn min_angle_of(p: &Point3<f64>, q: &Point3<f64>, r: &Point3<f64>) -> f64 {
    let at =
        |u: nalgebra::Vector3<f64>, v: nalgebra::Vector3<f64>| u.cross(&v).norm().atan2(u.dot(&v));
    at(q - p, r - p).min(at(r - q, p - q)).min(at(p - r, q - r))
}

/// Split all edges longer than the threshold (with progress reporting).
///
/// Uses batch processing: collect all long edges, split them simultaneously.
/// Returns whether the pass ran to completion; `false` means the mesh it produced was
/// rejected by [`build_from_triangles`], so the mesh is left as it was.
fn split_long_edges_with_progress<I: MeshIndex>(
    mesh: &mut HalfEdgeMesh<I>,
    threshold: f64,
    preserve_boundary: bool,
    progress: Option<&Progress>,
    step: usize,
    total_steps: usize,
) -> bool {
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
            .iter()
            .filter(|((v0, v1), fis)| {
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
                // Prefer the longest edge of a face, and never split into a sliver.
                //
                // Splitting an edge halves it, but what that does to the *shape* of the
                // faces around it depends on which edge it is. For a thin triangle whose
                // apex `C` sits near the line `AB`, splitting the base `AB` halves the base
                // and doubles the minimum angle; splitting a side `AC` halves the
                // triangle's *height* and halves the angle. Both sides of a sliver are
                // long, so both were being split, and this loop runs up to 20 passes: the
                // Stanford bunny's worst triangle fell from 1.50° to 5.5e-8°, which put
                // curvature values of 6e7 on the vertices around it.
                //
                // Preferring longest edges (Rivara's longest-edge bisection) is most of the
                // answer, but not all of it: an edge can be the longest edge of one of its
                // faces and a *side* of the other, and splitting it still flattens that
                // other face. Rivara handles this by recursively splitting the neighbour's
                // longest edge first; the cheaper guard used here is to look at what the
                // split would actually produce and decline if any resulting triangle would
                // be a sliver.
                let is_longest_somewhere = fis.iter().any(|&fi| {
                    let f = faces[fi];
                    let side =
                        |a: usize, b: usize| (vertices[f[b]] - vertices[f[a]]).norm_squared();
                    let longest = side(0, 1).max(side(1, 2)).max(side(2, 0));
                    // A relative tolerance, so an equilateral triangle splits rather than
                    // deadlocking on exact ties.
                    len_sq >= longest * (1.0 - 1e-9)
                });
                if !is_longest_somewhere {
                    return false;
                }
                let midpoint = Point3::from((vertices[*v0].coords + vertices[*v1].coords) * 0.5);
                fis.iter().all(|&fi| {
                    let f = faces[fi];
                    // The corner of this face opposite the edge being split.
                    let Some(&w) = f.iter().find(|&&x| x != *v0 && x != *v1) else {
                        return false; // a face repeating a vertex has no shape to preserve
                    };
                    min_angle_of(&vertices[*v0], &midpoint, &vertices[w]) >= MIN_ANGLE_FLOOR
                        && min_angle_of(&midpoint, &vertices[*v1], &vertices[w]) >= MIN_ANGLE_FLOOR
                })
            })
            .map(|(edge, fis)| (*edge, fis.clone()))
            .collect();

        if long_edges.is_empty() {
            break;
        }

        // Report progress
        if let Some(p) = progress {
            p.report_sub(
                iteration + 1,
                max_iterations,
                step,
                total_steps,
                "Splitting edges",
            );
        }

        // Longest first, then by edge so the order is total.
        //
        // The tiebreak matters more than it looks: midpoint vertices are appended in this
        // order, so without it the *indices* of every new vertex depend on the hash
        // iteration order that produced `long_edges`, and two runs of the same input give
        // differently numbered — and then differently smoothed — meshes.
        long_edges.sort_by(|a, b| {
            let (v0a, v1a) = a.0;
            let (v0b, v1b) = b.0;
            let len_a = (vertices[v1a] - vertices[v0a]).norm_squared();
            let len_b = (vertices[v1b] - vertices[v0b]).norm_squared();
            len_b
                .total_cmp(&len_a)
                .then_with(|| v0a.cmp(&v0b))
                .then_with(|| v1a.cmp(&v1b))
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

        for (fi, &face) in faces.iter().enumerate() {
            if !faces_to_process.contains(&fi) {
                new_faces.push(face);
                continue;
            }

            // This face has at least one long edge - need to subdivide
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
    eprintln!(
        "Split phase done: {} faces, {} vertices. Building mesh...",
        faces.len(),
        vertices.len()
    );

    let built = match build_from_triangles::<I>(&vertices, &faces) {
        Ok(new_mesh) => {
            *mesh = new_mesh;
            true
        }
        Err(_) => false,
    };

    #[cfg(debug_assertions)]
    eprintln!("Split phase: mesh built");

    built
}

/// Collapse all edges shorter than the threshold (with progress reporting).
///
/// Uses batch processing: finds all collapsible edges, selects independent ones
/// (no shared vertices), and collapses them all at once before rebuilding topology.
/// Returns whether the pass ran to completion. See
/// [`split_long_edges_with_progress`].
fn collapse_short_edges_with_progress<I: MeshIndex>(
    mesh: &mut HalfEdgeMesh<I>,
    low_threshold: f64,
    high_threshold: f64,
    preserve_boundary: bool,
    progress: Option<&Progress>,
    step: usize,
    total_steps: usize,
) -> bool {
    let (mut vertices, mut faces) = to_face_vertex(mesh);

    let max_iterations = 30; // Batches, not individual collapses
    let mut _total_collapses = 0usize;

    for iteration in 0..max_iterations {
        #[cfg(debug_assertions)]
        eprintln!(
            "Collapse iter {}: {} faces, {} vertices",
            iteration,
            faces.len(),
            vertices.len()
        );

        // Build topology once per batch
        let topology = MeshTopology::from_faces(&faces, vertices.len());

        // Find ALL collapsible short edges
        let mut candidate_edges: Vec<(usize, usize, f64)> = Vec::new();

        for &(v0, v1) in topology.edge_faces.keys() {
            let length = (vertices[v1] - vertices[v0]).norm();

            if length < low_threshold
                && can_collapse_edge_fast(
                    &vertices,
                    &faces,
                    &topology,
                    v0,
                    v1,
                    high_threshold,
                    preserve_boundary,
                )
            {
                candidate_edges.push((v0, v1, length));
            }
        }

        if candidate_edges.is_empty() {
            break;
        }

        // Shortest first, then by edge so the order is total. `candidate_edges` was
        // gathered from a `HashMap`'s keys, so without the tiebreak equal-length edges are
        // ordered by hash iteration and the result varies between runs.
        candidate_edges.sort_by(|a, b| {
            a.2.total_cmp(&b.2)
                .then_with(|| a.0.cmp(&b.0))
                .then_with(|| a.1.cmp(&b.1))
        });

        // Select collapses that cannot interfere with each other.
        //
        // Marking only the two endpoints as used is not enough, and this pass used to do
        // exactly that — the comment here even said so. `can_collapse_edge_fast` checks a
        // link condition against the *pre-batch* topology, but collapsing an edge changes
        // the link of every vertex in its 1-ring: if `a` is a common neighbour of `c` and
        // `d`, merging `a` into `b` changes how many common neighbours `(c, d)` has, and a
        // collapse that was legal when checked is not legal when performed. On the torus
        // that produced a face list `build_from_triangles` rejected, so every collapse pass
        // was discarded and the mesh came back unchanged, five iterations running.
        //
        // Marking each endpoint's neighbours as used as well is sufficient. A later edge
        // `(c, d)` is then guaranteed that `c, d ∉ {a, b} ∪ N(a) ∪ N(b)`, which by symmetry
        // means `a, b ∉ N(c) ∪ N(d)` — so neither the endpoints nor the link of `(c, d)` is
        // touched by collapsing `(a, b)`, and its check still holds. The cost is smaller
        // batches, which the batch loop absorbs.
        let mut used_vertices: HashSet<usize> = HashSet::new();
        let mut edges_to_collapse: Vec<(usize, usize)> = Vec::new();

        for (v0, v1, _len) in candidate_edges {
            if used_vertices.contains(&v0) || used_vertices.contains(&v1) {
                continue;
            }
            edges_to_collapse.push((v0, v1));
            for v in [v0, v1] {
                used_vertices.insert(v);
                if let Some(neighbors) = topology.vertex_neighbors.get(v) {
                    used_vertices.extend(neighbors.iter().copied());
                }
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
            p.report_sub(
                iteration + 1,
                max_iterations,
                step,
                total_steps,
                "Collapsing edges",
            );
        }
    }

    // Final progress update for this step
    if let Some(p) = progress {
        p.report_sub(1, 1, step, total_steps, "Collapsing edges");
    }

    let (clean_vertices, clean_faces) = cleanup_mesh(&vertices, &faces);

    #[cfg(debug_assertions)]
    {
        eprintln!(
            "Collapse done. Building mesh from {} faces, {} vertices",
            clean_faces.len(),
            clean_vertices.len()
        );
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

    if clean_faces.is_empty() {
        // Collapsing consumed every face. Whatever that is, it is not a remeshed mesh.
        return false;
    }
    match build_from_triangles::<I>(&clean_vertices, &clean_faces) {
        Ok(new_mesh) => {
            #[cfg(debug_assertions)]
            eprintln!(
                "Collapse: mesh built successfully with {} halfedges",
                new_mesh.num_halfedges()
            );
            *mesh = new_mesh;
            true
        }
        Err(_e) => {
            #[cfg(debug_assertions)]
            eprintln!("Collapse: ERROR building mesh: {:?}", _e);
            false
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
    faces: &[[usize; 3]],
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

    // Every surviving face that used either endpoint changes shape, because the merged
    // vertex lands at the midpoint. Decline if any of them would become a sliver.
    //
    // Nothing here checked shape before, so collapsing was free to manufacture
    // near-degenerate triangles the same way splitting was. It matters more now than it
    // used to: while the batch selection was unsound, collapse passes were being rejected
    // wholesale and never applied, which hid this.
    let mid = Point3::from(midpoint);
    let mut checked: HashSet<usize> = HashSet::new();
    for &v in &[v0, v1] {
        for &n in topology.neighbors(v).iter() {
            let key = if v < n { (v, n) } else { (n, v) };
            let Some(face_indices) = topology.edge_faces.get(&key) else {
                continue;
            };
            for &fi in face_indices {
                if !checked.insert(fi) {
                    continue;
                }
                let f = faces[fi];
                // The two faces along the collapsed edge disappear; they have no shape to
                // preserve.
                if f.contains(&v0) && f.contains(&v1) {
                    continue;
                }
                let at = |x: usize| if x == v0 || x == v1 { mid } else { vertices[x] };
                if min_angle_of(&at(f[0]), &at(f[1]), &at(f[2])) < MIN_ANGLE_FLOOR {
                    return false;
                }
            }
        }
    }

    true
}

/// Flip edges to improve vertex valence.
///
/// Returns whether the pass ran to completion. See
/// [`split_long_edges_with_progress`].
fn flip_edges_to_improve_valence<I: MeshIndex>(
    mesh: &mut HalfEdgeMesh<I>,
    preserve_boundary: bool,
) -> bool {
    let (vertices, mut faces) = to_face_vertex(mesh);

    flip_edges_for_valence_faces(&vertices, &mut faces, preserve_boundary);

    if !validate_face_list(&vertices, &faces) {
        return false;
    }

    match build_from_triangles::<I>(&vertices, &faces) {
        Ok(new_mesh) => {
            *mesh = new_mesh;
            true
        }
        Err(_) => false,
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
        let original_euler = mesh.num_vertices() as i32 - (mesh.num_halfedges() / 2) as i32
            + mesh.num_faces() as i32;

        let options = RemeshOptions::with_target_length(0.5).with_iterations(2);
        let _ = isotropic_remesh(&mut mesh, &options);

        assert!(mesh.is_valid());

        let new_euler = mesh.num_vertices() as i32 - (mesh.num_halfedges() / 2) as i32
            + mesh.num_faces() as i32;
        assert_eq!(original_euler, new_euler);
    }

    #[test]
    fn test_isotropic_remesh_changes_edge_lengths() {
        let mut mesh = create_grid_mesh(3);
        let original_avg = average_edge_length(&mesh);

        let target = original_avg * 0.5;
        let options = RemeshOptions::with_target_length(target).with_iterations(3);
        let _ = isotropic_remesh(&mut mesh, &options);

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
        let _ = isotropic_remesh(&mut mesh, &options);

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
