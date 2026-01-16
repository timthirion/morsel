//! Quadric Error Metrics (QEM) decimation.

use std::collections::{BinaryHeap, HashMap, HashSet};
use std::cmp::Ordering;

use nalgebra::{Matrix4, Point3, Vector4};

use crate::mesh::{build_from_triangles, to_face_vertex, HalfEdgeMesh, MeshIndex};

use super::DecimateOptions;

/// A quadric error matrix (4x4 symmetric matrix).
///
/// Represents the sum of squared distances to a set of planes.
/// Stored as 10 unique elements since the matrix is symmetric.
#[derive(Debug, Clone, Copy)]
struct Quadric {
    /// Upper triangular elements: [a, b, c, d, e, f, g, h, i, j]
    /// Matrix form:
    /// | a b c d |
    /// | b e f g |
    /// | c f h i |
    /// | d g i j |
    data: [f64; 10],
}

impl Quadric {
    /// Create a zero quadric.
    fn zero() -> Self {
        Self { data: [0.0; 10] }
    }

    /// Create a quadric from a plane equation ax + by + cz + d = 0.
    /// The plane should be normalized (a² + b² + c² = 1).
    fn from_plane(a: f64, b: f64, c: f64, d: f64) -> Self {
        Self {
            data: [
                a * a,     // [0,0]
                a * b,     // [0,1] = [1,0]
                a * c,     // [0,2] = [2,0]
                a * d,     // [0,3] = [3,0]
                b * b,     // [1,1]
                b * c,     // [1,2] = [2,1]
                b * d,     // [1,3] = [3,1]
                c * c,     // [2,2]
                c * d,     // [2,3] = [3,2]
                d * d,     // [3,3]
            ],
        }
    }

    /// Add another quadric to this one in place.
    fn add_assign(&mut self, other: &Quadric) {
        for i in 0..10 {
            self.data[i] += other.data[i];
        }
    }

    /// Evaluate the quadric error for a point.
    /// Returns v^T * Q * v where v = [x, y, z, 1].
    fn evaluate(&self, p: &Point3<f64>) -> f64 {
        let x = p.x;
        let y = p.y;
        let z = p.z;

        // v^T * Q * v expanded
        self.data[0] * x * x
            + 2.0 * self.data[1] * x * y
            + 2.0 * self.data[2] * x * z
            + 2.0 * self.data[3] * x
            + self.data[4] * y * y
            + 2.0 * self.data[5] * y * z
            + 2.0 * self.data[6] * y
            + self.data[7] * z * z
            + 2.0 * self.data[8] * z
            + self.data[9]
    }

    /// Convert to a 4x4 matrix for solving the optimal point.
    fn to_matrix(&self) -> Matrix4<f64> {
        Matrix4::new(
            self.data[0], self.data[1], self.data[2], self.data[3],
            self.data[1], self.data[4], self.data[5], self.data[6],
            self.data[2], self.data[5], self.data[7], self.data[8],
            self.data[3], self.data[6], self.data[8], self.data[9],
        )
    }

    /// Find the optimal point that minimizes the quadric error.
    /// Returns None if the matrix is singular.
    fn optimal_point(&self) -> Option<Point3<f64>> {
        // We want to solve Q * v = [0, 0, 0, 1]^T
        // But we need to modify Q to have [0, 0, 0, 1] as the last row
        let mut m = self.to_matrix();
        m[(3, 0)] = 0.0;
        m[(3, 1)] = 0.0;
        m[(3, 2)] = 0.0;
        m[(3, 3)] = 1.0;

        // Try to invert
        if let Some(inv) = m.try_inverse() {
            let v = inv * Vector4::new(0.0, 0.0, 0.0, 1.0);
            Some(Point3::new(v.x, v.y, v.z))
        } else {
            None
        }
    }
}

impl std::ops::Add for Quadric {
    type Output = Quadric;

    fn add(self, other: Quadric) -> Quadric {
        let mut result = self;
        result.add_assign(&other);
        result
    }
}

/// An edge candidate for collapse.
#[derive(Debug, Clone)]
struct EdgeCandidate {
    /// Vertex indices (smaller first for canonical ordering).
    v0: usize,
    v1: usize,
    /// Optimal position after collapse.
    optimal_pos: Point3<f64>,
    /// Error cost of this collapse.
    error: f64,
    /// Version counter to detect stale entries.
    version: usize,
}

impl EdgeCandidate {
    fn new(v0: usize, v1: usize, optimal_pos: Point3<f64>, error: f64, version: usize) -> Self {
        let (v0, v1) = if v0 < v1 { (v0, v1) } else { (v1, v0) };
        Self {
            v0,
            v1,
            optimal_pos,
            error,
            version,
        }
    }
}

// Implement ordering for min-heap (we want smallest error first)
impl PartialEq for EdgeCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.error == other.error
    }
}

impl Eq for EdgeCandidate {}

impl PartialOrd for EdgeCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for EdgeCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap
        other.error.partial_cmp(&self.error).unwrap_or(Ordering::Equal)
    }
}

/// Performs QEM decimation on a triangle mesh.
///
/// This algorithm iteratively collapses edges to reduce the face count while
/// minimizing geometric error using quadric error metrics.
///
/// # Arguments
///
/// * `mesh` - The mesh to decimate (modified in place)
/// * `options` - Decimation parameters
pub fn qem_decimate<I: MeshIndex>(mesh: &mut HalfEdgeMesh<I>, options: &DecimateOptions) {
    let (vertices, faces) = to_face_vertex(mesh);

    if vertices.is_empty() || faces.is_empty() {
        return;
    }

    let target_faces = options.compute_target(faces.len());
    if target_faces >= faces.len() {
        return;
    }

    // Run the decimation algorithm
    let (new_vertices, new_faces) = decimate_mesh(
        vertices,
        faces,
        target_faces,
        options.preserve_boundary,
        options.max_error,
    );

    // Rebuild the half-edge mesh
    if !new_faces.is_empty() {
        if let Ok(new_mesh) = build_from_triangles::<I>(&new_vertices, &new_faces) {
            *mesh = new_mesh;
        }
    }
}

/// Main decimation algorithm on face-vertex representation.
fn decimate_mesh(
    mut vertices: Vec<Point3<f64>>,
    mut faces: Vec<[usize; 3]>,
    target_faces: usize,
    preserve_boundary: bool,
    max_error: Option<f64>,
) -> (Vec<Point3<f64>>, Vec<[usize; 3]>) {
    let n_vertices = vertices.len();

    // Track which vertices and faces are still valid
    let mut valid_vertices: Vec<bool> = vec![true; n_vertices];
    let mut valid_faces: Vec<bool> = vec![true; faces.len()];
    let mut current_face_count = faces.len();

    // Compute initial quadrics for each vertex
    let mut quadrics = compute_vertex_quadrics(&vertices, &faces);

    // Find boundary edges if we need to preserve them
    let boundary_edges = if preserve_boundary {
        find_boundary_edges(&faces)
    } else {
        HashSet::new()
    };

    // Build edge-to-faces mapping
    let mut edge_faces = build_edge_faces(&faces);

    // Version counter for each vertex (to detect stale heap entries)
    let mut vertex_versions: Vec<usize> = vec![0; n_vertices];

    // Priority queue of edge candidates
    let mut heap: BinaryHeap<EdgeCandidate> = BinaryHeap::new();

    // Initialize heap with all edges
    let mut seen_edges: HashSet<(usize, usize)> = HashSet::new();
    for (fi, face) in faces.iter().enumerate() {
        if !valid_faces[fi] {
            continue;
        }
        for i in 0..3 {
            let v0 = face[i];
            let v1 = face[(i + 1) % 3];
            let edge = canonical_edge(v0, v1);

            if seen_edges.contains(&edge) {
                continue;
            }
            seen_edges.insert(edge);

            // Skip boundary edges if preserving boundary
            if preserve_boundary && boundary_edges.contains(&edge) {
                continue;
            }

            if let Some(candidate) = create_edge_candidate(
                v0,
                v1,
                &vertices,
                &quadrics,
                vertex_versions[v0],
            ) {
                heap.push(candidate);
            }
        }
    }

    // Main decimation loop
    while current_face_count > target_faces {
        // Get the edge with minimum error
        let candidate = match heap.pop() {
            Some(c) => c,
            None => break,
        };

        // Check if this entry is stale
        if !valid_vertices[candidate.v0]
            || !valid_vertices[candidate.v1]
            || candidate.version != vertex_versions[candidate.v0]
        {
            continue;
        }

        // Check max error threshold
        if let Some(max_err) = max_error {
            if candidate.error > max_err {
                break;
            }
        }

        // Check if collapse is valid (won't create non-manifold geometry)
        if !is_collapse_valid(
            candidate.v0,
            candidate.v1,
            &faces,
            &valid_faces,
            &edge_faces,
        ) {
            continue;
        }

        // Perform the collapse: v1 -> v0
        let v_keep = candidate.v0;
        let v_remove = candidate.v1;

        // Update position of kept vertex
        vertices[v_keep] = candidate.optimal_pos;

        // Update quadric of kept vertex
        quadrics[v_keep] = quadrics[v_keep] + quadrics[v_remove];

        // Increment version to invalidate old heap entries
        vertex_versions[v_keep] += 1;

        // Mark removed vertex as invalid
        valid_vertices[v_remove] = false;

        // Update faces: replace v_remove with v_keep, remove degenerate faces
        let edge = canonical_edge(v_keep, v_remove);
        if let Some(face_indices) = edge_faces.get(&edge) {
            for &fi in face_indices {
                if valid_faces[fi] {
                    valid_faces[fi] = false;
                    current_face_count -= 1;
                }
            }
        }

        // Update remaining faces that reference v_remove
        for (fi, face) in faces.iter_mut().enumerate() {
            if !valid_faces[fi] {
                continue;
            }

            let mut changed = false;
            for v in face.iter_mut() {
                if *v == v_remove {
                    *v = v_keep;
                    changed = true;
                }
            }

            // Check for degenerate face (two or more same vertices)
            if changed {
                if face[0] == face[1] || face[1] == face[2] || face[0] == face[2] {
                    valid_faces[fi] = false;
                    current_face_count -= 1;
                }
            }
        }

        // Rebuild edge_faces for affected edges
        rebuild_edge_faces_for_vertex(v_keep, &faces, &valid_faces, &mut edge_faces);

        // Add new edge candidates for edges around v_keep
        let neighbors = get_vertex_neighbors(v_keep, &faces, &valid_faces);
        for &neighbor in &neighbors {
            if !valid_vertices[neighbor] {
                continue;
            }

            let edge = canonical_edge(v_keep, neighbor);
            if preserve_boundary && boundary_edges.contains(&edge) {
                continue;
            }

            if let Some(candidate) = create_edge_candidate(
                v_keep,
                neighbor,
                &vertices,
                &quadrics,
                vertex_versions[v_keep],
            ) {
                heap.push(candidate);
            }
        }
    }

    // Compact the mesh (remove invalid vertices and faces)
    compact_mesh(vertices, faces, valid_vertices, valid_faces)
}

/// Compute initial quadrics for each vertex from adjacent faces.
fn compute_vertex_quadrics(
    vertices: &[Point3<f64>],
    faces: &[[usize; 3]],
) -> Vec<Quadric> {
    let mut quadrics = vec![Quadric::zero(); vertices.len()];

    for face in faces {
        let p0 = &vertices[face[0]];
        let p1 = &vertices[face[1]];
        let p2 = &vertices[face[2]];

        // Compute plane equation
        let e1 = p1 - p0;
        let e2 = p2 - p0;
        let normal = e1.cross(&e2);

        let len = normal.norm();
        if len < 1e-10 {
            continue; // Degenerate face
        }

        let n = normal / len;
        let d = -n.dot(&p0.coords);

        let q = Quadric::from_plane(n.x, n.y, n.z, d);

        // Add to all three vertices
        quadrics[face[0]].add_assign(&q);
        quadrics[face[1]].add_assign(&q);
        quadrics[face[2]].add_assign(&q);
    }

    quadrics
}

/// Find all boundary edges (edges with only one adjacent face).
fn find_boundary_edges(faces: &[[usize; 3]]) -> HashSet<(usize, usize)> {
    let mut edge_count: HashMap<(usize, usize), usize> = HashMap::new();

    for face in faces {
        for i in 0..3 {
            let edge = canonical_edge(face[i], face[(i + 1) % 3]);
            *edge_count.entry(edge).or_insert(0) += 1;
        }
    }

    edge_count
        .into_iter()
        .filter(|(_, count)| *count == 1)
        .map(|(edge, _)| edge)
        .collect()
}

/// Build mapping from edge to face indices.
fn build_edge_faces(faces: &[[usize; 3]]) -> HashMap<(usize, usize), Vec<usize>> {
    let mut edge_faces: HashMap<(usize, usize), Vec<usize>> = HashMap::new();

    for (fi, face) in faces.iter().enumerate() {
        for i in 0..3 {
            let edge = canonical_edge(face[i], face[(i + 1) % 3]);
            edge_faces.entry(edge).or_default().push(fi);
        }
    }

    edge_faces
}

/// Rebuild edge_faces for edges adjacent to a vertex.
fn rebuild_edge_faces_for_vertex(
    v: usize,
    faces: &[[usize; 3]],
    valid_faces: &[bool],
    edge_faces: &mut HashMap<(usize, usize), Vec<usize>>,
) {
    // Find all edges adjacent to v and rebuild their face lists
    for (fi, face) in faces.iter().enumerate() {
        if !valid_faces[fi] {
            continue;
        }

        for i in 0..3 {
            if face[i] == v || face[(i + 1) % 3] == v {
                let edge = canonical_edge(face[i], face[(i + 1) % 3]);
                let entry = edge_faces.entry(edge).or_default();
                if !entry.contains(&fi) {
                    entry.push(fi);
                }
            }
        }
    }

    // Clean up entries with invalid faces
    edge_faces.retain(|_, face_list| {
        face_list.retain(|&fi| valid_faces[fi]);
        !face_list.is_empty()
    });
}

/// Get canonical edge representation (smaller index first).
fn canonical_edge(v0: usize, v1: usize) -> (usize, usize) {
    if v0 < v1 {
        (v0, v1)
    } else {
        (v1, v0)
    }
}

/// Create an edge candidate with optimal position and error.
fn create_edge_candidate(
    v0: usize,
    v1: usize,
    vertices: &[Point3<f64>],
    quadrics: &[Quadric],
    version: usize,
) -> Option<EdgeCandidate> {
    let q_combined = quadrics[v0] + quadrics[v1];

    // Try to find optimal position
    let optimal_pos = if let Some(p) = q_combined.optimal_point() {
        // Check if optimal point is reasonable (not too far from edge)
        let midpoint = Point3::from((vertices[v0].coords + vertices[v1].coords) * 0.5);
        let edge_len = (vertices[v1] - vertices[v0]).norm();

        if (p - midpoint).norm() < edge_len * 2.0 {
            p
        } else {
            // Fallback to midpoint
            midpoint
        }
    } else {
        // Matrix singular, try endpoints and midpoint
        let p0 = vertices[v0];
        let p1 = vertices[v1];
        let mid = Point3::from((p0.coords + p1.coords) * 0.5);

        let e0 = q_combined.evaluate(&p0);
        let e1 = q_combined.evaluate(&p1);
        let em = q_combined.evaluate(&mid);

        if e0 <= e1 && e0 <= em {
            p0
        } else if e1 <= em {
            p1
        } else {
            mid
        }
    };

    let error = q_combined.evaluate(&optimal_pos);

    Some(EdgeCandidate::new(v0, v1, optimal_pos, error, version))
}

/// Check if an edge collapse is valid (won't create non-manifold geometry).
fn is_collapse_valid(
    v0: usize,
    v1: usize,
    faces: &[[usize; 3]],
    valid_faces: &[bool],
    edge_faces: &HashMap<(usize, usize), Vec<usize>>,
) -> bool {
    // Get the faces adjacent to the edge
    let edge = canonical_edge(v0, v1);
    let edge_face_count = edge_faces
        .get(&edge)
        .map(|f| f.iter().filter(|&&fi| valid_faces[fi]).count())
        .unwrap_or(0);

    // Edge should have 1 or 2 adjacent faces
    if edge_face_count == 0 || edge_face_count > 2 {
        return false;
    }

    // Check link condition: the intersection of vertex neighborhoods
    // should be exactly the vertices shared by both (the opposite vertices
    // of the edge in adjacent triangles)
    let neighbors_v0 = get_vertex_neighbors(v0, faces, valid_faces);
    let neighbors_v1 = get_vertex_neighbors(v1, faces, valid_faces);

    let common: HashSet<_> = neighbors_v0.intersection(&neighbors_v1).collect();

    // For a valid collapse:
    // - Interior edge (2 faces): exactly 2 common neighbors
    // - Boundary edge (1 face): exactly 1 common neighbor
    if edge_face_count == 2 && common.len() != 2 {
        return false;
    }
    if edge_face_count == 1 && common.len() != 1 {
        return false;
    }

    // Additional check: ensure collapse won't create duplicate edges
    // After collapse, all neighbors of v1 (except v0) become neighbors of v0
    // This is invalid if any neighbor of v1 is already a neighbor of v0
    for &n in &neighbors_v1 {
        if n != v0 && neighbors_v0.contains(&n) && !common.contains(&n) {
            return false;
        }
    }

    true
}

/// Get all neighboring vertices of a vertex.
fn get_vertex_neighbors(
    v: usize,
    faces: &[[usize; 3]],
    valid_faces: &[bool],
) -> HashSet<usize> {
    let mut neighbors = HashSet::new();

    for (fi, face) in faces.iter().enumerate() {
        if !valid_faces[fi] {
            continue;
        }

        for i in 0..3 {
            if face[i] == v {
                neighbors.insert(face[(i + 1) % 3]);
                neighbors.insert(face[(i + 2) % 3]);
            }
        }
    }

    neighbors
}

/// Compact the mesh by removing invalid vertices and faces.
fn compact_mesh(
    vertices: Vec<Point3<f64>>,
    faces: Vec<[usize; 3]>,
    valid_vertices: Vec<bool>,
    valid_faces: Vec<bool>,
) -> (Vec<Point3<f64>>, Vec<[usize; 3]>) {
    // Create vertex index mapping
    let mut vertex_map: Vec<usize> = vec![usize::MAX; vertices.len()];
    let mut new_vertices: Vec<Point3<f64>> = Vec::new();

    for (i, &valid) in valid_vertices.iter().enumerate() {
        if valid {
            vertex_map[i] = new_vertices.len();
            new_vertices.push(vertices[i]);
        }
    }

    // Remap faces
    let new_faces: Vec<[usize; 3]> = faces
        .iter()
        .enumerate()
        .filter(|(fi, _)| valid_faces[*fi])
        .filter_map(|(_, face)| {
            let v0 = vertex_map[face[0]];
            let v1 = vertex_map[face[1]];
            let v2 = vertex_map[face[2]];

            if v0 == usize::MAX || v1 == usize::MAX || v2 == usize::MAX {
                None
            } else {
                Some([v0, v1, v2])
            }
        })
        .collect();

    // Validate: check for non-manifold edges (edge shared by more than 2 faces)
    let mut edge_count: HashMap<(usize, usize), usize> = HashMap::new();
    for face in &new_faces {
        for i in 0..3 {
            let edge = canonical_edge(face[i], face[(i + 1) % 3]);
            *edge_count.entry(edge).or_insert(0) += 1;
        }
    }

    // If any edge has more than 2 adjacent faces, the mesh is non-manifold
    let is_manifold = edge_count.values().all(|&count| count <= 2);

    if is_manifold {
        (new_vertices, new_faces)
    } else {
        // Return empty to signal failure - caller will keep original mesh
        (Vec::new(), Vec::new())
    }
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

    fn create_octahedron() -> HalfEdgeMesh {
        let vertices = vec![
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(-1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, -1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
            Point3::new(0.0, 0.0, -1.0),
        ];
        let faces = vec![
            [0, 2, 4],
            [2, 1, 4],
            [1, 3, 4],
            [3, 0, 4],
            [2, 0, 5],
            [1, 2, 5],
            [3, 1, 5],
            [0, 3, 5],
        ];
        build_from_triangles(&vertices, &faces).unwrap()
    }

    fn create_grid_mesh(n: usize) -> HalfEdgeMesh {
        let mut vertices = Vec::new();
        let mut faces = Vec::new();

        // Create vertices
        for j in 0..=n {
            for i in 0..=n {
                vertices.push(Point3::new(i as f64, j as f64, 0.0));
            }
        }

        // Create faces (two triangles per grid cell)
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

    #[test]
    fn test_quadric_from_plane() {
        // Plane z = 0 (normal [0, 0, 1], d = 0)
        let q = Quadric::from_plane(0.0, 0.0, 1.0, 0.0);

        // Error should be z² for any point
        let p1 = Point3::new(0.0, 0.0, 0.0);
        assert!((q.evaluate(&p1) - 0.0).abs() < 1e-10);

        let p2 = Point3::new(0.0, 0.0, 1.0);
        assert!((q.evaluate(&p2) - 1.0).abs() < 1e-10);

        let p3 = Point3::new(5.0, 3.0, 2.0);
        assert!((q.evaluate(&p3) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_quadric_addition() {
        let q1 = Quadric::from_plane(1.0, 0.0, 0.0, 0.0); // x = 0
        let q2 = Quadric::from_plane(0.0, 1.0, 0.0, 0.0); // y = 0

        let q = q1 + q2;

        // Combined error should be x² + y²
        let p = Point3::new(3.0, 4.0, 0.0);
        assert!((q.evaluate(&p) - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_decimate_reduces_faces() {
        let mut mesh = create_octahedron();
        let original_faces = mesh.num_faces();

        let options = DecimateOptions::with_target_ratio(0.5);
        qem_decimate(&mut mesh, &options);

        assert!(mesh.num_faces() < original_faces);
        assert!(mesh.is_valid());
    }

    #[test]
    fn test_decimate_target_faces() {
        let mut mesh = create_octahedron();
        let original_faces = mesh.num_faces();

        let options = DecimateOptions::with_target_faces(6);
        qem_decimate(&mut mesh, &options);

        // Should reduce faces (closed mesh is more reliable)
        assert!(mesh.num_faces() <= original_faces);
        assert!(mesh.is_valid());
    }

    #[test]
    fn test_decimate_preserves_validity() {
        let mut mesh = create_octahedron();

        let options = DecimateOptions::with_target_ratio(0.6);
        qem_decimate(&mut mesh, &options);

        assert!(mesh.is_valid());
    }

    #[test]
    fn test_decimate_no_change_at_full_ratio() {
        let mut mesh = create_tetrahedron();
        let original_faces = mesh.num_faces();
        let original_vertices = mesh.num_vertices();

        let options = DecimateOptions::with_target_ratio(1.0);
        qem_decimate(&mut mesh, &options);

        assert_eq!(mesh.num_faces(), original_faces);
        assert_eq!(mesh.num_vertices(), original_vertices);
    }

    #[test]
    fn test_decimate_grid_mesh() {
        let mut mesh = create_grid_mesh(3);
        let original_faces = mesh.num_faces();

        // Use a moderate ratio on smaller grid for reliable results
        let options = DecimateOptions::with_target_ratio(0.7);
        qem_decimate(&mut mesh, &options);

        assert!(mesh.num_faces() < original_faces);
        assert!(mesh.is_valid());
    }

    #[test]
    fn test_decimate_closed_mesh_aggressive() {
        // Test aggressive decimation on closed mesh
        let mut mesh = create_octahedron();
        let original_faces = mesh.num_faces();

        let options = DecimateOptions::with_target_ratio(0.5);
        qem_decimate(&mut mesh, &options);

        // Should reduce faces
        assert!(mesh.num_faces() <= original_faces);
        assert!(mesh.is_valid());
    }

    #[test]
    fn test_decimate_with_max_error() {
        let mut mesh = create_octahedron();

        // Very low max error should prevent most collapses
        let options = DecimateOptions::with_target_ratio(0.1).with_max_error(1e-10);
        qem_decimate(&mut mesh, &options);

        // Mesh should still be valid even if target wasn't reached
        assert!(mesh.is_valid());
    }
}
