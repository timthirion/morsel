//! Loop subdivision for triangle meshes.

use std::collections::HashMap;

use nalgebra::{Point3, Vector3};

use crate::algo::Progress;
use crate::mesh::{build_from_triangles, to_face_vertex, HalfEdgeMesh, MeshIndex};

use super::SubdivideOptions;

/// Performs Loop subdivision on a triangle mesh.
///
/// Loop subdivision is an approximating subdivision scheme that produces
/// smooth surfaces from triangle meshes. Each iteration quadruples the
/// number of triangles.
///
/// # Arguments
///
/// * `mesh` - The mesh to subdivide (modified in place)
/// * `options` - Subdivision parameters
///
/// # Algorithm
///
/// For each iteration:
/// 1. Compute new "edge vertices" at weighted positions along each edge
/// 2. Update original "vertex vertices" based on their neighbors
/// 3. Replace each triangle with 4 smaller triangles
///
/// # Vertex Rules
///
/// - **Interior edge vertex**: `3/8 * (v0 + v1) + 1/8 * (v_left + v_right)`
/// - **Boundary edge vertex**: `1/2 * (v0 + v1)`
/// - **Interior vertex**: `(1 - n*β) * v + β * Σ(neighbors)`
/// - **Boundary vertex**: `1/8 * (left + right) + 3/4 * v`
pub fn loop_subdivide<I: MeshIndex>(mesh: &mut HalfEdgeMesh<I>, options: &SubdivideOptions) {
    if options.iterations == 0 {
        return;
    }

    for _ in 0..options.iterations {
        loop_subdivide_once(mesh, options.preserve_boundary);
    }
}

/// Loop subdivision with progress reporting.
pub fn loop_subdivide_with_progress<I: MeshIndex>(
    mesh: &mut HalfEdgeMesh<I>,
    options: &SubdivideOptions,
    progress: &Progress,
) {
    if options.iterations == 0 {
        return;
    }

    for iter in 0..options.iterations {
        progress.report(iter, options.iterations, "Loop subdivision");
        loop_subdivide_once(mesh, options.preserve_boundary);
    }
    progress.report(options.iterations, options.iterations, "Loop subdivision");
}

/// Perform one iteration of Loop subdivision.
fn loop_subdivide_once<I: MeshIndex>(mesh: &mut HalfEdgeMesh<I>, preserve_boundary: bool) {
    let (vertices, faces) = to_face_vertex(mesh);

    if vertices.is_empty() || faces.is_empty() {
        return;
    }

    // Build edge information
    let edge_info = build_edge_info(&faces);

    // Compute new edge vertices
    let edge_vertices = compute_edge_vertices(&vertices, &faces, &edge_info, preserve_boundary);

    // Compute updated vertex positions
    let updated_vertices =
        compute_updated_vertices(&vertices, &faces, &edge_info, preserve_boundary);

    // Build the subdivided mesh
    let (new_vertices, new_faces) =
        build_subdivided_mesh(&updated_vertices, &edge_vertices, &faces, &edge_info);

    // Rebuild the half-edge mesh
    if let Ok(new_mesh) = build_from_triangles::<I>(&new_vertices, &new_faces) {
        *mesh = new_mesh;
    }
}

/// Information about an edge for subdivision.
#[derive(Debug, Clone)]
struct EdgeInfo {
    /// Index of the new vertex created for this edge.
    new_vertex_index: usize,
    /// Indices of the two triangles sharing this edge (second is None for boundary).
    faces: (usize, Option<usize>),
    /// The opposite vertex in the first triangle.
    opposite1: usize,
    /// The opposite vertex in the second triangle (None for boundary).
    opposite2: Option<usize>,
}

/// Build a map from edge (v0, v1) to edge information.
fn build_edge_info(faces: &[[usize; 3]]) -> HashMap<(usize, usize), EdgeInfo> {
    let mut edge_map: HashMap<(usize, usize), EdgeInfo> = HashMap::new();
    let mut next_edge_vertex = 0;

    for (face_idx, face) in faces.iter().enumerate() {
        for i in 0..3 {
            let v0 = face[i];
            let v1 = face[(i + 1) % 3];
            let opposite = face[(i + 2) % 3];

            // Canonical edge key (smaller index first)
            let edge_key = if v0 < v1 { (v0, v1) } else { (v1, v0) };

            if let Some(info) = edge_map.get_mut(&edge_key) {
                // Second face sharing this edge
                info.faces.1 = Some(face_idx);
                info.opposite2 = Some(opposite);
            } else {
                // First face for this edge
                edge_map.insert(
                    edge_key,
                    EdgeInfo {
                        new_vertex_index: next_edge_vertex,
                        faces: (face_idx, None),
                        opposite1: opposite,
                        opposite2: None,
                    },
                );
                next_edge_vertex += 1;
            }
        }
    }

    edge_map
}

/// Compute positions for new edge vertices.
fn compute_edge_vertices(
    vertices: &[Point3<f64>],
    _faces: &[[usize; 3]],
    edge_info: &HashMap<(usize, usize), EdgeInfo>,
    preserve_boundary: bool,
) -> Vec<Point3<f64>> {
    let mut edge_vertices = vec![Point3::origin(); edge_info.len()];

    for (&(v0, v1), info) in edge_info {
        let p0 = &vertices[v0];
        let p1 = &vertices[v1];

        let new_pos = if info.opposite2.is_none() {
            // Boundary edge: simple midpoint (or weighted for smooth boundary)
            if preserve_boundary {
                // Linear interpolation for sharp boundary
                Point3::from((p0.coords + p1.coords) * 0.5)
            } else {
                // Could use boundary subdivision rules here
                Point3::from((p0.coords + p1.coords) * 0.5)
            }
        } else {
            // Interior edge: weighted average
            // new_pos = 3/8 * (v0 + v1) + 1/8 * (opposite1 + opposite2)
            let p_opp1 = &vertices[info.opposite1];
            let p_opp2 = &vertices[info.opposite2.unwrap()];

            Point3::from(
                (p0.coords + p1.coords) * (3.0 / 8.0)
                    + (p_opp1.coords + p_opp2.coords) * (1.0 / 8.0),
            )
        };

        edge_vertices[info.new_vertex_index] = new_pos;
    }

    edge_vertices
}

/// Compute updated positions for original vertices.
fn compute_updated_vertices(
    vertices: &[Point3<f64>],
    faces: &[[usize; 3]],
    edge_info: &HashMap<(usize, usize), EdgeInfo>,
    preserve_boundary: bool,
) -> Vec<Point3<f64>> {
    let n = vertices.len();

    // Build adjacency and identify boundary vertices
    let (neighbors, boundary_neighbors) = build_vertex_adjacency(vertices.len(), faces, edge_info);

    let mut updated = Vec::with_capacity(n);

    for (i, pos) in vertices.iter().enumerate() {
        let is_boundary = !boundary_neighbors[i].is_empty();

        let new_pos = if is_boundary && preserve_boundary {
            // Boundary vertex rule
            if boundary_neighbors[i].len() == 2 {
                // Regular boundary vertex: 1/8 * (left + right) + 3/4 * v
                let left = &vertices[boundary_neighbors[i][0]];
                let right = &vertices[boundary_neighbors[i][1]];
                Point3::from((left.coords + right.coords) * (1.0 / 8.0) + pos.coords * (3.0 / 4.0))
            } else {
                // Corner or irregular boundary: keep position
                *pos
            }
        } else {
            // Interior vertex rule
            let n_neighbors = neighbors[i].len();
            if n_neighbors == 0 {
                *pos
            } else {
                let beta = compute_loop_beta(n_neighbors);
                let neighbor_sum: Vector3<f64> =
                    neighbors[i].iter().map(|&j| vertices[j].coords).sum();

                Point3::from(
                    pos.coords * (1.0 - n_neighbors as f64 * beta) + neighbor_sum * beta,
                )
            }
        };

        updated.push(new_pos);
    }

    updated
}

/// Compute the Loop subdivision beta coefficient for a vertex with n neighbors.
fn compute_loop_beta(n: usize) -> f64 {
    if n == 3 {
        // Special case for valence 3 (common optimization)
        3.0 / 16.0
    } else {
        // General formula: β = 1/n * (5/8 - (3/8 + 1/4 * cos(2π/n))²)
        let n_f = n as f64;
        let cos_term = (2.0 * std::f64::consts::PI / n_f).cos();
        let inner = 3.0 / 8.0 + 0.25 * cos_term;
        (1.0 / n_f) * (5.0 / 8.0 - inner * inner)
    }
}

/// Build vertex adjacency lists and identify boundary neighbors.
fn build_vertex_adjacency(
    num_vertices: usize,
    faces: &[[usize; 3]],
    edge_info: &HashMap<(usize, usize), EdgeInfo>,
) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    // All neighbors
    let mut neighbors: Vec<Vec<usize>> = vec![Vec::new(); num_vertices];
    // Boundary neighbors only
    let mut boundary_neighbors: Vec<Vec<usize>> = vec![Vec::new(); num_vertices];

    // Collect all neighbors from faces
    for face in faces {
        for i in 0..3 {
            let v = face[i];
            let next = face[(i + 1) % 3];
            let prev = face[(i + 2) % 3];

            if !neighbors[v].contains(&next) {
                neighbors[v].push(next);
            }
            if !neighbors[v].contains(&prev) {
                neighbors[v].push(prev);
            }
        }
    }

    // Find boundary edges and mark boundary neighbors
    for (&(v0, v1), info) in edge_info {
        if info.opposite2.is_none() {
            // This is a boundary edge
            if !boundary_neighbors[v0].contains(&v1) {
                boundary_neighbors[v0].push(v1);
            }
            if !boundary_neighbors[v1].contains(&v0) {
                boundary_neighbors[v1].push(v0);
            }
        }
    }

    (neighbors, boundary_neighbors)
}

/// Build the subdivided mesh with new connectivity.
fn build_subdivided_mesh(
    updated_vertices: &[Point3<f64>],
    edge_vertices: &[Point3<f64>],
    original_faces: &[[usize; 3]],
    edge_info: &HashMap<(usize, usize), EdgeInfo>,
) -> (Vec<Point3<f64>>, Vec<[usize; 3]>) {
    // New vertices: original (updated) + edge vertices
    let num_original = updated_vertices.len();
    let mut new_vertices: Vec<Point3<f64>> = updated_vertices.to_vec();
    new_vertices.extend(edge_vertices.iter().cloned());

    // Helper to get edge vertex index
    let get_edge_vertex = |v0: usize, v1: usize| -> usize {
        let key = if v0 < v1 { (v0, v1) } else { (v1, v0) };
        let info = edge_info.get(&key).unwrap();
        num_original + info.new_vertex_index
    };

    // Build new faces: each original triangle becomes 4 triangles
    let mut new_faces: Vec<[usize; 3]> = Vec::with_capacity(original_faces.len() * 4);

    for face in original_faces {
        let v0 = face[0];
        let v1 = face[1];
        let v2 = face[2];

        // Edge vertices
        let e01 = get_edge_vertex(v0, v1);
        let e12 = get_edge_vertex(v1, v2);
        let e20 = get_edge_vertex(v2, v0);

        // Four new triangles:
        // 1. Corner triangle at v0
        new_faces.push([v0, e01, e20]);
        // 2. Corner triangle at v1
        new_faces.push([v1, e12, e01]);
        // 3. Corner triangle at v2
        new_faces.push([v2, e20, e12]);
        // 4. Central triangle
        new_faces.push([e01, e12, e20]);
    }

    (new_vertices, new_faces)
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

    fn create_single_triangle() -> HalfEdgeMesh {
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.5, 1.0, 0.0),
        ];
        let faces = vec![[0, 1, 2]];
        build_from_triangles(&vertices, &faces).unwrap()
    }

    fn create_two_triangles() -> HalfEdgeMesh {
        // Two triangles sharing an edge
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.5, 1.0, 0.0),
            Point3::new(0.5, -1.0, 0.0),
        ];
        let faces = vec![[0, 1, 2], [0, 3, 1]];
        build_from_triangles(&vertices, &faces).unwrap()
    }

    #[test]
    fn test_loop_subdivide_single_triangle() {
        let mut mesh = create_single_triangle();

        let options = SubdivideOptions::new(1);
        loop_subdivide(&mut mesh, &options);

        // 1 triangle -> 4 triangles
        assert_eq!(mesh.num_faces(), 4);
        // 3 original + 3 edge vertices = 6
        assert_eq!(mesh.num_vertices(), 6);
        assert!(mesh.is_valid());
    }

    #[test]
    fn test_loop_subdivide_quadruples_faces() {
        let mut mesh = create_tetrahedron();
        let original_faces = mesh.num_faces();

        let options = SubdivideOptions::new(1);
        loop_subdivide(&mut mesh, &options);

        // Each face becomes 4 faces
        assert_eq!(mesh.num_faces(), original_faces * 4);
        assert!(mesh.is_valid());
    }

    #[test]
    fn test_loop_subdivide_two_iterations() {
        let mut mesh = create_tetrahedron();
        let original_faces = mesh.num_faces();

        let options = SubdivideOptions::new(2);
        loop_subdivide(&mut mesh, &options);

        // Each iteration quadruples: 4 * 4 = 16x
        assert_eq!(mesh.num_faces(), original_faces * 16);
        assert!(mesh.is_valid());
    }

    #[test]
    fn test_loop_subdivide_preserves_euler() {
        let mut mesh = create_tetrahedron();
        let original_euler =
            mesh.num_vertices() as i32 - (mesh.num_halfedges() / 2) as i32 + mesh.num_faces() as i32;

        let options = SubdivideOptions::new(1);
        loop_subdivide(&mut mesh, &options);

        let new_euler =
            mesh.num_vertices() as i32 - (mesh.num_halfedges() / 2) as i32 + mesh.num_faces() as i32;

        assert_eq!(original_euler, new_euler, "Euler characteristic should be preserved");
    }

    #[test]
    fn test_loop_subdivide_zero_iterations() {
        let mut mesh = create_tetrahedron();
        let original_faces = mesh.num_faces();
        let original_vertices = mesh.num_vertices();

        let options = SubdivideOptions::new(0);
        loop_subdivide(&mut mesh, &options);

        assert_eq!(mesh.num_faces(), original_faces);
        assert_eq!(mesh.num_vertices(), original_vertices);
    }

    #[test]
    fn test_loop_subdivide_two_triangles() {
        let mut mesh = create_two_triangles();

        let options = SubdivideOptions::new(1);
        loop_subdivide(&mut mesh, &options);

        // 2 triangles -> 8 triangles
        assert_eq!(mesh.num_faces(), 8);
        // 4 original + 5 edge vertices = 9
        assert_eq!(mesh.num_vertices(), 9);
        assert!(mesh.is_valid());
    }

    #[test]
    fn test_loop_beta_coefficient() {
        // Valence 3: β should be 3/16
        let beta3 = compute_loop_beta(3);
        assert!((beta3 - 3.0 / 16.0).abs() < 1e-10);

        // Valence 6 (regular): should be close to 1/16 (Warren's modification)
        let beta6 = compute_loop_beta(6);
        assert!(beta6 > 0.0 && beta6 < 0.2);
    }

    #[test]
    fn test_loop_subdivide_shrinks_closed_mesh() {
        // Loop subdivision is approximating, so closed meshes shrink toward their center
        let mut mesh = create_tetrahedron();

        // Compute original centroid
        let original_positions: Vec<Point3<f64>> =
            mesh.vertex_ids().map(|v| *mesh.position(v)).collect();
        let original_centroid: Vector3<f64> =
            original_positions.iter().map(|p| p.coords).sum::<Vector3<f64>>()
                / original_positions.len() as f64;

        let options = SubdivideOptions::new(2);
        loop_subdivide(&mut mesh, &options);

        // Compute new centroid
        let new_positions: Vec<Point3<f64>> =
            mesh.vertex_ids().map(|v| *mesh.position(v)).collect();
        let new_centroid: Vector3<f64> = new_positions.iter().map(|p| p.coords).sum::<Vector3<f64>>()
            / new_positions.len() as f64;

        // Centroids should be similar (subdivision doesn't drift the center much)
        assert!((new_centroid - original_centroid).norm() < 0.1);
    }

    #[test]
    fn test_interior_edge_vertex_position() {
        // Test that interior edge vertices are computed correctly
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
            Point3::new(1.0, 2.0, 0.0),
            Point3::new(1.0, -2.0, 0.0),
        ];
        let faces = vec![[0, 1, 2], [1, 0, 3]];

        let edge_info = build_edge_info(&faces);
        let edge_vertices = compute_edge_vertices(&vertices, &faces, &edge_info, true);

        // The shared edge (0, 1) should have an interior edge vertex
        let key = (0, 1);
        let info = edge_info.get(&key).unwrap();

        // Interior edge: 3/8 * (v0 + v1) + 1/8 * (v2 + v3)
        // = 3/8 * ((0,0,0) + (2,0,0)) + 1/8 * ((1,2,0) + (1,-2,0))
        // = 3/8 * (2,0,0) + 1/8 * (2,0,0)
        // = (0.75, 0, 0) + (0.25, 0, 0) = (1, 0, 0)
        let expected = Point3::new(1.0, 0.0, 0.0);
        let actual = edge_vertices[info.new_vertex_index];

        assert!((actual - expected).norm() < 1e-10);
    }
}
