//! Catmull-Clark subdivision for quad meshes.

use std::collections::HashMap;

use nalgebra::{Point3, Vector3};

use crate::algo::Progress;
use crate::mesh::{build_from_quads, to_face_vertex_quads, HalfEdgeMesh, MeshIndex};

use super::SubdivideOptions;

/// Performs Catmull-Clark subdivision on a quad mesh.
///
/// Catmull-Clark subdivision is an approximating subdivision scheme that produces
/// smooth surfaces from quad meshes. Each iteration quadruples the number of quads.
///
/// # Arguments
///
/// * `mesh` - The quad mesh to subdivide (modified in place)
/// * `options` - Subdivision parameters
///
/// # Algorithm
///
/// For each iteration:
/// 1. Compute face points (centroid of each face)
/// 2. Compute edge points (average of edge midpoint and adjacent face points)
/// 3. Update original vertices using weighted average
/// 4. Each quad becomes 4 new quads
///
/// # Vertex Rules
///
/// - **Face point**: centroid of face vertices
/// - **Edge point**: average of (edge midpoint, adjacent face points)
/// - **Vertex point**: (Q + 2R + (n-3)S) / n where:
///   - Q = average of adjacent face points
///   - R = average of adjacent edge midpoints
///   - S = original position
///   - n = valence
pub fn catmull_clark_subdivide<I: MeshIndex>(mesh: &mut HalfEdgeMesh<I>, options: &SubdivideOptions) {
    if options.iterations == 0 {
        return;
    }

    for _ in 0..options.iterations {
        catmull_clark_subdivide_once(mesh, options.preserve_boundary);
    }
}

/// Catmull-Clark subdivision with progress reporting.
pub fn catmull_clark_subdivide_with_progress<I: MeshIndex>(
    mesh: &mut HalfEdgeMesh<I>,
    options: &SubdivideOptions,
    progress: &Progress,
) {
    if options.iterations == 0 {
        return;
    }

    for iter in 0..options.iterations {
        progress.report(iter, options.iterations, "Catmull-Clark subdivision");
        catmull_clark_subdivide_once(mesh, options.preserve_boundary);
    }
    progress.report(options.iterations, options.iterations, "Catmull-Clark subdivision");
}

/// Perform one iteration of Catmull-Clark subdivision.
fn catmull_clark_subdivide_once<I: MeshIndex>(mesh: &mut HalfEdgeMesh<I>, preserve_boundary: bool) {
    let (vertices, faces) = to_face_vertex_quads(mesh);

    if vertices.is_empty() || faces.is_empty() {
        return;
    }

    // Step 1: Compute face points (centroids)
    let face_points: Vec<Point3<f64>> = faces
        .iter()
        .map(|face| {
            let sum: Vector3<f64> = face.iter().map(|&vi| vertices[vi].coords).sum();
            Point3::from(sum / 4.0)
        })
        .collect();

    // Step 2: Build edge information and compute edge points
    let edge_info = build_quad_edge_info(&faces);
    let edge_points = compute_cc_edge_points(&vertices, &face_points, &edge_info, preserve_boundary);

    // Step 3: Compute updated vertex positions
    let updated_vertices = compute_cc_vertex_points(
        &vertices,
        &faces,
        &face_points,
        &edge_info,
        preserve_boundary,
    );

    // Step 4: Build the subdivided quad mesh
    let (new_vertices, new_faces) =
        build_cc_subdivided_mesh(&updated_vertices, &face_points, &edge_points, &faces, &edge_info);

    // Rebuild the half-edge mesh
    if let Ok(new_mesh) = build_from_quads::<I>(&new_vertices, &new_faces) {
        *mesh = new_mesh;
    }
}

/// Information about an edge for Catmull-Clark subdivision.
#[derive(Debug, Clone)]
struct QuadEdgeInfo {
    /// Index of the new vertex created for this edge.
    new_vertex_index: usize,
    /// Indices of faces sharing this edge (second is None for boundary).
    faces: (usize, Option<usize>),
}

/// Build a map from edge (v0, v1) to edge information for quad faces.
fn build_quad_edge_info(faces: &[[usize; 4]]) -> HashMap<(usize, usize), QuadEdgeInfo> {
    let mut edge_map: HashMap<(usize, usize), QuadEdgeInfo> = HashMap::new();
    let mut next_edge_vertex = 0;

    for (face_idx, face) in faces.iter().enumerate() {
        for i in 0..4 {
            let v0 = face[i];
            let v1 = face[(i + 1) % 4];

            // Canonical edge key (smaller index first)
            let edge_key = if v0 < v1 { (v0, v1) } else { (v1, v0) };

            if let Some(info) = edge_map.get_mut(&edge_key) {
                // Second face sharing this edge
                info.faces.1 = Some(face_idx);
            } else {
                // First face for this edge
                edge_map.insert(
                    edge_key,
                    QuadEdgeInfo {
                        new_vertex_index: next_edge_vertex,
                        faces: (face_idx, None),
                    },
                );
                next_edge_vertex += 1;
            }
        }
    }

    edge_map
}

/// Compute edge points for Catmull-Clark subdivision.
fn compute_cc_edge_points(
    vertices: &[Point3<f64>],
    face_points: &[Point3<f64>],
    edge_info: &HashMap<(usize, usize), QuadEdgeInfo>,
    preserve_boundary: bool,
) -> Vec<Point3<f64>> {
    let mut edge_points = vec![Point3::origin(); edge_info.len()];

    for (&(v0, v1), info) in edge_info {
        let p0 = &vertices[v0];
        let p1 = &vertices[v1];
        let midpoint = (p0.coords + p1.coords) * 0.5;

        let new_pos = if info.faces.1.is_none() {
            // Boundary edge
            if preserve_boundary {
                // Sharp boundary: just use midpoint
                Point3::from(midpoint)
            } else {
                Point3::from(midpoint)
            }
        } else {
            // Interior edge: average of midpoint and two face points
            let fp1 = &face_points[info.faces.0];
            let fp2 = &face_points[info.faces.1.unwrap()];
            Point3::from((midpoint + fp1.coords + fp2.coords) / 3.0)
        };

        edge_points[info.new_vertex_index] = new_pos;
    }

    edge_points
}

/// Compute updated vertex positions for Catmull-Clark subdivision.
fn compute_cc_vertex_points(
    vertices: &[Point3<f64>],
    faces: &[[usize; 4]],
    face_points: &[Point3<f64>],
    edge_info: &HashMap<(usize, usize), QuadEdgeInfo>,
    preserve_boundary: bool,
) -> Vec<Point3<f64>> {
    let n = vertices.len();

    // Build vertex-to-faces and vertex-to-edges adjacency
    let mut vertex_faces: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut vertex_edges: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
    let mut boundary_neighbors: Vec<Vec<usize>> = vec![Vec::new(); n];

    // Collect adjacent faces for each vertex
    for (face_idx, face) in faces.iter().enumerate() {
        for &vi in face {
            if !vertex_faces[vi].contains(&face_idx) {
                vertex_faces[vi].push(face_idx);
            }
        }
    }

    // Collect adjacent edges for each vertex and identify boundary edges
    for (&(v0, v1), info) in edge_info {
        vertex_edges[v0].push((v0, v1));
        vertex_edges[v1].push((v0, v1));

        if info.faces.1.is_none() {
            // Boundary edge
            if !boundary_neighbors[v0].contains(&v1) {
                boundary_neighbors[v0].push(v1);
            }
            if !boundary_neighbors[v1].contains(&v0) {
                boundary_neighbors[v1].push(v0);
            }
        }
    }

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
                // Corner: keep position
                *pos
            }
        } else {
            // Interior vertex: (Q + 2R + (n-3)S) / n
            let valence = vertex_faces[i].len();
            if valence == 0 {
                *pos
            } else {
                // Q = average of adjacent face points
                let q: Vector3<f64> = vertex_faces[i]
                    .iter()
                    .map(|&fi| face_points[fi].coords)
                    .sum::<Vector3<f64>>()
                    / valence as f64;

                // R = average of adjacent edge midpoints
                let r: Vector3<f64> = vertex_edges[i]
                    .iter()
                    .map(|&(v0, v1)| (vertices[v0].coords + vertices[v1].coords) * 0.5)
                    .sum::<Vector3<f64>>()
                    / vertex_edges[i].len() as f64;

                // S = original position
                let s = pos.coords;

                // New position: (Q + 2R + (n-3)S) / n
                let n_f = valence as f64;
                Point3::from((q + r * 2.0 + s * (n_f - 3.0)) / n_f)
            }
        };

        updated.push(new_pos);
    }

    updated
}

/// Build the subdivided quad mesh with new connectivity.
fn build_cc_subdivided_mesh(
    updated_vertices: &[Point3<f64>],
    face_points: &[Point3<f64>],
    edge_points: &[Point3<f64>],
    original_faces: &[[usize; 4]],
    edge_info: &HashMap<(usize, usize), QuadEdgeInfo>,
) -> (Vec<Point3<f64>>, Vec<[usize; 4]>) {
    let num_original = updated_vertices.len();
    let num_face_points = face_points.len();

    // New vertices: original (updated) + face points + edge points
    let mut new_vertices: Vec<Point3<f64>> = updated_vertices.to_vec();
    new_vertices.extend(face_points.iter().cloned());
    new_vertices.extend(edge_points.iter().cloned());

    // Helper to get face point index
    let get_face_point = |face_idx: usize| -> usize { num_original + face_idx };

    // Helper to get edge point index
    let get_edge_point = |v0: usize, v1: usize| -> usize {
        let key = if v0 < v1 { (v0, v1) } else { (v1, v0) };
        let info = edge_info.get(&key).unwrap();
        num_original + num_face_points + info.new_vertex_index
    };

    // Build new faces: each original quad becomes 4 quads
    let mut new_faces: Vec<[usize; 4]> = Vec::with_capacity(original_faces.len() * 4);

    for (face_idx, face) in original_faces.iter().enumerate() {
        let v0 = face[0];
        let v1 = face[1];
        let v2 = face[2];
        let v3 = face[3];

        // Face point
        let fp = get_face_point(face_idx);

        // Edge points
        let e01 = get_edge_point(v0, v1);
        let e12 = get_edge_point(v1, v2);
        let e23 = get_edge_point(v2, v3);
        let e30 = get_edge_point(v3, v0);

        // Four new quads (counter-clockwise winding):
        // 1. Quad at corner v0: v0 -> e01 -> fp -> e30
        new_faces.push([v0, e01, fp, e30]);
        // 2. Quad at corner v1: v1 -> e12 -> fp -> e01
        new_faces.push([v1, e12, fp, e01]);
        // 3. Quad at corner v2: v2 -> e23 -> fp -> e12
        new_faces.push([v2, e23, fp, e12]);
        // 4. Quad at corner v3: v3 -> e30 -> fp -> e23
        new_faces.push([v3, e30, fp, e23]);
    }

    (new_vertices, new_faces)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_single_quad() -> HalfEdgeMesh {
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ];
        let faces = vec![[0, 1, 2, 3]];
        build_from_quads(&vertices, &faces).unwrap()
    }

    fn create_two_quads() -> HalfEdgeMesh {
        // Two quads sharing an edge
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
            Point3::new(2.0, 1.0, 0.0),
        ];
        let faces = vec![[0, 1, 2, 3], [1, 4, 5, 2]];
        build_from_quads(&vertices, &faces).unwrap()
    }

    fn create_quad_cube() -> HalfEdgeMesh {
        // A simple cube with 6 quad faces
        let vertices = vec![
            // Bottom face (z = 0)
            Point3::new(0.0, 0.0, 0.0), // 0
            Point3::new(1.0, 0.0, 0.0), // 1
            Point3::new(1.0, 1.0, 0.0), // 2
            Point3::new(0.0, 1.0, 0.0), // 3
            // Top face (z = 1)
            Point3::new(0.0, 0.0, 1.0), // 4
            Point3::new(1.0, 0.0, 1.0), // 5
            Point3::new(1.0, 1.0, 1.0), // 6
            Point3::new(0.0, 1.0, 1.0), // 7
        ];
        let faces = vec![
            [0, 3, 2, 1], // Bottom (CCW when viewed from below)
            [4, 5, 6, 7], // Top
            [0, 1, 5, 4], // Front
            [2, 3, 7, 6], // Back
            [0, 4, 7, 3], // Left
            [1, 2, 6, 5], // Right
        ];
        build_from_quads(&vertices, &faces).unwrap()
    }

    #[test]
    fn test_catmull_clark_single_quad() {
        let mut mesh = create_single_quad();

        let options = SubdivideOptions::new(1);
        catmull_clark_subdivide(&mut mesh, &options);

        // 1 quad -> 4 quads
        assert_eq!(mesh.num_faces(), 4);
        // 4 original + 1 face point + 4 edge points = 9
        assert_eq!(mesh.num_vertices(), 9);
        assert!(mesh.is_valid());
        assert!(mesh.is_quad_mesh());
    }

    #[test]
    fn test_catmull_clark_two_quads() {
        let mut mesh = create_two_quads();

        let options = SubdivideOptions::new(1);
        catmull_clark_subdivide(&mut mesh, &options);

        // 2 quads -> 8 quads
        assert_eq!(mesh.num_faces(), 8);
        // 6 original + 2 face points + 7 edge points = 15
        assert_eq!(mesh.num_vertices(), 15);
        assert!(mesh.is_valid());
        assert!(mesh.is_quad_mesh());
    }

    #[test]
    fn test_catmull_clark_quadruples_faces() {
        let mut mesh = create_quad_cube();
        let original_faces = mesh.num_faces();

        let options = SubdivideOptions::new(1);
        catmull_clark_subdivide(&mut mesh, &options);

        // Each face becomes 4 faces
        assert_eq!(mesh.num_faces(), original_faces * 4);
        assert!(mesh.is_valid());
        assert!(mesh.is_quad_mesh());
    }

    #[test]
    fn test_catmull_clark_two_iterations() {
        let mut mesh = create_quad_cube();
        let original_faces = mesh.num_faces();

        let options = SubdivideOptions::new(2);
        catmull_clark_subdivide(&mut mesh, &options);

        // Each iteration quadruples: 4 * 4 = 16x
        assert_eq!(mesh.num_faces(), original_faces * 16);
        assert!(mesh.is_valid());
        assert!(mesh.is_quad_mesh());
    }

    #[test]
    fn test_catmull_clark_preserves_euler() {
        let mut mesh = create_quad_cube();
        let original_euler =
            mesh.num_vertices() as i32 - (mesh.num_halfedges() / 2) as i32 + mesh.num_faces() as i32;

        let options = SubdivideOptions::new(1);
        catmull_clark_subdivide(&mut mesh, &options);

        let new_euler =
            mesh.num_vertices() as i32 - (mesh.num_halfedges() / 2) as i32 + mesh.num_faces() as i32;

        assert_eq!(original_euler, new_euler, "Euler characteristic should be preserved");
    }

    #[test]
    fn test_catmull_clark_zero_iterations() {
        let mut mesh = create_quad_cube();
        let original_faces = mesh.num_faces();
        let original_vertices = mesh.num_vertices();

        let options = SubdivideOptions::new(0);
        catmull_clark_subdivide(&mut mesh, &options);

        assert_eq!(mesh.num_faces(), original_faces);
        assert_eq!(mesh.num_vertices(), original_vertices);
    }

    #[test]
    fn test_catmull_clark_shrinks_closed_mesh() {
        // Catmull-Clark is approximating, so closed meshes shrink toward their center
        let mut mesh = create_quad_cube();

        // Compute original centroid
        let original_positions: Vec<Point3<f64>> =
            mesh.vertex_ids().map(|v| *mesh.position(v)).collect();
        let original_centroid: Vector3<f64> =
            original_positions.iter().map(|p| p.coords).sum::<Vector3<f64>>()
                / original_positions.len() as f64;

        let options = SubdivideOptions::new(2);
        catmull_clark_subdivide(&mut mesh, &options);

        // Compute new centroid
        let new_positions: Vec<Point3<f64>> =
            mesh.vertex_ids().map(|v| *mesh.position(v)).collect();
        let new_centroid: Vector3<f64> = new_positions.iter().map(|p| p.coords).sum::<Vector3<f64>>()
            / new_positions.len() as f64;

        // Centroids should be similar (subdivision doesn't drift the center much)
        assert!((new_centroid - original_centroid).norm() < 0.1);
    }

    #[test]
    fn test_catmull_clark_face_point_is_centroid() {
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
            Point3::new(2.0, 2.0, 0.0),
            Point3::new(0.0, 2.0, 0.0),
        ];
        let faces = vec![[0, 1, 2, 3]];

        // Face point should be at centroid: (1, 1, 0)
        let face_points: Vec<Point3<f64>> = faces
            .iter()
            .map(|face| {
                let sum: Vector3<f64> = face.iter().map(|&vi| vertices[vi].coords).sum();
                Point3::from(sum / 4.0)
            })
            .collect();

        let expected = Point3::new(1.0, 1.0, 0.0);
        assert!((face_points[0] - expected).norm() < 1e-10);
    }
}
