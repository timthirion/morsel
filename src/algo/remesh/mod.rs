//! Mesh remeshing algorithms.
//!
//! This module provides algorithms for remeshing triangle meshes:
//!
//! - **Isotropic remeshing**: Uniform edge lengths throughout the mesh
//! - **Anisotropic remeshing**: Edge lengths adapted to local curvature
//! - **CVT remeshing**: Centroidal Voronoi Tessellation for optimal vertex distribution
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
//! # CVT Remeshing
//!
//! CVT remeshing uses Lloyd's algorithm to optimize vertex positions:
//! - Creates well-shaped, nearly equilateral triangles
//! - Vertices are distributed according to a density function
//! - Iteratively moves vertices to Voronoi cell centroids
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
//! - Du, Q., et al. (1999). "Centroidal Voronoi tessellations."
//!   SIAM Review.

mod anisotropic;
mod cvt;
mod isotropic;

pub use anisotropic::{
    anisotropic_remesh, anisotropic_remesh_with_progress, compute_sizing_field, AnisotropicOptions,
    SizingField,
};
pub use cvt::{cvt_remesh, cvt_remesh_with_progress, CvtOptions};
pub use isotropic::{
    average_edge_length, isotropic_remesh, isotropic_remesh_with_progress, RemeshOptions,
};

use std::collections::HashSet;

use nalgebra::{Point3, Vector3};

use crate::mesh::{build_from_triangles, to_face_vertex, HalfEdgeMesh, MeshIndex};

// ============================================================================
// Shared Helpers - Edge/Face Queries
// ============================================================================

/// Check if an edge is on the boundary (appears in only one face).
pub(crate) fn is_boundary_edge_in_faces(faces: &[[usize; 3]], v0: usize, v1: usize) -> bool {
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

/// Check if a vertex is on the boundary.
pub(crate) fn is_boundary_vertex_in_faces(faces: &[[usize; 3]], v: usize) -> bool {
    for &neighbor in &get_vertex_neighbors(faces, v) {
        if is_boundary_edge_in_faces(faces, v, neighbor) {
            return true;
        }
    }
    false
}

/// Check if an edge exists in the face list.
pub(crate) fn edge_exists_in_faces(faces: &[[usize; 3]], v0: usize, v1: usize) -> bool {
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

/// Get all vertex neighbors from the face list.
pub(crate) fn get_vertex_neighbors(faces: &[[usize; 3]], v: usize) -> HashSet<usize> {
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

// ============================================================================
// Shared Helpers - Edge Operations
// ============================================================================

/// Split an edge by inserting a vertex at its midpoint.
pub(crate) fn split_edge(
    vertices: &mut Vec<Point3<f64>>,
    faces: &mut Vec<[usize; 3]>,
    v0: usize,
    v1: usize,
) {
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

/// Collapse an edge by merging v1 into v0.
pub(crate) fn collapse_edge(
    vertices: &mut Vec<Point3<f64>>,
    faces: &mut Vec<[usize; 3]>,
    v0: usize,
    v1: usize,
) {
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
    faces.retain(|face| face[0] != face[1] && face[1] != face[2] && face[0] != face[2]);
}

/// Remove unused vertices and reindex faces.
pub(crate) fn cleanup_mesh(
    vertices: &[Point3<f64>],
    faces: &[[usize; 3]],
) -> (Vec<Point3<f64>>, Vec<[usize; 3]>) {
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
// Shared Helpers - Edge Flipping
// ============================================================================

/// Validate a face list for manifold properties.
pub(crate) fn validate_face_list(vertices: &[Point3<f64>], faces: &[[usize; 3]]) -> bool {
    // Check vertex indices
    for face in faces {
        for &vi in face {
            if vi >= vertices.len() {
                return false;
            }
        }
    }

    // Check for non-manifold edges (>2 faces sharing an edge)
    let mut edge_counts: std::collections::HashMap<(usize, usize), usize> =
        std::collections::HashMap::new();
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
    let mut directed_counts: std::collections::HashMap<(usize, usize), usize> =
        std::collections::HashMap::new();
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
pub(crate) fn flip_edges_for_valence_faces(
    vertices: &[Point3<f64>],
    faces: &mut Vec<[usize; 3]>,
    preserve_boundary: bool,
) {
    let max_iterations = faces.len() * 3; // Prevent infinite loops
    let mut failed_edges: HashSet<(usize, usize)> = HashSet::new();

    for _iteration in 0..max_iterations {
        // Find the first edge that should be flipped (excluding failed ones)
        let edge_to_flip =
            find_edge_to_flip_excluding(vertices, faces, preserve_boundary, &failed_edges);

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
fn is_convex_quad(
    p0: &Point3<f64>,
    p1: &Point3<f64>,
    p2: &Point3<f64>,
    p3: &Point3<f64>,
) -> bool {
    let v01 = p1 - p0;
    let v12 = p2 - p1;
    let v23 = p3 - p2;
    let v30 = p0 - p3;

    let n0 = v01.cross(&(-v30));
    let n1 = v12.cross(&(-v01));
    let n2 = v23.cross(&(-v12));
    let n3 = v30.cross(&(-v23));

    let d01 = n0.dot(&n1);
    let d12 = n1.dot(&n2);
    let d23 = n2.dot(&n3);

    d01 > 0.0 && d12 > 0.0 && d23 > 0.0
}

/// Flip an edge in the face list.
pub(crate) fn flip_edge(faces: &mut Vec<[usize; 3]>, v0: usize, v1: usize) -> bool {
    // Find the two faces sharing this edge
    let mut face_info: Vec<(usize, usize)> = Vec::new();

    for (idx, face) in faces.iter().enumerate() {
        for i in 0..3 {
            let a = face[i];
            let b = face[(i + 1) % 3];
            if (a == v0 && b == v1) || (a == v1 && b == v0) {
                face_info.push((idx, i));
                break;
            }
        }
    }

    if face_info.len() != 2 {
        return false;
    }

    let (idx0, edge_idx0) = face_info[0];
    let (idx1, edge_idx1) = face_info[1];

    let face0 = faces[idx0];
    let face1 = faces[idx1];

    let opp0 = face0[(edge_idx0 + 2) % 3];
    let opp1 = face1[(edge_idx1 + 2) % 3];

    let a0 = face0[edge_idx0];

    if a0 == v0 {
        faces[idx0] = [opp0, opp1, v0];
        faces[idx1] = [opp1, opp0, v1];
    } else {
        faces[idx0] = [opp0, opp1, v1];
        faces[idx1] = [opp1, opp0, v0];
    }

    true
}

// ============================================================================
// Shared Helpers - Smoothing
// ============================================================================

/// Apply tangential smoothing to regularize vertex positions.
pub(crate) fn tangential_smooth<I: MeshIndex>(
    mesh: &mut HalfEdgeMesh<I>,
    lambda: f64,
    preserve_boundary: bool,
) {
    let (vertices, faces) = to_face_vertex(mesh);

    let normals = compute_vertex_normals_from_faces(&vertices, &faces);
    let boundary = compute_boundary_vertices(&faces, vertices.len());
    let neighbors = build_vertex_neighbors(&faces, vertices.len());

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

        let mut centroid = Vector3::zeros();
        for &neighbor in vertex_neighbors {
            centroid += vertices[neighbor].coords;
        }
        centroid /= vertex_neighbors.len() as f64;

        let displacement = centroid - pos.coords;
        let normal = &normals[idx];
        let tangent_displacement = displacement - normal.dot(&displacement) * normal;

        new_positions.push(Point3::from(pos.coords + lambda * tangent_displacement));
    }

    if let Ok(new_mesh) = build_from_triangles::<I>(&new_positions, &faces) {
        *mesh = new_mesh;
    }
}

/// Compute vertex normals from face data.
pub(crate) fn compute_vertex_normals_from_faces(
    vertices: &[Point3<f64>],
    faces: &[[usize; 3]],
) -> Vec<Vector3<f64>> {
    let mut normals: Vec<Vector3<f64>> = vec![Vector3::zeros(); vertices.len()];

    for face in faces {
        let p0 = &vertices[face[0]];
        let p1 = &vertices[face[1]];
        let p2 = &vertices[face[2]];

        let e1 = p1 - p0;
        let e2 = p2 - p0;
        let face_normal = e1.cross(&e2);

        normals[face[0]] += face_normal;
        normals[face[1]] += face_normal;
        normals[face[2]] += face_normal;
    }

    for n in &mut normals {
        let len = n.norm();
        if len > 1e-10 {
            *n /= len;
        }
    }

    normals
}

/// Compute which vertices are on the boundary.
pub(crate) fn compute_boundary_vertices(faces: &[[usize; 3]], num_vertices: usize) -> Vec<bool> {
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
pub(crate) fn build_vertex_neighbors(
    faces: &[[usize; 3]],
    num_vertices: usize,
) -> Vec<Vec<usize>> {
    let mut neighbors: Vec<HashSet<usize>> = vec![HashSet::new(); num_vertices];

    for face in faces {
        neighbors[face[0]].insert(face[1]);
        neighbors[face[0]].insert(face[2]);
        neighbors[face[1]].insert(face[0]);
        neighbors[face[1]].insert(face[2]);
        neighbors[face[2]].insert(face[0]);
        neighbors[face[2]].insert(face[1]);
    }

    neighbors
        .into_iter()
        .map(|s| s.into_iter().collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn create_tetrahedron() -> HalfEdgeMesh {
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.5, 1.0, 0.0),
            Point3::new(0.5, 0.5, 1.0),
        ];
        let faces = vec![[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]];
        build_from_triangles(&vertices, &faces).unwrap()
    }

    pub(crate) fn create_grid_mesh(n: usize) -> HalfEdgeMesh {
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
    fn test_edge_split() {
        let mut vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
            Point3::new(1.0, 2.0, 0.0),
        ];
        let mut faces = vec![[0, 1, 2]];

        split_edge(&mut vertices, &mut faces, 0, 1);

        assert_eq!(vertices.len(), 4);
        assert_eq!(faces.len(), 2);

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
        let faces = vec![[0, 2, 3]];

        let (clean_verts, clean_faces) = cleanup_mesh(&vertices, &faces);

        assert_eq!(clean_verts.len(), 3);
        assert_eq!(clean_faces.len(), 1);
    }

    #[test]
    fn test_edge_flip_two_triangles() {
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(1.0, -1.0, 0.0),
        ];
        let mut faces = vec![[0, 1, 2], [1, 0, 3]];

        let flipped = flip_edge(&mut faces, 0, 1);
        assert!(flipped, "Edge should be flippable");

        assert_eq!(faces.len(), 2);

        let mut seen_edges: HashSet<(usize, usize)> = HashSet::new();
        for face in &faces {
            for i in 0..3 {
                let v0 = face[i];
                let v1 = face[(i + 1) % 3];
                let edge = if v0 < v1 { (v0, v1) } else { (v1, v0) };
                seen_edges.insert(edge);
            }
        }

        assert_eq!(seen_edges.len(), 5);
        assert!(!seen_edges.contains(&(0, 1)));
        assert!(seen_edges.contains(&(2, 3)));

        let result = build_from_triangles::<u32>(&vertices, &faces);
        assert!(result.is_ok());
        assert!(result.unwrap().is_valid());
    }

    #[test]
    fn test_edge_flip_valence_check() {
        let mesh = create_grid_mesh(2);
        let (vertices, mut faces) = to_face_vertex(&mesh);

        flip_edges_for_valence_faces(&vertices, &mut faces, true);

        let result = build_from_triangles::<u32>(&vertices, &faces);
        assert!(result.is_ok());
    }
}
