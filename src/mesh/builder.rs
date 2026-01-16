//! Mesh construction utilities.
//!
//! This module provides functions for building half-edge meshes from
//! various input formats, primarily face-vertex lists as commonly found
//! in mesh file formats.

use std::collections::HashMap;

use nalgebra::Point3;

use super::halfedge::HalfEdgeMesh;
use super::index::{FaceId, HalfEdgeId, MeshIndex, VertexId};
use crate::error::{MeshError, Result};

/// Build a half-edge mesh from vertices and triangle faces.
///
/// # Arguments
/// * `vertices` - List of vertex positions
/// * `faces` - List of triangle faces, each as [v0, v1, v2] indices
///
/// # Returns
/// A half-edge mesh, or an error if the input is invalid.
///
/// # Example
/// ```
/// use morsel::mesh::{build_from_triangles, HalfEdgeMesh};
/// use nalgebra::Point3;
///
/// let vertices = vec![
///     Point3::new(0.0, 0.0, 0.0),
///     Point3::new(1.0, 0.0, 0.0),
///     Point3::new(0.5, 1.0, 0.0),
/// ];
/// let faces = vec![[0, 1, 2]];
///
/// let mesh: HalfEdgeMesh = build_from_triangles(&vertices, &faces).unwrap();
/// assert_eq!(mesh.num_vertices(), 3);
/// assert_eq!(mesh.num_faces(), 1);
/// ```
pub fn build_from_triangles<I: MeshIndex>(
    vertices: &[Point3<f64>],
    faces: &[[usize; 3]],
) -> Result<HalfEdgeMesh<I>> {
    if faces.is_empty() {
        return Err(MeshError::EmptyMesh);
    }

    // Validate vertex indices
    for (fi, face) in faces.iter().enumerate() {
        for &vi in face {
            if vi >= vertices.len() {
                return Err(MeshError::InvalidVertexIndex { face: fi, vertex: vi });
            }
        }
        // Check for degenerate faces
        if face[0] == face[1] || face[1] == face[2] || face[0] == face[2] {
            return Err(MeshError::DegenerateFace { face: fi });
        }
    }

    let mut mesh = HalfEdgeMesh::with_capacity(vertices.len(), faces.len());

    // Add vertices
    let vertex_ids: Vec<VertexId<I>> = vertices
        .iter()
        .map(|&pos| mesh.add_vertex(pos))
        .collect();

    // Map from directed edge (v0, v1) to half-edge ID
    let mut edge_map: HashMap<(usize, usize), HalfEdgeId<I>> = HashMap::new();

    // First pass: create all half-edges and faces
    for face in faces {
        let v0 = face[0];
        let v1 = face[1];
        let v2 = face[2];

        // Create three half-edges for this face
        let he0 = HalfEdgeId::<I>::new(mesh.num_halfedges());
        let he1 = HalfEdgeId::<I>::new(mesh.num_halfedges() + 1);
        let he2 = HalfEdgeId::<I>::new(mesh.num_halfedges() + 2);

        // Add half-edges to mesh storage
        for _ in 0..3 {
            mesh.halfedges.push(super::halfedge::HalfEdge::new());
        }

        // Create face
        let face_id = FaceId::<I>::new(mesh.num_faces());
        mesh.faces.push(super::halfedge::Face::new(he0));

        // Set up half-edge connectivity within the face
        {
            let he = mesh.halfedge_mut(he0);
            he.origin = vertex_ids[v0];
            he.next = he1;
            he.prev = he2;
            he.face = face_id;
        }
        {
            let he = mesh.halfedge_mut(he1);
            he.origin = vertex_ids[v1];
            he.next = he2;
            he.prev = he0;
            he.face = face_id;
        }
        {
            let he = mesh.halfedge_mut(he2);
            he.origin = vertex_ids[v2];
            he.next = he0;
            he.prev = he1;
            he.face = face_id;
        }

        // Set vertex half-edges (will be overwritten for shared vertices)
        mesh.vertex_mut(vertex_ids[v0]).halfedge = he0;
        mesh.vertex_mut(vertex_ids[v1]).halfedge = he1;
        mesh.vertex_mut(vertex_ids[v2]).halfedge = he2;

        // Record edges for twin linking
        edge_map.insert((v0, v1), he0);
        edge_map.insert((v1, v2), he1);
        edge_map.insert((v2, v0), he2);
    }

    // Second pass: link twins
    for (&(v0, v1), &he) in &edge_map {
        if let Some(&twin) = edge_map.get(&(v1, v0)) {
            mesh.halfedge_mut(he).twin = twin;
        } else {
            // Boundary edge - create boundary half-edge
            let boundary_he = HalfEdgeId::<I>::new(mesh.num_halfedges());
            mesh.halfedges.push(super::halfedge::HalfEdge::new());

            mesh.halfedge_mut(he).twin = boundary_he;
            {
                let bhe = mesh.halfedge_mut(boundary_he);
                bhe.origin = vertex_ids[v1];
                bhe.twin = he;
                // Face is invalid (boundary)
            }
        }
    }

    // Third pass: link boundary half-edges into loops
    link_boundary_loops(&mut mesh)?;

    // Fourth pass: ensure boundary vertices point to boundary half-edges
    fix_boundary_vertex_halfedges(&mut mesh);

    Ok(mesh)
}

/// Link boundary half-edges into proper loops.
fn link_boundary_loops<I: MeshIndex>(mesh: &mut HalfEdgeMesh<I>) -> Result<()> {
    // Find all boundary half-edges
    let boundary_hes: Vec<HalfEdgeId<I>> = mesh
        .halfedge_ids()
        .filter(|&he| mesh.is_boundary_halfedge(he))
        .collect();

    // Group by origin vertex for quick lookup
    let mut outgoing: HashMap<usize, HalfEdgeId<I>> = HashMap::new();
    for he in &boundary_hes {
        let origin = mesh.origin(*he).index();
        outgoing.insert(origin, *he);
    }

    // Link next/prev for boundary half-edges
    for &he in &boundary_hes {
        // The next boundary half-edge starts where this one ends
        let dest = mesh.dest(he).index();
        if let Some(&next_he) = outgoing.get(&dest) {
            mesh.halfedge_mut(he).next = next_he;
            mesh.halfedge_mut(next_he).prev = he;
        }
    }

    Ok(())
}

/// Ensure boundary vertices point to a boundary half-edge.
fn fix_boundary_vertex_halfedges<I: MeshIndex>(mesh: &mut HalfEdgeMesh<I>) {
    for vid in mesh.vertex_ids().collect::<Vec<_>>() {
        let start_he = mesh.vertex(vid).halfedge;
        if !start_he.is_valid() {
            continue;
        }

        // Walk around the vertex to find a boundary half-edge
        // Uses the same iteration pattern as VertexHalfEdgeIter: twin -> next
        let mut he = start_he;
        loop {
            if mesh.is_boundary_halfedge(he) {
                mesh.vertex_mut(vid).halfedge = he;
                break;
            }
            he = mesh.next(mesh.twin(he));
            if he == start_he {
                break;
            }
        }
    }
}

/// Convert a half-edge mesh back to a face-vertex representation.
///
/// Returns (vertices, faces) tuple.
pub fn to_face_vertex<I: MeshIndex>(mesh: &HalfEdgeMesh<I>) -> (Vec<Point3<f64>>, Vec<[usize; 3]>) {
    let vertices: Vec<Point3<f64>> = mesh.vertex_ids().map(|v| *mesh.position(v)).collect();

    let faces: Vec<[usize; 3]> = mesh
        .face_ids()
        .map(|f| {
            let [v0, v1, v2] = mesh.face_triangle(f);
            [v0.index(), v1.index(), v2.index()]
        })
        .collect();

    (vertices, faces)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn single_triangle() -> (Vec<Point3<f64>>, Vec<[usize; 3]>) {
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.5, 1.0, 0.0),
        ];
        let faces = vec![[0, 1, 2]];
        (vertices, faces)
    }

    fn two_triangles() -> (Vec<Point3<f64>>, Vec<[usize; 3]>) {
        // Two triangles sharing an edge
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.5, 1.0, 0.0),
            Point3::new(0.5, -1.0, 0.0),
        ];
        let faces = vec![[0, 1, 2], [1, 0, 3]];
        (vertices, faces)
    }

    #[test]
    fn test_single_triangle() {
        let (vertices, faces) = single_triangle();
        let mesh: HalfEdgeMesh<u32> = build_from_triangles(&vertices, &faces).unwrap();

        assert_eq!(mesh.num_vertices(), 3);
        assert_eq!(mesh.num_faces(), 1);
        // 3 interior half-edges + 3 boundary half-edges
        assert_eq!(mesh.num_halfedges(), 6);
        assert!(mesh.is_valid());

        // All vertices should be on boundary
        for v in mesh.vertex_ids() {
            assert!(mesh.is_boundary_vertex(v));
        }
    }

    #[test]
    fn test_two_triangles() {
        let (vertices, faces) = two_triangles();
        let mesh: HalfEdgeMesh<u32> = build_from_triangles(&vertices, &faces).unwrap();

        assert_eq!(mesh.num_vertices(), 4);
        assert_eq!(mesh.num_faces(), 2);
        // 6 interior half-edges + 4 boundary half-edges
        assert_eq!(mesh.num_halfedges(), 10);
        assert!(mesh.is_valid());
    }

    #[test]
    fn test_roundtrip() {
        let (vertices, faces) = two_triangles();
        let mesh: HalfEdgeMesh<u32> = build_from_triangles(&vertices, &faces).unwrap();

        let (out_verts, out_faces) = to_face_vertex(&mesh);

        assert_eq!(vertices.len(), out_verts.len());
        assert_eq!(faces.len(), out_faces.len());

        // Positions should match
        for (v_in, v_out) in vertices.iter().zip(out_verts.iter()) {
            assert!((v_in - v_out).norm() < 1e-10);
        }
    }

    #[test]
    fn test_invalid_vertex_index() {
        let vertices = vec![Point3::new(0.0, 0.0, 0.0)];
        let faces = vec![[0, 1, 2]]; // Indices 1 and 2 are invalid

        let result: Result<HalfEdgeMesh<u32>> = build_from_triangles(&vertices, &faces);
        assert!(result.is_err());
    }

    #[test]
    fn test_degenerate_face() {
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.5, 1.0, 0.0),
        ];
        let faces = vec![[0, 0, 2]]; // Degenerate: v0 == v1

        let result: Result<HalfEdgeMesh<u32>> = build_from_triangles(&vertices, &faces);
        assert!(result.is_err());
    }
}
