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
///
/// A half-edge mesh, or an error if the input is invalid.
///
/// # Validation
///
/// A half-edge mesh can only represent an *orientable manifold* surface, so input
/// that isn't one is rejected rather than built. Without these checks the builder
/// returned `Ok` with a structurally broken mesh — a half-edge whose `twin` was
/// never assigned, because `edge_map.insert` silently overwrote the earlier face's
/// entry — and every algorithm downstream inherited the corruption. Rejected:
///
/// - a vertex index out of range, or a face repeating a vertex;
/// - two faces traversing the same edge in the *same* direction, which means they
///   are duplicates or inconsistently oriented (in a consistently oriented mesh
///   the two faces on an interior edge traverse it in opposite directions);
/// - an edge with more than two incident faces;
/// - a vertex whose incident faces do not form a single fan or cycle — a
///   "bowtie". Circulating such a vertex cannot reach all of its faces, so the
///   structure has no valid representation for it.
///
/// Note what is *not* rejected: a mesh may be disconnected, may have several
/// boundary loops, may contain zero-area or sliver faces, and may have vertices
/// no face references. Those are all representable.
///
/// # Errors
///
/// [`MeshError::EmptyMesh`], [`MeshError::InvalidVertexIndex`],
/// [`MeshError::DegenerateFace`], [`MeshError::NonManifoldEdge`], or
/// [`MeshError::NonManifold`].
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

    validate_manifold_soup(vertices.len(), faces)?;

    let mut mesh = HalfEdgeMesh::with_capacity(vertices.len(), faces.len());

    // Add vertices
    let vertex_ids: Vec<VertexId<I>> = vertices.iter().map(|&pos| mesh.add_vertex(pos)).collect();

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

/// Reject face-vertex input that a half-edge mesh cannot represent.
///
/// See [`build_from_triangles`] for what is and isn't rejected, and why.
fn validate_manifold_soup(num_vertices: usize, faces: &[[usize; 3]]) -> Result<()> {
    // ---- per-face sanity ------------------------------------------------
    for (fi, face) in faces.iter().enumerate() {
        for &vi in face {
            if vi >= num_vertices {
                return Err(MeshError::InvalidVertexIndex {
                    face: fi,
                    vertex: vi,
                });
            }
        }
        if face[0] == face[1] || face[1] == face[2] || face[0] == face[2] {
            return Err(MeshError::DegenerateFace { face: fi });
        }
    }

    // ---- edges ----------------------------------------------------------
    // A directed edge may be used once. Two faces traversing `a -> b` the same
    // way are duplicates or disagree about orientation; either way the twin
    // linking below has nowhere to put the second one.
    let mut directed: HashMap<(usize, usize), usize> = HashMap::with_capacity(faces.len() * 3);
    // Undirected edge -> the (at most two) faces on it.
    let mut undirected: HashMap<(usize, usize), [Option<usize>; 2]> =
        HashMap::with_capacity(faces.len() * 3);

    for (fi, face) in faces.iter().enumerate() {
        for k in 0..3 {
            let (a, b) = (face[k], face[(k + 1) % 3]);

            if let Some(&other) = directed.get(&(a, b)) {
                return Err(MeshError::NonManifold {
                    details: format!(
                        "faces {other} and {fi} both traverse edge ({a} -> {b}) in the same \
                         direction, so they are duplicates or inconsistently oriented"
                    ),
                });
            }
            directed.insert((a, b), fi);

            let key = if a < b { (a, b) } else { (b, a) };
            let slot = undirected.entry(key).or_insert([None, None]);
            match slot {
                [None, _] => slot[0] = Some(fi),
                [Some(_), None] => slot[1] = Some(fi),
                _ => {
                    return Err(MeshError::NonManifoldEdge {
                        v0: key.0,
                        v1: key.1,
                    })
                }
            }
        }
    }

    // ---- vertices -------------------------------------------------------
    // Each vertex's incident faces must form a single fan or cycle. Work with
    // *corners* — one per (face, slot) — and union the two corners at a vertex
    // whenever a pair of faces shares an edge through it. A vertex whose corners
    // end up in more than one component is a bowtie.
    let mut uf = UnionFind::new(faces.len() * 3);
    let corner_of = |fi: usize, v: usize| -> Option<usize> {
        faces[fi].iter().position(|&x| x == v).map(|k| fi * 3 + k)
    };

    for (&(a, b), slot) in &undirected {
        if let [Some(f0), Some(f1)] = slot {
            for v in [a, b] {
                if let (Some(c0), Some(c1)) = (corner_of(*f0, v), corner_of(*f1, v)) {
                    uf.union(c0, c1);
                }
            }
        }
    }

    // Group corners by vertex and check each group is one component.
    let mut representative: Vec<Option<usize>> = vec![None; num_vertices];
    for (fi, face) in faces.iter().enumerate() {
        for (k, &v) in face.iter().enumerate() {
            let root = uf.find(fi * 3 + k);
            match representative[v] {
                None => representative[v] = Some(root),
                Some(r) if r == root => {}
                Some(_) => {
                    return Err(MeshError::NonManifold {
                        details: format!(
                            "vertex {v} is non-manifold: its incident faces form more than one \
                             fan (a bowtie), so circulating the vertex cannot reach all of them"
                        ),
                    })
                }
            }
        }
    }

    Ok(())
}

/// Minimal union-find, used only by [`validate_manifold_soup`].
struct UnionFind {
    parent: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            // Path halving keeps this near-constant without recursion.
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            self.parent[ra] = rb;
        }
    }
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

/// Build a half-edge mesh from vertices and quad faces.
///
/// # Arguments
/// * `vertices` - List of vertex positions
/// * `faces` - List of quad faces, each as [v0, v1, v2, v3] indices (counter-clockwise)
///
/// # Returns
/// A half-edge mesh, or an error if the input is invalid.
pub fn build_from_quads<I: MeshIndex>(
    vertices: &[Point3<f64>],
    faces: &[[usize; 4]],
) -> Result<HalfEdgeMesh<I>> {
    if faces.is_empty() {
        return Err(MeshError::EmptyMesh);
    }

    // Validate vertex indices
    for (fi, face) in faces.iter().enumerate() {
        for &vi in face {
            if vi >= vertices.len() {
                return Err(MeshError::InvalidVertexIndex {
                    face: fi,
                    vertex: vi,
                });
            }
        }
        // Check for degenerate faces (duplicate vertices)
        if face[0] == face[1]
            || face[1] == face[2]
            || face[2] == face[3]
            || face[3] == face[0]
            || face[0] == face[2]
            || face[1] == face[3]
        {
            return Err(MeshError::DegenerateFace { face: fi });
        }
    }

    let mut mesh = HalfEdgeMesh::with_capacity(vertices.len(), faces.len());

    // Add vertices
    let vertex_ids: Vec<VertexId<I>> = vertices.iter().map(|&pos| mesh.add_vertex(pos)).collect();

    // Map from directed edge (v0, v1) to half-edge ID
    let mut edge_map: HashMap<(usize, usize), HalfEdgeId<I>> = HashMap::new();

    // First pass: create all half-edges and faces
    for face in faces {
        let v0 = face[0];
        let v1 = face[1];
        let v2 = face[2];
        let v3 = face[3];

        // Create four half-edges for this face
        let he0 = HalfEdgeId::<I>::new(mesh.num_halfedges());
        let he1 = HalfEdgeId::<I>::new(mesh.num_halfedges() + 1);
        let he2 = HalfEdgeId::<I>::new(mesh.num_halfedges() + 2);
        let he3 = HalfEdgeId::<I>::new(mesh.num_halfedges() + 3);

        // Add half-edges to mesh storage
        for _ in 0..4 {
            mesh.halfedges.push(super::halfedge::HalfEdge::new());
        }

        // Create face
        let face_id = FaceId::<I>::new(mesh.num_faces());
        mesh.faces.push(super::halfedge::Face::new(he0));

        // Set up half-edge connectivity within the face (counter-clockwise)
        {
            let he = mesh.halfedge_mut(he0);
            he.origin = vertex_ids[v0];
            he.next = he1;
            he.prev = he3;
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
            he.next = he3;
            he.prev = he1;
            he.face = face_id;
        }
        {
            let he = mesh.halfedge_mut(he3);
            he.origin = vertex_ids[v3];
            he.next = he0;
            he.prev = he2;
            he.face = face_id;
        }

        // Set vertex half-edges (will be overwritten for shared vertices)
        mesh.vertex_mut(vertex_ids[v0]).halfedge = he0;
        mesh.vertex_mut(vertex_ids[v1]).halfedge = he1;
        mesh.vertex_mut(vertex_ids[v2]).halfedge = he2;
        mesh.vertex_mut(vertex_ids[v3]).halfedge = he3;

        // Record edges for twin linking
        edge_map.insert((v0, v1), he0);
        edge_map.insert((v1, v2), he1);
        edge_map.insert((v2, v3), he2);
        edge_map.insert((v3, v0), he3);
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

/// Convert a half-edge mesh (quad) back to a face-vertex representation.
///
/// Returns (vertices, faces) tuple where each face has 4 vertices.
pub fn to_face_vertex_quads<I: MeshIndex>(
    mesh: &HalfEdgeMesh<I>,
) -> (Vec<Point3<f64>>, Vec<[usize; 4]>) {
    let vertices: Vec<Point3<f64>> = mesh.vertex_ids().map(|v| *mesh.position(v)).collect();

    let faces: Vec<[usize; 4]> = mesh
        .face_ids()
        .map(|f| {
            let [v0, v1, v2, v3] = mesh.face_quad(f);
            [v0.index(), v1.index(), v2.index(), v3.index()]
        })
        .collect();

    (vertices, faces)
}

#[cfg(test)]
mod tests {
    // ---- manifold validation ------------------------------------------

    /// Three faces on one edge cannot be represented, so it must be rejected
    /// rather than built into a half-edge with no twin.
    #[test]
    fn rejects_nonmanifold_edge() {
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.5, 1.0, 0.0),
            Point3::new(0.5, -1.0, 0.0),
            Point3::new(0.5, 0.0, 1.0),
        ];
        // All three faces traverse 0 -> 1, so this trips the directed-edge check
        // first; either rejection is correct.
        let faces = vec![[0, 1, 2], [0, 1, 3], [0, 1, 4]];
        let err = build_from_triangles::<u32>(&vertices, &faces).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("non-manifold"),
            "expected a non-manifold rejection, got: {msg}"
        );
    }

    /// Two fans meeting at a single vertex: circulating it cannot reach both.
    #[test]
    fn rejects_bowtie_vertex() {
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.5, 1.0, 0.0),
            Point3::new(-1.0, 0.0, 0.0),
            Point3::new(-0.5, -1.0, 0.0),
        ];
        let faces = vec![[0, 1, 2], [0, 3, 4]];
        let err = build_from_triangles::<u32>(&vertices, &faces).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("vertex 0") && msg.contains("non-manifold"),
            "expected vertex 0 named as non-manifold, got: {msg}"
        );
    }

    #[test]
    fn rejects_duplicate_face() {
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.5, 1.0, 0.0),
        ];
        let faces = vec![[0, 1, 2], [0, 1, 2]];
        assert!(build_from_triangles::<u32>(&vertices, &faces).is_err());
    }

    /// Two faces wound the same way across a shared edge disagree about which
    /// side is out, so the surface has no consistent orientation.
    #[test]
    fn rejects_inconsistent_winding() {
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ];
        // [0,2,3] and [0,3,2] both use the edge between 2 and 3, wound oppositely
        // relative to each other.
        let faces = vec![[0, 1, 2], [0, 2, 3], [0, 3, 2]];
        let err = build_from_triangles::<u32>(&vertices, &faces).unwrap_err();
        assert!(err.to_string().contains("non-manifold"));
    }

    /// The validator must not over-reach. All of these are representable and must
    /// still build.
    #[test]
    fn accepts_awkward_but_representable_input() {
        // Disconnected components.
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.5, 1.0, 0.0),
            Point3::new(5.0, 0.0, 0.0),
            Point3::new(6.0, 0.0, 0.0),
            Point3::new(5.5, 1.0, 0.0),
        ];
        assert!(
            build_from_triangles::<u32>(&vertices, &[[0, 1, 2], [3, 4, 5]]).is_ok(),
            "a disconnected mesh is still a manifold"
        );

        // A vertex no face references.
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.5, 1.0, 0.0),
            Point3::new(9.0, 9.0, 9.0),
        ];
        assert!(
            build_from_triangles::<u32>(&vertices, &[[0, 1, 2]]).is_ok(),
            "an isolated vertex is representable"
        );

        // A zero-area face: geometrically degenerate, topologically fine.
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
        ];
        assert!(
            build_from_triangles::<u32>(&vertices, &[[0, 1, 2], [0, 3, 1]]).is_ok(),
            "a collinear face is degenerate geometry, not invalid topology"
        );

        // Two vertices sharing a position but forming a single fan.
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0), // coincident with index 1
            Point3::new(0.5, 1.0, 0.0),
        ];
        assert!(
            build_from_triangles::<u32>(&vertices, &[[0, 1, 3], [1, 2, 3]]).is_ok(),
            "coincident positions are a geometric matter, not a topological one"
        );
    }

    /// A hole punched in a grid gives two boundary loops. Still a manifold.
    #[test]
    fn accepts_two_boundary_loops() {
        let n = 4usize;
        let mut vertices = Vec::new();
        let mut faces = Vec::new();
        for j in 0..=n {
            for i in 0..=n {
                vertices.push(Point3::new(i as f64, j as f64, 0.0));
            }
        }
        for j in 0..n {
            for i in 0..n {
                if i == 1 && j == 1 {
                    continue; // the hole
                }
                let v00 = j * (n + 1) + i;
                let v10 = j * (n + 1) + i + 1;
                let v01 = (j + 1) * (n + 1) + i;
                let v11 = (j + 1) * (n + 1) + i + 1;
                faces.push([v00, v10, v11]);
                faces.push([v00, v11, v01]);
            }
        }
        assert!(build_from_triangles::<u32>(&vertices, &faces).is_ok());
    }

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

    // ==================== Quad Tests ====================

    fn single_quad() -> (Vec<Point3<f64>>, Vec<[usize; 4]>) {
        // A simple planar quad
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ];
        let faces = vec![[0, 1, 2, 3]];
        (vertices, faces)
    }

    fn two_quads() -> (Vec<Point3<f64>>, Vec<[usize; 4]>) {
        // Two quads sharing an edge (1-2)
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
            Point3::new(2.0, 1.0, 0.0),
        ];
        let faces = vec![[0, 1, 2, 3], [1, 4, 5, 2]];
        (vertices, faces)
    }

    #[test]
    fn test_build_from_quads_single_quad() {
        let (vertices, faces) = single_quad();
        let mesh: HalfEdgeMesh<u32> = build_from_quads(&vertices, &faces).unwrap();

        assert_eq!(mesh.num_vertices(), 4);
        assert_eq!(mesh.num_faces(), 1);
        // 4 interior half-edges + 4 boundary half-edges
        assert_eq!(mesh.num_halfedges(), 8);
        assert!(mesh.is_valid());

        // All vertices should be on boundary
        for v in mesh.vertex_ids() {
            assert!(mesh.is_boundary_vertex(v));
        }

        // Check it's recognized as a quad mesh
        assert!(mesh.is_quad_mesh());
        assert!(!mesh.is_triangle_mesh());
    }

    #[test]
    fn test_build_from_quads_two_quads() {
        let (vertices, faces) = two_quads();
        let mesh: HalfEdgeMesh<u32> = build_from_quads(&vertices, &faces).unwrap();

        assert_eq!(mesh.num_vertices(), 6);
        assert_eq!(mesh.num_faces(), 2);
        // 8 interior half-edges + 6 boundary half-edges (shared edge has no boundary)
        assert_eq!(mesh.num_halfedges(), 14);
        assert!(mesh.is_valid());
    }

    #[test]
    fn test_roundtrip_quads() {
        let (vertices, faces) = two_quads();
        let mesh: HalfEdgeMesh<u32> = build_from_quads(&vertices, &faces).unwrap();

        let (out_verts, out_faces) = to_face_vertex_quads(&mesh);

        assert_eq!(vertices.len(), out_verts.len());
        assert_eq!(faces.len(), out_faces.len());

        // Positions should match
        for (v_in, v_out) in vertices.iter().zip(out_verts.iter()) {
            assert!((v_in - v_out).norm() < 1e-10);
        }
    }

    #[test]
    fn test_quad_degenerate_face() {
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ];
        // Degenerate: v0 == v2 (diagonal vertices the same)
        let faces = vec![[0, 1, 0, 3]];

        let result: Result<HalfEdgeMesh<u32>> = build_from_quads(&vertices, &faces);
        assert!(result.is_err());
    }

    #[test]
    fn test_quad_geometry() {
        let (vertices, faces) = single_quad();
        let mesh: HalfEdgeMesh<u32> = build_from_quads(&vertices, &faces).unwrap();

        let f = FaceId::<u32>::new(0);

        // Check vertex count
        assert_eq!(mesh.face_vertex_count(f), 4);

        // Check area (1x1 quad = area 1.0)
        let area = mesh.face_area_quad(f);
        assert!((area - 1.0).abs() < 1e-10);

        // Check centroid (should be at (0.5, 0.5, 0))
        let centroid = mesh.face_centroid_quad(f);
        assert!((centroid.x - 0.5).abs() < 1e-10);
        assert!((centroid.y - 0.5).abs() < 1e-10);
        assert!(centroid.z.abs() < 1e-10);

        // Check normal (should be (0, 0, 1) for CCW winding)
        let normal = mesh.face_normal_quad(f);
        assert!(normal.z.abs() > 0.99);
    }
}
