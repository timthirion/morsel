//! Half-edge mesh data structure.
//!
//! This module provides a half-edge (doubly-connected edge list) representation
//! for triangle meshes. This structure enables O(1) adjacency queries and is
//! the foundation for most geometry processing algorithms.
//!
//! # Structure
//!
//! - Each edge is split into two **half-edges** pointing in opposite directions
//! - Each half-edge knows its **twin** (opposite half-edge), **next** (next half-edge
//!   around the face), **origin vertex**, and **incident face**
//! - Each vertex stores one outgoing half-edge
//! - Each face stores one half-edge on its boundary
//!
//! # Boundary Handling
//!
//! Boundary half-edges (on mesh boundaries) have an invalid face ID. Their twins
//! are the interior half-edges. Boundary loops can be traversed using the `next`
//! pointer on boundary half-edges.

use nalgebra::{Point3, Vector3};

use super::index::{FaceId, HalfEdgeId, MeshIndex, VertexId};

/// A vertex in the half-edge mesh.
#[derive(Debug, Clone)]
pub struct Vertex<I: MeshIndex = u32> {
    /// The 3D position of this vertex.
    pub position: Point3<f64>,

    /// One outgoing half-edge from this vertex.
    /// For boundary vertices, this is guaranteed to be a boundary half-edge.
    pub halfedge: HalfEdgeId<I>,
}

impl<I: MeshIndex> Vertex<I> {
    /// Create a new vertex at the given position.
    pub fn new(position: Point3<f64>) -> Self {
        Self {
            position,
            halfedge: HalfEdgeId::invalid(),
        }
    }

    /// Create a new vertex from coordinates.
    pub fn from_coords(x: f64, y: f64, z: f64) -> Self {
        Self::new(Point3::new(x, y, z))
    }
}

/// A half-edge in the mesh.
#[derive(Debug, Clone, Copy)]
pub struct HalfEdge<I: MeshIndex = u32> {
    /// The vertex this half-edge originates from.
    pub origin: VertexId<I>,

    /// The opposite half-edge (pointing in the reverse direction).
    pub twin: HalfEdgeId<I>,

    /// The next half-edge around the face (counter-clockwise).
    pub next: HalfEdgeId<I>,

    /// The previous half-edge around the face (clockwise).
    /// This is redundant but speeds up many operations.
    pub prev: HalfEdgeId<I>,

    /// The face this half-edge belongs to.
    /// Invalid for boundary half-edges.
    pub face: FaceId<I>,
}

impl<I: MeshIndex> HalfEdge<I> {
    /// Create a new uninitialized half-edge.
    pub fn new() -> Self {
        Self {
            origin: VertexId::invalid(),
            twin: HalfEdgeId::invalid(),
            next: HalfEdgeId::invalid(),
            prev: HalfEdgeId::invalid(),
            face: FaceId::invalid(),
        }
    }

    /// Check if this half-edge is on the boundary.
    #[inline]
    pub fn is_boundary(&self) -> bool {
        !self.face.is_valid()
    }
}

impl<I: MeshIndex> Default for HalfEdge<I> {
    fn default() -> Self {
        Self::new()
    }
}

/// A face in the half-edge mesh.
#[derive(Debug, Clone, Copy)]
pub struct Face<I: MeshIndex = u32> {
    /// One half-edge on the boundary of this face.
    pub halfedge: HalfEdgeId<I>,
}

impl<I: MeshIndex> Face<I> {
    /// Create a new face with the given half-edge.
    pub fn new(halfedge: HalfEdgeId<I>) -> Self {
        Self { halfedge }
    }
}

impl<I: MeshIndex> Default for Face<I> {
    fn default() -> Self {
        Self {
            halfedge: HalfEdgeId::invalid(),
        }
    }
}

/// A half-edge mesh data structure for triangle meshes.
///
/// This structure stores vertices, half-edges, and faces with full connectivity
/// information, enabling O(1) adjacency queries.
#[derive(Debug, Clone)]
pub struct HalfEdgeMesh<I: MeshIndex = u32> {
    /// All vertices in the mesh.
    pub(crate) vertices: Vec<Vertex<I>>,

    /// All half-edges in the mesh.
    pub(crate) halfedges: Vec<HalfEdge<I>>,

    /// All faces in the mesh.
    pub(crate) faces: Vec<Face<I>>,
}

impl<I: MeshIndex> Default for HalfEdgeMesh<I> {
    fn default() -> Self {
        Self::new()
    }
}

impl<I: MeshIndex> HalfEdgeMesh<I> {
    /// Create a new empty mesh.
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            halfedges: Vec::new(),
            faces: Vec::new(),
        }
    }

    /// Create a mesh with pre-allocated capacity.
    pub fn with_capacity(num_vertices: usize, num_faces: usize) -> Self {
        // Each triangle has 3 half-edges, but each interior edge is shared
        // For a closed mesh: E = 3F/2, so HE = 3F
        // For a mesh with boundary, slightly more
        let num_halfedges = num_faces * 3 + num_faces / 2;

        Self {
            vertices: Vec::with_capacity(num_vertices),
            halfedges: Vec::with_capacity(num_halfedges),
            faces: Vec::with_capacity(num_faces),
        }
    }

    // ==================== Accessors ====================

    /// Get the number of vertices.
    #[inline]
    pub fn num_vertices(&self) -> usize {
        self.vertices.len()
    }

    /// Get the number of half-edges.
    #[inline]
    pub fn num_halfedges(&self) -> usize {
        self.halfedges.len()
    }

    /// Get the number of faces.
    #[inline]
    pub fn num_faces(&self) -> usize {
        self.faces.len()
    }

    /// Get a vertex by ID.
    #[inline]
    pub fn vertex(&self, id: VertexId<I>) -> &Vertex<I> {
        &self.vertices[id.index()]
    }

    /// Get a mutable vertex by ID.
    #[inline]
    pub fn vertex_mut(&mut self, id: VertexId<I>) -> &mut Vertex<I> {
        &mut self.vertices[id.index()]
    }

    /// Get a half-edge by ID.
    #[inline]
    pub fn halfedge(&self, id: HalfEdgeId<I>) -> &HalfEdge<I> {
        &self.halfedges[id.index()]
    }

    /// Get a mutable half-edge by ID.
    #[inline]
    pub fn halfedge_mut(&mut self, id: HalfEdgeId<I>) -> &mut HalfEdge<I> {
        &mut self.halfedges[id.index()]
    }

    /// Get a face by ID.
    #[inline]
    pub fn face(&self, id: FaceId<I>) -> &Face<I> {
        &self.faces[id.index()]
    }

    /// Get a mutable face by ID.
    #[inline]
    pub fn face_mut(&mut self, id: FaceId<I>) -> &mut Face<I> {
        &mut self.faces[id.index()]
    }

    /// Get the position of a vertex.
    #[inline]
    pub fn position(&self, v: VertexId<I>) -> &Point3<f64> {
        &self.vertex(v).position
    }

    /// Set the position of a vertex.
    #[inline]
    pub fn set_position(&mut self, v: VertexId<I>, pos: Point3<f64>) {
        self.vertex_mut(v).position = pos;
    }

    // ==================== Topology Queries ====================

    /// Get the twin (opposite) half-edge.
    #[inline]
    pub fn twin(&self, he: HalfEdgeId<I>) -> HalfEdgeId<I> {
        self.halfedge(he).twin
    }

    /// Get the next half-edge around the face.
    #[inline]
    pub fn next(&self, he: HalfEdgeId<I>) -> HalfEdgeId<I> {
        self.halfedge(he).next
    }

    /// Get the previous half-edge around the face.
    #[inline]
    pub fn prev(&self, he: HalfEdgeId<I>) -> HalfEdgeId<I> {
        self.halfedge(he).prev
    }

    /// Get the origin vertex of a half-edge.
    #[inline]
    pub fn origin(&self, he: HalfEdgeId<I>) -> VertexId<I> {
        self.halfedge(he).origin
    }

    /// Get the destination vertex of a half-edge.
    #[inline]
    pub fn dest(&self, he: HalfEdgeId<I>) -> VertexId<I> {
        self.origin(self.twin(he))
    }

    /// Get the face of a half-edge.
    #[inline]
    pub fn face_of(&self, he: HalfEdgeId<I>) -> FaceId<I> {
        self.halfedge(he).face
    }

    /// Check if a half-edge is on the boundary.
    #[inline]
    pub fn is_boundary_halfedge(&self, he: HalfEdgeId<I>) -> bool {
        self.halfedge(he).is_boundary()
    }

    /// Check if a vertex is on the boundary.
    pub fn is_boundary_vertex(&self, v: VertexId<I>) -> bool {
        let start = self.vertex(v).halfedge;
        if !start.is_valid() {
            return true; // Isolated vertex
        }

        // Walk around the vertex using the same logic as VertexHalfEdgeIter
        let mut he = start;
        loop {
            if self.is_boundary_halfedge(he) {
                return true;
            }
            he = self.next(self.twin(he));
            if he == start {
                break;
            }
        }
        false
    }

    /// Check if an edge (represented by one of its half-edges) is on the boundary.
    #[inline]
    pub fn is_boundary_edge(&self, he: HalfEdgeId<I>) -> bool {
        self.is_boundary_halfedge(he) || self.is_boundary_halfedge(self.twin(he))
    }

    // ==================== Iteration ====================

    /// Iterate over all vertex IDs.
    pub fn vertex_ids(&self) -> impl Iterator<Item = VertexId<I>> + '_ {
        (0..self.vertices.len()).map(|i| VertexId::new(i))
    }

    /// Iterate over all vertices with their IDs.
    pub fn vertices(&self) -> impl Iterator<Item = (VertexId<I>, &Vertex<I>)> + '_ {
        self.vertices
            .iter()
            .enumerate()
            .map(|(i, v)| (VertexId::new(i), v))
    }

    /// Iterate over all half-edge IDs.
    pub fn halfedge_ids(&self) -> impl Iterator<Item = HalfEdgeId<I>> + '_ {
        (0..self.halfedges.len()).map(|i| HalfEdgeId::new(i))
    }

    /// Iterate over all half-edges with their IDs.
    pub fn halfedges(&self) -> impl Iterator<Item = (HalfEdgeId<I>, &HalfEdge<I>)> + '_ {
        self.halfedges
            .iter()
            .enumerate()
            .map(|(i, he)| (HalfEdgeId::new(i), he))
    }

    /// Iterate over all face IDs.
    pub fn face_ids(&self) -> impl Iterator<Item = FaceId<I>> + '_ {
        (0..self.faces.len()).map(|i| FaceId::new(i))
    }

    /// Iterate over all faces with their IDs.
    pub fn faces(&self) -> impl Iterator<Item = (FaceId<I>, &Face<I>)> + '_ {
        self.faces
            .iter()
            .enumerate()
            .map(|(i, f)| (FaceId::new(i), f))
    }

    /// Iterate over half-edges around a vertex (outgoing half-edges).
    pub fn vertex_halfedges(&self, v: VertexId<I>) -> VertexHalfEdgeIter<'_, I> {
        VertexHalfEdgeIter::new(self, v)
    }

    /// Iterate over vertices adjacent to a vertex.
    pub fn vertex_neighbors(&self, v: VertexId<I>) -> impl Iterator<Item = VertexId<I>> + '_ {
        self.vertex_halfedges(v).map(|he| self.dest(he))
    }

    /// Iterate over faces adjacent to a vertex.
    pub fn vertex_faces(&self, v: VertexId<I>) -> impl Iterator<Item = FaceId<I>> + '_ {
        self.vertex_halfedges(v)
            .filter_map(|he| {
                let f = self.face_of(he);
                if f.is_valid() {
                    Some(f)
                } else {
                    None
                }
            })
    }

    /// Iterate over half-edges around a face.
    pub fn face_halfedges(&self, f: FaceId<I>) -> FaceHalfEdgeIter<'_, I> {
        FaceHalfEdgeIter::new(self, f)
    }

    /// Iterate over vertices of a face.
    pub fn face_vertices(&self, f: FaceId<I>) -> impl Iterator<Item = VertexId<I>> + '_ {
        self.face_halfedges(f).map(|he| self.origin(he))
    }

    /// Get the three vertices of a triangular face.
    pub fn face_triangle(&self, f: FaceId<I>) -> [VertexId<I>; 3] {
        let he0 = self.face(f).halfedge;
        let he1 = self.next(he0);
        let he2 = self.next(he1);
        [self.origin(he0), self.origin(he1), self.origin(he2)]
    }

    /// Get the positions of the three vertices of a triangular face.
    pub fn face_positions(&self, f: FaceId<I>) -> [Point3<f64>; 3] {
        let [v0, v1, v2] = self.face_triangle(f);
        [*self.position(v0), *self.position(v1), *self.position(v2)]
    }

    // ==================== Geometry ====================

    /// Compute the normal of a face.
    pub fn face_normal(&self, f: FaceId<I>) -> Vector3<f64> {
        let [p0, p1, p2] = self.face_positions(f);
        let e1 = p1 - p0;
        let e2 = p2 - p0;
        e1.cross(&e2).normalize()
    }

    /// Compute the area of a face.
    pub fn face_area(&self, f: FaceId<I>) -> f64 {
        let [p0, p1, p2] = self.face_positions(f);
        let e1 = p1 - p0;
        let e2 = p2 - p0;
        0.5 * e1.cross(&e2).norm()
    }

    /// Compute the area-weighted normal at a vertex.
    pub fn vertex_normal(&self, v: VertexId<I>) -> Vector3<f64> {
        let mut normal = Vector3::zeros();
        for f in self.vertex_faces(v) {
            let [p0, p1, p2] = self.face_positions(f);
            let e1 = p1 - p0;
            let e2 = p2 - p0;
            normal += e1.cross(&e2); // Area-weighted (not normalized)
        }
        normal.normalize()
    }

    /// Compute the length of an edge.
    pub fn edge_length(&self, he: HalfEdgeId<I>) -> f64 {
        let p0 = self.position(self.origin(he));
        let p1 = self.position(self.dest(he));
        (p1 - p0).norm()
    }

    /// Compute the edge vector (from origin to destination).
    pub fn edge_vector(&self, he: HalfEdgeId<I>) -> Vector3<f64> {
        let p0 = self.position(self.origin(he));
        let p1 = self.position(self.dest(he));
        p1 - p0
    }

    /// Compute the midpoint of an edge.
    pub fn edge_midpoint(&self, he: HalfEdgeId<I>) -> Point3<f64> {
        let p0 = self.position(self.origin(he));
        let p1 = self.position(self.dest(he));
        Point3::from((p0.coords + p1.coords) * 0.5)
    }

    /// Compute the valence (degree) of a vertex.
    pub fn valence(&self, v: VertexId<I>) -> usize {
        self.vertex_halfedges(v).count()
    }

    /// Compute the centroid of a face.
    pub fn face_centroid(&self, f: FaceId<I>) -> Point3<f64> {
        let [p0, p1, p2] = self.face_positions(f);
        Point3::from((p0.coords + p1.coords + p2.coords) / 3.0)
    }

    /// Compute the bounding box of the mesh.
    pub fn bounding_box(&self) -> Option<(Point3<f64>, Point3<f64>)> {
        if self.vertices.is_empty() {
            return None;
        }

        let mut min = self.vertices[0].position;
        let mut max = self.vertices[0].position;

        for v in &self.vertices {
            for i in 0..3 {
                min[i] = min[i].min(v.position[i]);
                max[i] = max[i].max(v.position[i]);
            }
        }

        Some((min, max))
    }

    /// Compute the total surface area of the mesh.
    pub fn surface_area(&self) -> f64 {
        self.face_ids().map(|f| self.face_area(f)).sum()
    }

    // ==================== Construction ====================

    /// Add a new vertex and return its ID.
    pub fn add_vertex(&mut self, position: Point3<f64>) -> VertexId<I> {
        let id = VertexId::new(self.vertices.len());
        self.vertices.push(Vertex::new(position));
        id
    }

    // ==================== Validation ====================

    /// Check if the mesh is valid (all connectivity is consistent).
    pub fn is_valid(&self) -> bool {
        // Check vertices
        for (vid, v) in self.vertices() {
            if v.halfedge.is_valid() {
                let he = self.halfedge(v.halfedge);
                if he.origin != vid {
                    return false;
                }
            }
        }

        // Check half-edges
        for (heid, he) in self.halfedges() {
            // Twin consistency
            if he.twin.is_valid() {
                let twin = self.halfedge(he.twin);
                if twin.twin != heid {
                    return false;
                }
            }

            // Next/prev consistency
            if he.next.is_valid() {
                if self.halfedge(he.next).prev != heid {
                    return false;
                }
            }

            if he.prev.is_valid() {
                if self.halfedge(he.prev).next != heid {
                    return false;
                }
            }
        }

        // Check faces
        for (_fid, f) in self.faces() {
            if !f.halfedge.is_valid() {
                return false;
            }
        }

        true
    }
}

/// Iterator over half-edges around a vertex.
pub struct VertexHalfEdgeIter<'a, I: MeshIndex = u32> {
    mesh: &'a HalfEdgeMesh<I>,
    start: HalfEdgeId<I>,
    current: HalfEdgeId<I>,
    done: bool,
}

impl<'a, I: MeshIndex> VertexHalfEdgeIter<'a, I> {
    fn new(mesh: &'a HalfEdgeMesh<I>, v: VertexId<I>) -> Self {
        let start = mesh.vertex(v).halfedge;
        Self {
            mesh,
            start,
            current: start,
            done: !start.is_valid(),
        }
    }
}

impl<'a, I: MeshIndex> Iterator for VertexHalfEdgeIter<'a, I> {
    type Item = HalfEdgeId<I>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let result = self.current;

        // Move to next outgoing half-edge: twin -> next
        // If he goes v -> w, then twin(he) goes w -> v.
        // next(twin(he)) is the half-edge after twin(he) in its face,
        // which originates at v (the next outgoing half-edge from v).
        self.current = self.mesh.next(self.mesh.twin(self.current));

        if self.current == self.start {
            self.done = true;
        }

        Some(result)
    }
}

/// Iterator over half-edges around a face.
pub struct FaceHalfEdgeIter<'a, I: MeshIndex = u32> {
    mesh: &'a HalfEdgeMesh<I>,
    start: HalfEdgeId<I>,
    current: HalfEdgeId<I>,
    done: bool,
}

impl<'a, I: MeshIndex> FaceHalfEdgeIter<'a, I> {
    fn new(mesh: &'a HalfEdgeMesh<I>, f: FaceId<I>) -> Self {
        let start = mesh.face(f).halfedge;
        Self {
            mesh,
            start,
            current: start,
            done: !start.is_valid(),
        }
    }
}

impl<'a, I: MeshIndex> Iterator for FaceHalfEdgeIter<'a, I> {
    type Item = HalfEdgeId<I>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let result = self.current;
        self.current = self.mesh.next(self.current);

        if self.current == self.start {
            self.done = true;
        }

        Some(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vertex_creation() {
        let v = Vertex::<u32>::from_coords(1.0, 2.0, 3.0);
        assert_eq!(v.position, Point3::new(1.0, 2.0, 3.0));
        assert!(!v.halfedge.is_valid());
    }

    #[test]
    fn test_empty_mesh() {
        let mesh = HalfEdgeMesh::<u32>::new();
        assert_eq!(mesh.num_vertices(), 0);
        assert_eq!(mesh.num_halfedges(), 0);
        assert_eq!(mesh.num_faces(), 0);
        assert!(mesh.is_valid());
    }

    #[test]
    fn test_add_vertex() {
        let mut mesh = HalfEdgeMesh::<u32>::new();
        let v0 = mesh.add_vertex(Point3::new(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(Point3::new(1.0, 0.0, 0.0));

        assert_eq!(mesh.num_vertices(), 2);
        assert_eq!(v0.index(), 0);
        assert_eq!(v1.index(), 1);
    }
}
