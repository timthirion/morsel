//! # Morsel
//!
//! A comprehensive 3D mesh processing library for geometry processing research.
//!
//! Morsel provides a robust half-edge mesh data structure and a collection of
//! geometry processing algorithms, designed for research and experimentation
//! in computational geometry.
//!
//! ## Features
//!
//! - **Half-edge data structure**: O(1) adjacency queries with type-safe indices
//! - **Flexible indexing**: Support for 16-bit, 32-bit, and 64-bit indices
//! - **Multiple file formats**: OBJ, STL, PLY, glTF
//! - **Robust predicates**: Integration with exactum for exact geometric predicates
//! - **Approximation algorithms**: Integration with approxum for efficient approximate operations
//!
//! ## Quick Start
//!
//! ```no_run
//! use morsel::prelude::*;
//!
//! // Load a mesh
//! let mesh: HalfEdgeMesh = morsel::io::load("model.obj").unwrap();
//!
//! // Query mesh properties
//! println!("Vertices: {}", mesh.num_vertices());
//! println!("Faces: {}", mesh.num_faces());
//!
//! // Iterate over faces
//! for face_id in mesh.face_ids() {
//!     let normal = mesh.face_normal(face_id);
//!     let area = mesh.face_area(face_id);
//!     println!("Face {:?}: normal={:?}, area={}", face_id, normal, area);
//! }
//!
//! // Save the mesh
//! morsel::io::save(&mesh, "output.stl").unwrap();
//! ```
//!
//! ## Building Meshes Programmatically
//!
//! ```
//! use morsel::prelude::*;
//! use nalgebra::Point3;
//!
//! // Define vertices and faces
//! let vertices = vec![
//!     Point3::new(0.0, 0.0, 0.0),
//!     Point3::new(1.0, 0.0, 0.0),
//!     Point3::new(0.5, 1.0, 0.0),
//!     Point3::new(0.5, 0.5, 1.0),
//! ];
//!
//! let faces = vec![
//!     [0, 2, 1],  // bottom
//!     [0, 1, 3],  // front
//!     [1, 2, 3],  // right
//!     [2, 0, 3],  // left
//! ];
//!
//! // Build the mesh
//! let mesh: HalfEdgeMesh = build_from_triangles(&vertices, &faces).unwrap();
//! assert_eq!(mesh.num_vertices(), 4);
//! assert_eq!(mesh.num_faces(), 4);
//! ```
//!
//! ## Mesh Traversal
//!
//! The half-edge structure enables efficient traversal of mesh elements:
//!
//! ```
//! use morsel::prelude::*;
//! use nalgebra::Point3;
//!
//! # let vertices = vec![
//! #     Point3::new(0.0, 0.0, 0.0),
//! #     Point3::new(1.0, 0.0, 0.0),
//! #     Point3::new(0.5, 1.0, 0.0),
//! # ];
//! # let faces = vec![[0, 1, 2]];
//! # let mesh: HalfEdgeMesh = build_from_triangles(&vertices, &faces).unwrap();
//! // Iterate over neighbors of a vertex
//! let v = VertexId::new(0);
//! for neighbor in mesh.vertex_neighbors(v) {
//!     println!("Neighbor: {:?}", neighbor);
//! }
//!
//! // Iterate over faces around a vertex
//! for face in mesh.vertex_faces(v) {
//!     println!("Adjacent face: {:?}", face);
//! }
//!
//! // Get vertices of a face
//! let f = FaceId::new(0);
//! let [v0, v1, v2] = mesh.face_triangle(f);
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod algo;
pub mod error;
pub mod io;
pub mod mesh;

/// Prelude module for convenient imports.
///
/// This module re-exports the most commonly used types and functions:
///
/// ```
/// use morsel::prelude::*;
/// ```
pub mod prelude {
    pub use crate::error::{MeshError, Result};
    pub use crate::mesh::{
        build_from_triangles, to_face_vertex, EdgeId, Face, FaceId, HalfEdge, HalfEdgeId,
        HalfEdgeMesh, MeshIndex, Vertex, VertexId,
    };
}

// Re-export nalgebra types for convenience
pub use nalgebra;

#[cfg(test)]
mod tests {
    use super::prelude::*;
    use nalgebra::Point3;

    #[test]
    fn test_tetrahedron() {
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.5, 1.0, 0.0),
            Point3::new(0.5, 0.5, 1.0),
        ];

        let faces = vec![
            [0, 2, 1], // bottom
            [0, 1, 3], // front
            [1, 2, 3], // right
            [2, 0, 3], // left
        ];

        let mesh: HalfEdgeMesh = build_from_triangles(&vertices, &faces).unwrap();

        assert_eq!(mesh.num_vertices(), 4);
        assert_eq!(mesh.num_faces(), 4);
        // Closed mesh: 4 faces * 3 half-edges per face / 2 (each edge has 2 half-edges) * 2 = 12
        // Actually: 4 faces * 3 = 12 half-edges, no boundary
        assert_eq!(mesh.num_halfedges(), 12);
        assert!(mesh.is_valid());

        // Check that it's a closed mesh (no boundary vertices)
        for v in mesh.vertex_ids() {
            assert!(!mesh.is_boundary_vertex(v), "vertex {:?} should not be on boundary", v);
        }
    }
}
