//! Core mesh data structures.
//!
//! This module provides the half-edge mesh representation and related types
//! for representing and manipulating triangle meshes.
//!
//! # Overview
//!
//! The primary type is [`HalfEdgeMesh`], which represents a triangle mesh using
//! a half-edge (doubly-connected edge list) data structure. This representation
//! provides O(1) adjacency queries, making it efficient for geometry processing
//! algorithms.
//!
//! # Index Types
//!
//! Mesh elements are identified by type-safe index wrappers:
//! - [`VertexId`] - Identifies a vertex
//! - [`HalfEdgeId`] - Identifies a half-edge
//! - [`FaceId`] - Identifies a face
//! - [`EdgeId`] - Identifies a full edge
//!
//! These indices are generic over the underlying integer type ([`MeshIndex`] trait),
//! allowing you to choose `u16`, `u32`, or `u64` based on mesh size.
//!
//! # Construction
//!
//! Meshes are typically constructed from file I/O or from face-vertex lists:
//!
//! ```
//! use morsel::mesh::{HalfEdgeMesh, build_from_triangles};
//! use nalgebra::Point3;
//!
//! let vertices = vec![
//!     Point3::new(0.0, 0.0, 0.0),
//!     Point3::new(1.0, 0.0, 0.0),
//!     Point3::new(0.5, 1.0, 0.0),
//! ];
//! let faces = vec![[0, 1, 2]];
//!
//! let mesh: HalfEdgeMesh = build_from_triangles(&vertices, &faces).unwrap();
//! ```

mod builder;
mod halfedge;
mod index;

pub use builder::{build_from_quads, build_from_triangles, to_face_vertex, to_face_vertex_quads};
pub use halfedge::{Face, HalfEdge, HalfEdgeMesh, Vertex};
pub use index::{
    DefaultIndexFamily, EdgeId, FaceId, HalfEdgeId, LargeIndexFamily, MeshIndex, MeshIndexFamily,
    SmallIndexFamily, VertexId,
};
