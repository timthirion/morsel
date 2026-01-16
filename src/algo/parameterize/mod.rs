//! UV parameterization algorithms.
//!
//! This module provides algorithms for computing UV coordinates (2D parameterization)
//! for triangle meshes. Parameterization maps the 3D mesh surface to a 2D domain,
//! which is essential for texture mapping and many geometry processing operations.
//!
//! # Available Algorithms
//!
//! - [`lscm`]: Least Squares Conformal Maps - minimizes angle distortion
//!
//! # Requirements
//!
//! Parameterization algorithms typically require the mesh to have boundary (disk topology).
//! Closed meshes must first be cut to create a boundary.
//!
//! # Example
//!
//! ```no_run
//! use morsel::prelude::*;
//! use morsel::algo::parameterize::{lscm, LSCMOptions, UVMap};
//!
//! let mesh: HalfEdgeMesh = morsel::io::load("input.obj").unwrap();
//!
//! // Compute LSCM parameterization
//! let uv_map = lscm(&mesh, &LSCMOptions::default()).unwrap();
//!
//! // Access UV coordinates
//! for vid in mesh.vertex_ids() {
//!     let uv = uv_map.get(vid);
//!     println!("Vertex {:?}: u={:.3}, v={:.3}", vid, uv.x, uv.y);
//! }
//! ```
//!
//! # References
//!
//! - Lévy, B., Petitjean, S., Ray, N., & Maillot, J. (2002). "Least squares
//!   conformal maps for automatic texture atlas generation." ACM SIGGRAPH.

mod lscm;
mod sparse;
mod uv;

pub use lscm::{lscm, LSCMOptions, PinStrategy, PinnedVertex};
pub use uv::UVMap;
