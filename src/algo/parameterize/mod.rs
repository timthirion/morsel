//! UV parameterization algorithms.
//!
//! This module provides algorithms for computing UV coordinates (2D parameterization)
//! for triangle meshes. Parameterization maps the 3D mesh surface to a 2D domain,
//! which is essential for texture mapping and many geometry processing operations.
//!
//! # Available Algorithms
//!
//! - [`lscm`]: Least Squares Conformal Maps - minimizes angle distortion
//! - [`arap`]: As-Rigid-As-Possible - minimizes both angle and area distortion
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
//! use morsel::algo::parameterize::{lscm, LSCMOptions, arap, ARAPOptions, UVMap};
//!
//! let mesh: HalfEdgeMesh = morsel::io::load("input.obj").unwrap();
//!
//! // Compute LSCM parameterization (fast, angle-preserving)
//! let uv_lscm = lscm(&mesh, &LSCMOptions::default()).unwrap();
//!
//! // Compute ARAP parameterization (iterative, preserves angles and areas)
//! let uv_arap = arap(&mesh, &ARAPOptions::default()).unwrap();
//!
//! // Access UV coordinates
//! for vid in mesh.vertex_ids() {
//!     let uv = uv_arap.get(vid);
//!     println!("Vertex {:?}: u={:.3}, v={:.3}", vid, uv.x, uv.y);
//! }
//! ```
//!
//! # References
//!
//! - Lévy, B., et al. (2002). "Least squares conformal maps for automatic
//!   texture atlas generation." ACM SIGGRAPH.
//! - Liu, L., et al. (2008). "A Local/Global Approach to Mesh Parameterization."
//!   SGP 2008.

mod arap;
mod lscm;
pub mod sparse;
mod uv;

pub use arap::{arap, ARAPOptions};
pub use lscm::{lscm, LSCMOptions, PinStrategy, PinnedVertex};
pub use uv::UVMap;
