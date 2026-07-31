//! Mesh processing algorithms.
//!
//! This module contains various algorithms for mesh processing, including:
//!
//! - **Smoothing**: Laplacian smoothing, bilateral smoothing, mean curvature flow
//! - **Remeshing**: Isotropic remeshing, anisotropic remeshing
//! - **Decimation**: Edge collapse driven by quadric error metrics
//! - **Subdivision**: Loop subdivision, Catmull-Clark subdivision
//! - **Parameterization**: cylindrical projection, LSCM, ARAP, optimal mass transport
//! - **Cutting**: cutting a surface to disk topology, which parameterization needs
//! - **Geodesics**: Dijkstra, heat method
//! - **Curvature**: Discrete curvature estimation
//! - **Quality**: Triangle-quality measurement, for checking the claims the others make
//! - **Distance**: point-to-surface queries and Hausdorff distance, for checking that a
//!   mesh is still the same *shape* after being processed
//!
//! Algorithms are added incrementally as the library develops. Vertex clustering,
//! exact polyhedral geodesics, and repair (non-manifold repair, hole filling) are
//! not implemented.

pub mod curvature;
pub mod cut;
pub mod decimate;
pub mod distance;
pub mod geodesic;
pub mod parameterize;
pub mod progress;
pub mod quality;
pub mod remesh;
pub mod smooth;
pub mod subdivide;

pub use progress::Progress;

// Modules will be added as algorithms are implemented:
// pub mod repair;
