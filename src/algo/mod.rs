//! Mesh processing algorithms.
//!
//! This module contains various algorithms for mesh processing, including:
//!
//! - **Smoothing**: Laplacian smoothing, bilateral smoothing, mean curvature flow
//! - **Remeshing**: Isotropic remeshing, anisotropic remeshing
//! - **Decimation**: Edge collapse, vertex clustering, quadric error metrics
//! - **Subdivision**: Loop subdivision, Catmull-Clark subdivision
//! - **Parameterization**: LSCM, ARAP, conformal maps
//! - **Geodesics**: Dijkstra, heat method, exact polyhedral
//! - **Curvature**: Discrete curvature estimation
//! - **Repair**: Non-manifold repair, hole filling
//!
//! Algorithms are added incrementally as the library develops.

pub mod decimate;
pub mod geodesic;
pub mod parameterize;
pub mod remesh;
pub mod smooth;
pub mod subdivide;

// Modules will be added as algorithms are implemented:
// pub mod curvature;
// pub mod repair;
