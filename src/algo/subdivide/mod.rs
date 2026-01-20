//! Mesh subdivision algorithms.
//!
//! This module provides algorithms for subdividing meshes to create
//! smoother surfaces.
//!
//! # Loop Subdivision (Triangle Meshes)
//!
//! Loop subdivision (Loop, 1987) is an approximating subdivision scheme for
//! triangle meshes. Each iteration:
//!
//! 1. Inserts new vertices at edge midpoints (weighted positions)
//! 2. Updates original vertex positions based on neighbors
//! 3. Splits each triangle into 4 smaller triangles
//!
//! The result converges to a C² continuous surface (C¹ at extraordinary vertices).
//!
//! # Catmull-Clark Subdivision (Quad Meshes)
//!
//! Catmull-Clark subdivision (Catmull & Clark, 1978) is an approximating
//! subdivision scheme for quad meshes. Each iteration:
//!
//! 1. Creates a face point at each face centroid
//! 2. Creates edge points as average of edge midpoint and adjacent face points
//! 3. Updates original vertices using weighted average of neighbors
//! 4. Connects to form new quads
//!
//! The result converges to a C² continuous surface (C¹ at extraordinary vertices).
//!
//! # Example
//!
//! ```no_run
//! use morsel::prelude::*;
//! use morsel::algo::subdivide::{loop_subdivide, SubdivideOptions};
//!
//! let mut mesh: HalfEdgeMesh = morsel::io::load("input.obj").unwrap();
//!
//! let options = SubdivideOptions::new(2); // 2 iterations
//! loop_subdivide(&mut mesh, &options);
//!
//! morsel::io::save(&mesh, "output.obj").unwrap();
//! ```
//!
//! # References
//!
//! - Loop, C. (1987). "Smooth Subdivision Surfaces Based on Triangles."
//!   Master's thesis, University of Utah.
//! - Catmull, E. & Clark, J. (1978). "Recursively generated B-spline surfaces
//!   on arbitrary topological meshes." Computer-Aided Design, 10(6), 350-355.

mod catmull_clark;
mod loop_subdivision;

pub use catmull_clark::{catmull_clark_subdivide, catmull_clark_subdivide_with_progress};
pub use loop_subdivision::{loop_subdivide, loop_subdivide_with_progress};

/// Options for subdivision algorithms.
#[derive(Debug, Clone)]
pub struct SubdivideOptions {
    /// Number of subdivision iterations.
    pub iterations: usize,

    /// Whether to preserve sharp boundary edges.
    /// If true, boundary edges use simpler linear interpolation.
    pub preserve_boundary: bool,

    /// Whether to use parallel execution (default: true).
    pub parallel: bool,
}

impl SubdivideOptions {
    /// Create options with the specified number of iterations.
    pub fn new(iterations: usize) -> Self {
        Self {
            iterations,
            preserve_boundary: true,
            parallel: true,
        }
    }

    /// Set whether to preserve boundary edges.
    pub fn with_preserve_boundary(mut self, preserve: bool) -> Self {
        self.preserve_boundary = preserve;
        self
    }

    /// Set whether to use parallel execution.
    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    /// Create options for single-threaded execution.
    pub fn sequential(mut self) -> Self {
        self.parallel = false;
        self
    }
}
