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

/// How a subdivision ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubdivideOutcome {
    /// Every requested level ran.
    Completed,
    /// Nothing was asked for: zero iterations.
    NothingRequested,
    /// Catmull-Clark was handed a mesh that is not all quads. It is a quad scheme and
    /// its per-face data assumes four corners, so it declines rather than reading past
    /// the end — which is what it used to do, panicking on the invalid-index sentinel.
    NotAQuadMesh,
    /// A level produced a mesh the half-edge representation will not accept: a repeated
    /// directed edge, an edge with three faces, or a bowtie vertex. The mesh is left at
    /// the last level that did succeed.
    RebuildRejected,
}

/// What a subdivision did.
///
/// Worth checking rather than ignoring: subdivision rebuilds the mesh from a face list,
/// and a rejected rebuild leaves the mesh at whatever level last worked. Before this
/// reported anything, that was indistinguishable from success.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use = "subdivision can stop early; check the report"]
pub struct SubdivideReport {
    /// Levels that actually ran.
    pub iterations_run: usize,
    /// How it ended.
    pub outcome: SubdivideOutcome,
    /// Face count before.
    pub faces_before: usize,
    /// Face count after.
    pub faces_after: usize,
}

impl SubdivideReport {
    /// Whether every requested level ran.
    pub fn completed(&self) -> bool {
        self.outcome == SubdivideOutcome::Completed
    }
}

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
