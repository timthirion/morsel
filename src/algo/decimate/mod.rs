//! Mesh decimation (simplification) algorithms.
//!
//! This module provides algorithms for reducing the number of triangles in a mesh
//! while preserving its overall shape as much as possible.
//!
//! # Quadric Error Metrics (QEM)
//!
//! The QEM algorithm (Garland & Heckbert, 1997) is a widely-used decimation method
//! that minimizes geometric error during edge collapses. Each vertex maintains a
//! quadric matrix that represents the sum of squared distances to its original
//! adjacent planes.
//!
//! # Example
//!
//! ```no_run
//! use morsel::prelude::*;
//! use morsel::algo::decimate::{qem_decimate, DecimateOptions};
//!
//! let mut mesh: HalfEdgeMesh = morsel::io::load("input.obj").unwrap();
//!
//! // Reduce to 50% of original faces
//! let options = DecimateOptions::with_target_ratio(0.5);
//! let report = qem_decimate(&mut mesh, &options);
//!
//! // Worth checking: the target is not always reachable. Most often no remaining
//! // collapse is topologically safe, so decimation stops above the requested count.
//! if !report.completed() {
//!     eprintln!(
//!         "wanted {} faces, got {} ({:?})",
//!         report.faces_requested, report.faces_after, report.outcome
//!     );
//! }
//!
//! morsel::io::save(&mesh, "output.obj").unwrap();
//! ```
//!
//! # References
//!
//! - Garland, M. & Heckbert, P. (1997). "Surface Simplification Using Quadric
//!   Error Metrics." SIGGRAPH '97.

mod qem;

pub use qem::{qem_decimate, qem_decimate_with_progress};

/// How a decimation ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecimateOutcome {
    /// The requested target was reached.
    Completed,
    /// Nothing was asked for: an empty mesh, or a target at or above the current face
    /// count.
    NothingRequested,
    /// Every remaining collapse would break the surface, so decimation stopped above the
    /// requested target. The mesh is valid and as reduced as it can safely get.
    ///
    /// Usually this means the request was impossible rather than that anything failed: a
    /// two-triangle mesh cannot become one triangle, and `annulus_two_boundary_loops` has
    /// only so many interior edges to give. It also fires where a boundary constrains
    /// things — an interior edge with both endpoints on the boundary can never be
    /// collapsed, because pinching two stretches of boundary together makes a bowtie.
    ///
    /// The reduction achieved may be nothing at all: `examples/cube.obj` is six
    /// disconnected quads whose only interior edges are the diagonals, and collapsing one
    /// deletes a whole patch, so it exhausts at its input size of 12 faces.
    Exhausted,
    /// The requested target produced a mesh the half-edge representation would not
    /// accept, and a milder reduction was used instead. The result is valid but coarser
    /// than asked for — ask for 50% and you may get 75%.
    ///
    /// This should now be unreachable in practice. It existed to paper over
    /// `is_collapse_valid` missing a case, and the case it was missing — stale boundary
    /// flags — is fixed. It is kept because the validity check is a topological argument,
    /// not a proof, and silently emitting a corrupt mesh is the one outcome worth ruling
    /// out unconditionally.
    BackedOff,
    /// No attempted reduction produced a mesh that would rebuild, so the input is
    /// untouched. Distinct from [`DecimateOutcome::Exhausted`], where the collapses that
    /// *were* made are kept: this means even the back-off found nothing usable.
    ///
    /// Like `BackedOff`, this should no longer be reachable — a mesh that cannot be
    /// reduced safely now exhausts instead, which is the same result described honestly.
    /// `examples/cube.obj`, six disconnected quads whose only interior edges are the
    /// diagonals, used to land here and now reports `Exhausted` at its input size.
    Refused,
}

/// What a decimation did.
///
/// Reaching the requested face count is not guaranteed, and the outcome says which of
/// the several reasons applies. Most often it is [`DecimateOutcome::Exhausted`]: no
/// remaining collapse is topologically safe, so the mesh stopped above the target.
///
/// Whatever it reports is reproducible: the collapse order is deterministic, so the same
/// input and options give the same mesh down to the bit, run after run. See
/// `tests/decimate_determinism.rs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use = "decimation can back off or refuse; check the report"]
pub struct DecimateReport {
    /// How it ended.
    pub outcome: DecimateOutcome,
    /// Face count before.
    pub faces_before: usize,
    /// Face count the options asked for.
    pub faces_requested: usize,
    /// Face count after.
    pub faces_after: usize,
    /// Targets tried, including the first. More than one means the back-off ran.
    pub attempts: usize,
}

impl DecimateReport {
    /// Whether the requested target was reached.
    pub fn completed(&self) -> bool {
        self.outcome == DecimateOutcome::Completed
    }
}

/// Options for mesh decimation.
#[derive(Debug, Clone)]
pub struct DecimateOptions {
    /// Target number of faces after decimation.
    /// If None, uses target_ratio instead.
    pub target_faces: Option<usize>,

    /// Target ratio of faces to keep (0.0 to 1.0).
    /// Only used if target_faces is None.
    pub target_ratio: f64,

    /// Whether to preserve boundary edges (don't collapse them).
    pub preserve_boundary: bool,

    /// Maximum allowed error for a single edge collapse.
    /// Edges with error above this threshold won't be collapsed.
    pub max_error: Option<f64>,

    /// Whether to use parallel execution for initialization (default: true).
    /// Note: The main decimation loop is inherently sequential.
    pub parallel: bool,
}

impl DecimateOptions {
    /// Create options to reduce to a target number of faces.
    pub fn with_target_faces(target: usize) -> Self {
        Self {
            target_faces: Some(target),
            target_ratio: 0.5,
            preserve_boundary: true,
            max_error: None,
            parallel: true,
        }
    }

    /// Create options to reduce to a ratio of the original face count.
    pub fn with_target_ratio(ratio: f64) -> Self {
        Self {
            target_faces: None,
            target_ratio: ratio.clamp(0.0, 1.0),
            preserve_boundary: true,
            max_error: None,
            parallel: true,
        }
    }

    /// Set whether to preserve boundary edges.
    pub fn with_preserve_boundary(mut self, preserve: bool) -> Self {
        self.preserve_boundary = preserve;
        self
    }

    /// Set maximum error threshold for edge collapses.
    pub fn with_max_error(mut self, max_error: f64) -> Self {
        self.max_error = Some(max_error);
        self
    }

    /// Set whether to use parallel execution for initialization.
    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    /// Create options for single-threaded execution.
    pub fn sequential(mut self) -> Self {
        self.parallel = false;
        self
    }

    /// Compute the target number of faces given the original count.
    pub fn compute_target(&self, original_faces: usize) -> usize {
        if let Some(target) = self.target_faces {
            target.min(original_faces)
        } else {
            ((original_faces as f64) * self.target_ratio).round() as usize
        }
    }
}
