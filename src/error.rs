//! Error types for morsel.
//!
//! This module defines all error types used throughout the library.

use std::path::PathBuf;
use thiserror::Error;

/// Result type alias using [`MeshError`].
pub type Result<T> = std::result::Result<T, MeshError>;

/// Errors that can occur during mesh operations.
#[derive(Error, Debug)]
pub enum MeshError {
    /// The mesh has no faces.
    #[error("mesh has no faces")]
    EmptyMesh,

    /// A face references an invalid vertex index.
    #[error("face {face} references invalid vertex index {vertex}")]
    InvalidVertexIndex {
        /// The face index.
        face: usize,
        /// The invalid vertex index.
        vertex: usize,
    },

    /// A face has duplicate vertex indices (degenerate triangle).
    #[error("face {face} is degenerate (has duplicate vertices)")]
    DegenerateFace {
        /// The face index.
        face: usize,
    },

    /// The mesh has non-manifold topology.
    #[error("mesh has non-manifold topology: {details}")]
    NonManifold {
        /// Description of the non-manifold condition.
        details: String,
    },

    /// An edge has more than two incident faces.
    #[error("edge ({v0}, {v1}) has more than two incident faces")]
    NonManifoldEdge {
        /// First vertex of the edge.
        v0: usize,
        /// Second vertex of the edge.
        v1: usize,
    },

    /// File I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Error loading mesh from file.
    #[error("failed to load mesh from {path}: {message}")]
    LoadError {
        /// The file path.
        path: PathBuf,
        /// Error message.
        message: String,
    },

    /// Error saving mesh to file.
    #[error("failed to save mesh to {path}: {message}")]
    SaveError {
        /// The file path.
        path: PathBuf,
        /// Error message.
        message: String,
    },

    /// Unsupported file format.
    #[error("unsupported file format: {extension}")]
    UnsupportedFormat {
        /// The file extension.
        extension: String,
    },

    /// Invalid mesh state for the requested operation.
    #[error("invalid mesh state: {0}")]
    InvalidState(String),

    /// Algorithm failed to converge.
    #[error("algorithm failed to converge after {iterations} iterations")]
    ConvergenceFailed {
        /// Number of iterations attempted.
        iterations: usize,
    },

    /// Invalid parameter value.
    #[error("invalid parameter: {name} = {value} ({reason})")]
    InvalidParameter {
        /// Parameter name.
        name: &'static str,
        /// The invalid value (as string).
        value: String,
        /// Reason the value is invalid.
        reason: &'static str,
    },
}

impl MeshError {
    /// Create an invalid parameter error.
    pub fn invalid_param<T: std::fmt::Display>(
        name: &'static str,
        value: T,
        reason: &'static str,
    ) -> Self {
        MeshError::InvalidParameter {
            name,
            value: value.to_string(),
            reason,
        }
    }
}
