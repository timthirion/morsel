//! Mesh file I/O.
//!
//! This module provides functions for loading and saving meshes in various formats.
//!
//! # Supported Formats
//!
//! | Format | Extension | Load | Save | Notes |
//! |--------|-----------|------|------|-------|
//! | Wavefront OBJ | `.obj` | ✓ | ✓ | Most common format |
//! | STL | `.stl` | ✓ | ✓ | Binary and ASCII |
//! | PLY | `.ply` | ✓ | ✓ | Stanford polygon format |
//! | glTF | `.gltf`, `.glb` | ✓ | ✗ | Modern 3D format |
//!
//! # Usage
//!
//! The easiest way to load and save meshes is using the automatic format detection:
//!
//! ```no_run
//! use morsel::io::{load, save};
//! use morsel::mesh::HalfEdgeMesh;
//!
//! // Load with automatic format detection
//! let mesh: HalfEdgeMesh = load("model.obj").unwrap();
//!
//! // Save with automatic format detection
//! save(&mesh, "output.stl").unwrap();
//! ```
//!
//! You can also use format-specific functions:
//!
//! ```no_run
//! use morsel::io::obj;
//! use morsel::mesh::HalfEdgeMesh;
//!
//! let mesh: HalfEdgeMesh = obj::load("model.obj").unwrap();
//! obj::save(&mesh, "output.obj").unwrap();
//! ```

pub mod gltf;
pub mod obj;
pub mod ply;
pub mod stl;

use std::path::Path;

use crate::error::{MeshError, Result};
use crate::mesh::{HalfEdgeMesh, MeshIndex};

/// Supported mesh file formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// Wavefront OBJ format.
    Obj,
    /// STL (stereolithography) format.
    Stl,
    /// PLY (Stanford polygon) format.
    Ply,
    /// glTF format.
    Gltf,
    /// glTF binary format.
    Glb,
}

impl Format {
    /// Detect format from file extension.
    pub fn from_extension(ext: &str) -> Option<Format> {
        match ext.to_lowercase().as_str() {
            "obj" => Some(Format::Obj),
            "stl" => Some(Format::Stl),
            "ply" => Some(Format::Ply),
            "gltf" => Some(Format::Gltf),
            "glb" => Some(Format::Glb),
            _ => None,
        }
    }

    /// Detect format from file path.
    pub fn from_path<P: AsRef<Path>>(path: P) -> Option<Format> {
        path.as_ref()
            .extension()
            .and_then(|ext| ext.to_str())
            .and_then(Format::from_extension)
    }
}

/// Load a mesh from a file with automatic format detection.
///
/// The format is determined by the file extension.
///
/// # Example
///
/// ```no_run
/// use morsel::io::load;
/// use morsel::mesh::HalfEdgeMesh;
///
/// let mesh: HalfEdgeMesh = load("model.obj").unwrap();
/// ```
pub fn load<P: AsRef<Path>, I: MeshIndex>(path: P) -> Result<HalfEdgeMesh<I>> {
    let path = path.as_ref();
    let format = Format::from_path(path).ok_or_else(|| MeshError::UnsupportedFormat {
        extension: path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("(none)")
            .to_string(),
    })?;

    match format {
        Format::Obj => obj::load(path),
        Format::Stl => stl::load(path),
        Format::Ply => ply::load(path),
        Format::Gltf | Format::Glb => gltf::load(path),
    }
}

/// Save a mesh to a file with automatic format detection.
///
/// The format is determined by the file extension.
///
/// # Example
///
/// ```no_run
/// use morsel::io::save;
/// use morsel::mesh::HalfEdgeMesh;
///
/// let mesh: HalfEdgeMesh = HalfEdgeMesh::new();
/// save(&mesh, "output.obj").unwrap();
/// ```
pub fn save<P: AsRef<Path>, I: MeshIndex>(mesh: &HalfEdgeMesh<I>, path: P) -> Result<()> {
    let path = path.as_ref();
    let format = Format::from_path(path).ok_or_else(|| MeshError::UnsupportedFormat {
        extension: path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("(none)")
            .to_string(),
    })?;

    match format {
        Format::Obj => obj::save(mesh, path),
        Format::Stl => stl::save(mesh, path),
        Format::Ply => ply::save(mesh, path),
        Format::Gltf | Format::Glb => Err(MeshError::SaveError {
            path: path.to_path_buf(),
            message: "glTF saving is not yet supported".to_string(),
        }),
    }
}
