//! glTF format support.
//!
//! This module provides loading of meshes from glTF and GLB files.
//! glTF is a modern 3D format designed for efficient transmission and loading.
//!
//! Note: Saving to glTF is not yet supported.

use std::path::Path;

use nalgebra::Point3;

use crate::error::{MeshError, Result};
use crate::mesh::{build_from_triangles, HalfEdgeMesh, MeshIndex};

/// Load a mesh from a glTF or GLB file.
///
/// This function loads all meshes from the file and combines them into a single mesh.
///
/// # Example
///
/// ```no_run
/// use morsel::io::gltf;
/// use morsel::mesh::HalfEdgeMesh;
///
/// let mesh: HalfEdgeMesh = gltf::load("model.gltf").unwrap();
/// ```
pub fn load<P: AsRef<Path>, I: MeshIndex>(path: P) -> Result<HalfEdgeMesh<I>> {
    let path = path.as_ref();

    let (document, buffers, _images) = ::gltf::import(path).map_err(|e| MeshError::LoadError {
        path: path.to_path_buf(),
        message: e.to_string(),
    })?;

    let mut all_vertices: Vec<Point3<f64>> = Vec::new();
    let mut all_faces: Vec<[usize; 3]> = Vec::new();

    for mesh in document.meshes() {
        for primitive in mesh.primitives() {
            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

            let vertex_offset = all_vertices.len();

            // Read positions
            if let Some(positions) = reader.read_positions() {
                for pos in positions {
                    all_vertices.push(Point3::new(pos[0] as f64, pos[1] as f64, pos[2] as f64));
                }
            }

            // Read indices
            if let Some(indices) = reader.read_indices() {
                let indices: Vec<usize> = indices.into_u32().map(|i| i as usize).collect();

                // Convert to triangles based on primitive mode
                match primitive.mode() {
                    ::gltf::mesh::Mode::Triangles => {
                        for chunk in indices.chunks(3) {
                            if chunk.len() == 3 {
                                all_faces.push([
                                    chunk[0] + vertex_offset,
                                    chunk[1] + vertex_offset,
                                    chunk[2] + vertex_offset,
                                ]);
                            }
                        }
                    }
                    ::gltf::mesh::Mode::TriangleStrip => {
                        for i in 0..indices.len().saturating_sub(2) {
                            if i % 2 == 0 {
                                all_faces.push([
                                    indices[i] + vertex_offset,
                                    indices[i + 1] + vertex_offset,
                                    indices[i + 2] + vertex_offset,
                                ]);
                            } else {
                                // Reverse winding for odd triangles
                                all_faces.push([
                                    indices[i] + vertex_offset,
                                    indices[i + 2] + vertex_offset,
                                    indices[i + 1] + vertex_offset,
                                ]);
                            }
                        }
                    }
                    ::gltf::mesh::Mode::TriangleFan => {
                        for i in 1..indices.len().saturating_sub(1) {
                            all_faces.push([
                                indices[0] + vertex_offset,
                                indices[i] + vertex_offset,
                                indices[i + 1] + vertex_offset,
                            ]);
                        }
                    }
                    _ => {
                        // Skip non-triangle primitives (points, lines)
                    }
                }
            }
        }
    }

    if all_faces.is_empty() {
        return Err(MeshError::LoadError {
            path: path.to_path_buf(),
            message: "glTF file contains no triangle meshes".to_string(),
        });
    }

    build_from_triangles(&all_vertices, &all_faces)
}
