//! STL (stereolithography) format support.
//!
//! This module provides loading and saving of meshes in the STL format,
//! commonly used for 3D printing. Both binary and ASCII formats are supported.

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use nalgebra::Point3;

use crate::error::{MeshError, Result};
use crate::mesh::{build_from_triangles, to_face_vertex, HalfEdgeMesh, MeshIndex};

/// Load a mesh from an STL file.
///
/// Automatically detects binary vs ASCII format.
///
/// # Example
///
/// ```no_run
/// use morsel::io::stl;
/// use morsel::mesh::HalfEdgeMesh;
///
/// let mesh: HalfEdgeMesh = stl::load("model.stl").unwrap();
/// ```
pub fn load<P: AsRef<Path>, I: MeshIndex>(path: P) -> Result<HalfEdgeMesh<I>> {
    let path = path.as_ref();
    let mut file = File::open(path)?;

    let stl = stl_io::read_stl(&mut file).map_err(|e| MeshError::LoadError {
        path: path.to_path_buf(),
        message: e.to_string(),
    })?;

    // STL stores vertices per-triangle, so we need to deduplicate
    let mut vertices: Vec<Point3<f64>> = Vec::new();
    let mut faces: Vec<[usize; 3]> = Vec::new();

    // Simple vertex deduplication using a tolerance
    const EPSILON: f64 = 1e-10;

    fn find_or_add_vertex(vertices: &mut Vec<Point3<f64>>, p: Point3<f64>) -> usize {
        for (i, v) in vertices.iter().enumerate() {
            if (v - p).norm() < EPSILON {
                return i;
            }
        }
        let idx = vertices.len();
        vertices.push(p);
        idx
    }

    for tri in &stl.faces {
        // tri.vertices contains indices into stl.vertices
        let vtx0 = &stl.vertices[tri.vertices[0]];
        let vtx1 = &stl.vertices[tri.vertices[1]];
        let vtx2 = &stl.vertices[tri.vertices[2]];

        let v0 = Point3::new(vtx0[0] as f64, vtx0[1] as f64, vtx0[2] as f64);
        let v1 = Point3::new(vtx1[0] as f64, vtx1[1] as f64, vtx1[2] as f64);
        let v2 = Point3::new(vtx2[0] as f64, vtx2[1] as f64, vtx2[2] as f64);

        let i0 = find_or_add_vertex(&mut vertices, v0);
        let i1 = find_or_add_vertex(&mut vertices, v1);
        let i2 = find_or_add_vertex(&mut vertices, v2);

        // Skip degenerate triangles
        if i0 != i1 && i1 != i2 && i0 != i2 {
            faces.push([i0, i1, i2]);
        }
    }

    if faces.is_empty() {
        return Err(MeshError::LoadError {
            path: path.to_path_buf(),
            message: "STL file contains no valid triangles".to_string(),
        });
    }

    build_from_triangles(&vertices, &faces)
}

/// Save a mesh to a binary STL file.
///
/// # Example
///
/// ```no_run
/// use morsel::io::stl;
/// use morsel::mesh::HalfEdgeMesh;
///
/// let mesh: HalfEdgeMesh = HalfEdgeMesh::new();
/// stl::save(&mesh, "output.stl").unwrap();
/// ```
pub fn save<P: AsRef<Path>, I: MeshIndex>(mesh: &HalfEdgeMesh<I>, path: P) -> Result<()> {
    let path = path.as_ref();
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    let (vertices, faces) = to_face_vertex(mesh);

    let triangles: Vec<stl_io::Triangle> = faces
        .iter()
        .map(|f| {
            let p0 = &vertices[f[0]];
            let p1 = &vertices[f[1]];
            let p2 = &vertices[f[2]];

            // Compute normal
            let e1 = p1 - p0;
            let e2 = p2 - p0;
            let n = e1.cross(&e2).normalize();

            stl_io::Triangle {
                normal: stl_io::Normal::new([n.x as f32, n.y as f32, n.z as f32]),
                vertices: [
                    stl_io::Vertex::new([p0.x as f32, p0.y as f32, p0.z as f32]),
                    stl_io::Vertex::new([p1.x as f32, p1.y as f32, p1.z as f32]),
                    stl_io::Vertex::new([p2.x as f32, p2.y as f32, p2.z as f32]),
                ],
            }
        })
        .collect();

    stl_io::write_stl(&mut writer, triangles.iter()).map_err(|e| MeshError::SaveError {
        path: path.to_path_buf(),
        message: e.to_string(),
    })?;

    Ok(())
}
