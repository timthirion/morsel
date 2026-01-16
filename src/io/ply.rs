//! PLY (Stanford polygon) format support.
//!
//! This module provides loading and saving of meshes in the PLY format,
//! also known as the Polygon File Format or Stanford Triangle Format.

use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

use nalgebra::Point3;
use ply_rs::parser::Parser;
use ply_rs::ply::{DefaultElement, Property};

use crate::error::{MeshError, Result};
use crate::mesh::{build_from_triangles, to_face_vertex, HalfEdgeMesh, MeshIndex};

/// Load a mesh from a PLY file.
///
/// # Example
///
/// ```no_run
/// use morsel::io::ply;
/// use morsel::mesh::HalfEdgeMesh;
///
/// let mesh: HalfEdgeMesh = ply::load("model.ply").unwrap();
/// ```
pub fn load<P: AsRef<Path>, I: MeshIndex>(path: P) -> Result<HalfEdgeMesh<I>> {
    let path = path.as_ref();
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    let parser = Parser::<DefaultElement>::new();
    let ply = parser.read_ply(&mut reader).map_err(|e| MeshError::LoadError {
        path: path.to_path_buf(),
        message: e.to_string(),
    })?;

    // Extract vertices
    let vertex_element = ply.payload.get("vertex").ok_or_else(|| MeshError::LoadError {
        path: path.to_path_buf(),
        message: "PLY file has no vertex element".to_string(),
    })?;

    let mut vertices: Vec<Point3<f64>> = Vec::with_capacity(vertex_element.len());
    for vertex in vertex_element {
        let x = get_float_property(vertex, "x").ok_or_else(|| MeshError::LoadError {
            path: path.to_path_buf(),
            message: "vertex missing x coordinate".to_string(),
        })?;
        let y = get_float_property(vertex, "y").ok_or_else(|| MeshError::LoadError {
            path: path.to_path_buf(),
            message: "vertex missing y coordinate".to_string(),
        })?;
        let z = get_float_property(vertex, "z").ok_or_else(|| MeshError::LoadError {
            path: path.to_path_buf(),
            message: "vertex missing z coordinate".to_string(),
        })?;
        vertices.push(Point3::new(x, y, z));
    }

    // Extract faces
    let face_element = ply.payload.get("face").ok_or_else(|| MeshError::LoadError {
        path: path.to_path_buf(),
        message: "PLY file has no face element".to_string(),
    })?;

    let mut faces: Vec<[usize; 3]> = Vec::with_capacity(face_element.len());
    for face in face_element {
        let indices = get_list_property(face, "vertex_indices")
            .or_else(|| get_list_property(face, "vertex_index"))
            .ok_or_else(|| MeshError::LoadError {
                path: path.to_path_buf(),
                message: "face missing vertex_indices property".to_string(),
            })?;

        if indices.len() == 3 {
            faces.push([indices[0], indices[1], indices[2]]);
        } else if indices.len() > 3 {
            // Triangulate polygon by fan triangulation
            for i in 1..indices.len() - 1 {
                faces.push([indices[0], indices[i], indices[i + 1]]);
            }
        }
    }

    if faces.is_empty() {
        return Err(MeshError::LoadError {
            path: path.to_path_buf(),
            message: "PLY file contains no faces".to_string(),
        });
    }

    build_from_triangles(&vertices, &faces)
}

fn get_float_property(element: &DefaultElement, name: &str) -> Option<f64> {
    match element.get(name)? {
        Property::Float(v) => Some(*v as f64),
        Property::Double(v) => Some(*v),
        Property::Int(v) => Some(*v as f64),
        Property::UInt(v) => Some(*v as f64),
        Property::Short(v) => Some(*v as f64),
        Property::UShort(v) => Some(*v as f64),
        Property::Char(v) => Some(*v as f64),
        Property::UChar(v) => Some(*v as f64),
        _ => None,
    }
}

fn get_list_property(element: &DefaultElement, name: &str) -> Option<Vec<usize>> {
    match element.get(name)? {
        Property::ListInt(v) => Some(v.iter().map(|&x| x as usize).collect()),
        Property::ListUInt(v) => Some(v.iter().map(|&x| x as usize).collect()),
        Property::ListShort(v) => Some(v.iter().map(|&x| x as usize).collect()),
        Property::ListUShort(v) => Some(v.iter().map(|&x| x as usize).collect()),
        Property::ListChar(v) => Some(v.iter().map(|&x| x as usize).collect()),
        Property::ListUChar(v) => Some(v.iter().map(|&x| x as usize).collect()),
        _ => None,
    }
}

/// Save a mesh to a PLY file (ASCII format).
///
/// # Example
///
/// ```no_run
/// use morsel::io::ply;
/// use morsel::mesh::HalfEdgeMesh;
///
/// let mesh: HalfEdgeMesh = HalfEdgeMesh::new();
/// ply::save(&mesh, "output.ply").unwrap();
/// ```
pub fn save<P: AsRef<Path>, I: MeshIndex>(mesh: &HalfEdgeMesh<I>, path: P) -> Result<()> {
    let path = path.as_ref();
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    let (vertices, faces) = to_face_vertex(mesh);

    // Write header
    writeln!(writer, "ply")?;
    writeln!(writer, "format ascii 1.0")?;
    writeln!(writer, "comment Generated by morsel")?;
    writeln!(writer, "element vertex {}", vertices.len())?;
    writeln!(writer, "property float x")?;
    writeln!(writer, "property float y")?;
    writeln!(writer, "property float z")?;
    writeln!(writer, "element face {}", faces.len())?;
    writeln!(writer, "property list uchar int vertex_indices")?;
    writeln!(writer, "end_header")?;

    // Write vertices
    for v in &vertices {
        writeln!(writer, "{} {} {}", v.x, v.y, v.z)?;
    }

    // Write faces
    for f in &faces {
        writeln!(writer, "3 {} {} {}", f[0], f[1], f[2])?;
    }

    writer.flush()?;
    Ok(())
}
