//! Mesh remeshing algorithms.
//!
//! This module provides algorithms for remeshing triangle meshes:
//!
//! - **Isotropic remeshing**: Uniform edge lengths throughout the mesh
//! - **Anisotropic remeshing**: Edge lengths adapted to local curvature
//! - **CVT remeshing**: Centroidal Voronoi Tessellation for optimal vertex distribution
//!
//! # Isotropic Remeshing
//!
//! The isotropic remeshing algorithm (Botsch & Kobbelt, 2004) iteratively applies:
//!
//! 1. **Split** edges longer than 4/3 × target_length
//! 2. **Collapse** edges shorter than 4/5 × target_length
//! 3. **Flip** edges to improve vertex valence
//! 4. **Tangential smoothing** to regularize vertex positions
//!
//! # Anisotropic Remeshing
//!
//! Anisotropic remeshing adapts edge lengths based on local curvature:
//! - High curvature regions get shorter edges (more detail)
//! - Low curvature regions get longer edges (fewer triangles)
//!
//! # CVT Remeshing
//!
//! CVT remeshing uses Lloyd's algorithm to optimize vertex positions:
//! - Creates well-shaped, nearly equilateral triangles
//! - Vertices are distributed according to a density function
//! - Iteratively moves vertices to Voronoi cell centroids
//!
//! # Example
//!
//! ```no_run
//! use morsel::prelude::*;
//! use morsel::algo::remesh::{isotropic_remesh, RemeshOptions};
//!
//! let mut mesh: HalfEdgeMesh = morsel::io::load("input.obj").unwrap();
//!
//! let options = RemeshOptions::with_target_length(0.1)
//!     .with_iterations(5);
//! isotropic_remesh(&mut mesh, &options);
//!
//! morsel::io::save(&mesh, "output.obj").unwrap();
//! ```
//!
//! # References
//!
//! - Botsch, M., & Kobbelt, L. (2004). "A remeshing approach to multiresolution modeling."
//!   Symposium on Geometry Processing.
//! - Dunyach, M., et al. (2013). "Adaptive remeshing for real-time mesh deformation."
//!   Eurographics.
//! - Du, Q., et al. (1999). "Centroidal Voronoi tessellations."
//!   SIAM Review.

mod anisotropic;
mod cvt;
mod isotropic;

pub use anisotropic::{
    anisotropic_remesh, anisotropic_remesh_with_progress, compute_sizing_field, AnisotropicOptions,
    SizingField,
};
pub use cvt::{cvt_remesh, cvt_remesh_with_progress, CvtOptions};
pub use isotropic::{
    average_edge_length, isotropic_remesh, isotropic_remesh_with_progress, RemeshOptions,
};

use std::collections::{HashMap, HashSet};

use nalgebra::{Point3, Vector3};
use rayon::prelude::*;

use crate::mesh::{build_from_triangles, to_face_vertex, HalfEdgeMesh, MeshIndex};

// ============================================================================
// Pre-computed Mesh Topology for O(1) Lookups
// ============================================================================

/// Pre-computed mesh topology for efficient remeshing operations.
///
/// This structure caches edge and vertex relationships to avoid O(n) scans
/// through the face list for each query.
#[derive(Debug, Clone)]
pub struct MeshTopology {
    /// Map from edge (v0, v1) where v0 < v1 to list of face indices containing it
    pub edge_faces: HashMap<(usize, usize), Vec<usize>>,
    /// Set of boundary edges (edges with only one adjacent face)
    pub boundary_edges: HashSet<(usize, usize)>,
    /// Set of boundary vertices
    pub boundary_vertices: HashSet<usize>,
    /// Neighbors for each vertex
    pub vertex_neighbors: Vec<HashSet<usize>>,
    /// Number of vertices
    pub num_vertices: usize,
}

impl MeshTopology {
    /// Build topology from a face list.
    pub fn from_faces(faces: &[[usize; 3]], num_vertices: usize) -> Self {
        // Build edge to faces mapping
        let mut edge_faces: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
        for (face_idx, face) in faces.iter().enumerate() {
            for i in 0..3 {
                let v0 = face[i];
                let v1 = face[(i + 1) % 3];
                let edge = if v0 < v1 { (v0, v1) } else { (v1, v0) };
                edge_faces.entry(edge).or_default().push(face_idx);
            }
        }

        // Find boundary edges (only one adjacent face)
        let boundary_edges: HashSet<(usize, usize)> = edge_faces
            .iter()
            .filter(|(_, faces)| faces.len() == 1)
            .map(|(&edge, _)| edge)
            .collect();

        // Build vertex neighbors
        let mut vertex_neighbors: Vec<HashSet<usize>> = vec![HashSet::new(); num_vertices];
        for face in faces {
            vertex_neighbors[face[0]].insert(face[1]);
            vertex_neighbors[face[0]].insert(face[2]);
            vertex_neighbors[face[1]].insert(face[0]);
            vertex_neighbors[face[1]].insert(face[2]);
            vertex_neighbors[face[2]].insert(face[0]);
            vertex_neighbors[face[2]].insert(face[1]);
        }

        // Find boundary vertices (vertices incident to boundary edges)
        let mut boundary_vertices: HashSet<usize> = HashSet::new();
        for &(v0, v1) in &boundary_edges {
            boundary_vertices.insert(v0);
            boundary_vertices.insert(v1);
        }

        Self {
            edge_faces,
            boundary_edges,
            boundary_vertices,
            vertex_neighbors,
            num_vertices,
        }
    }

    /// Check if an edge is on the boundary.
    #[inline]
    pub fn is_boundary_edge(&self, v0: usize, v1: usize) -> bool {
        let edge = if v0 < v1 { (v0, v1) } else { (v1, v0) };
        self.boundary_edges.contains(&edge)
    }

    /// Check if a vertex is on the boundary.
    #[inline]
    pub fn is_boundary_vertex(&self, v: usize) -> bool {
        self.boundary_vertices.contains(&v)
    }

    /// Get neighbors of a vertex.
    #[inline]
    pub fn neighbors(&self, v: usize) -> &HashSet<usize> {
        &self.vertex_neighbors[v]
    }

    /// Check if an edge exists.
    #[inline]
    pub fn edge_exists(&self, v0: usize, v1: usize) -> bool {
        let edge = if v0 < v1 { (v0, v1) } else { (v1, v0) };
        self.edge_faces.contains_key(&edge)
    }

    /// Get faces adjacent to an edge.
    #[inline]
    pub fn get_edge_faces(&self, v0: usize, v1: usize) -> Option<&Vec<usize>> {
        let edge = if v0 < v1 { (v0, v1) } else { (v1, v0) };
        self.edge_faces.get(&edge)
    }
}

// ============================================================================
// Shared Helpers - Edge/Face Queries
// ============================================================================

/// Check if an edge is on the boundary (appears in only one face).
pub(crate) fn is_boundary_edge_in_faces(faces: &[[usize; 3]], v0: usize, v1: usize) -> bool {
    let mut count = 0;
    for face in faces {
        for i in 0..3 {
            let a = face[i];
            let b = face[(i + 1) % 3];
            if (a == v0 && b == v1) || (a == v1 && b == v0) {
                count += 1;
            }
        }
    }
    count == 1
}

/// Check if a vertex is on the boundary.
pub(crate) fn is_boundary_vertex_in_faces(faces: &[[usize; 3]], v: usize) -> bool {
    for &neighbor in &get_vertex_neighbors(faces, v) {
        if is_boundary_edge_in_faces(faces, v, neighbor) {
            return true;
        }
    }
    false
}

/// Check if an edge exists in the face list.
#[allow(dead_code)]
pub(crate) fn edge_exists_in_faces(faces: &[[usize; 3]], v0: usize, v1: usize) -> bool {
    for face in faces {
        for i in 0..3 {
            let a = face[i];
            let b = face[(i + 1) % 3];
            if (a == v0 && b == v1) || (a == v1 && b == v0) {
                return true;
            }
        }
    }
    false
}

/// Get all vertex neighbors from the face list.
pub(crate) fn get_vertex_neighbors(faces: &[[usize; 3]], v: usize) -> HashSet<usize> {
    let mut neighbors = HashSet::new();
    for face in faces {
        for i in 0..3 {
            if face[i] == v {
                neighbors.insert(face[(i + 1) % 3]);
                neighbors.insert(face[(i + 2) % 3]);
            }
        }
    }
    neighbors
}

// ============================================================================
// Shared Helpers - Edge Operations
// ============================================================================

/// Split an edge by inserting a vertex at its midpoint.
pub(crate) fn split_edge(
    vertices: &mut Vec<Point3<f64>>,
    faces: &mut Vec<[usize; 3]>,
    v0: usize,
    v1: usize,
) {
    // Add midpoint vertex
    let midpoint = Point3::from((vertices[v0].coords + vertices[v1].coords) * 0.5);
    let mid_idx = vertices.len();
    vertices.push(midpoint);

    // Find and replace faces containing this edge
    let mut new_faces: Vec<[usize; 3]> = Vec::new();
    let mut i = 0;
    while i < faces.len() {
        let face = faces[i];

        // Check if this face contains the edge
        let mut edge_idx = None;
        for j in 0..3 {
            let a = face[j];
            let b = face[(j + 1) % 3];
            if (a == v0 && b == v1) || (a == v1 && b == v0) {
                edge_idx = Some(j);
                break;
            }
        }

        if let Some(j) = edge_idx {
            // Split this face into two
            let a = face[j];
            let b = face[(j + 1) % 3];
            let c = face[(j + 2) % 3];

            // Remove old face
            faces.swap_remove(i);

            // Add two new faces
            new_faces.push([a, mid_idx, c]);
            new_faces.push([mid_idx, b, c]);
        } else {
            i += 1;
        }
    }

    faces.extend(new_faces);
}

/// Collapse an edge by merging v1 into v0.
pub(crate) fn collapse_edge(
    vertices: &mut Vec<Point3<f64>>,
    faces: &mut Vec<[usize; 3]>,
    v0: usize,
    v1: usize,
) {
    // Move v0 to midpoint
    vertices[v0] = Point3::from((vertices[v0].coords + vertices[v1].coords) * 0.5);

    // Replace all references to v1 with v0
    for face in faces.iter_mut() {
        for v in face.iter_mut() {
            if *v == v1 {
                *v = v0;
            }
        }
    }

    // Remove degenerate faces (faces with duplicate vertices)
    faces.retain(|face| face[0] != face[1] && face[1] != face[2] && face[0] != face[2]);
}

/// Remove unused vertices, duplicate faces, non-manifold edges, and reindex faces.
pub(crate) fn cleanup_mesh(
    vertices: &[Point3<f64>],
    faces: &[[usize; 3]],
) -> (Vec<Point3<f64>>, Vec<[usize; 3]>) {
    // First, remove duplicate faces (faces with identical vertex lists, same winding)
    // We normalize by rotating so the smallest index is first (preserves winding)
    let mut unique_faces: Vec<[usize; 3]> = Vec::with_capacity(faces.len());
    let mut seen_faces: HashSet<[usize; 3]> = HashSet::new();

    for &face in faces {
        // Normalize face for comparison: rotate so smallest index is first (preserves winding)
        let min_idx = if face[0] <= face[1] && face[0] <= face[2] {
            0
        } else if face[1] <= face[2] {
            1
        } else {
            2
        };
        let normalized = [face[min_idx], face[(min_idx + 1) % 3], face[(min_idx + 2) % 3]];
        if seen_faces.insert(normalized) {
            unique_faces.push(face);
        }
    }

    // Remove faces that create non-manifold edges (edges shared by >2 faces)
    // or duplicate directed edges (inconsistent winding)
    // Keep removing until all edges are manifold
    loop {
        // Count edge occurrences (undirected)
        let mut edge_counts: HashMap<(usize, usize), usize> = HashMap::new();
        // Count directed edge occurrences (for winding consistency)
        let mut directed_counts: HashMap<(usize, usize), Vec<usize>> = HashMap::new();

        for (fi, face) in unique_faces.iter().enumerate() {
            for i in 0..3 {
                let v0 = face[i];
                let v1 = face[(i + 1) % 3];
                let edge = if v0 < v1 { (v0, v1) } else { (v1, v0) };
                *edge_counts.entry(edge).or_insert(0) += 1;
                directed_counts.entry((v0, v1)).or_default().push(fi);
            }
        }

        // Find non-manifold edges
        let non_manifold_edges: HashSet<(usize, usize)> = edge_counts
            .iter()
            .filter(|(_, &count)| count > 2)
            .map(|(&edge, _)| edge)
            .collect();

        // Find duplicate directed edges (inconsistent winding)
        let duplicate_directed: Vec<((usize, usize), Vec<usize>)> = directed_counts
            .into_iter()
            .filter(|(_, faces)| faces.len() > 1)
            .collect();

        if non_manifold_edges.is_empty() && duplicate_directed.is_empty() {
            break;
        }

        let mut faces_to_remove: HashSet<usize> = HashSet::new();

        // Handle duplicate directed edges first (remove all but one face for each)
        for ((_v0, _v1), face_indices) in &duplicate_directed {
            // Keep the first face, remove the rest
            for &fi in face_indices.iter().skip(1) {
                faces_to_remove.insert(fi);
            }
            #[cfg(debug_assertions)]
            eprintln!(
                "cleanup_mesh: removing {} faces with duplicate directed edge ({}, {})",
                face_indices.len() - 1,
                _v0,
                _v1
            );
        }

        // Handle non-manifold edges
        for &nm_edge in &non_manifold_edges {
            // Find all faces containing this edge
            let mut edge_faces: Vec<(usize, f64)> = Vec::new();
            for (fi, face) in unique_faces.iter().enumerate() {
                if faces_to_remove.contains(&fi) {
                    continue;
                }
                for i in 0..3 {
                    let v0 = face[i];
                    let v1 = face[(i + 1) % 3];
                    let edge = if v0 < v1 { (v0, v1) } else { (v1, v0) };
                    if edge == nm_edge {
                        // Compute face area for priority
                        let p0 = &vertices[face[0]];
                        let p1 = &vertices[face[1]];
                        let p2 = &vertices[face[2]];
                        let area = (p1 - p0).cross(&(p2 - p0)).norm();
                        edge_faces.push((fi, area));
                        break;
                    }
                }
            }

            // Sort by area (smallest first) and remove excess faces
            edge_faces.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            // Keep at most 2 faces per edge
            for (fi, _) in edge_faces.iter().skip(2) {
                faces_to_remove.insert(*fi);
            }
        }

        if faces_to_remove.is_empty() {
            break;
        }

        // Remove marked faces
        unique_faces = unique_faces
            .into_iter()
            .enumerate()
            .filter(|(i, _)| !faces_to_remove.contains(i))
            .map(|(_, f)| f)
            .collect();
    }

    // Find used vertices
    let mut used: HashSet<usize> = HashSet::new();
    for face in &unique_faces {
        used.insert(face[0]);
        used.insert(face[1]);
        used.insert(face[2]);
    }

    // Create mapping from old to new indices
    let mut old_to_new: Vec<Option<usize>> = vec![None; vertices.len()];
    let mut new_vertices: Vec<Point3<f64>> = Vec::new();

    for old_idx in 0..vertices.len() {
        if used.contains(&old_idx) {
            old_to_new[old_idx] = Some(new_vertices.len());
            new_vertices.push(vertices[old_idx]);
        }
    }

    // Reindex faces
    let new_faces: Vec<[usize; 3]> = unique_faces
        .iter()
        .map(|face| {
            [
                old_to_new[face[0]].unwrap(),
                old_to_new[face[1]].unwrap(),
                old_to_new[face[2]].unwrap(),
            ]
        })
        .collect();

    (new_vertices, new_faces)
}

// ============================================================================
// Shared Helpers - Edge Flipping
// ============================================================================

/// Validate a face list for manifold properties.
pub(crate) fn validate_face_list(vertices: &[Point3<f64>], faces: &[[usize; 3]]) -> bool {
    // Check vertex indices
    for face in faces {
        for &vi in face {
            if vi >= vertices.len() {
                return false;
            }
        }
    }

    // Check for non-manifold edges (>2 faces sharing an edge)
    let mut edge_counts: std::collections::HashMap<(usize, usize), usize> =
        std::collections::HashMap::new();
    for face in faces {
        for i in 0..3 {
            let v0 = face[i];
            let v1 = face[(i + 1) % 3];
            let edge = if v0 < v1 { (v0, v1) } else { (v1, v0) };
            *edge_counts.entry(edge).or_insert(0) += 1;
        }
    }
    if edge_counts.values().any(|&c| c > 2) {
        return false;
    }

    // Check for duplicate directed edges (inconsistent winding)
    let mut directed_counts: std::collections::HashMap<(usize, usize), usize> =
        std::collections::HashMap::new();
    for face in faces {
        for i in 0..3 {
            let v0 = face[i];
            let v1 = face[(i + 1) % 3];
            *directed_counts.entry((v0, v1)).or_insert(0) += 1;
        }
    }
    if directed_counts.values().any(|&c| c > 1) {
        return false;
    }

    true
}

/// Flip edges to improve vertex valence (operates on face list).
///
/// Uses batch processing: finds all edges that should be flipped, selects
/// independent ones, and flips them all at once before rebuilding topology.
pub(crate) fn flip_edges_for_valence_faces(
    vertices: &[Point3<f64>],
    faces: &mut Vec<[usize; 3]>,
    preserve_boundary: bool,
) {
    let max_batches = 50; // Maximum number of batches
    let mut failed_edges: HashSet<(usize, usize)> = HashSet::new();

    for _batch in 0..max_batches {
        // Build topology once per batch
        let topology = MeshTopology::from_faces(faces, vertices.len());

        // Find ALL edges that should be flipped
        let mut candidate_edges: Vec<(usize, usize)> = Vec::new();

        for &(v0, v1) in topology.edge_faces.keys() {
            if failed_edges.contains(&(v0, v1)) {
                continue;
            }

            // Skip boundary edges
            if preserve_boundary && topology.is_boundary_edge(v0, v1) {
                continue;
            }

            // Check if flip improves valence
            if should_flip_edge_fast(vertices, faces, &topology, v0, v1) {
                candidate_edges.push((v0, v1));
            }
        }

        if candidate_edges.is_empty() {
            break;
        }

        // Select independent edges (no shared vertices or adjacent faces)
        let mut used_vertices: HashSet<usize> = HashSet::new();
        let mut edges_to_flip: Vec<(usize, usize)> = Vec::new();

        for (v0, v1) in candidate_edges {
            if !used_vertices.contains(&v0) && !used_vertices.contains(&v1) {
                // Get opposite vertices to ensure independence
                if let Some(face_indices) = topology.get_edge_faces(v0, v1) {
                    if face_indices.len() == 2 {
                        let mut opp_verts: Vec<usize> = Vec::new();
                        for &fi in face_indices {
                            for &v in &faces[fi] {
                                if v != v0 && v != v1 {
                                    opp_verts.push(v);
                                }
                            }
                        }

                        // Check if any of the 4 involved vertices are already used
                        let all_verts_free = !used_vertices.contains(&v0)
                            && !used_vertices.contains(&v1)
                            && opp_verts.iter().all(|v| !used_vertices.contains(v));

                        if all_verts_free {
                            edges_to_flip.push((v0, v1));
                            used_vertices.insert(v0);
                            used_vertices.insert(v1);
                            for &v in &opp_verts {
                                used_vertices.insert(v);
                            }
                        }
                    }
                }
            }
        }

        if edges_to_flip.is_empty() {
            // All candidates conflict, try single flip
            let topology = MeshTopology::from_faces(faces, vertices.len());
            let edge = find_edge_to_flip_fast(vertices, faces, &topology, preserve_boundary, &failed_edges);

            match edge {
                Some((v0, v1)) => {
                    let saved_faces = faces.clone();
                    flip_edge(faces, v0, v1);
                    if !validate_face_list(vertices, faces) {
                        *faces = saved_faces;
                        let edge = if v0 < v1 { (v0, v1) } else { (v1, v0) };
                        failed_edges.insert(edge);
                    }
                }
                None => break,
            }
        } else {
            // Batch flip all selected edges
            let saved_faces = faces.clone();
            let mut any_failed = false;

            for (v0, v1) in &edges_to_flip {
                flip_edge(faces, *v0, *v1);
            }

            // Validate the result
            if !validate_face_list(vertices, faces) {
                // Batch failed, revert and try one by one
                *faces = saved_faces;
                any_failed = true;
            }

            if any_failed {
                // Fall back to one-by-one with validation
                for (v0, v1) in edges_to_flip {
                    let saved = faces.clone();
                    flip_edge(faces, v0, v1);
                    if !validate_face_list(vertices, faces) {
                        *faces = saved;
                        let edge = if v0 < v1 { (v0, v1) } else { (v1, v0) };
                        failed_edges.insert(edge);
                    }
                }
            }
        }
    }
}

/// Find the first edge that should be flipped (O(1) per edge using topology).
fn find_edge_to_flip_fast(
    vertices: &[Point3<f64>],
    faces: &[[usize; 3]],
    topology: &MeshTopology,
    preserve_boundary: bool,
    exclude: &HashSet<(usize, usize)>,
) -> Option<(usize, usize)> {
    for &(v0, v1) in topology.edge_faces.keys() {
        if exclude.contains(&(v0, v1)) {
            continue;
        }

        // Skip boundary edges
        if preserve_boundary && topology.is_boundary_edge(v0, v1) {
            continue;
        }

        // Check if flip improves valence
        if should_flip_edge_fast(vertices, faces, topology, v0, v1) {
            return Some((v0, v1));
        }
    }

    None
}

/// Compute target valence for a vertex (O(1) using topology).
#[inline]
fn target_valence_fast(topology: &MeshTopology, v: usize) -> usize {
    if topology.is_boundary_vertex(v) {
        4 // Boundary vertices should have valence 4
    } else {
        6 // Interior vertices should have valence 6
    }
}

/// Compute current valence of a vertex (O(1) using topology).
#[inline]
fn vertex_valence_fast(topology: &MeshTopology, v: usize) -> usize {
    topology.neighbors(v).len()
}

/// Check if flipping an edge would improve vertex valences (O(1) using topology).
fn should_flip_edge_fast(
    vertices: &[Point3<f64>],
    faces: &[[usize; 3]],
    topology: &MeshTopology,
    v0: usize,
    v1: usize,
) -> bool {
    // Get adjacent faces from topology
    let adjacent_face_indices = match topology.get_edge_faces(v0, v1) {
        Some(f) if f.len() == 2 => f,
        _ => return false, // Not an interior edge
    };

    // Find the opposite vertices
    let mut v2 = None;
    let mut v3 = None;

    for &face_idx in adjacent_face_indices {
        let face = faces[face_idx];
        for &v in &face {
            if v != v0 && v != v1 {
                if v2.is_none() {
                    v2 = Some(v);
                } else {
                    v3 = Some(v);
                }
            }
        }
    }

    let (v2, v3) = match (v2, v3) {
        (Some(a), Some(b)) => (a, b),
        _ => return false,
    };

    // Check if the new edge (v2-v3) already exists in the mesh
    if topology.edge_exists(v2, v3) {
        return false;
    }

    // Compute valence deviation before flip
    let val_v0 = vertex_valence_fast(topology, v0) as i32;
    let val_v1 = vertex_valence_fast(topology, v1) as i32;
    let val_v2 = vertex_valence_fast(topology, v2) as i32;
    let val_v3 = vertex_valence_fast(topology, v3) as i32;

    let target_v0 = target_valence_fast(topology, v0) as i32;
    let target_v1 = target_valence_fast(topology, v1) as i32;
    let target_v2 = target_valence_fast(topology, v2) as i32;
    let target_v3 = target_valence_fast(topology, v3) as i32;

    let dev_before = (val_v0 - target_v0).abs()
        + (val_v1 - target_v1).abs()
        + (val_v2 - target_v2).abs()
        + (val_v3 - target_v3).abs();

    // After flip: v0 and v1 lose one neighbor, v2 and v3 gain one
    let dev_after = (val_v0 - 1 - target_v0).abs()
        + (val_v1 - 1 - target_v1).abs()
        + (val_v2 + 1 - target_v2).abs()
        + (val_v3 + 1 - target_v3).abs();

    // Also check that flip doesn't create degenerate geometry
    let p0 = &vertices[v0];
    let p1 = &vertices[v1];
    let p2 = &vertices[v2];
    let p3 = &vertices[v3];

    // Check if the quad is convex (required for valid flip)
    if !is_convex_quad(p0, p2, p1, p3) {
        return false;
    }

    dev_after < dev_before
}

/// Check if a quad is convex.
fn is_convex_quad(
    p0: &Point3<f64>,
    p1: &Point3<f64>,
    p2: &Point3<f64>,
    p3: &Point3<f64>,
) -> bool {
    let v01 = p1 - p0;
    let v12 = p2 - p1;
    let v23 = p3 - p2;
    let v30 = p0 - p3;

    let n0 = v01.cross(&(-v30));
    let n1 = v12.cross(&(-v01));
    let n2 = v23.cross(&(-v12));
    let n3 = v30.cross(&(-v23));

    let d01 = n0.dot(&n1);
    let d12 = n1.dot(&n2);
    let d23 = n2.dot(&n3);

    d01 > 0.0 && d12 > 0.0 && d23 > 0.0
}

/// Flip an edge in the face list.
pub(crate) fn flip_edge(faces: &mut Vec<[usize; 3]>, v0: usize, v1: usize) -> bool {
    // Find the two faces sharing this edge
    let mut face_info: Vec<(usize, usize)> = Vec::new();

    for (idx, face) in faces.iter().enumerate() {
        for i in 0..3 {
            let a = face[i];
            let b = face[(i + 1) % 3];
            if (a == v0 && b == v1) || (a == v1 && b == v0) {
                face_info.push((idx, i));
                break;
            }
        }
    }

    if face_info.len() != 2 {
        return false;
    }

    let (idx0, edge_idx0) = face_info[0];
    let (idx1, edge_idx1) = face_info[1];

    let face0 = faces[idx0];
    let face1 = faces[idx1];

    let opp0 = face0[(edge_idx0 + 2) % 3];
    let opp1 = face1[(edge_idx1 + 2) % 3];

    let a0 = face0[edge_idx0];

    if a0 == v0 {
        faces[idx0] = [opp0, opp1, v0];
        faces[idx1] = [opp1, opp0, v1];
    } else {
        faces[idx0] = [opp0, opp1, v1];
        faces[idx1] = [opp1, opp0, v0];
    }

    true
}

// ============================================================================
// Shared Helpers - Smoothing
// ============================================================================

/// Apply tangential smoothing to regularize vertex positions.
pub(crate) fn tangential_smooth<I: MeshIndex>(
    mesh: &mut HalfEdgeMesh<I>,
    lambda: f64,
    preserve_boundary: bool,
    parallel: bool,
) {
    let (vertices, faces) = to_face_vertex(mesh);

    let normals = compute_vertex_normals_from_faces(&vertices, &faces, parallel);
    let boundary = compute_boundary_vertices(&faces, vertices.len(), parallel);
    let neighbors = build_vertex_neighbors(&faces, vertices.len());

    let compute_position = |idx: usize| -> Point3<f64> {
        let pos = &vertices[idx];
        if preserve_boundary && boundary[idx] {
            return *pos;
        }

        let vertex_neighbors = &neighbors[idx];
        if vertex_neighbors.is_empty() {
            return *pos;
        }

        let mut centroid = Vector3::zeros();
        for &neighbor in vertex_neighbors {
            centroid += vertices[neighbor].coords;
        }
        centroid /= vertex_neighbors.len() as f64;

        let displacement = centroid - pos.coords;
        let normal = &normals[idx];
        let tangent_displacement = displacement - normal.dot(&displacement) * normal;

        Point3::from(pos.coords + lambda * tangent_displacement)
    };

    let new_positions: Vec<Point3<f64>> = if parallel {
        (0..vertices.len())
            .into_par_iter()
            .map(compute_position)
            .collect()
    } else {
        (0..vertices.len())
            .map(compute_position)
            .collect()
    };

    if let Ok(new_mesh) = build_from_triangles::<I>(&new_positions, &faces) {
        *mesh = new_mesh;
    }
}

/// Compute vertex normals from face data.
pub(crate) fn compute_vertex_normals_from_faces(
    vertices: &[Point3<f64>],
    faces: &[[usize; 3]],
    parallel: bool,
) -> Vec<Vector3<f64>> {
    if parallel {
        // Parallel: compute per-vertex by gathering face contributions
        // First build vertex-to-face mapping
        let mut vertex_faces: Vec<Vec<usize>> = vec![Vec::new(); vertices.len()];
        for (face_idx, face) in faces.iter().enumerate() {
            vertex_faces[face[0]].push(face_idx);
            vertex_faces[face[1]].push(face_idx);
            vertex_faces[face[2]].push(face_idx);
        }

        (0..vertices.len())
            .into_par_iter()
            .map(|vi| {
                let mut normal = Vector3::zeros();
                for &face_idx in &vertex_faces[vi] {
                    let face = &faces[face_idx];
                    let p0 = &vertices[face[0]];
                    let p1 = &vertices[face[1]];
                    let p2 = &vertices[face[2]];
                    let e1 = p1 - p0;
                    let e2 = p2 - p0;
                    normal += e1.cross(&e2);
                }
                let len = normal.norm();
                if len > 1e-10 {
                    normal /= len;
                }
                normal
            })
            .collect()
    } else {
        // Sequential: scatter face normals to vertices
        let mut normals: Vec<Vector3<f64>> = vec![Vector3::zeros(); vertices.len()];

        for face in faces {
            let p0 = &vertices[face[0]];
            let p1 = &vertices[face[1]];
            let p2 = &vertices[face[2]];

            let e1 = p1 - p0;
            let e2 = p2 - p0;
            let face_normal = e1.cross(&e2);

            normals[face[0]] += face_normal;
            normals[face[1]] += face_normal;
            normals[face[2]] += face_normal;
        }

        for n in &mut normals {
            let len = n.norm();
            if len > 1e-10 {
                *n /= len;
            }
        }

        normals
    }
}

/// Compute which vertices are on the boundary.
pub(crate) fn compute_boundary_vertices(
    faces: &[[usize; 3]],
    num_vertices: usize,
    parallel: bool,
) -> Vec<bool> {
    // Pre-compute edge counts for boundary detection
    // An edge is boundary if it appears only once in the face list
    let mut edge_counts: std::collections::HashMap<(usize, usize), usize> =
        std::collections::HashMap::new();
    for face in faces {
        for i in 0..3 {
            let v0 = face[i];
            let v1 = face[(i + 1) % 3];
            let edge = if v0 < v1 { (v0, v1) } else { (v1, v0) };
            *edge_counts.entry(edge).or_insert(0) += 1;
        }
    }

    // Build neighbor lists for each vertex
    let neighbors = build_vertex_neighbors(faces, num_vertices);

    let is_boundary_vertex = |v: usize| -> bool {
        for &neighbor in &neighbors[v] {
            let edge = if v < neighbor {
                (v, neighbor)
            } else {
                (neighbor, v)
            };
            if edge_counts.get(&edge) == Some(&1) {
                return true;
            }
        }
        false
    };

    if parallel {
        (0..num_vertices)
            .into_par_iter()
            .map(is_boundary_vertex)
            .collect()
    } else {
        (0..num_vertices)
            .map(is_boundary_vertex)
            .collect()
    }
}

/// Build adjacency list from faces.
pub(crate) fn build_vertex_neighbors(
    faces: &[[usize; 3]],
    num_vertices: usize,
) -> Vec<Vec<usize>> {
    let mut neighbors: Vec<HashSet<usize>> = vec![HashSet::new(); num_vertices];

    for face in faces {
        neighbors[face[0]].insert(face[1]);
        neighbors[face[0]].insert(face[2]);
        neighbors[face[1]].insert(face[0]);
        neighbors[face[1]].insert(face[2]);
        neighbors[face[2]].insert(face[0]);
        neighbors[face[2]].insert(face[1]);
    }

    neighbors
        .into_iter()
        .map(|s| s.into_iter().collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn create_tetrahedron() -> HalfEdgeMesh {
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.5, 1.0, 0.0),
            Point3::new(0.5, 0.5, 1.0),
        ];
        let faces = vec![[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]];
        build_from_triangles(&vertices, &faces).unwrap()
    }

    pub(crate) fn create_grid_mesh(n: usize) -> HalfEdgeMesh {
        let mut vertices = Vec::new();
        let mut faces = Vec::new();

        for j in 0..=n {
            for i in 0..=n {
                vertices.push(Point3::new(i as f64, j as f64, 0.0));
            }
        }

        for j in 0..n {
            for i in 0..n {
                let v00 = j * (n + 1) + i;
                let v10 = v00 + 1;
                let v01 = v00 + (n + 1);
                let v11 = v01 + 1;

                faces.push([v00, v10, v11]);
                faces.push([v00, v11, v01]);
            }
        }

        build_from_triangles(&vertices, &faces).unwrap()
    }

    #[test]
    fn test_edge_split() {
        let mut vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
            Point3::new(1.0, 2.0, 0.0),
        ];
        let mut faces = vec![[0, 1, 2]];

        split_edge(&mut vertices, &mut faces, 0, 1);

        assert_eq!(vertices.len(), 4);
        assert_eq!(faces.len(), 2);

        let mid = &vertices[3];
        assert!((mid.x - 1.0).abs() < 1e-10);
        assert!((mid.y - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_cleanup_mesh() {
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0), // unused
            Point3::new(2.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
        ];
        let faces = vec![[0, 2, 3]];

        let (clean_verts, clean_faces) = cleanup_mesh(&vertices, &faces);

        assert_eq!(clean_verts.len(), 3);
        assert_eq!(clean_faces.len(), 1);
    }

    #[test]
    fn test_edge_flip_two_triangles() {
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(1.0, -1.0, 0.0),
        ];
        let mut faces = vec![[0, 1, 2], [1, 0, 3]];

        let flipped = flip_edge(&mut faces, 0, 1);
        assert!(flipped, "Edge should be flippable");

        assert_eq!(faces.len(), 2);

        let mut seen_edges: HashSet<(usize, usize)> = HashSet::new();
        for face in &faces {
            for i in 0..3 {
                let v0 = face[i];
                let v1 = face[(i + 1) % 3];
                let edge = if v0 < v1 { (v0, v1) } else { (v1, v0) };
                seen_edges.insert(edge);
            }
        }

        assert_eq!(seen_edges.len(), 5);
        assert!(!seen_edges.contains(&(0, 1)));
        assert!(seen_edges.contains(&(2, 3)));

        let result = build_from_triangles::<u32>(&vertices, &faces);
        assert!(result.is_ok());
        assert!(result.unwrap().is_valid());
    }

    #[test]
    fn test_edge_flip_valence_check() {
        let mesh = create_grid_mesh(2);
        let (vertices, mut faces) = to_face_vertex(&mesh);

        flip_edges_for_valence_faces(&vertices, &mut faces, true);

        let result = build_from_triangles::<u32>(&vertices, &faces);
        assert!(result.is_ok());
    }
}
