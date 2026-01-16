//! Anisotropic remeshing algorithm.

use std::collections::HashSet;

use nalgebra::{Point3, Vector3};

use crate::mesh::{build_from_triangles, to_face_vertex, HalfEdgeMesh, MeshIndex};

use super::{
    cleanup_mesh, collapse_edge, flip_edges_for_valence_faces, get_vertex_neighbors,
    is_boundary_edge_in_faces, is_boundary_vertex_in_faces, split_edge, tangential_smooth,
    validate_face_list,
};

/// Options for anisotropic remeshing.
///
/// Anisotropic remeshing adapts edge lengths based on local curvature,
/// using shorter edges in high-curvature regions and longer edges in
/// flat regions.
#[derive(Debug, Clone)]
pub struct AnisotropicOptions {
    /// Minimum allowed edge length.
    pub min_length: f64,

    /// Maximum allowed edge length.
    pub max_length: f64,

    /// Number of remeshing iterations.
    pub iterations: usize,

    /// Whether to preserve boundary edges.
    pub preserve_boundary: bool,

    /// Number of tangential smoothing iterations per remeshing iteration.
    pub smoothing_iterations: usize,

    /// Smoothing factor for tangential relaxation.
    pub smoothing_lambda: f64,

    /// Curvature adaptation strength (0.0 = uniform, 1.0 = fully adaptive).
    pub adaptation: f64,
}

impl AnisotropicOptions {
    /// Create options with the specified edge length bounds.
    pub fn new(min_length: f64, max_length: f64) -> Self {
        Self {
            min_length,
            max_length,
            iterations: 5,
            preserve_boundary: true,
            smoothing_iterations: 3,
            smoothing_lambda: 0.5,
            adaptation: 1.0,
        }
    }

    /// Set the number of remeshing iterations.
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Set whether to preserve boundary edges.
    pub fn with_preserve_boundary(mut self, preserve: bool) -> Self {
        self.preserve_boundary = preserve;
        self
    }

    /// Set the curvature adaptation strength.
    pub fn with_adaptation(mut self, adaptation: f64) -> Self {
        self.adaptation = adaptation.clamp(0.0, 1.0);
        self
    }

    /// Set the number of smoothing iterations per remeshing iteration.
    pub fn with_smoothing_iterations(mut self, iterations: usize) -> Self {
        self.smoothing_iterations = iterations;
        self
    }
}

/// Performs anisotropic remeshing on a triangle mesh.
///
/// This algorithm produces a mesh with edge lengths adapted to local curvature:
/// shorter edges in high-curvature regions, longer edges in flat regions.
pub fn anisotropic_remesh<I: MeshIndex>(mesh: &mut HalfEdgeMesh<I>, options: &AnisotropicOptions) {
    if options.iterations == 0
        || options.min_length <= 0.0
        || options.max_length <= options.min_length
    {
        return;
    }

    for _iter in 0..options.iterations {
        let sizing = compute_sizing_field(mesh, options);

        split_long_edges_anisotropic(mesh, &sizing, options.preserve_boundary);

        let sizing = compute_sizing_field(mesh, options);

        collapse_short_edges_anisotropic(mesh, &sizing, options.preserve_boundary);

        flip_edges_to_improve_valence(mesh, options.preserve_boundary);

        for _ in 0..options.smoothing_iterations {
            tangential_smooth(mesh, options.smoothing_lambda, options.preserve_boundary);
        }

        let _ = sizing;
    }
}

/// Per-vertex sizing field (target edge length at each vertex).
#[derive(Debug, Clone)]
pub struct SizingField {
    /// Target edge length for each vertex.
    pub vertex_sizes: Vec<f64>,
}

impl SizingField {
    /// Get the target edge length at a vertex.
    pub fn get(&self, vertex_idx: usize) -> f64 {
        self.vertex_sizes.get(vertex_idx).copied().unwrap_or(1.0)
    }

    /// Get the target edge length for an edge (average of endpoint sizes).
    pub fn edge_target(&self, v0: usize, v1: usize) -> f64 {
        (self.get(v0) + self.get(v1)) * 0.5
    }
}

/// Compute the sizing field based on local curvature.
pub fn compute_sizing_field<I: MeshIndex>(
    mesh: &HalfEdgeMesh<I>,
    options: &AnisotropicOptions,
) -> SizingField {
    let (vertices, faces) = to_face_vertex(mesh);

    let curvatures = compute_mean_curvature_magnitudes(&vertices, &faces);

    let max_curv = curvatures
        .iter()
        .copied()
        .filter(|c| c.is_finite())
        .fold(0.0_f64, f64::max);

    let min_curv = 0.0;

    let vertex_sizes: Vec<f64> = curvatures
        .iter()
        .map(|&curv| {
            if !curv.is_finite() || max_curv <= min_curv {
                (options.min_length + options.max_length) * 0.5
            } else {
                let t = ((curv - min_curv) / (max_curv - min_curv)).clamp(0.0, 1.0);
                let t = t * options.adaptation;
                options.max_length * (1.0 - t) + options.min_length * t
            }
        })
        .collect();

    SizingField { vertex_sizes }
}

/// Compute mean curvature magnitude at each vertex using the cotangent Laplacian.
fn compute_mean_curvature_magnitudes(vertices: &[Point3<f64>], faces: &[[usize; 3]]) -> Vec<f64> {
    let n = vertices.len();
    let mut curvatures = vec![0.0; n];
    let mut areas = vec![0.0; n];

    for face in faces {
        let i0 = face[0];
        let i1 = face[1];
        let i2 = face[2];

        let p0 = &vertices[i0];
        let p1 = &vertices[i1];
        let p2 = &vertices[i2];

        let e0 = p1 - p0;
        let e1 = p2 - p1;
        let e2 = p0 - p2;

        let area = e0.cross(&(-e2)).norm() * 0.5;
        if area < 1e-12 {
            continue;
        }

        let cot0 = cotangent_weight(&e1, &e2);
        let cot1 = cotangent_weight(&e2, &e0);
        let cot2 = cotangent_weight(&e0, &e1);

        curvatures[i0] += cot2 * (p1 - p0).norm();
        curvatures[i0] += cot1 * (p2 - p0).norm();

        curvatures[i1] += cot2 * (p0 - p1).norm();
        curvatures[i1] += cot0 * (p2 - p1).norm();

        curvatures[i2] += cot1 * (p0 - p2).norm();
        curvatures[i2] += cot0 * (p1 - p2).norm();

        let area_third = area / 3.0;
        areas[i0] += area_third;
        areas[i1] += area_third;
        areas[i2] += area_third;
    }

    for i in 0..n {
        if areas[i] > 1e-12 {
            curvatures[i] /= 4.0 * areas[i];
        }
    }

    curvatures
}

/// Compute cotangent of angle between two edge vectors.
pub(crate) fn cotangent_weight(e1: &Vector3<f64>, e2: &Vector3<f64>) -> f64 {
    let dot = -e1.dot(e2);
    let cross_norm = e1.cross(e2).norm();

    if cross_norm < 1e-12 {
        0.0
    } else {
        (dot / cross_norm).clamp(-100.0, 100.0)
    }
}

/// Split edges longer than their local target.
fn split_long_edges_anisotropic<I: MeshIndex>(
    mesh: &mut HalfEdgeMesh<I>,
    sizing: &SizingField,
    preserve_boundary: bool,
) {
    let (mut vertices, mut faces) = to_face_vertex(mesh);
    let mut vertex_sizes = sizing.vertex_sizes.clone();

    let mut changed = true;
    while changed {
        changed = false;

        let mut edges_to_split: Vec<(usize, usize)> = Vec::new();
        let mut seen_edges: HashSet<(usize, usize)> = HashSet::new();

        for face in &faces {
            for i in 0..3 {
                let v0 = face[i];
                let v1 = face[(i + 1) % 3];
                let edge = if v0 < v1 { (v0, v1) } else { (v1, v0) };

                if seen_edges.contains(&edge) {
                    continue;
                }
                seen_edges.insert(edge);

                let p0 = &vertices[v0];
                let p1 = &vertices[v1];
                let length = (p1 - p0).norm();

                let target = (vertex_sizes[v0] + vertex_sizes[v1]) * 0.5;
                let threshold = target * 4.0 / 3.0;

                if length > threshold {
                    if preserve_boundary && is_boundary_edge_in_faces(&faces, v0, v1) {
                        continue;
                    }
                    edges_to_split.push((v0, v1));
                }
            }
        }

        if edges_to_split.is_empty() {
            break;
        }

        for (v0, v1) in edges_to_split {
            let new_size = (vertex_sizes[v0] + vertex_sizes[v1]) * 0.5;

            split_edge(&mut vertices, &mut faces, v0, v1);
            vertex_sizes.push(new_size);
            changed = true;
        }
    }

    if let Ok(new_mesh) = build_from_triangles::<I>(&vertices, &faces) {
        *mesh = new_mesh;
    }
}

/// Collapse edges shorter than their local target.
fn collapse_short_edges_anisotropic<I: MeshIndex>(
    mesh: &mut HalfEdgeMesh<I>,
    sizing: &SizingField,
    preserve_boundary: bool,
) {
    let (mut vertices, mut faces) = to_face_vertex(mesh);
    let mut vertex_sizes = sizing.vertex_sizes.clone();

    let mut changed = true;
    while changed {
        changed = false;

        let mut edge_to_collapse: Option<(usize, usize)> = None;
        let mut seen_edges: HashSet<(usize, usize)> = HashSet::new();

        for face in &faces {
            for i in 0..3 {
                let v0 = face[i];
                let v1 = face[(i + 1) % 3];
                let edge = if v0 < v1 { (v0, v1) } else { (v1, v0) };

                if seen_edges.contains(&edge) {
                    continue;
                }
                seen_edges.insert(edge);

                let p0 = &vertices[v0];
                let p1 = &vertices[v1];
                let length = (p1 - p0).norm();

                let target = (vertex_sizes.get(v0).copied().unwrap_or(1.0)
                    + vertex_sizes.get(v1).copied().unwrap_or(1.0))
                    * 0.5;
                let low_threshold = target * 4.0 / 5.0;

                if length < low_threshold {
                    if can_collapse_edge_anisotropic(
                        &vertices,
                        &faces,
                        &vertex_sizes,
                        v0,
                        v1,
                        preserve_boundary,
                    ) {
                        edge_to_collapse = Some((v0, v1));
                        break;
                    }
                }
            }
            if edge_to_collapse.is_some() {
                break;
            }
        }

        if let Some((v0, v1)) = edge_to_collapse {
            let new_size = (vertex_sizes.get(v0).copied().unwrap_or(1.0)
                + vertex_sizes.get(v1).copied().unwrap_or(1.0))
                * 0.5;

            collapse_edge(&mut vertices, &mut faces, v0, v1);
            if v0 < vertex_sizes.len() {
                vertex_sizes[v0] = new_size;
            }
            changed = true;
        }
    }

    let (clean_vertices, clean_faces) = cleanup_mesh(&vertices, &faces);

    if !clean_faces.is_empty() {
        if let Ok(new_mesh) = build_from_triangles::<I>(&clean_vertices, &clean_faces) {
            *mesh = new_mesh;
        }
    }
}

/// Check if an edge can be safely collapsed (anisotropic version).
fn can_collapse_edge_anisotropic(
    vertices: &[Point3<f64>],
    faces: &[[usize; 3]],
    vertex_sizes: &[f64],
    v0: usize,
    v1: usize,
    preserve_boundary: bool,
) -> bool {
    if preserve_boundary && is_boundary_edge_in_faces(faces, v0, v1) {
        return false;
    }

    if preserve_boundary {
        let v0_boundary = is_boundary_vertex_in_faces(faces, v0);
        let v1_boundary = is_boundary_vertex_in_faces(faces, v1);
        if v0_boundary && v1_boundary && !is_boundary_edge_in_faces(faces, v0, v1) {
            return false;
        }
    }

    let midpoint = (vertices[v0].coords + vertices[v1].coords) * 0.5;

    let neighbors: HashSet<usize> = get_vertex_neighbors(faces, v0)
        .union(&get_vertex_neighbors(faces, v1))
        .copied()
        .filter(|&v| v != v0 && v != v1)
        .collect();

    let new_size = (vertex_sizes.get(v0).copied().unwrap_or(1.0)
        + vertex_sizes.get(v1).copied().unwrap_or(1.0))
        * 0.5;

    for &neighbor in &neighbors {
        let neighbor_size = vertex_sizes.get(neighbor).copied().unwrap_or(1.0);
        let edge_target = (new_size + neighbor_size) * 0.5;
        let high_threshold = edge_target * 4.0 / 3.0;

        let new_length = (vertices[neighbor].coords - midpoint).norm();
        if new_length > high_threshold {
            return false;
        }
    }

    let neighbors_v0 = get_vertex_neighbors(faces, v0);
    let neighbors_v1 = get_vertex_neighbors(faces, v1);
    let common: HashSet<_> = neighbors_v0.intersection(&neighbors_v1).collect();

    if !is_boundary_edge_in_faces(faces, v0, v1) && common.len() != 2 {
        return false;
    }

    true
}

/// Flip edges to improve vertex valence.
fn flip_edges_to_improve_valence<I: MeshIndex>(mesh: &mut HalfEdgeMesh<I>, preserve_boundary: bool) {
    let (vertices, mut faces) = to_face_vertex(mesh);

    flip_edges_for_valence_faces(&vertices, &mut faces, preserve_boundary);

    if !validate_face_list(&vertices, &faces) {
        return;
    }

    if let Ok(new_mesh) = build_from_triangles::<I>(&vertices, &faces) {
        *mesh = new_mesh;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algo::remesh::tests::create_tetrahedron;
    use crate::mesh::build_from_triangles;

    /// Create a mesh with varying curvature: a bumpy surface.
    fn create_bumpy_mesh() -> HalfEdgeMesh {
        let mut vertices = Vec::new();
        let mut faces = Vec::new();

        let n = 5;
        for j in 0..=n {
            for i in 0..=n {
                let x = i as f64;
                let y = j as f64;
                let cx = n as f64 / 2.0;
                let cy = n as f64 / 2.0;
                let dist2 = (x - cx).powi(2) + (y - cy).powi(2);
                let z = 2.0 * (-dist2 / 2.0).exp();
                vertices.push(Point3::new(x, y, z));
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
    fn test_compute_sizing_field() {
        let mesh = create_bumpy_mesh();
        let options = AnisotropicOptions::new(0.3, 1.5);

        let sizing = compute_sizing_field(&mesh, &options);

        for &size in &sizing.vertex_sizes {
            assert!(size >= options.min_length && size <= options.max_length);
        }

        let center_idx = 14;
        let corner_idx = 0;

        assert!(sizing.vertex_sizes[center_idx] <= sizing.vertex_sizes[corner_idx] + 0.5);
    }

    #[test]
    fn test_anisotropic_remesh_preserves_validity() {
        let mut mesh = create_tetrahedron();

        let options = AnisotropicOptions::new(0.3, 1.0).with_iterations(1);
        anisotropic_remesh(&mut mesh, &options);

        assert!(mesh.is_valid());
        assert!(mesh.num_faces() > 0);
        assert!(mesh.num_vertices() > 0);
    }

    #[test]
    fn test_anisotropic_remesh_preserves_euler() {
        let mut mesh = create_tetrahedron();
        let original_euler =
            mesh.num_vertices() as i32 - (mesh.num_halfedges() / 2) as i32 + mesh.num_faces() as i32;

        let options = AnisotropicOptions::new(0.3, 0.8).with_iterations(2);
        anisotropic_remesh(&mut mesh, &options);

        let new_euler =
            mesh.num_vertices() as i32 - (mesh.num_halfedges() / 2) as i32 + mesh.num_faces() as i32;

        assert_eq!(original_euler, new_euler);
    }

    #[test]
    fn test_anisotropic_zero_iterations_no_change() {
        let mut mesh = create_tetrahedron();
        let original_vertices: Vec<Point3<f64>> =
            mesh.vertex_ids().map(|v| *mesh.position(v)).collect();
        let original_face_count = mesh.num_faces();

        let options = AnisotropicOptions::new(0.3, 1.0).with_iterations(0);
        anisotropic_remesh(&mut mesh, &options);

        assert_eq!(mesh.num_faces(), original_face_count);
        for (vid, orig) in mesh.vertex_ids().zip(original_vertices.iter()) {
            assert_eq!(mesh.position(vid), orig);
        }
    }

    #[test]
    fn test_anisotropic_adaptation_strength() {
        let mesh = create_bumpy_mesh();

        let options_full = AnisotropicOptions::new(0.3, 1.5).with_adaptation(1.0);
        let sizing_full = compute_sizing_field(&mesh, &options_full);

        let options_none = AnisotropicOptions::new(0.3, 1.5).with_adaptation(0.0);
        let sizing_none = compute_sizing_field(&mesh, &options_none);

        for &size in &sizing_none.vertex_sizes {
            assert!((size - 1.5).abs() < 0.01);
        }

        let min_full = sizing_full
            .vertex_sizes
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let max_full = sizing_full
            .vertex_sizes
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let range_full = max_full - min_full;

        assert!(range_full > 0.1);
    }

    #[test]
    fn test_cotangent_weight_computation() {
        let e1 = Vector3::new(1.0, 0.0, 0.0);
        let e2 = Vector3::new(0.0, 1.0, 0.0);
        let cot = cotangent_weight(&e1, &e2);
        assert!(cot.abs() < 0.01);

        let e1 = Vector3::new(1.0, 0.0, 0.0);
        let e2 = Vector3::new(-1.0, 1.0, 0.0);
        let cot = cotangent_weight(&e1, &e2);
        assert!((cot - 1.0).abs() < 0.01);
    }
}
