//! CVT-based remeshing using Lloyd's algorithm.

use std::collections::HashSet;

use nalgebra::{Point3, Vector3};

use crate::algo::Progress;
use crate::mesh::{build_from_triangles, to_face_vertex, HalfEdgeMesh, MeshIndex};

use super::is_boundary_vertex_in_faces;

/// Options for CVT-based remeshing.
///
/// CVT (Centroidal Voronoi Tessellation) remeshing uses Lloyd's algorithm
/// to create high-quality meshes with well-distributed vertices.
#[derive(Debug, Clone)]
pub struct CvtOptions {
    /// Target number of vertices in the output mesh.
    pub target_vertices: Option<usize>,

    /// Number of Lloyd relaxation iterations.
    pub iterations: usize,

    /// Whether to preserve boundary vertices.
    pub preserve_boundary: bool,

    /// Convergence threshold for Lloyd iterations.
    pub convergence_threshold: f64,

    /// Whether to retriangulate after Lloyd relaxation.
    pub retriangulate: bool,
}

impl Default for CvtOptions {
    fn default() -> Self {
        Self {
            target_vertices: None,
            iterations: 10,
            preserve_boundary: true,
            convergence_threshold: 1e-6,
            retriangulate: true,
        }
    }
}

impl CvtOptions {
    /// Create options with the specified number of Lloyd iterations.
    pub fn new(iterations: usize) -> Self {
        Self {
            iterations,
            ..Default::default()
        }
    }

    /// Set the target number of vertices.
    pub fn with_target_vertices(mut self, count: usize) -> Self {
        self.target_vertices = Some(count);
        self
    }

    /// Set whether to preserve boundary vertices.
    pub fn with_preserve_boundary(mut self, preserve: bool) -> Self {
        self.preserve_boundary = preserve;
        self
    }

    /// Set the convergence threshold.
    pub fn with_convergence_threshold(mut self, threshold: f64) -> Self {
        self.convergence_threshold = threshold;
        self
    }

    /// Set whether to retriangulate after relaxation.
    pub fn with_retriangulate(mut self, retriangulate: bool) -> Self {
        self.retriangulate = retriangulate;
        self
    }
}

/// Performs CVT-based remeshing using Lloyd's algorithm.
///
/// This algorithm optimizes vertex positions by iteratively moving each vertex
/// to the centroid of its Voronoi cell on the mesh surface.
pub fn cvt_remesh<I: MeshIndex>(mesh: &mut HalfEdgeMesh<I>, options: &CvtOptions) {
    cvt_remesh_internal(mesh, options, None);
}

/// Performs CVT-based remeshing with progress reporting.
///
/// See [`cvt_remesh`] for algorithm details.
pub fn cvt_remesh_with_progress<I: MeshIndex>(
    mesh: &mut HalfEdgeMesh<I>,
    options: &CvtOptions,
    progress: &Progress,
) {
    cvt_remesh_internal(mesh, options, Some(progress));
}

fn cvt_remesh_internal<I: MeshIndex>(
    mesh: &mut HalfEdgeMesh<I>,
    options: &CvtOptions,
    progress: Option<&Progress>,
) {
    if options.iterations == 0 {
        return;
    }

    let (vertices, faces) = to_face_vertex(mesh);
    if vertices.is_empty() || faces.is_empty() {
        return;
    }

    let target_count = options.target_vertices.unwrap_or(vertices.len());
    let mut seeds = initialize_seeds(&vertices, &faces, target_count);

    let boundary_seeds: HashSet<usize> = if options.preserve_boundary {
        seeds
            .iter()
            .enumerate()
            .filter(|(_, pos)| is_near_boundary_position(&vertices, &faces, pos, 1e-6))
            .map(|(i, _)| i)
            .collect()
    } else {
        HashSet::new()
    };

    for iter in 0..options.iterations {
        if let Some(p) = progress {
            p.report(iter, options.iterations, "CVT remeshing (Lloyd iteration)");
        }

        let assignments = assign_to_nearest_seeds(&vertices, &seeds);
        let centroids = compute_voronoi_centroids(&vertices, &faces, &seeds, &assignments);

        let mut max_movement = 0.0_f64;
        for (i, centroid) in centroids.into_iter().enumerate() {
            if options.preserve_boundary && boundary_seeds.contains(&i) {
                continue;
            }
            if let Some(c) = centroid {
                let movement = (c - seeds[i]).norm();
                max_movement = max_movement.max(movement);
                seeds[i] = c;
            }
        }

        if max_movement < options.convergence_threshold {
            // Report early convergence
            if let Some(p) = progress {
                p.report(iter + 1, options.iterations, "CVT converged early");
            }
            break;
        }
    }

    if options.retriangulate {
        let projected_seeds = project_points_to_surface(&vertices, &faces, &seeds);

        if let Some((new_verts, new_faces)) =
            triangulate_seeds_on_surface(&vertices, &faces, &projected_seeds, options.preserve_boundary)
        {
            if let Ok(new_mesh) = build_from_triangles::<I>(&new_verts, &new_faces) {
                *mesh = new_mesh;
            }
        }
    } else {
        let mut new_vertices = vertices.clone();

        for (i, seed) in seeds.iter().enumerate() {
            if i < new_vertices.len() {
                new_vertices[i] = *seed;
            }
        }

        if let Ok(new_mesh) = build_from_triangles::<I>(&new_vertices, &faces) {
            *mesh = new_mesh;
        }
    }

    // Report completion
    if let Some(p) = progress {
        p.report(options.iterations, options.iterations, "CVT remeshing complete");
    }
}

/// Initialize seed positions for CVT.
fn initialize_seeds(
    vertices: &[Point3<f64>],
    faces: &[[usize; 3]],
    target_count: usize,
) -> Vec<Point3<f64>> {
    if target_count >= vertices.len() {
        return vertices.to_vec();
    }

    farthest_point_sampling(vertices, faces, target_count)
}

/// Farthest point sampling to select well-distributed seed points.
pub(crate) fn farthest_point_sampling(
    vertices: &[Point3<f64>],
    _faces: &[[usize; 3]],
    count: usize,
) -> Vec<Point3<f64>> {
    if count == 0 || vertices.is_empty() {
        return Vec::new();
    }

    let mut selected: Vec<usize> = Vec::with_capacity(count);
    let mut distances: Vec<f64> = vec![f64::INFINITY; vertices.len()];

    selected.push(0);

    while selected.len() < count && selected.len() < vertices.len() {
        let last = selected[selected.len() - 1];
        let last_pos = &vertices[last];

        for (i, dist) in distances.iter_mut().enumerate() {
            let d = (vertices[i] - last_pos).norm();
            *dist = dist.min(d);
        }

        let mut farthest_idx = 0;
        let mut farthest_dist = 0.0_f64;

        for (i, &dist) in distances.iter().enumerate() {
            if !selected.contains(&i) && dist > farthest_dist {
                farthest_dist = dist;
                farthest_idx = i;
            }
        }

        if farthest_dist <= 0.0 {
            break;
        }

        selected.push(farthest_idx);
    }

    selected.iter().map(|&i| vertices[i]).collect()
}

/// Assign each vertex to its nearest seed.
pub(crate) fn assign_to_nearest_seeds(vertices: &[Point3<f64>], seeds: &[Point3<f64>]) -> Vec<usize> {
    vertices
        .iter()
        .map(|v| {
            let mut best_seed = 0;
            let mut best_dist = f64::INFINITY;

            for (i, seed) in seeds.iter().enumerate() {
                let dist = (v - seed).norm_squared();
                if dist < best_dist {
                    best_dist = dist;
                    best_seed = i;
                }
            }

            best_seed
        })
        .collect()
}

/// Compute centroids of Voronoi cells.
fn compute_voronoi_centroids(
    vertices: &[Point3<f64>],
    faces: &[[usize; 3]],
    seeds: &[Point3<f64>],
    assignments: &[usize],
) -> Vec<Option<Point3<f64>>> {
    let mut centroids: Vec<Option<Point3<f64>>> = vec![None; seeds.len()];
    let mut weights: Vec<f64> = vec![0.0; seeds.len()];
    let mut sums: Vec<Vector3<f64>> = vec![Vector3::zeros(); seeds.len()];

    for face in faces {
        let i0 = face[0];
        let i1 = face[1];
        let i2 = face[2];

        let p0 = &vertices[i0];
        let p1 = &vertices[i1];
        let p2 = &vertices[i2];

        let area = compute_triangle_area_pts(p0, p1, p2);
        if area < 1e-12 {
            continue;
        }

        let tri_centroid = (p0.coords + p1.coords + p2.coords) / 3.0;

        let s0 = assignments[i0];
        let s1 = assignments[i1];
        let s2 = assignments[i2];

        let owner = if s0 == s1 || s0 == s2 {
            s0
        } else if s1 == s2 {
            s1
        } else {
            let mut best = s0;
            let mut best_dist = (Point3::from(tri_centroid) - seeds[s0]).norm_squared();
            for &s in &[s1, s2] {
                let d = (Point3::from(tri_centroid) - seeds[s]).norm_squared();
                if d < best_dist {
                    best_dist = d;
                    best = s;
                }
            }
            best
        };

        sums[owner] += tri_centroid * area;
        weights[owner] += area;
    }

    for (i, vertex) in vertices.iter().enumerate() {
        let owner = assignments[i];
        let vertex_weight = 0.01;
        sums[owner] += vertex.coords * vertex_weight;
        weights[owner] += vertex_weight;
    }

    for i in 0..seeds.len() {
        if weights[i] > 1e-12 {
            centroids[i] = Some(Point3::from(sums[i] / weights[i]));
        }
    }

    centroids
}

/// Compute triangle area from three points.
fn compute_triangle_area_pts(p0: &Point3<f64>, p1: &Point3<f64>, p2: &Point3<f64>) -> f64 {
    let e1 = p1 - p0;
    let e2 = p2 - p0;
    e1.cross(&e2).norm() * 0.5
}

/// Check if a position is near the mesh boundary.
fn is_near_boundary_position(
    vertices: &[Point3<f64>],
    faces: &[[usize; 3]],
    pos: &Point3<f64>,
    _tolerance: f64,
) -> bool {
    let mut nearest_idx = 0;
    let mut nearest_dist = f64::INFINITY;

    for (i, v) in vertices.iter().enumerate() {
        let d = (v - pos).norm_squared();
        if d < nearest_dist {
            nearest_dist = d;
            nearest_idx = i;
        }
    }

    is_boundary_vertex_in_faces(faces, nearest_idx)
}

/// Project points onto the mesh surface.
fn project_points_to_surface(
    vertices: &[Point3<f64>],
    faces: &[[usize; 3]],
    points: &[Point3<f64>],
) -> Vec<Point3<f64>> {
    points
        .iter()
        .map(|p| project_point_to_surface(vertices, faces, p))
        .collect()
}

/// Project a single point onto the nearest triangle of the mesh.
fn project_point_to_surface(
    vertices: &[Point3<f64>],
    faces: &[[usize; 3]],
    point: &Point3<f64>,
) -> Point3<f64> {
    let mut best_proj = *point;
    let mut best_dist = f64::INFINITY;

    for face in faces {
        let p0 = &vertices[face[0]];
        let p1 = &vertices[face[1]];
        let p2 = &vertices[face[2]];

        let proj = project_point_to_triangle(point, p0, p1, p2);
        let dist = (proj - point).norm_squared();

        if dist < best_dist {
            best_dist = dist;
            best_proj = proj;
        }
    }

    best_proj
}

/// Project a point onto a triangle.
pub(crate) fn project_point_to_triangle(
    point: &Point3<f64>,
    p0: &Point3<f64>,
    p1: &Point3<f64>,
    p2: &Point3<f64>,
) -> Point3<f64> {
    let e1 = p1 - p0;
    let e2 = p2 - p0;
    let normal = e1.cross(&e2);
    let area2 = normal.norm();

    if area2 < 1e-12 {
        return Point3::from((p0.coords + p1.coords + p2.coords) / 3.0);
    }

    let n = normal / area2;

    let v = point - p0;
    let dist = v.dot(&n);
    let proj = point - n * dist;

    let v0 = p2 - p0;
    let v1 = p1 - p0;
    let v2 = proj - p0;

    let dot00 = v0.dot(&v0);
    let dot01 = v0.dot(&v1);
    let dot02 = v0.dot(&v2);
    let dot11 = v1.dot(&v1);
    let dot12 = v1.dot(&v2);

    let denom = dot00 * dot11 - dot01 * dot01;
    if denom.abs() < 1e-12 {
        return Point3::from((p0.coords + p1.coords + p2.coords) / 3.0);
    }

    let u = (dot11 * dot02 - dot01 * dot12) / denom;
    let v = (dot00 * dot12 - dot01 * dot02) / denom;

    if u >= 0.0 && v >= 0.0 && u + v <= 1.0 {
        proj
    } else {
        let candidates = [
            closest_point_on_segment(point, p0, p1),
            closest_point_on_segment(point, p1, p2),
            closest_point_on_segment(point, p2, p0),
        ];

        let mut best = candidates[0];
        let mut best_dist = (candidates[0] - point).norm_squared();

        for &c in &candidates[1..] {
            let d = (c - point).norm_squared();
            if d < best_dist {
                best_dist = d;
                best = c;
            }
        }

        best
    }
}

/// Find closest point on a line segment.
pub(crate) fn closest_point_on_segment(
    point: &Point3<f64>,
    a: &Point3<f64>,
    b: &Point3<f64>,
) -> Point3<f64> {
    let ab = b - a;
    let len2 = ab.norm_squared();

    if len2 < 1e-12 {
        return *a;
    }

    let t = ((point - a).dot(&ab) / len2).clamp(0.0, 1.0);
    Point3::from(a.coords + ab * t)
}

/// Triangulate seed points on the mesh surface.
fn triangulate_seeds_on_surface(
    original_vertices: &[Point3<f64>],
    original_faces: &[[usize; 3]],
    seeds: &[Point3<f64>],
    _preserve_boundary: bool,
) -> Option<(Vec<Point3<f64>>, Vec<[usize; 3]>)> {
    if seeds.len() < 3 {
        return None;
    }

    let mut new_faces: Vec<[usize; 3]> = Vec::new();

    for face in original_faces {
        let p0 = &original_vertices[face[0]];
        let p1 = &original_vertices[face[1]];
        let p2 = &original_vertices[face[2]];

        let s0 = find_nearest_seed(p0, seeds);
        let s1 = find_nearest_seed(p1, seeds);
        let s2 = find_nearest_seed(p2, seeds);

        if s0 != s1 && s1 != s2 && s0 != s2 {
            new_faces.push([s0, s1, s2]);
        }
    }

    let mut seen: HashSet<[usize; 3]> = HashSet::new();
    new_faces.retain(|face| {
        let mut sorted = *face;
        sorted.sort();
        seen.insert(sorted)
    });

    let new_faces = fix_face_winding(seeds, &new_faces);

    if new_faces.is_empty() {
        return None;
    }

    Some((seeds.to_vec(), new_faces))
}

/// Find the index of the nearest seed to a point.
fn find_nearest_seed(point: &Point3<f64>, seeds: &[Point3<f64>]) -> usize {
    let mut best = 0;
    let mut best_dist = f64::INFINITY;

    for (i, seed) in seeds.iter().enumerate() {
        let d = (point - seed).norm_squared();
        if d < best_dist {
            best_dist = d;
            best = i;
        }
    }

    best
}

/// Fix face winding to be consistent.
fn fix_face_winding(vertices: &[Point3<f64>], faces: &[[usize; 3]]) -> Vec<[usize; 3]> {
    if faces.is_empty() {
        return Vec::new();
    }

    let centroid: Vector3<f64> =
        vertices.iter().map(|v| v.coords).sum::<Vector3<f64>>() / vertices.len() as f64;

    faces
        .iter()
        .map(|face| {
            let p0 = &vertices[face[0]];
            let p1 = &vertices[face[1]];
            let p2 = &vertices[face[2]];

            let e1 = p1 - p0;
            let e2 = p2 - p0;
            let normal = e1.cross(&e2);

            let face_center = (p0.coords + p1.coords + p2.coords) / 3.0;
            let to_center = centroid - face_center;

            if normal.dot(&to_center) > 0.0 {
                [face[0], face[2], face[1]]
            } else {
                *face
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algo::remesh::tests::{create_grid_mesh, create_tetrahedron};

    #[test]
    fn test_farthest_point_sampling() {
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
            Point3::new(3.0, 0.0, 0.0),
            Point3::new(4.0, 0.0, 0.0),
        ];
        let faces: Vec<[usize; 3]> = vec![];

        let samples = farthest_point_sampling(&vertices, &faces, 3);
        assert_eq!(samples.len(), 3);

        assert_eq!(samples[0], vertices[0]);
        assert_eq!(samples[1], vertices[4]);
        assert_eq!(samples[2], vertices[2]);
    }

    #[test]
    fn test_assign_to_nearest_seeds() {
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.1, 0.0, 0.0),
            Point3::new(0.9, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
        ];
        let seeds = vec![Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)];

        let assignments = assign_to_nearest_seeds(&vertices, &seeds);

        assert_eq!(assignments[0], 0);
        assert_eq!(assignments[1], 0);
        assert_eq!(assignments[2], 1);
        assert_eq!(assignments[3], 1);
    }

    #[test]
    fn test_project_point_to_triangle() {
        let p0 = Point3::new(0.0, 0.0, 0.0);
        let p1 = Point3::new(1.0, 0.0, 0.0);
        let p2 = Point3::new(0.0, 1.0, 0.0);

        let point = Point3::new(0.25, 0.25, 1.0);
        let proj = project_point_to_triangle(&point, &p0, &p1, &p2);
        assert!((proj.z - 0.0).abs() < 1e-6);
        assert!((proj.x - 0.25).abs() < 1e-6);
        assert!((proj.y - 0.25).abs() < 1e-6);

        let point_outside = Point3::new(-1.0, -1.0, 0.0);
        let proj_outside = project_point_to_triangle(&point_outside, &p0, &p1, &p2);
        assert!((proj_outside - p0).norm() < 1e-6);
    }

    #[test]
    fn test_cvt_remesh_preserves_validity() {
        let mut mesh = create_tetrahedron();

        let options = CvtOptions::new(3).with_retriangulate(true);
        cvt_remesh(&mut mesh, &options);

        assert!(mesh.is_valid());
        assert!(mesh.num_faces() > 0);
        assert!(mesh.num_vertices() > 0);
    }

    #[test]
    fn test_cvt_remesh_without_retriangulation() {
        let mut mesh = create_tetrahedron();
        let original_face_count = mesh.num_faces();

        let options = CvtOptions::new(3).with_retriangulate(false);
        cvt_remesh(&mut mesh, &options);

        assert!(mesh.is_valid());
        assert_eq!(mesh.num_faces(), original_face_count);
    }

    #[test]
    fn test_cvt_zero_iterations_no_change() {
        let mut mesh = create_tetrahedron();
        let original_vertices: Vec<Point3<f64>> =
            mesh.vertex_ids().map(|v| *mesh.position(v)).collect();
        let original_face_count = mesh.num_faces();

        let options = CvtOptions::new(0);
        cvt_remesh(&mut mesh, &options);

        assert_eq!(mesh.num_faces(), original_face_count);
        for (vid, orig) in mesh.vertex_ids().zip(original_vertices.iter()) {
            assert_eq!(mesh.position(vid), orig);
        }
    }

    #[test]
    fn test_cvt_with_target_vertices() {
        let mut mesh = create_grid_mesh(3);
        let original_count = mesh.num_vertices();
        assert!(original_count > 8);

        let options = CvtOptions::new(5)
            .with_target_vertices(8)
            .with_retriangulate(true);
        cvt_remesh(&mut mesh, &options);

        assert!(mesh.is_valid());
        assert!(mesh.num_vertices() <= original_count);
    }

    #[test]
    fn test_closest_point_on_segment() {
        let a = Point3::new(0.0, 0.0, 0.0);
        let b = Point3::new(2.0, 0.0, 0.0);

        let p1 = Point3::new(1.0, 1.0, 0.0);
        let c1 = closest_point_on_segment(&p1, &a, &b);
        assert!((c1.x - 1.0).abs() < 1e-6);
        assert!((c1.y - 0.0).abs() < 1e-6);

        let p2 = Point3::new(-1.0, 0.0, 0.0);
        let c2 = closest_point_on_segment(&p2, &a, &b);
        assert!((c2 - a).norm() < 1e-6);

        let p3 = Point3::new(3.0, 0.0, 0.0);
        let c3 = closest_point_on_segment(&p3, &a, &b);
        assert!((c3 - b).norm() < 1e-6);
    }
}
