//! Discrete curvature computation on meshes.
//!
//! This module provides algorithms for computing discrete curvature on triangle meshes,
//! including Gaussian curvature, mean curvature, and principal curvatures.
//!
//! # Curvature Types
//!
//! - **Gaussian curvature K**: Intrinsic curvature computed via angle defect
//! - **Mean curvature H**: Extrinsic curvature from the Laplace-Beltrami operator
//! - **Principal curvatures k1, k2**: Maximum and minimum normal curvatures
//!
//! # Example
//!
//! ```no_run
//! use morsel::prelude::*;
//! use morsel::algo::curvature::{compute_curvature, gaussian_curvature};
//!
//! let mesh: HalfEdgeMesh = morsel::io::load("mesh.obj").unwrap();
//!
//! // Compute all curvatures
//! let result = compute_curvature(&mesh);
//!
//! // Access curvature at a vertex
//! let v = VertexId::new(0);
//! println!("Gaussian: {}", result.gaussian(v));
//! println!("Mean: {}", result.mean(v));
//! let (k1, k2) = result.principal(v);
//! println!("Principal: k1={}, k2={}", k1, k2);
//!
//! // Or compute just Gaussian curvature (faster)
//! let gaussian = gaussian_curvature(&mesh);
//! ```
//!
//! # References
//!
//! - Meyer, M., et al. (2003). "Discrete Differential-Geometry Operators for
//!   Triangulated 2-Manifolds." Visualization and Mathematics III.

use std::f64::consts::PI;
use std::marker::PhantomData;

use nalgebra::{Point3, Vector3};
use rayon::prelude::*;

use crate::mesh::{HalfEdgeMesh, MeshIndex, VertexId};

/// Result of curvature computation.
///
/// Contains per-vertex curvature values for all vertices in the mesh.
#[derive(Debug, Clone)]
pub struct CurvatureResult<I: MeshIndex = u32> {
    /// Gaussian curvature (K) per vertex.
    gaussian: Vec<f64>,
    /// Mean curvature (H) per vertex (signed).
    mean: Vec<f64>,
    /// Maximum principal curvature (k1) per vertex.
    principal_max: Vec<f64>,
    /// Minimum principal curvature (k2) per vertex.
    principal_min: Vec<f64>,
    /// Phantom data for index type.
    _marker: PhantomData<I>,
}

impl<I: MeshIndex> CurvatureResult<I> {
    /// Get Gaussian curvature at a vertex.
    #[inline]
    pub fn gaussian(&self, v: VertexId<I>) -> f64 {
        self.gaussian[v.index()]
    }

    /// Get mean curvature at a vertex.
    #[inline]
    pub fn mean(&self, v: VertexId<I>) -> f64 {
        self.mean[v.index()]
    }

    /// Get principal curvatures at a vertex.
    ///
    /// Returns (k1, k2) where k1 >= k2.
    #[inline]
    pub fn principal(&self, v: VertexId<I>) -> (f64, f64) {
        (self.principal_max[v.index()], self.principal_min[v.index()])
    }

    /// Get all Gaussian curvatures as a slice.
    #[inline]
    pub fn gaussian_values(&self) -> &[f64] {
        &self.gaussian
    }

    /// Get all mean curvatures as a slice.
    #[inline]
    pub fn mean_values(&self) -> &[f64] {
        &self.mean
    }

    /// Get the number of vertices.
    #[inline]
    pub fn len(&self) -> usize {
        self.gaussian.len()
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.gaussian.is_empty()
    }

    /// Compute shape index at a vertex.
    ///
    /// Shape index is a scale-invariant measure: (2/π) * atan((k1+k2)/(k1-k2))
    /// Range: [-1, 1], where -1 = cup, 0 = saddle, 1 = cap
    pub fn shape_index(&self, v: VertexId<I>) -> f64 {
        let k1 = self.principal_max[v.index()];
        let k2 = self.principal_min[v.index()];
        let diff = k1 - k2;
        if diff.abs() < 1e-10 {
            0.0 // Umbilical point
        } else {
            (2.0 / PI) * ((k1 + k2) / diff).atan()
        }
    }

    /// Compute curvedness at a vertex.
    ///
    /// Curvedness measures the magnitude of curvature: sqrt((k1² + k2²) / 2)
    pub fn curvedness(&self, v: VertexId<I>) -> f64 {
        let k1 = self.principal_max[v.index()];
        let k2 = self.principal_min[v.index()];
        ((k1 * k1 + k2 * k2) / 2.0).sqrt()
    }
}

/// Compute the angle at vertex `a` in triangle (a, b, c).
fn triangle_angle(a: &Point3<f64>, b: &Point3<f64>, c: &Point3<f64>) -> f64 {
    let ab = (b - a).normalize();
    let ac = (c - a).normalize();
    let dot = ab.dot(&ac).clamp(-1.0, 1.0);
    dot.acos()
}

/// Compute the cotangent of the angle at vertex `a` in triangle (a, b, c).
fn cotangent_angle(a: &Point3<f64>, b: &Point3<f64>, c: &Point3<f64>) -> f64 {
    let ab = b - a;
    let ac = c - a;
    let dot = ab.dot(&ac);
    let cross_norm = ab.cross(&ac).norm();
    if cross_norm < 1e-10 {
        0.0
    } else {
        dot / cross_norm
    }
}

/// Check if a triangle is obtuse and return the index (0, 1, 2) of the obtuse vertex.
fn obtuse_vertex(p0: &Point3<f64>, p1: &Point3<f64>, p2: &Point3<f64>) -> Option<usize> {
    let angle0 = triangle_angle(p0, p1, p2);
    let angle1 = triangle_angle(p1, p0, p2);
    let angle2 = triangle_angle(p2, p0, p1);

    let half_pi = PI / 2.0;
    if angle0 > half_pi {
        Some(0)
    } else if angle1 > half_pi {
        Some(1)
    } else if angle2 > half_pi {
        Some(2)
    } else {
        None
    }
}

/// Compute Voronoi area contribution for a vertex in a non-obtuse triangle.
fn voronoi_area_contribution(
    p_vertex: &Point3<f64>,
    p_prev: &Point3<f64>,
    p_next: &Point3<f64>,
) -> f64 {
    // Voronoi area = (1/8) * (|PR|² * cot(Q) + |PQ|² * cot(R))
    // where P is the vertex, Q and R are the other vertices
    let pr = p_next - p_vertex;
    let pq = p_prev - p_vertex;

    let cot_q = cotangent_angle(p_prev, p_vertex, p_next);
    let cot_r = cotangent_angle(p_next, p_vertex, p_prev);

    0.125 * (pr.norm_squared() * cot_q + pq.norm_squared() * cot_r)
}

/// Compute mixed Voronoi area for a vertex.
///
/// Uses the Meyer et al. formulation:
/// - Non-obtuse triangles: Voronoi area
/// - Obtuse at vertex: triangle_area / 2
/// - Obtuse elsewhere: triangle_area / 4
fn compute_mixed_area<I: MeshIndex>(mesh: &HalfEdgeMesh<I>, v: VertexId<I>) -> f64 {
    let mut area = 0.0;

    for f in mesh.vertex_faces(v) {
        let verts = mesh.face_triangle(f);
        let [p0, p1, p2] = mesh.face_positions(f);
        let tri_area = mesh.face_area(f);

        // Find which vertex in the face is v
        let (local_idx, p_vertex, p_prev, p_next) = if verts[0] == v {
            (0, &p0, &p2, &p1)
        } else if verts[1] == v {
            (1, &p1, &p0, &p2)
        } else {
            (2, &p2, &p1, &p0)
        };

        // Check if triangle is obtuse
        match obtuse_vertex(&p0, &p1, &p2) {
            None => {
                // Non-obtuse: use Voronoi area
                area += voronoi_area_contribution(p_vertex, p_prev, p_next);
            }
            Some(obtuse_idx) => {
                if obtuse_idx == local_idx {
                    // Obtuse at our vertex: area / 2
                    area += tri_area / 2.0;
                } else {
                    // Obtuse elsewhere: area / 4
                    area += tri_area / 4.0;
                }
            }
        }
    }

    // Ensure we have a valid area (handle boundary vertices)
    if area < 1e-10 {
        // Fallback: use simple area weighting
        let mut fallback_area = 0.0;
        for f in mesh.vertex_faces(v) {
            fallback_area += mesh.face_area(f) / 3.0;
        }
        if fallback_area > 1e-10 {
            return fallback_area;
        }
    }

    area
}

/// Compute the sum of angles at a vertex.
fn compute_angle_sum<I: MeshIndex>(mesh: &HalfEdgeMesh<I>, v: VertexId<I>) -> f64 {
    let mut angle_sum = 0.0;

    for f in mesh.vertex_faces(v) {
        let verts = mesh.face_triangle(f);
        let [p0, p1, p2] = mesh.face_positions(f);

        // Find the angle at v
        let angle = if verts[0] == v {
            triangle_angle(&p0, &p1, &p2)
        } else if verts[1] == v {
            triangle_angle(&p1, &p0, &p2)
        } else {
            triangle_angle(&p2, &p0, &p1)
        };

        angle_sum += angle;
    }

    angle_sum
}

/// Compute the mean curvature normal vector at a vertex.
///
/// This is the unnormalized Laplace-Beltrami: Δx = (1/2A) * Σ (cot α + cot β) * (x_j - x_i)
/// The magnitude is 2H and the direction is the normal.
fn compute_mean_curvature_normal<I: MeshIndex>(
    mesh: &HalfEdgeMesh<I>,
    v: VertexId<I>,
) -> Vector3<f64> {
    let p_v = mesh.position(v);
    let mut laplacian = Vector3::zeros();

    for he in mesh.vertex_halfedges(v) {
        let v_j = mesh.dest(he);
        let p_j = mesh.position(v_j);

        // Get cotangent weights from the two adjacent triangles
        let mut cot_sum = 0.0;

        // First triangle (face of this half-edge)
        let f1 = mesh.face_of(he);
        if f1.is_valid() {
            let fverts = mesh.face_triangle(f1);
            let [fp0, fp1, fp2] = mesh.face_positions(f1);

            // Find the opposite vertex
            let p_opp = if fverts[0] != v && fverts[0] != v_j {
                fp0
            } else if fverts[1] != v && fverts[1] != v_j {
                fp1
            } else {
                fp2
            };

            cot_sum += cotangent_angle(&p_opp, p_v, p_j);
        }

        // Second triangle (face of twin half-edge)
        let twin = mesh.twin(he);
        let f2 = mesh.face_of(twin);
        if f2.is_valid() {
            let fverts = mesh.face_triangle(f2);
            let [fp0, fp1, fp2] = mesh.face_positions(f2);

            // Find the opposite vertex
            let p_opp = if fverts[0] != v && fverts[0] != v_j {
                fp0
            } else if fverts[1] != v && fverts[1] != v_j {
                fp1
            } else {
                fp2
            };

            cot_sum += cotangent_angle(&p_opp, p_v, p_j);
        }

        // Clamp to avoid negative weights from degenerate triangles
        cot_sum = cot_sum.max(0.0);

        laplacian += cot_sum * (p_j - p_v);
    }

    0.5 * laplacian
}

/// Compute Gaussian curvature for all vertices.
///
/// Uses the angle defect formula: K = (2π - Σθ) / A_mixed
///
/// This function uses parallel computation by default. Use
/// [`gaussian_curvature_sequential`] for single-threaded execution.
///
/// # Example
///
/// ```no_run
/// use morsel::prelude::*;
/// use morsel::algo::curvature::gaussian_curvature;
///
/// let mesh: HalfEdgeMesh = morsel::io::load("mesh.obj").unwrap();
/// let curvatures = gaussian_curvature(&mesh);
/// ```
pub fn gaussian_curvature<I: MeshIndex + Sync>(mesh: &HalfEdgeMesh<I>) -> Vec<f64> {
    gaussian_curvature_impl(mesh, true)
}

/// Compute Gaussian curvature for all vertices (sequential version).
///
/// Uses single-threaded execution. Useful for benchmarking.
pub fn gaussian_curvature_sequential<I: MeshIndex>(mesh: &HalfEdgeMesh<I>) -> Vec<f64> {
    gaussian_curvature_impl(mesh, false)
}

fn gaussian_curvature_impl<I: MeshIndex + Sync>(mesh: &HalfEdgeMesh<I>, parallel: bool) -> Vec<f64> {
    let n = mesh.num_vertices();
    let vertex_indices: Vec<usize> = (0..n).collect();

    let compute_vertex = |idx: usize| -> f64 {
        let v = VertexId::<I>::new(idx);
        let angle_sum = compute_angle_sum(mesh, v);
        let area = compute_mixed_area(mesh, v);

        let angle_defect = 2.0 * PI - angle_sum;

        if area > 1e-10 {
            angle_defect / area
        } else {
            0.0
        }
    };

    if parallel {
        vertex_indices
            .par_iter()
            .map(|&idx| compute_vertex(idx))
            .collect()
    } else {
        vertex_indices
            .iter()
            .map(|&idx| compute_vertex(idx))
            .collect()
    }
}

/// Compute mean curvature for all vertices.
///
/// Uses the cotangent Laplacian: H = ||Δx|| / (2 * A_mixed)
/// The sign is determined by the direction relative to the vertex normal.
///
/// This function uses parallel computation by default. Use
/// [`mean_curvature_sequential`] for single-threaded execution.
///
/// # Example
///
/// ```no_run
/// use morsel::prelude::*;
/// use morsel::algo::curvature::mean_curvature;
///
/// let mesh: HalfEdgeMesh = morsel::io::load("mesh.obj").unwrap();
/// let curvatures = mean_curvature(&mesh);
/// ```
pub fn mean_curvature<I: MeshIndex + Sync>(mesh: &HalfEdgeMesh<I>) -> Vec<f64> {
    mean_curvature_impl(mesh, true)
}

/// Compute mean curvature for all vertices (sequential version).
///
/// Uses single-threaded execution. Useful for benchmarking.
pub fn mean_curvature_sequential<I: MeshIndex>(mesh: &HalfEdgeMesh<I>) -> Vec<f64> {
    mean_curvature_impl(mesh, false)
}

fn mean_curvature_impl<I: MeshIndex + Sync>(mesh: &HalfEdgeMesh<I>, parallel: bool) -> Vec<f64> {
    let n = mesh.num_vertices();
    let vertex_indices: Vec<usize> = (0..n).collect();

    let compute_vertex = |idx: usize| -> f64 {
        let v = VertexId::<I>::new(idx);
        let laplacian = compute_mean_curvature_normal(mesh, v);
        let area = compute_mixed_area(mesh, v);

        if area > 1e-10 {
            let laplacian_normalized = laplacian / area;
            let h_unsigned = laplacian_normalized.norm() / 2.0;

            // Determine sign: positive if laplacian points in same direction as normal
            let normal = mesh.vertex_normal(v);
            let sign = if laplacian_normalized.dot(&normal) >= 0.0 {
                1.0
            } else {
                -1.0
            };

            sign * h_unsigned
        } else {
            0.0
        }
    };

    if parallel {
        vertex_indices
            .par_iter()
            .map(|&idx| compute_vertex(idx))
            .collect()
    } else {
        vertex_indices
            .iter()
            .map(|&idx| compute_vertex(idx))
            .collect()
    }
}

/// Compute all curvatures (Gaussian, mean, and principal) for all vertices.
///
/// This is more efficient than computing each separately when you need all of them.
///
/// This function uses parallel computation by default. Use
/// [`compute_curvature_sequential`] for single-threaded execution.
///
/// # Example
///
/// ```no_run
/// use morsel::prelude::*;
/// use morsel::algo::curvature::compute_curvature;
///
/// let mesh: HalfEdgeMesh = morsel::io::load("mesh.obj").unwrap();
/// let result = compute_curvature(&mesh);
///
/// for v in mesh.vertex_ids() {
///     let k = result.gaussian(v);
///     let h = result.mean(v);
///     let (k1, k2) = result.principal(v);
///     println!("v{}: K={:.4}, H={:.4}, k1={:.4}, k2={:.4}",
///              v.index(), k, h, k1, k2);
/// }
/// ```
pub fn compute_curvature<I: MeshIndex + Sync>(mesh: &HalfEdgeMesh<I>) -> CurvatureResult<I> {
    compute_curvature_impl(mesh, true)
}

/// Compute all curvatures (sequential version).
///
/// Uses single-threaded execution. Useful for benchmarking.
pub fn compute_curvature_sequential<I: MeshIndex>(mesh: &HalfEdgeMesh<I>) -> CurvatureResult<I> {
    compute_curvature_impl(mesh, false)
}

/// Per-vertex curvature data computed in parallel.
#[derive(Debug, Clone)]
struct VertexCurvature {
    gaussian: f64,
    mean: f64,
    principal_max: f64,
    principal_min: f64,
}

fn compute_curvature_impl<I: MeshIndex + Sync>(
    mesh: &HalfEdgeMesh<I>,
    parallel: bool,
) -> CurvatureResult<I> {
    let n = mesh.num_vertices();
    let vertex_indices: Vec<usize> = (0..n).collect();

    let compute_vertex = |idx: usize| -> VertexCurvature {
        let v = VertexId::<I>::new(idx);

        // Compute area once
        let area = compute_mixed_area(mesh, v);

        // Gaussian curvature
        let angle_sum = compute_angle_sum(mesh, v);
        let angle_defect = 2.0 * PI - angle_sum;
        let k = if area > 1e-10 {
            angle_defect / area
        } else {
            0.0
        };

        // Mean curvature
        let laplacian = compute_mean_curvature_normal(mesh, v);
        let h = if area > 1e-10 {
            let laplacian_normalized = laplacian / area;
            let h_unsigned = laplacian_normalized.norm() / 2.0;

            let normal = mesh.vertex_normal(v);
            let sign = if laplacian_normalized.dot(&normal) >= 0.0 {
                1.0
            } else {
                -1.0
            };

            sign * h_unsigned
        } else {
            0.0
        };

        // Principal curvatures: k1, k2 = H ± sqrt(H² - K)
        let discriminant = h * h - k;
        let (principal_max, principal_min) = if discriminant >= 0.0 {
            let sqrt_disc = discriminant.sqrt();
            (h + sqrt_disc, h - sqrt_disc)
        } else {
            // Numerical issues: fall back to H for both
            (h, h)
        };

        VertexCurvature {
            gaussian: k,
            mean: h,
            principal_max,
            principal_min,
        }
    };

    let results: Vec<VertexCurvature> = if parallel {
        vertex_indices
            .par_iter()
            .map(|&idx| compute_vertex(idx))
            .collect()
    } else {
        vertex_indices
            .iter()
            .map(|&idx| compute_vertex(idx))
            .collect()
    };

    // Unpack results into separate vectors
    let mut gaussian = Vec::with_capacity(n);
    let mut mean = Vec::with_capacity(n);
    let mut principal_max = Vec::with_capacity(n);
    let mut principal_min = Vec::with_capacity(n);

    for vc in results {
        gaussian.push(vc.gaussian);
        mean.push(vc.mean);
        principal_max.push(vc.principal_max);
        principal_min.push(vc.principal_min);
    }

    CurvatureResult {
        gaussian,
        mean,
        principal_max,
        principal_min,
        _marker: PhantomData,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{build_from_triangles, HalfEdgeMesh};

    fn create_flat_grid(n: usize) -> HalfEdgeMesh {
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
                let v10 = j * (n + 1) + i + 1;
                let v01 = (j + 1) * (n + 1) + i;
                let v11 = (j + 1) * (n + 1) + i + 1;

                faces.push([v00, v10, v11]);
                faces.push([v00, v11, v01]);
            }
        }

        build_from_triangles(&vertices, &faces).unwrap()
    }

    fn create_icosphere(subdivisions: usize) -> HalfEdgeMesh {
        // Start with icosahedron
        let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
        let scale = 1.0 / (1.0 + phi * phi).sqrt();

        let mut vertices = vec![
            Point3::new(-1.0, phi, 0.0) * scale,
            Point3::new(1.0, phi, 0.0) * scale,
            Point3::new(-1.0, -phi, 0.0) * scale,
            Point3::new(1.0, -phi, 0.0) * scale,
            Point3::new(0.0, -1.0, phi) * scale,
            Point3::new(0.0, 1.0, phi) * scale,
            Point3::new(0.0, -1.0, -phi) * scale,
            Point3::new(0.0, 1.0, -phi) * scale,
            Point3::new(phi, 0.0, -1.0) * scale,
            Point3::new(phi, 0.0, 1.0) * scale,
            Point3::new(-phi, 0.0, -1.0) * scale,
            Point3::new(-phi, 0.0, 1.0) * scale,
        ];

        let mut faces = vec![
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ];

        // Subdivide
        for _ in 0..subdivisions {
            let mut new_faces = Vec::new();
            let mut edge_midpoints: std::collections::HashMap<(usize, usize), usize> =
                std::collections::HashMap::new();

            for face in &faces {
                let mut mids = [0usize; 3];

                for i in 0..3 {
                    let v0 = face[i];
                    let v1 = face[(i + 1) % 3];
                    let key = if v0 < v1 { (v0, v1) } else { (v1, v0) };

                    mids[i] = *edge_midpoints.entry(key).or_insert_with(|| {
                        let mid = Point3::from((vertices[v0].coords + vertices[v1].coords) / 2.0);
                        let normalized = Point3::from(mid.coords.normalize());
                        vertices.push(normalized);
                        vertices.len() - 1
                    });
                }

                new_faces.push([face[0], mids[0], mids[2]]);
                new_faces.push([face[1], mids[1], mids[0]]);
                new_faces.push([face[2], mids[2], mids[1]]);
                new_faces.push([mids[0], mids[1], mids[2]]);
            }

            faces = new_faces;
        }

        build_from_triangles(&vertices, &faces).unwrap()
    }

    #[test]
    fn test_curvature_flat_plane() {
        let mesh = create_flat_grid(3);
        let result = compute_curvature(&mesh);

        // Interior vertices should have K ≈ 0, H ≈ 0
        // Check vertex (1,1) which is interior (index 5)
        let v_interior = VertexId::new(5);
        assert!(
            result.gaussian(v_interior).abs() < 0.1,
            "Gaussian curvature should be ~0 for flat plane, got {}",
            result.gaussian(v_interior)
        );
        assert!(
            result.mean(v_interior).abs() < 0.1,
            "Mean curvature should be ~0 for flat plane, got {}",
            result.mean(v_interior)
        );
    }

    #[test]
    fn test_curvature_sphere() {
        // Unit sphere approximation
        let mesh = create_icosphere(2);
        let result = compute_curvature(&mesh);

        // For a unit sphere: K = 1, H = 1
        // With a discrete approximation, we won't get exact values
        let mut total_gaussian = 0.0;
        for v in mesh.vertex_ids() {
            total_gaussian += result.gaussian(v) * compute_mixed_area(&mesh, v);
        }

        // Gauss-Bonnet: ∫K dA = 2π * χ = 2π * 2 = 4π for a sphere
        let expected_total = 4.0 * PI;
        assert!(
            (total_gaussian - expected_total).abs() < 0.5,
            "Total Gaussian curvature should be ~4π, got {}",
            total_gaussian
        );

        // Check that mean curvature has consistent sign (may be positive or negative
        // depending on face orientation)
        let v = VertexId::new(0);
        let h = result.mean(v);
        assert!(
            h.abs() > 0.1,
            "Mean curvature should be non-zero for sphere, got {}",
            h
        );
    }

    #[test]
    fn test_principal_curvatures_relation() {
        // For any surface: K = k1 * k2, H = (k1 + k2) / 2
        let mesh = create_icosphere(1);
        let result = compute_curvature(&mesh);

        for v in mesh.vertex_ids() {
            let k = result.gaussian(v);
            let h = result.mean(v);
            let (k1, k2) = result.principal(v);

            // k1 * k2 should equal K
            let product = k1 * k2;
            assert!(
                (product - k).abs() < 0.1,
                "k1*k2 should equal K: {} * {} = {} vs K = {}",
                k1,
                k2,
                product,
                k
            );

            // (k1 + k2) / 2 should equal H
            let avg = (k1 + k2) / 2.0;
            assert!(
                (avg - h).abs() < 0.1,
                "(k1+k2)/2 should equal H: ({} + {}) / 2 = {} vs H = {}",
                k1,
                k2,
                avg,
                h
            );

            // k1 >= k2
            assert!(
                k1 >= k2 - 1e-10,
                "k1 should be >= k2: {} vs {}",
                k1,
                k2
            );
        }
    }

    #[test]
    fn test_curvature_boundary() {
        // Grid mesh has boundary vertices
        let mesh = create_flat_grid(2);

        // Should not panic on boundary vertices
        let gaussian = gaussian_curvature(&mesh);
        let mean = mean_curvature(&mesh);

        assert_eq!(gaussian.len(), mesh.num_vertices());
        assert_eq!(mean.len(), mesh.num_vertices());

        // All values should be finite
        for &k in &gaussian {
            assert!(k.is_finite(), "Gaussian curvature should be finite");
        }
        for &h in &mean {
            assert!(h.is_finite(), "Mean curvature should be finite");
        }
    }

    #[test]
    fn test_curvature_single_triangle() {
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.5, 1.0, 0.0),
        ];
        let faces = vec![[0, 1, 2]];
        let mesh: HalfEdgeMesh = build_from_triangles(&vertices, &faces).unwrap();

        let result = compute_curvature(&mesh);

        // Should not panic and should have values for all vertices
        assert_eq!(result.len(), 3);

        for v in mesh.vertex_ids() {
            assert!(result.gaussian(v).is_finite());
            assert!(result.mean(v).is_finite());
        }
    }

    #[test]
    fn test_shape_index_and_curvedness() {
        let mesh = create_icosphere(1);
        let result = compute_curvature(&mesh);

        for v in mesh.vertex_ids() {
            let si = result.shape_index(v);
            let curv = result.curvedness(v);

            // Shape index should be in [-1, 1]
            assert!(
                si >= -1.0 - 1e-10 && si <= 1.0 + 1e-10,
                "Shape index out of range: {}",
                si
            );

            // Curvedness should be non-negative
            assert!(curv >= 0.0, "Curvedness should be non-negative: {}", curv);
        }
    }

    #[test]
    fn test_gauss_bonnet() {
        // Gauss-Bonnet theorem: ∫K dA = 2π * χ(M)
        // For a closed mesh (sphere): χ = 2, so ∫K dA = 4π

        let mesh = create_icosphere(2);
        let gaussian = gaussian_curvature(&mesh);

        let mut total = 0.0;
        for v in mesh.vertex_ids() {
            let area = compute_mixed_area(&mesh, v);
            total += gaussian[v.index()] * area;
        }

        let expected = 4.0 * PI; // 2π * χ where χ = 2 for sphere
        assert!(
            (total - expected).abs() < 0.5,
            "Gauss-Bonnet violated: got {}, expected {}",
            total,
            expected
        );
    }
}
