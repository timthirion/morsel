//! Measuring triangle quality, so that claims about it can be checked.
//!
//! Remeshing exists to improve triangle quality and decimation to trade it away
//! cheaply, but neither claim means anything without a number attached. This module
//! supplies the numbers.
//!
//! # What "quality" means here
//!
//! Two per-triangle measures, both standard and both with a known best case:
//!
//! - **Minimum angle.** An equilateral triangle's is 60°; a sliver's approaches 0.
//!   This is the quantity Delaunay refinement maximises, and the one that governs the
//!   conditioning of a finite-element stiffness matrix built on the mesh.
//! - **Radius ratio**, `2r/R`, the inradius over the circumradius, doubled so that it
//!   lands in `(0, 1]` with `1` for an equilateral triangle. Euler's inequality
//!   `R ≥ 2r` is what makes that bound tight. It degrades faster than the minimum
//!   angle on needles, which makes the two worth reporting together.
//!
//! Plus two aggregate measures of what *isotropic* remeshing specifically aims at:
//! edge lengths of a uniform size, and interior vertices of valence 6, which is the
//! valence a triangulated plane has everywhere.
//!
//! # What this does not measure
//!
//! Geometric fidelity. A remesher could score perfectly here by discarding the shape
//! and returning a nicely triangulated blob, so quality numbers are only half of any
//! claim — they need pairing with a distance to the original surface. The tests in
//! `tests/remesh_quality.rs` sidestep that for now by remeshing a sphere of known
//! radius, where drift off the surface is directly measurable, but a general Hausdorff
//! distance is still missing. It matters: all three remeshers score better on this
//! metric partly *by* shrinking the mesh.

use crate::mesh::{FaceId, HalfEdgeMesh, MeshIndex};

/// Quality measures for a single triangle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TriangleQuality {
    /// Smallest interior angle, in radians. `π/3` for an equilateral triangle.
    pub min_angle: f64,
    /// Largest interior angle, in radians. `π/3` for an equilateral triangle.
    pub max_angle: f64,
    /// `2r/R`, the inradius over the circumradius. In `(0, 1]`, and `1` exactly for an
    /// equilateral triangle. Zero for a degenerate triangle.
    pub radius_ratio: f64,
    /// Area, which the ratios deliberately ignore — a tiny triangle can be perfectly
    /// shaped.
    pub area: f64,
}

impl TriangleQuality {
    /// The quality of a triangle with no area: angles pinned to the degenerate
    /// extremes and a radius ratio of zero.
    fn degenerate() -> Self {
        Self {
            min_angle: 0.0,
            max_angle: std::f64::consts::PI,
            radius_ratio: 0.0,
            area: 0.0,
        }
    }
}

/// Measure one triangle.
///
/// Angles come from the cross and dot products of the edge vectors rather than from
/// the law of cosines on squared lengths, which loses precision badly on the thin
/// triangles this is most often used to diagnose.
pub fn triangle_quality<I: MeshIndex>(mesh: &HalfEdgeMesh<I>, face: FaceId<I>) -> TriangleQuality {
    let [i, j, k] = mesh.face_triangle(face);
    let (p, q, r) = (mesh.position(i), mesh.position(j), mesh.position(k));

    // Side lengths, named for the vertex each is opposite.
    let a = (q - r).norm();
    let b = (r - p).norm();
    let c = (p - q).norm();
    if !(a.is_finite() && b.is_finite() && c.is_finite()) {
        return TriangleQuality::degenerate();
    }

    // `is_finite` first so a NaN area is caught: every comparison against NaN is false,
    // so `area <= 0.0` alone would let it through.
    let area = 0.5 * (q - p).cross(&(r - p)).norm();
    if !area.is_finite() || area <= 0.0 {
        return TriangleQuality::degenerate();
    }

    // atan2(‖u×v‖, u·v) is stable across the whole range, unlike acos of a dot
    // product that rounds past ±1 on near-degenerate input.
    let angle_at =
        |u: nalgebra::Vector3<f64>, v: nalgebra::Vector3<f64>| u.cross(&v).norm().atan2(u.dot(&v));
    let angles = [
        angle_at(q - p, r - p),
        angle_at(r - q, p - q),
        angle_at(p - r, q - r),
    ];
    let min_angle = angles.iter().copied().fold(f64::INFINITY, f64::min);
    let max_angle = angles.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    // r = area/s and R = abc/(4·area), so 2r/R = 8·area²/(s·a·b·c). Written that way
    // it needs no division by a circumradius that may be enormous.
    let s = 0.5 * (a + b + c);
    let radius_ratio = 8.0 * area * area / (s * a * b * c);

    TriangleQuality {
        min_angle,
        max_angle,
        radius_ratio: radius_ratio.min(1.0),
        area,
    }
}

/// Aggregate quality of a whole mesh.
///
/// Worst-case and mean are both reported because they answer different questions: a
/// single sliver ruins a finite-element solve regardless of how good the mean is, and a
/// good worst case says nothing about the bulk.
#[derive(Debug, Clone, PartialEq)]
pub struct QualityReport {
    /// Triangles measured. Degenerate faces are included, and drag the minima down.
    pub num_faces: usize,
    /// Smallest minimum angle anywhere, in degrees. The worst sliver in the mesh.
    pub min_angle_deg: f64,
    /// Mean over triangles of each triangle's minimum angle, in degrees.
    pub mean_min_angle_deg: f64,
    /// Largest angle anywhere, in degrees.
    pub max_angle_deg: f64,
    /// Worst radius ratio.
    pub min_radius_ratio: f64,
    /// Mean radius ratio.
    pub mean_radius_ratio: f64,
    /// Triangles whose minimum angle is under 30°, the usual threshold for calling a
    /// triangle poorly shaped.
    pub faces_under_30_deg: usize,
    /// Counts of minimum angle in 10° bins: `[0,10), [10,20), … [80,90)`. Nine bins,
    /// since a triangle's minimum angle cannot reach 60° unless equilateral and can
    /// never reach 90°.
    pub min_angle_histogram: [usize; 9],
    /// Shortest, mean and longest edge length.
    pub edge_length: (f64, f64, f64),
    /// Standard deviation of edge length divided by the mean. Zero means perfectly
    /// uniform edges, which is what isotropic remeshing is aiming for.
    pub edge_length_cv: f64,
    /// Fraction of *interior* vertices with valence exactly 6, the valence of a
    /// regularly triangulated plane. Boundary vertices are excluded, since their
    /// ideal valence is 4 and counting them would penalise open meshes for their
    /// boundary.
    pub regular_interior_fraction: f64,
}

/// Measure a whole mesh.
///
/// Returns `None` for a mesh with no faces, where every statistic would be vacuous
/// rather than merely bad.
pub fn mesh_quality<I: MeshIndex>(mesh: &HalfEdgeMesh<I>) -> Option<QualityReport> {
    if mesh.num_faces() == 0 {
        return None;
    }

    let mut min_angle_deg = f64::INFINITY;
    let mut max_angle_deg = f64::NEG_INFINITY;
    let mut min_radius_ratio = f64::INFINITY;
    let mut sum_min_angle = 0.0;
    let mut sum_radius_ratio = 0.0;
    let mut faces_under_30_deg = 0;
    let mut min_angle_histogram = [0usize; 9];

    for face in mesh.face_ids() {
        let q = triangle_quality(mesh, face);
        let lo = q.min_angle.to_degrees();
        let hi = q.max_angle.to_degrees();

        min_angle_deg = min_angle_deg.min(lo);
        max_angle_deg = max_angle_deg.max(hi);
        min_radius_ratio = min_radius_ratio.min(q.radius_ratio);
        sum_min_angle += lo;
        sum_radius_ratio += q.radius_ratio;
        if lo < 30.0 {
            faces_under_30_deg += 1;
        }
        // A minimum angle is at most 60°, so clamping to the last bin only catches
        // rounding at the equilateral limit.
        let bin = ((lo / 10.0) as usize).min(8);
        min_angle_histogram[bin] += 1;
    }

    let n = mesh.num_faces() as f64;

    let mut shortest = f64::INFINITY;
    let mut longest: f64 = 0.0;
    let mut sum_len = 0.0;
    let mut sum_len_sq = 0.0;
    let mut edges = 0usize;
    for he in mesh.halfedge_ids() {
        // One of each pair, so an edge is not measured twice.
        if he.index() > mesh.twin(he).index() {
            continue;
        }
        let len = (mesh.position(mesh.dest(he)) - mesh.position(mesh.origin(he))).norm();
        if !len.is_finite() {
            continue;
        }
        shortest = shortest.min(len);
        longest = longest.max(len);
        sum_len += len;
        sum_len_sq += len * len;
        edges += 1;
    }
    let (mean_len, edge_length_cv) = if edges == 0 {
        (0.0, 0.0)
    } else {
        let m = sum_len / edges as f64;
        let var = (sum_len_sq / edges as f64 - m * m).max(0.0);
        (m, if m > 0.0 { var.sqrt() / m } else { 0.0 })
    };
    if !shortest.is_finite() {
        shortest = 0.0;
    }

    let mut interior = 0usize;
    let mut regular = 0usize;
    for v in mesh.vertex_ids() {
        if mesh.is_boundary_vertex(v) {
            continue;
        }
        interior += 1;
        if mesh.vertex_neighbors(v).count() == 6 {
            regular += 1;
        }
    }

    Some(QualityReport {
        num_faces: mesh.num_faces(),
        min_angle_deg,
        mean_min_angle_deg: sum_min_angle / n,
        max_angle_deg,
        min_radius_ratio,
        mean_radius_ratio: sum_radius_ratio / n,
        faces_under_30_deg,
        min_angle_histogram,
        edge_length: (shortest, mean_len, longest),
        edge_length_cv,
        regular_interior_fraction: if interior == 0 {
            0.0
        } else {
            regular as f64 / interior as f64
        },
    })
}

impl QualityReport {
    /// A compact multi-line summary, for CLI output.
    pub fn summary(&self) -> String {
        let (lo, mean, hi) = self.edge_length;
        let mut out = format!(
            "  faces:            {}\n\
               min angle:        {:.2}° (worst), {:.2}° (mean per triangle)\n\
               max angle:        {:.2}°\n\
               radius ratio:     {:.4} (worst), {:.4} (mean), 1 = equilateral\n\
               under 30°:        {} faces ({:.1}%)\n\
               edge length:      {:.4} / {:.4} / {:.4} (min/mean/max), cv {:.4}\n\
               valence-6 interior: {:.1}%\n\
               min-angle bins:   ",
            self.num_faces,
            self.min_angle_deg,
            self.mean_min_angle_deg,
            self.max_angle_deg,
            self.min_radius_ratio,
            self.mean_radius_ratio,
            self.faces_under_30_deg,
            100.0 * self.faces_under_30_deg as f64 / self.num_faces as f64,
            lo,
            mean,
            hi,
            self.edge_length_cv,
            100.0 * self.regular_interior_fraction,
        );
        for (i, count) in self.min_angle_histogram.iter().enumerate() {
            if *count > 0 {
                out.push_str(&format!("{}-{}°:{} ", i * 10, (i + 1) * 10, count));
            }
        }
        out
    }
}
