//! Optimal Mass Transport (OMT) based area-preserving parameterization.
//!
//! This module implements semi-discrete optimal mass transport for
//! area-preserving UV parameterization. It takes an existing conformal
//! parameterization (typically LSCM) and corrects its area distortion.
//!
//! # Algorithm
//!
//! Following the semi-discrete formulation, let `D` be the planar domain
//! covered by the initial parameterization and let each vertex `i` carry a
//! target mass `ν_i` — its share of the 3D surface area, normalized so that
//! `Σ ν_i = |D|`. We seek weights `w` such that the power diagram of the
//! sites (the initial UVs) partitions `D` into cells with
//! `|Pow_i(w) ∩ D| = ν_i`.
//!
//! The Kantorovich dual is concave in `w` with gradient
//! `∂Φ/∂w_i = ν_i − |Pow_i(w) ∩ D|`, so ascent converges. Once the cell areas
//! match the target masses, vertex `i` is moved to the **centroid of its own
//! cell**. That is the barycentric projection of the Brenier map, and it is
//! applied exactly once — it is a transport step, not a relaxation.
//!
//! # Cells are computed exactly, as polygons
//!
//! A power cell is the intersection of half-planes — one per rival site, each
//! boundary a radical axis — so it is a convex polygon and its area and centroid
//! are exact. They are computed that way here: clip a bounding rectangle by each
//! rival's half-plane to find which constraints bound the cell, then clip each
//! overlapping UV triangle by those same half-planes and measure the pieces with
//! [`approxum`]'s `polygon_area` and `polygon_centroid`.
//!
//! Everything is a half-plane clip, decided by the signed distances of an edge's
//! endpoints. That matters for robustness: a general polygon-polygon
//! intersection needs line-line intersections, and on a regular grid — where
//! cocircular quadruples of sites are everywhere and cell edges run exactly
//! parallel to triangle edges — those degenerate and silently drop vertices,
//! losing area. `test_power_cells_partition_the_domain` pins this down.
//!
//! An earlier version estimated cell areas by counting samples on a
//! `resolution²` grid. That is worth recording, because it failed in a specific
//! and instructive way: a cell covering `k` samples has an area known only to
//! about `1/√k`, so the estimator's noise floor sat *above* the convergence
//! tolerance it was compared against. The solve could therefore never report
//! convergence, and the useful sample count grew with the vertex count, so
//! resolution had to scale like `√n`. Below roughly 25 samples per cell OMT
//! made area distortion *worse* rather than better. Exact polygons remove the
//! floor, the resolution parameter, and the mesh-density ceiling together.
//!
//! # A note on Lloyd relaxation
//!
//! An earlier version also iterated the centroid step as Lloyd relaxation. That
//! is incorrect and actively harmful: iterating drives the configuration toward
//! a *centroidal* power diagram, whose fixed point is uniformly spaced sites —
//! the opposite of respecting the target masses. Measured on a flat grid (where
//! the conformal map is already isometric and the correct answer is to do
//! nothing), each extra iteration made area distortion monotonically worse. The
//! centroid step is applied once.
//!
//! # References
//!
//! - Gu, X., et al. (2013). "Area-Preserving Parameterization via Optimal
//!   Mass Transport." IEEE TVCG.
//! - Mérigot, Q. (2011). "A Multiscale Approach to Optimal Transport."
//!   Computer Graphics Forum.
//! - Kitagawa, J., Mérigot, Q., Thibert, B. (2019). "Convergence of a Newton
//!   algorithm for semi-discrete optimal transport." JEMS.
//! - Aurenhammer, F. (1987). "Power diagrams: properties, algorithms and
//!   applications." SIAM J. Comput.

use approxum::polygon::{polygon_area, polygon_centroid};
use approxum::Point2 as AxPoint2;
use nalgebra::Point2;
use rayon::prelude::*;

use crate::error::{MeshError, Result};
use crate::mesh::{HalfEdgeMesh, MeshIndex};

use super::UVMap;

/// Options for OMT-based area-preserving parameterization.
#[derive(Debug, Clone)]
pub struct OMTOptions {
    /// Maximum number of weight-ascent iterations.
    ///
    /// The ascent is first-order, and with exact cells it is now the limiting
    /// factor rather than the measurement. Iterations needed to reach
    /// `tolerance = 1e-6` on a paraboloid patch grow roughly linearly in the
    /// vertex count, and the result keeps improving until it gets there:
    ///
    /// | vertices | 100 iters | converged | iters to converge |
    /// |----------|-----------|-----------|-------------------|
    /// | 81       | 35.6%     | 35.7%     | 488               |
    /// | 289      | 30.3%     | **23.1%** | 2137              |
    /// | 1089     | 56.9%     | **15.7%** | 8968              |
    ///
    /// (Percentages are area distortion relative to the conformal input.) Note
    /// that the converged quality *improves* with refinement, as a consistent
    /// discretization should — so stopping early is the main thing costing
    /// quality here.
    ///
    /// The default is a compromise, not the best this can do; [`Self::high_quality`]
    /// raises it enough to actually converge on moderate meshes. Two things would
    /// make a generous budget affordable, and both are open:
    ///
    /// - **A damped Newton step on the dual** (Kitagawa–Mérigot–Thibert) converges
    ///   in tens of iterations rather than thousands, replacing the linear growth
    ///   above.
    /// - **Spatial pruning of rivals.** Each cell currently scans every other site
    ///   — cheap per test, but `O(n²)` per iteration. Only nearby sites can bound
    ///   a cell, so a k-d tree or Delaunay neighbourhood would cut this to
    ///   `O(n·k)`. This is what makes a 2601-vertex mesh take tens of seconds.
    pub max_iterations: usize,

    /// Convergence tolerance on the worst per-cell relative area error.
    ///
    /// Cell areas are computed exactly (as polygons), so this is limited by f64
    /// arithmetic rather than by any estimator's noise, and asking for a tight
    /// value is meaningful. What bounds it in practice is the ascent: cells with
    /// very small target mass are the last to converge.
    pub tolerance: f64,

    /// Step size for the ascent on the dual. The step is preconditioned by the
    /// target mass and scaled by the mean cell area, so this is dimensionless;
    /// it is also the ceiling that the backtracking line search may grow back to.
    pub step_size: f64,

    /// Hold boundary vertices at their initial UV positions.
    ///
    /// Defaults to `true`, and should normally stay there. A boundary site's
    /// power cell is clipped by the edge of the domain, so its centroid lies
    /// strictly *inside* the domain — letting the transport step move it
    /// contracts the boundary inward on every vertex at once. Measured against a
    /// correct conformal baseline (`rms` of log area ratio):
    ///
    /// | patch          | LSCM  | `false` | `true` |
    /// |----------------|-------|---------|--------|
    /// | paraboloid 8×8 | 0.168 | 0.207   | 0.060  |
    /// | saddle 8×8     | 0.238 | 0.224   | 0.081  |
    ///
    /// With the boundary free, OMT is no better than doing nothing. Letting
    /// boundary vertices *slide along* the boundary curve — rather than pinning
    /// them or freeing them entirely — is the principled treatment and is not
    /// implemented yet.
    pub fix_boundary: bool,
}

impl Default for OMTOptions {
    fn default() -> Self {
        Self {
            max_iterations: 500,
            tolerance: 1e-4,
            step_size: 0.5,
            fix_boundary: true,
        }
    }
}

impl OMTOptions {
    /// Fewer iterations and a looser tolerance, for interactive use.
    pub fn fast() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-2,
            step_size: 0.8,
            fix_boundary: true,
        }
    }

    /// More iterations and a tighter tolerance.
    pub fn high_quality() -> Self {
        Self {
            max_iterations: 20_000,
            tolerance: 1e-6,
            step_size: 0.3,
            fix_boundary: true,
        }
    }
}

/// Diagnostics from an OMT solve, useful for verifying convergence.
#[derive(Debug, Clone, Copy)]
pub struct OMTReport {
    /// Number of weight-ascent iterations actually performed.
    pub iterations: usize,

    /// Worst per-cell relative area error when the loop stopped.
    pub max_relative_error: f64,

    /// Whether `max_relative_error` reached the requested tolerance.
    pub converged: bool,

    /// Exact area of the transport domain (the UV image of the mesh).
    pub domain_area: f64,
}

/// Apply area-preserving correction to an existing UV parameterization using OMT.
///
/// This function takes an initial UV map (typically from LSCM or another
/// conformal method) and adjusts it to better preserve surface area.
///
/// # Arguments
///
/// * `mesh` - The input mesh (must have boundary)
/// * `initial_uvs` - Initial UV coordinates (e.g., from LSCM)
/// * `options` - Algorithm options
///
/// # Returns
///
/// Adjusted UV coordinates with improved area preservation.
///
/// # Example
///
/// ```no_run
/// use morsel::prelude::*;
/// use morsel::algo::parameterize::{lscm, LSCMOptions, omt, OMTOptions};
///
/// let mesh: HalfEdgeMesh = morsel::io::load("disk.obj").unwrap();
///
/// // First compute conformal map
/// let uv_conformal = lscm(&mesh, &LSCMOptions::default()).unwrap();
///
/// // Then apply area-preserving correction
/// let uv_area = omt(&mesh, &uv_conformal, &OMTOptions::default()).unwrap();
/// ```
pub fn omt<I: MeshIndex>(
    mesh: &HalfEdgeMesh<I>,
    initial_uvs: &UVMap<I>,
    options: &OMTOptions,
) -> Result<UVMap<I>> {
    omt_with_report(mesh, initial_uvs, options).map(|(uvs, _)| uvs)
}

/// Like [`omt`], but also returns convergence diagnostics.
pub fn omt_with_report<I: MeshIndex>(
    mesh: &HalfEdgeMesh<I>,
    initial_uvs: &UVMap<I>,
    options: &OMTOptions,
) -> Result<(UVMap<I>, OMTReport)> {
    let n_vertices = mesh.num_vertices();
    if n_vertices == 0 {
        return Err(MeshError::EmptyMesh);
    }

    if initial_uvs.len() != n_vertices {
        return Err(MeshError::InvalidState(
            "UV map size doesn't match mesh vertex count".to_string(),
        ));
    }

    // Target masses from 3D triangle areas.
    let target_masses = compute_target_masses(mesh);
    let total_target_mass: f64 = target_masses.iter().sum();
    if total_target_mass < 1e-10 {
        return Err(MeshError::InvalidState(
            "Mesh has zero surface area".to_string(),
        ));
    }

    let is_boundary = find_boundary_vertices(mesh);
    let mut positions: Vec<Point2<f64>> = initial_uvs.as_slice().to_vec();

    // The transport domain is the region the parameterization actually covers —
    // the union of its UV triangles — not the bounding box. Using the box would
    // hand the sites mass from empty corners that no choice of weights can
    // account for. Taking it as the triangle union also makes the domain area
    // exact and handles a non-convex UV footprint without special cases.
    let domain = DomainMesh::new(mesh, initial_uvs);
    if domain.is_empty() {
        return Err(MeshError::InvalidState(
            "UV parameterization has no non-degenerate triangles".to_string(),
        ));
    }
    let domain_area = domain.area();

    // Normalize target masses so they sum to the domain measure; only then can
    // the cell areas match them.
    let mass_scale = domain_area / total_target_mass;
    let target_masses: Vec<f64> = target_masses.iter().map(|m| m * mass_scale).collect();

    // Ascent on the concave Kantorovich dual. The raw gradient `ν_i − A_i` is
    // poorly scaled when the target masses vary, so it is preconditioned by
    // `1 / ν_i` (a positive diagonal scaling, so still an ascent direction) and
    // expressed in units of the mean cell area — the natural scale for weights,
    // which have units of length².
    //
    // Step length is chosen by backtracking on the RMS relative area error. A
    // plain fixed step does not converge here: it overshoots the cells with
    // small targets and stalls on the large ones.
    let cell_scale = domain_area / n_vertices as f64;
    let mut weights = vec![0.0; n_vertices];
    let mut cells = domain.power_cells(&positions, &weights);
    let mut merit = rms_relative_error(&cells.areas, &target_masses);
    let mut max_error = compute_max_relative_error(&cells.areas, &target_masses);
    let mut step = options.step_size;
    let mut iterations = 0;
    let mut converged = max_error < options.tolerance;

    while iterations < options.max_iterations && !converged {
        let mut trial = weights.clone();
        for i in 0..n_vertices {
            if target_masses[i] > 1e-12 {
                let relative_deficit = (target_masses[i] - cells.areas[i]) / target_masses[i];
                trial[i] += step * cell_scale * relative_deficit;
            }
        }
        // The dual is invariant to a constant shift; remove it to avoid drift.
        let mean_weight: f64 = trial.iter().sum::<f64>() / n_vertices as f64;
        for w in &mut trial {
            *w -= mean_weight;
        }

        let trial_cells = domain.power_cells(&positions, &trial);
        let trial_merit = rms_relative_error(&trial_cells.areas, &target_masses);
        iterations += 1;

        if trial_merit < merit {
            weights = trial;
            cells = trial_cells;
            merit = trial_merit;
            max_error = compute_max_relative_error(&cells.areas, &target_masses);
            converged = max_error < options.tolerance;
            // Creep the step back up so a single bad stretch doesn't permanently
            // cripple progress.
            step = (step * 1.3).min(options.step_size * 4.0);
        } else {
            step *= 0.5;
            if step < 1e-10 {
                break;
            }
        }
    }

    // Single transport step: move each site to the centroid of its own cell.
    // This is the barycentric projection of the Brenier map. Applying it more
    // than once turns it into Lloyd relaxation, which has a different (wrong)
    // fixed point — see the module docs.
    for i in 0..n_vertices {
        if options.fix_boundary && is_boundary[i] {
            continue;
        }
        if let Some(centroid) = cells.centroid(i) {
            positions[i] = centroid;
        }
    }

    let mut uv_map = UVMap::new(positions);
    uv_map.normalize();

    let report = OMTReport {
        iterations,
        max_relative_error: max_error,
        converged,
        domain_area,
    };

    Ok((uv_map, report))
}
/// Compute target mass for each vertex from 3D triangle areas, using the
/// mixed Voronoi area of Meyer et al. (2003).
///
/// Barycentric (`area / 3`) lumping is *not* usable here. The masses have to be
/// consistent with the dual cells the power diagram produces, and barycentric
/// lumping is not: on a flat `n = 4` grid it assigns a corner vertex mass
/// `0.0833` while its Voronoi cell is `0.0625`, so satisfying the mass
/// constraint forces vertices off an already-isometric map. The mixed Voronoi
/// area agrees with the true Voronoi area whenever the triangulation is
/// Delaunay, and falls back to a safe subdivision on obtuse triangles.
///
/// Both branches sum to the triangle's area, so the total mass is still exactly
/// the surface area (using `Σ ℓ² cot(opposite) = 4·area`).
fn compute_target_masses<I: MeshIndex>(mesh: &HalfEdgeMesh<I>) -> Vec<f64> {
    let n_vertices = mesh.num_vertices();
    let mut masses = vec![0.0; n_vertices];

    for fid in mesh.face_ids() {
        let area = mesh.face_area(fid);
        if !area.is_finite() || area < 1e-12 {
            continue;
        }

        let [v0, v1, v2] = mesh.face_triangle(fid);
        let a = *mesh.position(v0);
        let b = *mesh.position(v1);
        let c = *mesh.position(v2);

        let two_area = 2.0 * area;
        // cot of the angle at each vertex: (e1 · e2) / |e1 × e2|.
        let cot_a = (b - a).dot(&(c - a)) / two_area;
        let cot_b = (a - b).dot(&(c - b)) / two_area;
        let cot_c = (a - c).dot(&(b - c)) / two_area;

        if cot_a >= 0.0 && cot_b >= 0.0 && cot_c >= 0.0 {
            // Non-obtuse: exact Voronoi split. Each incident edge's squared
            // length is weighted by the cotangent of the angle opposite it.
            let l_ab = (b - a).norm_squared();
            let l_bc = (c - b).norm_squared();
            let l_ca = (a - c).norm_squared();

            masses[v0.index()] += (l_ab * cot_c + l_ca * cot_b) / 8.0;
            masses[v1.index()] += (l_ab * cot_c + l_bc * cot_a) / 8.0;
            masses[v2.index()] += (l_ca * cot_b + l_bc * cot_a) / 8.0;
        } else {
            // Obtuse: the circumcenter leaves the triangle, so the cotangent
            // split would go negative. Give the obtuse corner half, the others
            // a quarter each.
            let (half, quarter) = (area / 2.0, area / 4.0);
            if cot_a < 0.0 {
                masses[v0.index()] += half;
                masses[v1.index()] += quarter;
                masses[v2.index()] += quarter;
            } else if cot_b < 0.0 {
                masses[v0.index()] += quarter;
                masses[v1.index()] += half;
                masses[v2.index()] += quarter;
            } else {
                masses[v0.index()] += quarter;
                masses[v1.index()] += quarter;
                masses[v2.index()] += half;
            }
        }
    }

    masses
}

/// Find boundary vertices using half-edge structure.
fn find_boundary_vertices<I: MeshIndex>(mesh: &HalfEdgeMesh<I>) -> Vec<bool> {
    let n_vertices = mesh.num_vertices();
    let mut is_boundary = vec![false; n_vertices];

    for vid in mesh.vertex_ids() {
        if mesh.is_boundary_vertex(vid) {
            is_boundary[vid.index()] = true;
        }
    }

    is_boundary
}

/// Per-site cell area and centroid over the transport domain.
struct PowerCells {
    areas: Vec<f64>,
    /// Area-weighted position sums; divided by area to get the centroid.
    moments: Vec<Point2<f64>>,
}

impl PowerCells {
    fn centroid(&self, i: usize) -> Option<Point2<f64>> {
        let area = self.areas[i];
        if area <= 0.0 || !area.is_finite() {
            return None;
        }
        let m = self.moments[i];
        let c = Point2::new(m.x / area, m.y / area);
        if c.x.is_finite() && c.y.is_finite() {
            Some(c)
        } else {
            None
        }
    }
}

/// A half-plane `a·x ≤ b`.
#[derive(Clone, Copy)]
struct HalfPlane {
    ax: f64,
    ay: f64,
    b: f64,
}

/// The transport domain, held as the UV triangles of the mesh plus a uniform
/// grid over their bounding boxes.
///
/// The domain is the union of the triangles rather than a bounding box or a
/// convex hull, which makes its area exact and copes with a non-convex UV
/// footprint. The grid is what keeps cell evaluation local: a power cell is
/// about the size of one vertex's neighbourhood, so it meets only a handful of
/// triangles, and there is no reason to test it against all of them.
struct DomainMesh {
    /// UV triangles, wound counter-clockwise.
    triangles: Vec<[AxPoint2<f64>; 3]>,
    /// Per-triangle `(min_x, min_y, max_x, max_y)`.
    bounds: Vec<[f64; 4]>,
    /// Grid buckets holding triangle indices.
    buckets: Vec<Vec<u32>>,
    grid_min: (f64, f64),
    inv_cell: (f64, f64),
    nx: usize,
    ny: usize,
    /// Bounding rectangle of the whole domain, the seed for every cell.
    frame: [AxPoint2<f64>; 4],
    total_area: f64,
}

impl DomainMesh {
    fn new<I: MeshIndex>(mesh: &HalfEdgeMesh<I>, uvs: &UVMap<I>) -> Self {
        let mut triangles = Vec::new();
        let mut bounds = Vec::new();
        let mut total_area = 0.0;

        let (mut min_x, mut min_y) = (f64::INFINITY, f64::INFINITY);
        let (mut max_x, mut max_y) = (f64::NEG_INFINITY, f64::NEG_INFINITY);

        for fid in mesh.face_ids() {
            let [v0, v1, v2] = mesh.face_triangle(fid);
            let a = uvs.get(v0);
            let b = uvs.get(v1);
            let c = uvs.get(v2);

            let double_area = (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y);
            if !double_area.is_finite() || double_area.abs() < 1e-18 {
                continue;
            }

            // Wind counter-clockwise so signed areas come out positive.
            let tri = if double_area > 0.0 {
                [ax(a), ax(b), ax(c)]
            } else {
                [ax(a), ax(c), ax(b)]
            };

            let bx = [
                a.x.min(b.x).min(c.x),
                a.y.min(b.y).min(c.y),
                a.x.max(b.x).max(c.x),
                a.y.max(b.y).max(c.y),
            ];
            min_x = min_x.min(bx[0]);
            min_y = min_y.min(bx[1]);
            max_x = max_x.max(bx[2]);
            max_y = max_y.max(bx[3]);

            total_area += 0.5 * double_area.abs();
            triangles.push(tri);
            bounds.push(bx);
        }

        if triangles.is_empty() {
            return Self {
                triangles,
                bounds,
                buckets: Vec::new(),
                grid_min: (0.0, 0.0),
                inv_cell: (1.0, 1.0),
                nx: 0,
                ny: 0,
                frame: [AxPoint2::new(0.0, 0.0); 4],
                total_area: 0.0,
            };
        }

        // Pad so sites sitting exactly on the extent still have a cell frame
        // strictly containing them.
        let pad = ((max_x - min_x) + (max_y - min_y)) * 0.01 + 1e-12;
        let (fx0, fy0) = (min_x - pad, min_y - pad);
        let (fx1, fy1) = (max_x + pad, max_y + pad);

        // Roughly one triangle per bucket.
        let res = (triangles.len() as f64).sqrt().ceil().max(1.0) as usize;
        let (nx, ny) = (res, res);
        let cell_w = (fx1 - fx0) / nx as f64;
        let cell_h = (fy1 - fy0) / ny as f64;
        let inv_cell = (
            if cell_w > 0.0 { 1.0 / cell_w } else { 0.0 },
            if cell_h > 0.0 { 1.0 / cell_h } else { 0.0 },
        );

        let mut buckets = vec![Vec::new(); nx * ny];
        for (t, bx) in bounds.iter().enumerate() {
            let (gx0, gy0, gx1, gy1) = grid_range(bx, (fx0, fy0), inv_cell, nx, ny);
            for gy in gy0..=gy1 {
                for gx in gx0..=gx1 {
                    buckets[gy * nx + gx].push(t as u32);
                }
            }
        }

        Self {
            triangles,
            bounds,
            buckets,
            grid_min: (fx0, fy0),
            inv_cell,
            nx,
            ny,
            frame: [
                AxPoint2::new(fx0, fy0),
                AxPoint2::new(fx1, fy0),
                AxPoint2::new(fx1, fy1),
                AxPoint2::new(fx0, fy1),
            ],
            total_area,
        }
    }

    fn is_empty(&self) -> bool {
        self.triangles.is_empty()
    }

    /// Exact total area of the domain.
    fn area(&self) -> f64 {
        self.total_area
    }

    /// Exact area and centroid of every power cell, intersected with the domain.
    fn power_cells(&self, positions: &[Point2<f64>], weights: &[f64]) -> PowerCells {
        let n = positions.len();

        // A rival's radical axis lies at distance
        //   (‖p_j − p_i‖² + w_i − w_j) / (2‖p_j − p_i‖)
        // from site `i`. Using the largest weight in place of `w_j` makes that a
        // lower bound, so a rival whose bound already exceeds the cell's own
        // radius provably cannot cut it and can be skipped.
        let max_weight = weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        let results: Vec<(f64, Point2<f64>)> = (0..n)
            .into_par_iter()
            .map_init(
                || Scratch::new(self.triangles.len()),
                |scratch, i| self.cell(i, positions, weights, max_weight, scratch),
            )
            .collect();

        let mut areas = Vec::with_capacity(n);
        let mut moments = Vec::with_capacity(n);
        for (a, m) in results {
            areas.push(a);
            moments.push(m);
        }

        PowerCells { areas, moments }
    }

    /// One cell's area and area-weighted centroid, restricted to the domain.
    ///
    /// Two stages. First the cell is built by clipping the domain's bounding
    /// rectangle with each rival's half-plane, recording the ones that actually
    /// cut. Then every domain triangle the cell could meet is clipped by that
    /// same recorded set, and the pieces are measured.
    ///
    /// Clipping the *triangles* rather than intersecting them with the finished
    /// cell polygon is deliberate. It keeps every clip a half-plane test, decided
    /// by the signed distances of the endpoints, so a crossing point is only ever
    /// emitted when those signs genuinely differ. A general polygon-polygon
    /// intersection instead needs line-line intersections, and on a regular grid
    /// — where cocircular quadruples of sites are everywhere and cell edges run
    /// exactly parallel to triangle edges — those degenerate, drop vertices, and
    /// silently lose area. The flat-grid partition test covers precisely this.
    ///
    /// Restricting to the recorded half-planes is sound because every triangle
    /// lies inside the frame: a rival skipped by the radius bound has the whole
    /// cell (and so the whole frame-clipped region) already on its inner side.
    fn cell(
        &self,
        i: usize,
        positions: &[Point2<f64>],
        weights: &[f64],
        max_weight: f64,
        scratch: &mut Scratch,
    ) -> (f64, Point2<f64>) {
        let p = positions[i];

        scratch.poly.clear();
        scratch.poly.extend_from_slice(&self.frame);
        scratch.planes.clear();

        let mut radius = cell_radius(&scratch.poly, p);

        for (j, &q) in positions.iter().enumerate() {
            if j == i {
                continue;
            }
            let dx = q.x - p.x;
            let dy = q.y - p.y;
            let d2 = dx * dx + dy * dy;
            if d2 <= 0.0 {
                // Coincident sites: whichever has the larger weight takes
                // everything. Breaking the tie by index keeps this deterministic
                // and keeps the areas summing to the domain.
                if weights[j] > weights[i] || (weights[j] == weights[i] && j < i) {
                    return (0.0, Point2::origin());
                }
                continue;
            }

            let d = d2.sqrt();
            if (d2 + weights[i] - max_weight) / (2.0 * d) > radius {
                continue;
            }

            // Site i is nearer in power distance where
            //   2 (p_j − p_i) · x ≤ (‖p_j‖² − w_j) − (‖p_i‖² − w_i)
            let plane = HalfPlane {
                ax: 2.0 * dx,
                ay: 2.0 * dy,
                b: (q.x * q.x + q.y * q.y - weights[j]) - (p.x * p.x + p.y * p.y - weights[i]),
            };

            if clip_halfplane(&mut scratch.poly, &mut scratch.buf, plane) {
                scratch.planes.push(plane);
                if scratch.poly.len() < 3 {
                    return (0.0, Point2::origin());
                }
                radius = cell_radius(&scratch.poly, p);
            }
        }

        if scratch.poly.len() < 3 {
            return (0.0, Point2::origin());
        }

        let cell_bounds = polygon_bounds(&scratch.poly);
        let mut area = 0.0;
        let mut moment = Point2::new(0.0, 0.0);

        scratch.generation += 1;
        let gen = scratch.generation;
        let (gx0, gy0, gx1, gy1) =
            grid_range(&cell_bounds, self.grid_min, self.inv_cell, self.nx, self.ny);

        for gy in gy0..=gy1 {
            for gx in gx0..=gx1 {
                for &t in &self.buckets[gy * self.nx + gx] {
                    let t = t as usize;
                    if scratch.seen[t] == gen {
                        continue;
                    }
                    scratch.seen[t] = gen;

                    if !bounds_overlap(&cell_bounds, &self.bounds[t]) {
                        continue;
                    }

                    scratch.piece.clear();
                    scratch.piece.extend_from_slice(&self.triangles[t]);
                    for &plane in &scratch.planes {
                        clip_halfplane(&mut scratch.piece, &mut scratch.buf, plane);
                        if scratch.piece.len() < 3 {
                            break;
                        }
                    }
                    if scratch.piece.len() < 3 {
                        continue;
                    }

                    let piece_area = polygon_area(&scratch.piece);
                    if piece_area <= 0.0 || !piece_area.is_finite() {
                        continue;
                    }
                    if let Some(c) = polygon_centroid(&scratch.piece) {
                        area += piece_area;
                        moment.x += piece_area * c.x;
                        moment.y += piece_area * c.y;
                    }
                }
            }
        }

        (area, moment)
    }
}

/// Per-thread reusable buffers, so a solve doesn't reallocate per site.
struct Scratch {
    poly: Vec<AxPoint2<f64>>,
    piece: Vec<AxPoint2<f64>>,
    buf: Vec<AxPoint2<f64>>,
    planes: Vec<HalfPlane>,
    seen: Vec<u32>,
    generation: u32,
}

impl Scratch {
    fn new(n_triangles: usize) -> Self {
        Self {
            poly: Vec::with_capacity(16),
            piece: Vec::with_capacity(16),
            buf: Vec::with_capacity(16),
            planes: Vec::with_capacity(16),
            seen: vec![0; n_triangles],
            generation: 0,
        }
    }
}

#[inline]
fn ax(p: Point2<f64>) -> AxPoint2<f64> {
    AxPoint2::new(p.x, p.y)
}

/// Clip a convex polygon by the half-plane `a · x ≤ b`, in place.
///
/// Returns whether the polygon actually changed, so the caller can skip
/// recomputing the cell radius when a rival turned out not to cut it.
fn clip_halfplane(
    poly: &mut Vec<AxPoint2<f64>>,
    buf: &mut Vec<AxPoint2<f64>>,
    plane: HalfPlane,
) -> bool {
    let n = poly.len();
    if n == 0 {
        return false;
    }

    let dist = |v: &AxPoint2<f64>| plane.ax * v.x + plane.ay * v.y - plane.b;

    // Nothing to do when every vertex is already inside.
    if !poly.iter().any(|v| dist(v) > 0.0) {
        return false;
    }

    buf.clear();
    for k in 0..n {
        let cur = poly[k];
        let next = poly[(k + 1) % n];
        let d_cur = dist(&cur);
        let d_next = dist(&next);

        if d_cur <= 0.0 {
            buf.push(cur);
        }
        // Emit a crossing only when the signs genuinely differ, so the
        // denominator below is never zero.
        if (d_cur < 0.0 && d_next > 0.0) || (d_cur > 0.0 && d_next < 0.0) {
            let t = d_cur / (d_cur - d_next);
            buf.push(AxPoint2::new(
                cur.x + t * (next.x - cur.x),
                cur.y + t * (next.y - cur.y),
            ));
        }
    }

    std::mem::swap(poly, buf);
    true
}

/// Largest distance from `p` to any vertex of the polygon.
fn cell_radius(poly: &[AxPoint2<f64>], p: Point2<f64>) -> f64 {
    let mut r2 = 0.0_f64;
    for v in poly {
        let dx = v.x - p.x;
        let dy = v.y - p.y;
        r2 = r2.max(dx * dx + dy * dy);
    }
    r2.sqrt()
}

fn polygon_bounds(poly: &[AxPoint2<f64>]) -> [f64; 4] {
    let mut b = [
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    for v in poly {
        b[0] = b[0].min(v.x);
        b[1] = b[1].min(v.y);
        b[2] = b[2].max(v.x);
        b[3] = b[3].max(v.y);
    }
    b
}

#[inline]
fn bounds_overlap(a: &[f64; 4], b: &[f64; 4]) -> bool {
    a[0] <= b[2] && b[0] <= a[2] && a[1] <= b[3] && b[1] <= a[3]
}

/// Inclusive grid index range covering a bounding box, clamped to the grid.
fn grid_range(
    bounds: &[f64; 4],
    origin: (f64, f64),
    inv_cell: (f64, f64),
    nx: usize,
    ny: usize,
) -> (usize, usize, usize, usize) {
    if nx == 0 || ny == 0 {
        return (0, 0, 0, 0);
    }
    let to_ix = |v: f64, o: f64, inv: f64, n: usize| -> usize {
        let raw = ((v - o) * inv).floor();
        if !raw.is_finite() || raw < 0.0 {
            0
        } else {
            (raw as usize).min(n - 1)
        }
    };
    (
        to_ix(bounds[0], origin.0, inv_cell.0, nx),
        to_ix(bounds[1], origin.1, inv_cell.1, ny),
        to_ix(bounds[2], origin.0, inv_cell.0, nx),
        to_ix(bounds[3], origin.1, inv_cell.1, ny),
    )
}

/// RMS relative area error. Used as the line-search merit function because it is
/// smoother than the max, which is dominated by a single worst cell.
fn rms_relative_error(areas: &[f64], targets: &[f64]) -> f64 {
    let mut sum_sq = 0.0;
    let mut count = 0usize;

    for (area, target) in areas.iter().zip(targets.iter()) {
        if *target > 1e-12 {
            let e = (area - target) / target;
            sum_sq += e * e;
            count += 1;
        }
    }

    if count == 0 {
        return 0.0;
    }
    (sum_sq / count as f64).sqrt()
}

/// Compute maximum relative error between cell areas and target masses.
fn compute_max_relative_error(areas: &[f64], targets: &[f64]) -> f64 {
    let mut max_error = 0.0_f64;

    for (area, target) in areas.iter().zip(targets.iter()) {
        if *target > 1e-10 {
            let error = (area - target).abs() / target;
            max_error = max_error.max(error);
        }
    }

    max_error
}
/// Compute area distortion statistics for a UV map.
///
/// Returns `(min_ratio, max_ratio, rms_log_error)` where `ratio` is a
/// triangle's UV area divided by the UV area it would have under a perfectly
/// area-preserving map. The global scale is divided out, so the measure is
/// invariant to uniform scaling of the UVs; `rms_log_error` is the RMS of
/// `ln(ratio)`, which penalizes shrinking and stretching symmetrically.
/// A perfect map gives `(1, 1, 0)`.
pub fn compute_area_distortion<I: MeshIndex>(
    mesh: &HalfEdgeMesh<I>,
    uvs: &UVMap<I>,
) -> (f64, f64, f64) {
    let mut ratios = Vec::new();

    // Compute total areas for scale factor
    let mut total_3d_area = 0.0;
    let mut total_uv_area = 0.0;

    for fid in mesh.face_ids() {
        let area_3d = mesh.face_area(fid);
        if !area_3d.is_finite() || area_3d < 1e-12 {
            continue;
        }

        let [v0, v1, v2] = mesh.face_triangle(fid);
        let uv0 = uvs.get(v0);
        let uv1 = uvs.get(v1);
        let uv2 = uvs.get(v2);

        // 2D signed area
        let area_uv =
            0.5 * ((uv1.x - uv0.x) * (uv2.y - uv0.y) - (uv2.x - uv0.x) * (uv1.y - uv0.y)).abs();

        total_3d_area += area_3d;
        total_uv_area += area_uv;
    }

    if total_3d_area < 1e-10 || total_uv_area < 1e-10 {
        return (1.0, 1.0, 0.0);
    }

    let scale = total_uv_area / total_3d_area;

    // Compute per-triangle ratios
    for fid in mesh.face_ids() {
        let area_3d = mesh.face_area(fid);
        if !area_3d.is_finite() || area_3d < 1e-12 {
            continue;
        }

        let [v0, v1, v2] = mesh.face_triangle(fid);
        let uv0 = uvs.get(v0);
        let uv1 = uvs.get(v1);
        let uv2 = uvs.get(v2);

        let area_uv =
            0.5 * ((uv1.x - uv0.x) * (uv2.y - uv0.y) - (uv2.x - uv0.x) * (uv1.y - uv0.y)).abs();

        let expected_uv_area = area_3d * scale;
        if expected_uv_area > 1e-12 {
            ratios.push(area_uv / expected_uv_area);
        }
    }

    if ratios.is_empty() {
        return (1.0, 1.0, 0.0);
    }

    let min_ratio = ratios.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_ratio = ratios.iter().cloned().fold(0.0, f64::max);

    // RMS of log-ratio (symmetric measure of distortion)
    let rms_error: f64 = (ratios
        .iter()
        .filter(|r| **r > 0.0)
        .map(|r| (r.ln()).powi(2))
        .sum::<f64>()
        / ratios.len() as f64)
        .sqrt();

    (min_ratio, max_ratio, rms_error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algo::parameterize::{lscm, LSCMOptions};
    use crate::mesh::build_from_triangles;
    use nalgebra::Point3;

    fn create_disk_mesh() -> HalfEdgeMesh {
        // Simple disk: center vertex + 6 boundary vertices
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0), // center
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.5, 0.866, 0.0),
            Point3::new(-0.5, 0.866, 0.0),
            Point3::new(-1.0, 0.0, 0.0),
            Point3::new(-0.5, -0.866, 0.0),
            Point3::new(0.5, -0.866, 0.0),
        ];
        let faces = vec![
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 4],
            [0, 4, 5],
            [0, 5, 6],
            [0, 6, 1],
        ];
        build_from_triangles(&vertices, &faces).unwrap()
    }

    /// Flat `n × n` grid in the z = 0 plane. The conformal map is already
    /// isometric here, so a correct area-preserving step must do nothing.
    fn create_grid_mesh(n: usize) -> HalfEdgeMesh {
        create_height_grid(n, |_, _| 0.0)
    }

    /// Grid over `[-1, 1]²` lifted by `height`, giving a curved patch with disk
    /// topology whose conformal map has genuine area distortion.
    fn create_height_grid<F: Fn(f64, f64) -> f64>(n: usize, height: F) -> HalfEdgeMesh {
        let mut vertices = Vec::new();
        let mut faces = Vec::new();

        for j in 0..=n {
            for i in 0..=n {
                let x = -1.0 + 2.0 * (i as f64) / (n as f64);
                let y = -1.0 + 2.0 * (j as f64) / (n as f64);
                vertices.push(Point3::new(x, y, height(x, y)));
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

    /// A paraboloid patch: area element `sqrt(1 + x² + y²)` ranges from 1 at the
    /// center to `sqrt(3)` at the corners, so LSCM must distort area.
    fn create_paraboloid(n: usize) -> HalfEdgeMesh {
        create_height_grid(n, |x, y| 0.5 * (x * x + y * y))
    }

    #[test]
    fn test_compute_target_masses() {
        let mesh = create_disk_mesh();
        let masses = compute_target_masses(&mesh);

        assert_eq!(masses.len(), 7);

        // Center vertex should have highest mass (connected to all triangles)
        let center_mass = masses[0];
        for &m in &masses[1..7] {
            assert!(center_mass > m);
        }

        // Total mass should equal total surface area
        let total_mass: f64 = masses.iter().sum();
        let expected_area: f64 = mesh.face_ids().map(|f| mesh.face_area(f)).sum();
        assert!((total_mass - expected_area).abs() < 1e-10);
    }

    #[test]
    fn test_omt_basic() {
        let mesh = create_disk_mesh();

        let uv_lscm = lscm(&mesh, &LSCMOptions::default()).unwrap();
        let uv_omt = omt(&mesh, &uv_lscm, &OMTOptions::default()).unwrap();

        assert_eq!(uv_omt.len(), 7);

        // normalize() puts UVs in [0, 1].
        for (_, uv) in uv_omt.iter() {
            assert!(
                uv.x >= -1e-9 && uv.x <= 1.0 + 1e-9,
                "u out of range: {}",
                uv.x
            );
            assert!(
                uv.y >= -1e-9 && uv.y <= 1.0 + 1e-9,
                "v out of range: {}",
                uv.y
            );
        }
    }

    /// The power cells must tile the domain: their areas sum to the domain area
    /// exactly, for any weights. This is the invariant that the grid-sampling
    /// estimator could only approximate, and it is what makes the convergence
    /// tolerance meaningful.
    #[test]
    fn test_power_cells_partition_the_domain() {
        // The flat grids are the important cases: a regular lattice is maximally
        // degenerate, with cocircular quadruples of sites everywhere and cell
        // edges exactly parallel to triangle edges.
        for mesh in [
            create_grid_mesh(2),
            create_grid_mesh(3),
            create_grid_mesh(4),
            create_paraboloid(6),
        ] {
            check_partition(&mesh);
        }
    }

    fn check_partition(mesh: &HalfEdgeMesh) {
        let uvs = lscm(mesh, &LSCMOptions::default()).unwrap();
        let domain = DomainMesh::new(mesh, &uvs);
        let positions = uvs.as_slice().to_vec();
        let n = positions.len();

        // Zero weights (a plain Voronoi diagram), then a deliberately uneven set.
        let uneven: Vec<f64> = (0..n).map(|i| 0.002 * ((i % 7) as f64 - 3.0)).collect();

        for (label, weights) in [("zero", vec![0.0; n]), ("uneven", uneven)] {
            let cells = domain.power_cells(&positions, &weights);
            let total: f64 = cells.areas.iter().sum();
            let rel = (total - domain.area()).abs() / domain.area();
            println!(
                "{} verts, {label} weights: Σ cell areas = {:.15}, domain = {:.15}, rel err = {:.2e}",
                n,
                total,
                domain.area(),
                rel
            );
            assert!(
                rel < 1e-12,
                "{n} verts, {label} weights: cells should tile the domain, \
                 relative error {rel:.2e}"
            );
        }
    }

    /// Every cell centroid must lie inside the domain's bounding box, and a cell
    /// with positive area must produce a centroid at all.
    #[test]
    fn test_power_cell_centroids_are_sane() {
        let mesh = create_paraboloid(5);
        let uvs = lscm(&mesh, &LSCMOptions::default()).unwrap();
        let domain = DomainMesh::new(&mesh, &uvs);
        let positions = uvs.as_slice().to_vec();
        let cells = domain.power_cells(&positions, &vec![0.0; positions.len()]);

        let frame = domain.frame;
        for i in 0..positions.len() {
            if cells.areas[i] <= 0.0 {
                continue;
            }
            let c = cells
                .centroid(i)
                .unwrap_or_else(|| panic!("site {i} has area but no centroid"));
            assert!(
                c.x >= frame[0].x - 1e-9
                    && c.x <= frame[2].x + 1e-9
                    && c.y >= frame[0].y - 1e-9
                    && c.y <= frame[2].y + 1e-9,
                "centroid {c:?} outside domain frame"
            );
        }
    }

    /// Raising one site's weight must grow its cell at its neighbours' expense,
    /// while the total stays put.
    #[test]
    fn test_power_cell_weight_monotonicity() {
        let mesh = create_grid_mesh(4);
        let uvs = lscm(&mesh, &LSCMOptions::default()).unwrap();
        let domain = DomainMesh::new(&mesh, &uvs);
        let positions = uvs.as_slice().to_vec();
        let n = positions.len();

        // An interior site, so its cell is bounded by rivals on all sides.
        let target = n / 2;

        let base = domain.power_cells(&positions, &vec![0.0; n]);
        let mut raised = vec![0.0; n];
        raised[target] = 0.01;
        let bumped = domain.power_cells(&positions, &raised);

        assert!(
            bumped.areas[target] > base.areas[target],
            "raising a site's weight should grow its cell: {} -> {}",
            base.areas[target],
            bumped.areas[target]
        );

        let base_total: f64 = base.areas.iter().sum();
        let bumped_total: f64 = bumped.areas.iter().sum();
        assert!(
            (base_total - bumped_total).abs() / base_total < 1e-12,
            "total area must be conserved: {base_total} vs {bumped_total}"
        );
    }

    /// The real acceptance criterion: on a curved patch, where the conformal map
    /// genuinely distorts area, OMT must *reduce* that distortion.
    #[test]
    fn test_omt_reduces_area_distortion_on_curved_patch() {
        let mesh = create_paraboloid(8);

        let uv_lscm = lscm(&mesh, &LSCMOptions::default()).unwrap();
        let (_, _, rms_lscm) = compute_area_distortion(&mesh, &uv_lscm);

        // Sanity: the baseline must actually have distortion to remove,
        // otherwise this test proves nothing.
        assert!(
            rms_lscm > 0.05,
            "paraboloid LSCM baseline should be visibly distorted, got rms={rms_lscm:.6}"
        );

        let (uv_omt, report) = omt_with_report(&mesh, &uv_lscm, &OMTOptions::default()).unwrap();
        let (_, _, rms_omt) = compute_area_distortion(&mesh, &uv_omt);

        println!(
            "paraboloid: LSCM rms={:.6} -> OMT rms={:.6} ({:.0}% of baseline); \
             {} iters, max_rel_err={:.3e}, converged={}",
            rms_lscm,
            rms_omt,
            100.0 * rms_omt / rms_lscm,
            report.iterations,
            report.max_relative_error,
            report.converged
        );

        assert!(
            rms_omt < 0.6 * rms_lscm,
            "OMT should cut area distortion substantially: {rms_lscm:.6} -> {rms_omt:.6}"
        );
    }

    /// Same acceptance criterion on a saddle, which distorts area with the
    /// opposite sign of Gaussian curvature.
    #[test]
    fn test_omt_reduces_area_distortion_on_saddle() {
        let mesh = create_height_grid(8, |x, y| 0.6 * (x * x - y * y));

        let uv_lscm = lscm(&mesh, &LSCMOptions::default()).unwrap();
        let (_, _, rms_lscm) = compute_area_distortion(&mesh, &uv_lscm);
        assert!(
            rms_lscm > 0.05,
            "saddle LSCM baseline should be visibly distorted, got rms={rms_lscm:.6}"
        );

        let uv_omt = omt(&mesh, &uv_lscm, &OMTOptions::default()).unwrap();
        let (_, _, rms_omt) = compute_area_distortion(&mesh, &uv_omt);

        println!(
            "saddle: LSCM rms={:.6} -> OMT rms={:.6} ({:.0}% of baseline)",
            rms_lscm,
            rms_omt,
            100.0 * rms_omt / rms_lscm
        );

        assert!(
            rms_omt < 0.6 * rms_lscm,
            "OMT should cut area distortion substantially: {rms_lscm:.6} -> {rms_omt:.6}"
        );
    }

    /// Regression guard for the Lloyd bug: on a flat grid the conformal map is
    /// already isometric, so OMT must leave it alone rather than drifting toward
    /// a uniformly-spaced (centroidal) configuration.
    #[test]
    fn test_omt_preserves_isometric_grid() {
        for n in [4usize, 8] {
            let mesh = create_grid_mesh(n);

            let uv_lscm = lscm(&mesh, &LSCMOptions::default()).unwrap();
            let (_, _, rms_lscm) = compute_area_distortion(&mesh, &uv_lscm);
            assert!(
                rms_lscm < 1e-5,
                "flat grid LSCM should be isometric, got rms={rms_lscm:.3e}"
            );

            let uv_omt = omt(&mesh, &uv_lscm, &OMTOptions::default()).unwrap();
            let (min_r, max_r, rms_omt) = compute_area_distortion(&mesh, &uv_omt);

            println!(
                "flat grid {n}x{n}: LSCM rms={:.3e} -> OMT rms={:.3e} (min={:.4}, max={:.4})",
                rms_lscm, rms_omt, min_r, max_r
            );

            // The grid-sampling implementation scored ~1.23 here; exact cells
            // should leave an isometric map essentially untouched.
            assert!(
                rms_omt < 1e-3,
                "OMT distorted an already-isometric {n}x{n} grid: \
                 {rms_lscm:.3e} -> {rms_omt:.6}"
            );
        }
    }
}
