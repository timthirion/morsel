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
//! `∂Φ/∂w_i = ν_i − |Pow_i(w) ∩ D|`, so plain gradient ascent converges.
//! Once the cell areas match the target masses, vertex `i` is moved to the
//! **centroid of its own cell**. That is the barycentric projection of the
//! Brenier map, and it is applied exactly once — it is a transport step, not
//! a relaxation.
//!
//! # A note on Lloyd relaxation
//!
//! An earlier version of this module iterated the centroid step as Lloyd
//! relaxation. That is incorrect and actively harmful: iterating drives the
//! configuration toward a *centroidal* power diagram, whose fixed point is
//! uniformly spaced sites — the opposite of respecting the target masses.
//! Measured on a flat grid (where the conformal map is already isometric and
//! the correct answer is to do nothing), each extra iteration made area
//! distortion monotonically worse. The centroid step is applied once.
//!
//! # References
//!
//! - Gu, X., et al. (2013). "Area-Preserving Parameterization via Optimal
//!   Mass Transport." IEEE TVCG.
//! - Mérigot, Q. (2011). "A Multiscale Approach to Optimal Transport."
//!   Computer Graphics Forum.
//! - Kitagawa, J., Mérigot, Q., Thibert, B. (2019). "Convergence of a Newton
//!   algorithm for semi-discrete optimal transport." JEMS.

use nalgebra::Point2;
use rayon::prelude::*;

use crate::error::{MeshError, Result};
use crate::mesh::{HalfEdgeMesh, MeshIndex};

use super::UVMap;

/// Options for OMT-based area-preserving parameterization.
#[derive(Debug, Clone)]
pub struct OMTOptions {
    /// Maximum number of weight-ascent iterations.
    pub max_iterations: usize,

    /// Convergence tolerance on the worst per-cell relative area error.
    ///
    /// Cell areas are estimated by counting grid samples, so the achievable
    /// accuracy is bounded by the sampling: a cell covering `k` samples has an
    /// area estimate good to roughly `1/√k`. At `grid_resolution = 256` over a
    /// few thousand cells, the worst-cell error bottoms out around `10⁻²`, so
    /// asking for much less than that just spends the whole iteration budget.
    /// Computing exact power-cell polygons by clipping would remove this floor.
    pub tolerance: f64,

    /// Grid resolution used to estimate power-cell areas and centroids.
    /// Higher values give more accurate cells but cost `O(resolution²)`
    /// samples per iteration, and each sample scans every site.
    pub grid_resolution: usize,

    /// Step size for the gradient ascent on the dual. The dual gradient has
    /// units of area and the weights units of length², so this is
    /// dimensionless; values near `0.5` track a Newton step reasonably well
    /// for well-shaped cells.
    pub step_size: f64,

    /// Hold boundary vertices at their initial UV positions.
    ///
    /// Defaults to `true`, and should normally stay there. A boundary site's
    /// power cell is clipped by the edge of the domain, so its centroid lies
    /// strictly *inside* the domain — letting the transport step move it
    /// contracts the boundary inward on every vertex at once. Measured against
    /// a correct conformal baseline (`rms` of log area ratio, `res = 256`):
    ///
    /// | patch          | LSCM  | `false` | `true` |
    /// |----------------|-------|---------|--------|
    /// | paraboloid 8×8 | 0.168 | 0.207   | 0.061  |
    /// | saddle 8×8     | 0.238 | 0.224   | 0.081  |
    ///
    /// With the boundary free, OMT is no better than doing nothing. Letting
    /// boundary vertices *slide along* the boundary curve — rather than
    /// pinning them or freeing them entirely — is the principled treatment and
    /// is not implemented yet.
    pub fix_boundary: bool,
}

impl Default for OMTOptions {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-2,
            grid_resolution: 256,
            step_size: 0.5,
            fix_boundary: true,
        }
    }
}

impl OMTOptions {
    /// Create options for fast but less accurate computation.
    pub fn fast() -> Self {
        Self {
            max_iterations: 50,
            tolerance: 1e-2,
            grid_resolution: 128,
            step_size: 0.8,
            fix_boundary: true,
        }
    }

    /// Create options for high-quality results.
    pub fn high_quality() -> Self {
        Self {
            max_iterations: 200,
            tolerance: 5e-3,
            grid_resolution: 512,
            step_size: 0.3,
            fix_boundary: true,
        }
    }

    /// Choose a grid resolution appropriate to a mesh's vertex count.
    ///
    /// Useful accuracy needs a few hundred grid samples per power cell, and the
    /// UV footprint covers only a fraction of the sampled square, so the
    /// resolution has to scale like `√n`. This targets ~300 samples per cell.
    ///
    /// The resolution is capped at 2048: the cost of a solve is
    /// `O(resolution² · n)` *per iteration* — every sample scans every site — so
    /// past a few thousand vertices no affordable resolution is sufficient. For
    /// meshes that large, check [`OMTReport::is_well_sampled`] on the result and
    /// treat a `false` as "OMT did not help here". Computing exact power-cell
    /// polygons by clipping, rather than sampling, is the way past this.
    pub fn for_vertex_count(n_vertices: usize) -> Self {
        // samples ≈ 0.38 · res², and we want ≈ 300 · n of them.
        let ideal = (300.0 * n_vertices as f64 / 0.38).sqrt().ceil() as usize;
        let resolution = ideal.clamp(64, 2048);
        Self {
            grid_resolution: resolution,
            ..Default::default()
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

    /// Number of grid samples that fell inside the UV footprint.
    pub domain_samples: usize,

    /// Number of sites (mesh vertices) sharing those samples.
    pub sites: usize,
}

impl OMTReport {
    /// Average number of grid samples backing each power cell.
    ///
    /// This is the single number that decides whether an OMT run is worth
    /// anything: cell areas are estimated by counting samples, so too few
    /// samples per cell means the areas driving the transport are noise.
    /// See [`OMTReport::is_well_sampled`].
    pub fn samples_per_cell(&self) -> f64 {
        if self.sites == 0 {
            return 0.0;
        }
        self.domain_samples as f64 / self.sites as f64
    }

    /// Whether the run had enough samples per cell to be trustworthy.
    ///
    /// Measured on a paraboloid patch, area distortion relative to the
    /// conformal baseline as a function of samples per cell:
    ///
    /// | samples/cell | result          |
    /// |--------------|-----------------|
    /// | 2–6          | 118–157% (worse)|
    /// | 10–23        | 99–107%         |
    /// | 40–90        | 50–83%          |
    /// | 300+         | 32–37%          |
    ///
    /// Below roughly 25 the estimator is bad enough that OMT *increases*
    /// distortion, so this reports `false` there. Since the useful sample count
    /// grows with the vertex count, `grid_resolution` has to grow like `√n` —
    /// see [`OMTOptions::for_vertex_count`].
    ///
    /// Adequate sampling is necessary but not sufficient. Holding samples per
    /// cell at ~300, an 81-vertex patch reaches 37% of its conformal baseline
    /// but a 2601-vertex one only 85%: the vertex dual-cell area that this
    /// method equalizes is a weaker proxy for per-triangle area as the mesh
    /// refines, and the one-shot barycentric projection does not close that gap.
    pub fn is_well_sampled(&self) -> bool {
        self.samples_per_cell() >= 25.0
    }
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

    if options.grid_resolution < 2 {
        return Err(MeshError::InvalidState(
            "OMT grid_resolution must be at least 2".to_string(),
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

    // The transport domain is the region the parameterization actually covers,
    // not its bounding box. Using the box would hand the sites mass from empty
    // corners, which no choice of weights can account for.
    let (domain_min, domain_max) = compute_bounding_box(&positions);
    let samples = rasterize_uv_footprint(
        mesh,
        initial_uvs,
        domain_min,
        domain_max,
        options.grid_resolution,
    );

    if samples.is_empty() {
        return Err(MeshError::InvalidState(
            "UV footprint did not cover any grid samples; try a higher grid_resolution".to_string(),
        ));
    }

    let dx = (domain_max.x - domain_min.x) / options.grid_resolution as f64;
    let dy = (domain_max.y - domain_min.y) / options.grid_resolution as f64;
    let sample_area = dx * dy;
    let domain_area = samples.len() as f64 * sample_area;

    // Normalize target masses so they sum to the domain measure; only then can
    // the cell areas match them.
    let mass_scale = domain_area / total_target_mass;
    let target_masses: Vec<f64> = target_masses.iter().map(|m| m * mass_scale).collect();

    // Ascent on the concave Kantorovich dual. The raw gradient
    // `ν_i − A_i` is poorly scaled when the target masses vary, so it is
    // preconditioned by `1 / ν_i` (a positive diagonal scaling, so still an
    // ascent direction) and expressed in units of the mean cell area — the
    // natural scale for weights, which have units of length².
    //
    // Step length is chosen by backtracking on the RMS relative area error.
    // A plain fixed step does not converge here: it overshoots the cells with
    // small targets and stalls on the large ones.
    let cell_scale = domain_area / n_vertices as f64;
    let mut weights = vec![0.0; n_vertices];
    let mut cells = compute_power_cells(&positions, &weights, &samples, sample_area);
    let mut merit = rms_relative_error(&cells.areas, &target_masses);
    let mut max_error = compute_max_relative_error(&cells.areas, &target_masses);
    let mut step = options.step_size;
    let mut iterations = 0;
    let mut converged = max_error < options.tolerance;

    while iterations < options.max_iterations && !converged {
        // Trial step.
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

        let trial_cells = compute_power_cells(&positions, &trial, &samples, sample_area);
        let trial_merit = rms_relative_error(&trial_cells.areas, &target_masses);
        iterations += 1;

        if trial_merit < merit {
            weights = trial;
            cells = trial_cells;
            merit = trial_merit;
            max_error = compute_max_relative_error(&cells.areas, &target_masses);
            converged = max_error < options.tolerance;
            // Creep the step back up so a single bad stretch doesn't
            // permanently cripple progress.
            step = (step * 1.3).min(options.step_size * 4.0);
        } else {
            step *= 0.5;
            // Below this the grid sampling noise dominates any real progress.
            if step < 1e-4 {
                break;
            }
        }
    }

    // Single transport step: move each site to the centroid of its own cell.
    // This is the barycentric projection of the Brenier map. Applying it more
    // than once turns it into Lloyd relaxation, which is a different (wrong)
    // fixed point — see the module docs.
    let cells = compute_power_cells(&positions, &weights, &samples, sample_area);
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
        domain_samples: samples.len(),
        sites: n_vertices,
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

/// Compute bounding box of UV positions, with a small padding so that sites
/// sitting exactly on the extent still have grid samples around them.
fn compute_bounding_box(positions: &[Point2<f64>]) -> (Point2<f64>, Point2<f64>) {
    let mut min = positions[0];
    let mut max = positions[0];

    for p in positions {
        min.x = min.x.min(p.x);
        min.y = min.y.min(p.y);
        max.x = max.x.max(p.x);
        max.y = max.y.max(p.y);
    }

    let padding = ((max.x - min.x) + (max.y - min.y)) * 0.01;
    min.x -= padding;
    min.y -= padding;
    max.x += padding;
    max.y += padding;

    (min, max)
}

/// Collect the grid sample centers that lie inside the UV image of the mesh.
///
/// Rasterizing the actual footprint — rather than the whole bounding box —
/// keeps the transport domain equal to the region the parameterization covers.
fn rasterize_uv_footprint<I: MeshIndex>(
    mesh: &HalfEdgeMesh<I>,
    uvs: &UVMap<I>,
    domain_min: Point2<f64>,
    domain_max: Point2<f64>,
    resolution: usize,
) -> Vec<Point2<f64>> {
    let dx = (domain_max.x - domain_min.x) / resolution as f64;
    let dy = (domain_max.y - domain_min.y) / resolution as f64;
    if dx <= 0.0 || dy <= 0.0 {
        return Vec::new();
    }

    let mut inside = vec![false; resolution * resolution];

    for fid in mesh.face_ids() {
        let [v0, v1, v2] = mesh.face_triangle(fid);
        let a = uvs.get(v0);
        let b = uvs.get(v1);
        let c = uvs.get(v2);

        // Signed area; skip UV-degenerate triangles.
        let double_area = (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y);
        if !double_area.is_finite() || double_area.abs() < 1e-18 {
            continue;
        }

        let tri_min_x = a.x.min(b.x).min(c.x);
        let tri_max_x = a.x.max(b.x).max(c.x);
        let tri_min_y = a.y.min(b.y).min(c.y);
        let tri_max_y = a.y.max(b.y).max(c.y);

        let gx0 = (((tri_min_x - domain_min.x) / dx).floor() as isize).max(0) as usize;
        let gx1 = ((((tri_max_x - domain_min.x) / dx).ceil() as isize).max(0) as usize)
            .min(resolution);
        let gy0 = (((tri_min_y - domain_min.y) / dy).floor() as isize).max(0) as usize;
        let gy1 = ((((tri_max_y - domain_min.y) / dy).ceil() as isize).max(0) as usize)
            .min(resolution);

        let inv = 1.0 / double_area;
        for gy in gy0..gy1 {
            let y = domain_min.y + (gy as f64 + 0.5) * dy;
            for gx in gx0..gx1 {
                let x = domain_min.x + (gx as f64 + 0.5) * dx;

                // Barycentric inside test, orientation-agnostic.
                let l0 = ((b.x - x) * (c.y - y) - (c.x - x) * (b.y - y)) * inv;
                let l1 = ((c.x - x) * (a.y - y) - (a.x - x) * (c.y - y)) * inv;
                let l2 = 1.0 - l0 - l1;
                if l0 >= 0.0 && l1 >= 0.0 && l2 >= 0.0 {
                    inside[gy * resolution + gx] = true;
                }
            }
        }
    }

    let mut samples = Vec::new();
    for gy in 0..resolution {
        let y = domain_min.y + (gy as f64 + 0.5) * dy;
        for gx in 0..resolution {
            if inside[gy * resolution + gx] {
                let x = domain_min.x + (gx as f64 + 0.5) * dx;
                samples.push(Point2::new(x, y));
            }
        }
    }

    samples
}

/// Per-site area and centroid accumulators for a power diagram.
struct PowerCells {
    areas: Vec<f64>,
    sum_x: Vec<f64>,
    sum_y: Vec<f64>,
    counts: Vec<usize>,
}

impl PowerCells {
    fn centroid(&self, i: usize) -> Option<Point2<f64>> {
        if self.counts[i] == 0 {
            return None;
        }
        let inv = 1.0 / self.counts[i] as f64;
        Some(Point2::new(self.sum_x[i] * inv, self.sum_y[i] * inv))
    }
}

/// Estimate power-diagram cell areas and centroids over the sampled domain.
///
/// The power distance from `x` to site `i` with weight `w_i` is
/// `‖x − p_i‖² − w_i`; `x` belongs to the cell minimizing it.
///
/// Areas and centroids are accumulated in a single pass — they are always
/// needed for the same weights, and the site scan dominates the cost.
fn compute_power_cells(
    positions: &[Point2<f64>],
    weights: &[f64],
    samples: &[Point2<f64>],
    sample_area: f64,
) -> PowerCells {
    let n = positions.len();

    let (counts, sum_x, sum_y) = samples
        .par_chunks(4096)
        .map(|chunk| {
            let mut counts = vec![0usize; n];
            let mut sum_x = vec![0.0f64; n];
            let mut sum_y = vec![0.0f64; n];

            for s in chunk {
                let mut min_power_dist = f64::INFINITY;
                let mut owner = 0usize;

                for i in 0..n {
                    let ddx = s.x - positions[i].x;
                    let ddy = s.y - positions[i].y;
                    let power_dist = ddx * ddx + ddy * ddy - weights[i];
                    if power_dist < min_power_dist {
                        min_power_dist = power_dist;
                        owner = i;
                    }
                }

                counts[owner] += 1;
                sum_x[owner] += s.x;
                sum_y[owner] += s.y;
            }

            (counts, sum_x, sum_y)
        })
        .reduce(
            || (vec![0usize; n], vec![0.0f64; n], vec![0.0f64; n]),
            |mut acc, part| {
                for i in 0..n {
                    acc.0[i] += part.0[i];
                    acc.1[i] += part.1[i];
                    acc.2[i] += part.2[i];
                }
                acc
            },
        );

    let areas = counts.iter().map(|&c| c as f64 * sample_area).collect();

    PowerCells {
        areas,
        sum_x,
        sum_y,
        counts,
    }
}

/// RMS relative area error. Used as the line-search merit function because it
/// is smoother than the max, which is dominated by a single worst cell.
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
        let area_uv = 0.5
            * ((uv1.x - uv0.x) * (uv2.y - uv0.y) - (uv2.x - uv0.x) * (uv1.y - uv0.y)).abs();

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

        let area_uv = 0.5
            * ((uv1.x - uv0.x) * (uv2.y - uv0.y) - (uv2.x - uv0.x) * (uv1.y - uv0.y)).abs();

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

    /// Grid over `[-1, 1]²` lifted by `height`, giving a curved patch with
    /// disk topology whose conformal map has genuine area distortion.
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

    /// A paraboloid patch: area element `sqrt(1 + x² + y²)` ranges from 1 at
    /// the center to `sqrt(3)` at the corners, so LSCM must distort area.
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
        let options = OMTOptions {
            max_iterations: 50,
            grid_resolution: 64,
            ..Default::default()
        };
        let uv_omt = omt(&mesh, &uv_lscm, &options).unwrap();

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

    /// The real acceptance criterion: on a curved patch, where the conformal
    /// map genuinely distorts area, OMT must *reduce* that distortion.
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
             {} iters, max_rel_err={:.4}, converged={}, samples={}",
            rms_lscm,
            rms_omt,
            100.0 * rms_omt / rms_lscm,
            report.iterations,
            report.max_relative_error,
            report.converged,
            report.domain_samples
        );

        // Measured ~0.061 vs a 0.168 baseline (36%). Require a real reduction,
        // not merely "not worse", so a regression to the old behaviour fails.
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

        // Measured ~0.081 vs a 0.238 baseline (34%).
        assert!(
            rms_omt < 0.6 * rms_lscm,
            "OMT should cut area distortion substantially: {rms_lscm:.6} -> {rms_omt:.6}"
        );
    }

    /// Regression guard for the Lloyd bug: on a flat grid the conformal map is
    /// already isometric, so OMT must leave it essentially alone rather than
    /// drifting toward a uniformly-spaced (centroidal) configuration.
    #[test]
    fn test_omt_preserves_isometric_grid() {
        let mesh = create_grid_mesh(4);

        let uv_lscm = lscm(&mesh, &LSCMOptions::default()).unwrap();
        let (_, _, rms_lscm) = compute_area_distortion(&mesh, &uv_lscm);
        assert!(
            rms_lscm < 0.01,
            "flat grid LSCM should be near-isometric, got rms={rms_lscm:.6}"
        );

        let uv_omt = omt(&mesh, &uv_lscm, &OMTOptions::default()).unwrap();
        let (min_r, max_r, rms_omt) = compute_area_distortion(&mesh, &uv_omt);

        println!(
            "flat grid: LSCM rms={:.6} -> OMT rms={:.6} (min={:.3}, max={:.3})",
            rms_lscm, rms_omt, min_r, max_r
        );

        // The old Lloyd-iterating implementation scored ~1.23 here, ~2000x the
        // baseline. Allow discretization noise but nothing structural.
        assert!(
            rms_omt < 0.05,
            "OMT distorted an already-isometric map: {rms_lscm:.6} -> {rms_omt:.6}"
        );
    }

    #[test]
    fn test_power_cell_areas() {
        // Four sites in the corners of a square domain, uniform weights: each
        // should own about a quarter of the samples.
        let positions = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(0.0, 1.0),
            Point2::new(1.0, 1.0),
        ];
        let weights = vec![0.0; 4];

        let resolution = 100;
        let (dmin, dmax) = (Point2::new(-0.1, -0.1), Point2::new(1.1, 1.1));
        let dx = (dmax.x - dmin.x) / resolution as f64;
        let dy = (dmax.y - dmin.y) / resolution as f64;

        let mut samples = Vec::new();
        for gy in 0..resolution {
            for gx in 0..resolution {
                samples.push(Point2::new(
                    dmin.x + (gx as f64 + 0.5) * dx,
                    dmin.y + (gy as f64 + 0.5) * dy,
                ));
            }
        }

        let cells = compute_power_cells(&positions, &weights, &samples, dx * dy);

        let total: f64 = cells.areas.iter().sum();
        for area in &cells.areas {
            let ratio = area / total;
            assert!((ratio - 0.25).abs() < 0.1, "Expected ~0.25, got {}", ratio);
        }
    }

    /// Raising the weight of one site must grow its cell at the others' expense.
    #[test]
    fn test_power_cell_weight_monotonicity() {
        let positions = vec![
            Point2::new(0.25, 0.5),
            Point2::new(0.75, 0.5),
        ];
        let resolution = 64;
        let mut samples = Vec::new();
        for gy in 0..resolution {
            for gx in 0..resolution {
                samples.push(Point2::new(
                    (gx as f64 + 0.5) / resolution as f64,
                    (gy as f64 + 0.5) / resolution as f64,
                ));
            }
        }
        let sample_area = 1.0 / (resolution * resolution) as f64;

        let balanced = compute_power_cells(&positions, &[0.0, 0.0], &samples, sample_area);
        assert!((balanced.areas[0] - balanced.areas[1]).abs() < 1e-9);

        let tilted = compute_power_cells(&positions, &[0.1, 0.0], &samples, sample_area);
        assert!(
            tilted.areas[0] > balanced.areas[0],
            "raising a site's weight should grow its cell"
        );
    }
}
