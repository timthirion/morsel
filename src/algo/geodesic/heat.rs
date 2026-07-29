//! Heat method for geodesic distances.
//!
//! Computes geodesic distances using the heat method (Crane et al. 2013).
//! This gives smooth approximate geodesic distances by solving heat diffusion
//! followed by a Poisson equation.

use nalgebra::{DVector, Point3, Vector3};

use crate::algo::parameterize::sparse::{
    preconditioned_conjugate_gradient, CsrMatrix, PinnedReduction,
};
use crate::error::Result;
use crate::mesh::{HalfEdgeMesh, MeshIndex, VertexId};

use super::GeodesicResult;

/// Options for the heat method geodesic distance computation.
#[derive(Debug, Clone)]
pub struct HeatMethodOptions {
    /// Time step for the heat diffusion, `t`.
    ///
    /// If `None`, defaults to `10 h²` where `h` is the mean edge length.
    ///
    /// The multiplier matters more than it looks, and `h²` — the bare short-time
    /// limit — is the wrong choice. Heat decays as `exp(-d²/4t)`, so a small `t`
    /// puts the far field below what the linear solve can resolve: at `t = h²` on a
    /// 6401-vertex cap the solution at distance `0.5` is around `exp(-156)`, some 58
    /// orders of magnitude beneath the solver's `1e-10` residual, and therefore
    /// numerical noise whose gradient direction is meaningless. Refining the mesh
    /// shrinks `t` and makes it worse.
    ///
    /// Measured mean relative error against exact geodesics on a spherical cap,
    /// source off-centre so the field is not radially symmetric:
    ///
    /// | vertices | `t = h²` | `t = 10 h²` | `t = 100 h²` |
    /// |----------|----------|-------------|--------------|
    /// | 101      | −7.0%    | −5.9%       | −5.8%        |
    /// | 401      | −4.1%    | −2.3%       | −1.5%        |
    /// | 1601     | −6.1%    | −0.8%       | +0.6%        |
    /// | 6401     | **−27%** | **−0.3%**   | +1.2%        |
    ///
    /// At `h²` the error *grows* with refinement, which is a diverging method. At
    /// `10 h²` it converges monotonically toward zero, which is what the heat
    /// method is supposed to do. Beyond that, over-smoothing sets in and the error
    /// settles at a small positive bias.
    pub time_step: Option<f64>,

    /// Maximum iterations for the conjugate gradient solver.
    ///
    /// The Poisson stage solves against a cotangent Laplacian, whose condition
    /// number grows with refinement, so the iteration count has to as well. The
    /// old default of 1000 was already insufficient at 6401 vertices, where it
    /// left enough solver error to make the method look *divergent* — the measured
    /// error stopped shrinking under refinement and turned back up. An unused
    /// allowance costs nothing, since each iteration is one sparse mat-vec.
    pub max_cg_iterations: usize,

    /// Convergence tolerance for the CG solver, as a relative residual.
    ///
    /// Tightened alongside the iteration cap for the same reason. Note the far
    /// field of the heat solution is genuinely tiny — see [`Self::time_step`] —
    /// so residual tolerance and time step have to be chosen together.
    pub cg_tolerance: f64,
}

impl Default for HeatMethodOptions {
    fn default() -> Self {
        Self {
            time_step: None,
            max_cg_iterations: 20_000,
            cg_tolerance: 1e-10,
        }
    }
}

impl HeatMethodOptions {
    /// Set custom time step (overrides auto-computation).
    pub fn with_time_step(mut self, t: f64) -> Self {
        self.time_step = Some(t);
        self
    }

    /// Set maximum CG iterations.
    pub fn with_max_cg_iterations(mut self, max_iter: usize) -> Self {
        self.max_cg_iterations = max_iter;
        self
    }

    /// Set CG convergence tolerance.
    pub fn with_cg_tolerance(mut self, tol: f64) -> Self {
        self.cg_tolerance = tol;
        self
    }
}

/// Compute the cotangent of the angle at vertex `a` in triangle (a, b, c).
fn cotangent_angle(a: &Point3<f64>, b: &Point3<f64>, c: &Point3<f64>) -> f64 {
    let ab = b - a;
    let ac = c - a;

    let dot = ab.dot(&ac);
    let cross_len = ab.cross(&ac).norm();

    if cross_len < 1e-10 {
        0.0
    } else {
        dot / cross_len
    }
}

/// Compute mean edge length of the mesh.
fn compute_mean_edge_length<I: MeshIndex>(mesh: &HalfEdgeMesh<I>) -> f64 {
    let mut sum = 0.0;
    let mut count = 0;

    for v in mesh.vertex_ids() {
        for he in mesh.vertex_halfedges(v) {
            let dest = mesh.dest(he);
            // Only count each edge once (when origin < dest)
            if v.index() < dest.index() {
                sum += mesh.edge_length(he);
                count += 1;
            }
        }
    }

    if count > 0 {
        sum / count as f64
    } else {
        1.0
    }
}

/// Build the lumped mass matrix (diagonal, stored as vector).
/// M[i,i] = (1/3) * sum of areas of incident triangles
fn build_mass_matrix<I: MeshIndex>(mesh: &HalfEdgeMesh<I>) -> DVector<f64> {
    let n = mesh.num_vertices();
    let mut mass = DVector::zeros(n);

    // Mixed Voronoi area, not barycentric thirds. The mass matrix has to pair with
    // the operator it accompanies, and the cotangent Laplacian is derived from the
    // mixed-Voronoi dual cells. Using `area / 3` instead mismatches the two, which
    // biased recovered distances by several percent in a way that did not shrink
    // under refinement — the same inconsistency that broke OMT's mass lumping.
    for v in mesh.vertex_ids() {
        mass[v.index()] = crate::algo::curvature::mixed_voronoi_area(mesh, v);
    }

    mass
}

/// Build both the cotangent Laplacian and heat diffusion matrix.
/// Returns (laplacian, heat_matrix) where heat_matrix = M + t*L.
fn build_matrices<I: MeshIndex>(
    mesh: &HalfEdgeMesh<I>,
    mass: &DVector<f64>,
    t: f64,
) -> (CsrMatrix, CsrMatrix) {
    let n = mesh.num_vertices();
    let mut laplacian_triplets: Vec<(usize, usize, f64)> = Vec::new();
    let mut heat_triplets: Vec<(usize, usize, f64)> = Vec::new();

    // Add mass diagonal to heat matrix
    for i in 0..n {
        heat_triplets.push((i, i, mass[i]));
    }

    // Process each face for cotangent weights
    for f in mesh.face_ids() {
        let [v0, v1, v2] = mesh.face_triangle(f);
        let [p0, p1, p2] = mesh.face_positions(f);

        // Cotangent at each vertex (opposite to the edge across from it)
        // cot0 is at vertex 0, opposite to edge v1-v2
        let cot0 = cotangent_angle(&p0, &p1, &p2).max(1e-8);
        let cot1 = cotangent_angle(&p1, &p0, &p2).max(1e-8);
        let cot2 = cotangent_angle(&p2, &p0, &p1).max(1e-8);

        // Edge v0-v1: weight is 0.5 * cot2 (angle at v2)
        add_edge_to_matrices(
            &mut laplacian_triplets,
            &mut heat_triplets,
            v0.index(),
            v1.index(),
            0.5 * cot2,
            t,
        );

        // Edge v1-v2: weight is 0.5 * cot0 (angle at v0)
        add_edge_to_matrices(
            &mut laplacian_triplets,
            &mut heat_triplets,
            v1.index(),
            v2.index(),
            0.5 * cot0,
            t,
        );

        // Edge v0-v2: weight is 0.5 * cot1 (angle at v1)
        add_edge_to_matrices(
            &mut laplacian_triplets,
            &mut heat_triplets,
            v0.index(),
            v2.index(),
            0.5 * cot1,
            t,
        );
    }

    let laplacian = CsrMatrix::from_triplets(n, n, laplacian_triplets);
    let heat_matrix = CsrMatrix::from_triplets(n, n, heat_triplets);

    (laplacian, heat_matrix)
}

/// Add an edge contribution to both Laplacian and heat matrix triplets.
fn add_edge_to_matrices(
    laplacian: &mut Vec<(usize, usize, f64)>,
    heat: &mut Vec<(usize, usize, f64)>,
    i: usize,
    j: usize,
    w: f64,
    t: f64,
) {
    // Laplacian: off-diagonal -w, diagonal +w
    laplacian.push((i, j, -w));
    laplacian.push((j, i, -w));
    laplacian.push((i, i, w));
    laplacian.push((j, j, w));

    // Heat matrix: same but scaled by t
    heat.push((i, j, -t * w));
    heat.push((j, i, -t * w));
    heat.push((i, i, t * w));
    heat.push((j, j, t * w));
}

/// Solve the heat equation to get heat distribution from sources.
fn solve_heat<I: MeshIndex>(
    heat_matrix: &CsrMatrix,
    sources: &[VertexId<I>],
    n: usize,
    options: &HeatMethodOptions,
) -> Result<DVector<f64>> {
    // Build right-hand side: delta function at sources
    let mut rhs = DVector::zeros(n);
    for &src in sources {
        rhs[src.index()] = 1.0;
    }

    // Solve (M + t*L) * u = delta
    preconditioned_conjugate_gradient(
        heat_matrix,
        &rhs,
        None,
        options.max_cg_iterations,
        options.cg_tolerance,
    )
}

/// Compute the normalized negative gradient of the heat function on each face.
fn compute_normalized_gradient<I: MeshIndex>(
    mesh: &HalfEdgeMesh<I>,
    u: &DVector<f64>,
) -> Vec<Vector3<f64>> {
    let mut gradients = Vec::with_capacity(mesh.num_faces());

    for f in mesh.face_ids() {
        let [v0, v1, v2] = mesh.face_triangle(f);
        let [p0, p1, p2] = mesh.face_positions(f);

        let normal = mesh.face_normal(f);
        let area = mesh.face_area(f);

        // Edge vectors (opposite to each vertex)
        let e0 = p2 - p1; // opposite to v0
        let e1 = p0 - p2; // opposite to v1
        let e2 = p1 - p0; // opposite to v2

        // Heat values at vertices
        let u0 = u[v0.index()];
        let u1 = u[v1.index()];
        let u2 = u[v2.index()];

        // Gradient: (1/2A) * sum(u_i * (N x e_i))
        let grad = if area > 1e-10 {
            let factor = 1.0 / (2.0 * area);
            factor * (u0 * normal.cross(&e0) + u1 * normal.cross(&e1) + u2 * normal.cross(&e2))
        } else {
            Vector3::zeros()
        };

        // Normalize and negate: X = -grad / |grad|.
        //
        // The threshold has to be near the bottom of the f64 range, not a
        // "small number" like 1e-10. Heat decays as exp(-d^2 / 4t), and with the
        // default t = h^2 the far field is genuinely minute: on a 6401-vertex cap,
        // h is about 0.02, so at distance 0.5 the solution is around exp(-156),
        // i.e. 1e-68. That is a perfectly representable normal double whose
        // direction is exact, but a 1e-10 cutoff discarded it and set X to zero
        // across most of the mesh — flattening phi and underestimating distance by
        // 33% at that resolution, worse the finer the mesh, since refining shrinks
        // t and steepens the decay.
        let grad_norm = grad.norm();
        let normalized = if grad_norm > 1e-300 && grad_norm.is_finite() {
            -grad / grad_norm
        } else {
            Vector3::zeros()
        };

        gradients.push(normalized);
    }

    gradients
}

/// Mark the vertices reachable from any source by walking mesh edges.
///
/// Used to keep the heat method's output honest on a disconnected mesh: anything
/// outside a source's component has no path to it.
fn reachable_from_sources<I: MeshIndex>(
    mesh: &HalfEdgeMesh<I>,
    sources: &[VertexId<I>],
) -> Vec<bool> {
    let mut seen = vec![false; mesh.num_vertices()];
    let mut stack: Vec<VertexId<I>> = Vec::new();

    for &s in sources {
        if !seen[s.index()] {
            seen[s.index()] = true;
            stack.push(s);
        }
    }

    while let Some(v) = stack.pop() {
        for n in mesh.vertex_neighbors(v) {
            if !seen[n.index()] {
                seen[n.index()] = true;
                stack.push(n);
            }
        }
    }

    seen
}

/// Compute the integrated divergence of the vector field at each vertex.
fn compute_divergence<I: MeshIndex>(mesh: &HalfEdgeMesh<I>, x: &[Vector3<f64>]) -> DVector<f64> {
    let n = mesh.num_vertices();
    let mut div = DVector::zeros(n);

    for (fi, f) in mesh.face_ids().enumerate() {
        let [v0, v1, v2] = mesh.face_triangle(f);
        let [p0, p1, p2] = mesh.face_positions(f);

        let x_f = x[fi]; // Normalized gradient for this face

        // Edge vectors (from each vertex to next)
        let e01 = p1 - p0;
        let e12 = p2 - p1;
        let e20 = p0 - p2;

        // Cotangent weights at each vertex
        let cot0 = cotangent_angle(&p0, &p1, &p2);
        let cot1 = cotangent_angle(&p1, &p0, &p2);
        let cot2 = cotangent_angle(&p2, &p0, &p1);

        // Divergence contribution at each vertex
        // At vertex v0: uses edges to v1 and v2
        div[v0.index()] += 0.5 * (cot2 * e01.dot(&x_f) - cot1 * e20.dot(&x_f));

        // At vertex v1: uses edges to v2 and v0
        div[v1.index()] += 0.5 * (cot0 * e12.dot(&x_f) - cot2 * e01.dot(&x_f));

        // At vertex v2: uses edges to v0 and v1
        div[v2.index()] += 0.5 * (cot1 * e20.dot(&x_f) - cot0 * e12.dot(&x_f));
    }

    div
}

/// Compute geodesic distances using the heat method.
///
/// The heat method provides approximate geodesic distances that are smooth
/// and globally consistent. It's faster than exact methods for large meshes.
///
/// # Arguments
///
/// * `mesh` - The input triangle mesh
/// * `source` - The source vertex
/// * `options` - Algorithm options
///
/// # Returns
///
/// A `GeodesicResult` containing distances from the source to all vertices.
///
/// # Example
///
/// ```no_run
/// use morsel::prelude::*;
/// use morsel::algo::geodesic::{heat_method, HeatMethodOptions};
///
/// let mesh: HalfEdgeMesh = morsel::io::load("mesh.obj").unwrap();
/// let source = VertexId::new(0);
///
/// let result = heat_method(&mesh, source, &HeatMethodOptions::default()).unwrap();
/// println!("Distance to vertex 10: {}", result.distance(VertexId::new(10)));
/// ```
pub fn heat_method<I: MeshIndex>(
    mesh: &HalfEdgeMesh<I>,
    source: VertexId<I>,
    options: &HeatMethodOptions,
) -> Result<GeodesicResult<I>> {
    heat_method_multiple(mesh, &[source], options)
}

/// Compute geodesic distances from multiple source vertices using the heat method.
///
/// All source vertices are treated as having distance 0. This is useful for
/// computing distance fields from a set of points.
///
/// # Arguments
///
/// * `mesh` - The input triangle mesh
/// * `sources` - The source vertices
/// * `options` - Algorithm options
///
/// # Returns
///
/// A `GeodesicResult` containing distances from the nearest source to all vertices.
///
/// # Example
///
/// ```no_run
/// use morsel::prelude::*;
/// use morsel::algo::geodesic::{heat_method_multiple, HeatMethodOptions};
///
/// let mesh: HalfEdgeMesh = morsel::io::load("mesh.obj").unwrap();
/// let sources = vec![VertexId::new(0), VertexId::new(10), VertexId::new(20)];
///
/// let result = heat_method_multiple(&mesh, &sources, &HeatMethodOptions::default()).unwrap();
/// ```
pub fn heat_method_multiple<I: MeshIndex>(
    mesh: &HalfEdgeMesh<I>,
    sources: &[VertexId<I>],
    options: &HeatMethodOptions,
) -> Result<GeodesicResult<I>> {
    let n = mesh.num_vertices();

    if n == 0 || sources.is_empty() {
        return Ok(GeodesicResult::new(vec![f64::INFINITY; n], None));
    }

    // Step 1: Compute time step
    let h = compute_mean_edge_length(mesh);
    // 10 h², not h². See `HeatMethodOptions::time_step` for the measurements
    // behind the multiplier.
    let t = options.time_step.unwrap_or(10.0 * h * h);

    // Step 2: Build mass matrix and system matrices
    let mass = build_mass_matrix(mesh);
    let (laplacian, heat_matrix) = build_matrices(mesh, &mass, t);

    // Step 3: Solve heat equation
    let u = solve_heat(&heat_matrix, sources, n, options)?;

    // Step 4: Compute normalized gradient
    let x = compute_normalized_gradient(mesh, &u);

    // Step 5: Compute divergence
    let div = compute_divergence(mesh, &x);

    // Step 6: Solve Poisson equation L * phi = -div
    // Note: Our L has convention L[i,i] > 0, L[i,j] < 0, which corresponds to -Δ.
    // The heat method requires Δφ = div(X), so we solve Lφ = -div(X).
    let neg_div = -&div;

    // The cotangent Laplacian is singular: only differences of `phi` appear, so
    // constants lie in its kernel. Conjugate gradient cannot drive the relative
    // residual below that null-space contribution, so it reported
    // `ConvergenceFailed` on every mesh larger than a few vertices regardless of
    // the iteration budget — 100,000 iterations still failed on a 178-vertex
    // sphere.
    //
    // Pinning one vertex removes exactly that one-dimensional kernel, and it costs
    // nothing here because the gauge is irrelevant: step 7 below shifts the whole
    // field so its minimum is zero, discarding any constant offset anyway.
    let phi = {
        let reduction = PinnedReduction::new(&laplacian.triplets(), n, 0);
        if reduction.n_free() == 0 {
            DVector::zeros(n)
        } else {
            let reduced = reduction.reduce_rhs(&neg_div, 0.0);
            let solved = preconditioned_conjugate_gradient(
                &reduction.matrix,
                &reduced,
                None,
                options.max_cg_iterations,
                options.cg_tolerance,
            )?;
            let mut full = vec![0.0; n];
            reduction.scatter(&solved, &mut full);
            DVector::from_vec(full)
        }
    };

    // Step 7: Restrict to the sources' connected components, then shift so the
    // minimum is 0.
    //
    // Both halves matter on a disconnected mesh. A vertex in another component has
    // no path to any source, so its distance is infinite — the same answer Dijkstra
    // gives — but the Laplacian is block diagonal there and pinning one vertex only
    // makes one block definite, so `phi` in the other blocks is arbitrary. Leaving
    // it in reported those vertices as reachable at plausible-looking distances,
    // and, worse, let their arbitrary values set the minimum used to shift the
    // component that actually contains the source.
    let reachable = reachable_from_sources(mesh, sources);

    let min_phi = phi
        .iter()
        .enumerate()
        .filter(|(i, _)| reachable[*i])
        .map(|(_, &p)| p)
        .fold(f64::INFINITY, f64::min);

    let distances: Vec<f64> = phi
        .iter()
        .enumerate()
        .map(|(i, &p)| {
            if reachable[i] {
                p - min_phi
            } else {
                f64::INFINITY
            }
        })
        .collect();

    // No predecessors for heat method (not a graph algorithm)
    Ok(GeodesicResult::new(distances, None))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algo::geodesic::{dijkstra, DijkstraOptions};
    use crate::mesh::build_from_triangles;

    fn create_single_triangle() -> HalfEdgeMesh {
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.5, 1.0, 0.0),
        ];
        let faces = vec![[0, 1, 2]];
        build_from_triangles(&vertices, &faces).unwrap()
    }

    fn create_grid_mesh(n: usize) -> HalfEdgeMesh {
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

    #[test]
    fn test_heat_method_single_triangle() {
        let mesh = create_single_triangle();
        let result = heat_method(&mesh, VertexId::new(0), &HeatMethodOptions::default());

        assert!(result.is_ok());
        let result = result.unwrap();

        assert_eq!(result.len(), 3);

        // Distance to self should be 0 (or very close)
        assert!(result.distance(VertexId::new(0)) < 1e-6);

        // Other vertices should have positive distances
        assert!(result.distance(VertexId::new(1)) > 0.0);
        assert!(result.distance(VertexId::new(2)) > 0.0);
    }

    #[test]
    fn test_heat_method_grid() {
        let mesh = create_grid_mesh(3);
        let result = heat_method(&mesh, VertexId::new(0), &HeatMethodOptions::default()).unwrap();

        // All vertices should be reachable
        assert_eq!(result.reachable_count(), 16);

        // Distance should increase along diagonal
        // Vertex 0 is at (0,0), vertex 15 is at (3,3)
        let d0 = result.distance(VertexId::new(0));
        let d5 = result.distance(VertexId::new(5)); // (1,1)
        let d10 = result.distance(VertexId::new(10)); // (2,2)
        let d15 = result.distance(VertexId::new(15)); // (3,3)

        assert!(d0 < d5);
        assert!(d5 < d10);
        assert!(d10 < d15);
    }

    #[test]
    fn test_heat_method_multiple_sources() {
        let mesh = create_grid_mesh(2);

        // Sources at opposite corners
        let sources = vec![VertexId::new(0), VertexId::new(8)];
        let result = heat_method_multiple(&mesh, &sources, &HeatMethodOptions::default()).unwrap();

        // Both sources should have distance ~0
        assert!(result.distance(VertexId::new(0)) < 1e-6);
        assert!(result.distance(VertexId::new(8)) < 1e-6);

        // Center vertex should have positive distance
        assert!(result.distance(VertexId::new(4)) > 0.0);
    }

    #[test]
    fn test_heat_method_vs_dijkstra_ordering() {
        // Distances should have same ordering as Dijkstra (monotonicity check)
        let mesh = create_grid_mesh(3);
        let source = VertexId::new(0);

        let heat_result = heat_method(&mesh, source, &HeatMethodOptions::default()).unwrap();
        let dijkstra_result = dijkstra(&mesh, source, &DijkstraOptions::default());

        // For vertices at different distances, ordering should match
        // Check a few specific pairs
        let pairs = [(0, 5), (5, 10), (10, 15), (0, 15)];
        for (a, b) in pairs {
            let dij_a = dijkstra_result.distance(VertexId::new(a));
            let dij_b = dijkstra_result.distance(VertexId::new(b));
            let heat_a = heat_result.distance(VertexId::new(a));
            let heat_b = heat_result.distance(VertexId::new(b));

            // If Dijkstra says a < b, heat should agree
            if (dij_a - dij_b).abs() > 0.1 {
                assert_eq!(
                    dij_a < dij_b,
                    heat_a < heat_b,
                    "Distance ordering mismatch for vertices {} and {}",
                    a,
                    b
                );
            }
        }
    }

    #[test]
    fn test_heat_method_custom_time_step() {
        let mesh = create_grid_mesh(2);
        let source = VertexId::new(0);

        let options_default = HeatMethodOptions::default();
        // Use a larger time step that's still different from default
        let options_custom = HeatMethodOptions::default().with_time_step(2.0);

        let result_default = heat_method(&mesh, source, &options_default).unwrap();
        let result_custom = heat_method(&mesh, source, &options_custom).unwrap();

        // Both should produce valid results (though may differ slightly)
        assert_eq!(result_default.reachable_count(), 9);
        assert_eq!(result_custom.reachable_count(), 9);

        // Source should still have distance 0 for both
        assert!(result_default.distance(VertexId::new(0)) < 1e-6);
        assert!(result_custom.distance(VertexId::new(0)) < 1e-6);
    }

    #[test]
    fn test_heat_method_empty_sources() {
        let mesh = create_grid_mesh(2);
        let result = heat_method_multiple(&mesh, &[], &HeatMethodOptions::default()).unwrap();

        // All vertices should be unreachable
        assert_eq!(result.reachable_count(), 0);
    }

    #[test]
    fn test_heat_method_symmetry() {
        // On a symmetric mesh, distances should be symmetric
        let mesh = create_grid_mesh(2);

        let result_0 = heat_method(&mesh, VertexId::new(0), &HeatMethodOptions::default()).unwrap();
        let result_8 = heat_method(&mesh, VertexId::new(8), &HeatMethodOptions::default()).unwrap();

        // d(0, 8) should equal d(8, 0) approximately
        let d_0_to_8 = result_0.distance(VertexId::new(8));
        let d_8_to_0 = result_8.distance(VertexId::new(0));

        let tolerance = 0.1 * d_0_to_8.max(d_8_to_0);
        assert!(
            (d_0_to_8 - d_8_to_0).abs() < tolerance,
            "Asymmetric distances: {} vs {}",
            d_0_to_8,
            d_8_to_0
        );
    }
}
