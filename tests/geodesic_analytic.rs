//! Geodesic distance against closed-form answers.
//!
//! The heat method (596 lines) and Dijkstra (460) had unit tests but nothing
//! comparing them to a distance known independently, and neither was reachable
//! from the CLI. Checking them against exact geodesics turned up four defects,
//! recorded here as regression guards.
//!
//! Where the exact answers come from:
//!
//! - **Cylinder** — developable, so it unrolls isometrically to a flat strip and a
//!   geodesic is a straight line there: `d = √((rΔθ)² + Δz²)` taking the short way
//!   round. Exact, with no discretization error in the *truth*.
//! - **Spherical cap** — spans 53° of polar angle, so it is geodesically convex and
//!   great-circle distance is the exact in-cap geodesic.
//! - **Sphere** — great-circle distance, `r·arccos(p̂·q̂)`.
//!
//! One subtlety that cost an hour: with the source at the centre of a rotationally
//! symmetric mesh, `u` is radially symmetric, so `∇u` points radially by symmetry
//! and normalizing discards its magnitude. The result is then provably independent
//! of the time step *and* the mass matrix, so such a configuration cannot detect an
//! error in either. Tests that need to see those must place the source off-centre.

use morsel::algo::geodesic::{dijkstra, heat_method, DijkstraOptions, HeatMethodOptions};
use morsel::mesh::{build_from_triangles, HalfEdgeMesh, VertexId};
use nalgebra::Point3;

fn load(path: &str) -> HalfEdgeMesh {
    morsel::io::load(path).unwrap_or_else(|e| panic!("{path}: {e}"))
}

/// Mean and extreme relative error against `truth`, over vertices other than the
/// source.
fn rel_error(got: &[f64], truth: &[f64], src: usize) -> (f64, f64, f64) {
    let mut errs = Vec::new();
    for (i, &t) in truth.iter().enumerate() {
        if i == src || t <= 1e-12 || !got[i].is_finite() {
            continue;
        }
        errs.push((got[i] - t) / t);
    }
    assert!(!errs.is_empty());
    (
        errs.iter().sum::<f64>() / errs.len() as f64,
        errs.iter().cloned().fold(f64::INFINITY, f64::min),
        errs.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    )
}

/// Polar-triangulated cap of the unit sphere, cut at radius `rmax`.
fn cap(nr: usize, na: usize, rmax: f64) -> HalfEdgeMesh {
    let mut v = vec![Point3::new(0.0, 0.0, 1.0)];
    for k in 1..=nr {
        let rad = rmax * k as f64 / nr as f64;
        let z = (1.0 - rad * rad).max(0.0).sqrt();
        for a in 0..na {
            let t = 2.0 * std::f64::consts::PI * a as f64 / na as f64;
            v.push(Point3::new(rad * t.cos(), rad * t.sin(), z));
        }
    }
    let ring = |k: usize, a: usize| 1 + (k - 1) * na + (a % na);
    let mut f = Vec::new();
    for a in 0..na {
        f.push([0, ring(1, a), ring(1, a + 1)]);
    }
    for k in 1..nr {
        for a in 0..na {
            f.push([ring(k, a), ring(k + 1, a), ring(k + 1, a + 1)]);
            f.push([ring(k, a), ring(k + 1, a + 1), ring(k, a + 1)]);
        }
    }
    build_from_triangles(&v, &f).unwrap()
}

/// Exact great-circle distances on the unit sphere from `src`.
fn great_circle_truth(mesh: &HalfEdgeMesh, src: VertexId, radius: f64) -> Vec<f64> {
    let s = mesh.position(src).coords.normalize();
    mesh.vertex_ids()
        .map(|v| {
            radius
                * s.dot(&mesh.position(v).coords.normalize())
                    .clamp(-1.0, 1.0)
                    .acos()
        })
        .collect()
}

/// The heat method must *return a result* on every asset.
///
/// It used to fail on all of them. The Poisson stage solved against the raw
/// cotangent Laplacian, which is singular — constants lie in its kernel — so CG
/// could never reduce the relative residual below the null-space contribution and
/// reported `ConvergenceFailed` at any iteration budget. 100,000 iterations still
/// failed on a 178-vertex sphere. Only an 8-vertex cube got through.
#[test]
fn heat_method_converges_on_every_example() {
    for path in [
        "examples/cube-closed.obj",
        "examples/sphere.obj",
        "examples/torus.obj",
        "examples/spherical-cap.obj",
        "examples/cylinder.obj",
        "examples/stanford-bunny.obj",
    ] {
        let mesh = load(path);
        let result = heat_method(&mesh, VertexId::new(0), &HeatMethodOptions::default());
        assert!(
            result.is_ok(),
            "{path}: heat method should converge, got {:?}",
            result.err()
        );
        let r = result.unwrap();
        assert!(
            r.distances().iter().all(|d| d.is_finite()),
            "{path}: distances should all be finite"
        );
    }
}

/// The cylinder is the sharpest case: developable, so the exact answer carries no
/// discretization error, and the source is off-axis so the field is not radially
/// symmetric.
#[test]
fn heat_method_is_accurate_on_the_cylinder() {
    let mesh = load("examples/cylinder.obj");
    let src = VertexId::new(0);
    let p0 = *mesh.position(src);
    let (t0, z0) = (p0.y.atan2(p0.x), p0.z);

    // Unroll: radius 1, so arc length equals angle. Take the short way round.
    let truth: Vec<f64> = mesh
        .vertex_ids()
        .map(|v| {
            let p = mesh.position(v);
            let mut dt = p.y.atan2(p.x) - t0;
            while dt > std::f64::consts::PI {
                dt -= 2.0 * std::f64::consts::PI;
            }
            while dt < -std::f64::consts::PI {
                dt += 2.0 * std::f64::consts::PI;
            }
            (dt * dt + (p.z - z0).powi(2)).sqrt()
        })
        .collect();

    let heat = heat_method(&mesh, src, &HeatMethodOptions::default()).unwrap();
    let (mean, lo, hi) = rel_error(heat.distances(), &truth, src.index());
    println!("cylinder heat: mean {mean:+.4}, range [{lo:+.4}, {hi:+.4}]");

    assert!(
        mean.abs() < 0.02,
        "mean relative error {mean:+.4} should be within 2%"
    );
}

/// Dijkstra walks along edges, so it can only *over*estimate a geodesic that cuts
/// across faces. On the cylinder, where geodesics are helices not aligned with any
/// edge, it is out by up to 41% — which is the case the heat method exists to fix.
#[test]
fn dijkstra_overestimates_where_edges_do_not_follow_geodesics() {
    let mesh = load("examples/cylinder.obj");
    let src = VertexId::new(0);
    let p0 = *mesh.position(src);
    let (t0, z0) = (p0.y.atan2(p0.x), p0.z);
    let truth: Vec<f64> = mesh
        .vertex_ids()
        .map(|v| {
            let p = mesh.position(v);
            let mut dt = p.y.atan2(p.x) - t0;
            while dt > std::f64::consts::PI {
                dt -= 2.0 * std::f64::consts::PI;
            }
            while dt < -std::f64::consts::PI {
                dt += 2.0 * std::f64::consts::PI;
            }
            (dt * dt + (p.z - z0).powi(2)).sqrt()
        })
        .collect();

    let d = dijkstra(&mesh, src, &DijkstraOptions::default());
    let (mean, lo, hi) = rel_error(d.distances(), &truth, src.index());
    println!("cylinder dijkstra: mean {mean:+.4}, range [{lo:+.4}, {hi:+.4}]");

    // Graph distance is never shorter than the true geodesic, up to the tolerance
    // of the discretization.
    assert!(
        lo > -0.01,
        "graph distance should not undercut the geodesic, got {lo:+.4}"
    );
    assert!(
        mean > 0.05,
        "expected a substantial overestimate on this mesh, got {mean:+.4}"
    );

    let heat = heat_method(&mesh, src, &HeatMethodOptions::default()).unwrap();
    let (heat_mean, _, _) = rel_error(heat.distances(), &truth, src.index());
    assert!(
        heat_mean.abs() < mean.abs() / 5.0,
        "the heat method should beat graph distance by a wide margin here: \
         {heat_mean:+.4} vs {mean:+.4}"
    );
}

/// Error must *shrink* under refinement. It did not: with the old default time step
/// of `h²` it grew from −7% at 101 vertices to −27% at 6401, because heat decays as
/// `exp(-d²/4t)` and a small `t` pushes the far field below what the linear solve
/// can resolve. The default is now `10 h²`.
#[test]
fn heat_method_converges_under_refinement() {
    let mut previous: Option<f64> = None;
    println!("off-centre source on a spherical cap:");

    for nr in [5usize, 10, 20, 40] {
        let mesh = cap(nr, 4 * nr, 0.8);
        // Off-centre deliberately: a source at the apex makes the field radially
        // symmetric, which hides any time-step or mass-matrix error.
        let src = mesh
            .vertex_ids()
            .min_by(|&a, &b| mesh.position(a).z.partial_cmp(&mesh.position(b).z).unwrap())
            .unwrap();
        let truth = great_circle_truth(&mesh, src, 1.0);

        let heat = heat_method(&mesh, src, &HeatMethodOptions::default()).unwrap();
        let (mean, _, _) = rel_error(heat.distances(), &truth, src.index());
        let err = mean.abs();
        println!(
            "  {:>5} verts: |mean rel err| {err:.4}",
            mesh.num_vertices()
        );

        if let Some(prev) = previous {
            assert!(
                err < prev * 1.2,
                "error should not grow with refinement: {prev:.4} -> {err:.4}"
            );
        }
        previous = Some(err);
    }

    let finest = previous.unwrap();
    assert!(
        finest < 0.01,
        "the finest mesh should be within 1%, got {finest:.4}"
    );
}

/// A rotationally symmetric configuration retains a few percent of bias that does
/// not shrink under refinement, unlike the asymmetric case above. Recorded rather
/// than fixed: with the source at the centre the normalized gradient field is
/// pinned by symmetry, so this residual comes from the divergence and Poisson
/// stages and is a separate defect.
#[test]
fn symmetric_source_retains_a_known_bias() {
    let mesh = load("examples/spherical-cap.obj");
    let apex = mesh
        .vertex_ids()
        .max_by(|&a, &b| mesh.position(a).z.partial_cmp(&mesh.position(b).z).unwrap())
        .unwrap();
    let truth: Vec<f64> = mesh
        .vertex_ids()
        .map(|v| mesh.position(v).z.clamp(-1.0, 1.0).acos())
        .collect();

    let heat = heat_method(&mesh, apex, &HeatMethodOptions::default()).unwrap();
    let (mean, lo, hi) = rel_error(heat.distances(), &truth, apex.index());
    println!("cap, apex source: mean {mean:+.4}, range [{lo:+.4}, {hi:+.4}]");

    // Underestimates by about 6%. If this ever tightens substantially the residual
    // has been found — update the bound and say so.
    assert!(
        mean < 0.0 && mean.abs() < 0.10,
        "expected a modest underestimate, got {mean:+.4}"
    );
}
