//! Smoothing and subdivision against properties that can be checked.
//!
//! Two of these test claims the code makes about itself. `taubin_smooth` is
//! described as shrinkage-resistant, and `mean_curvature_flow` cites Desbrun et
//! al.; both assertions were previously unverified. Loop and Catmull-Clark are
//! checked on invariants instead: subdivision must not change topology.
//!
//! Where an exact answer exists it is used. Mean curvature flow of a sphere has a
//! closed-form solution — the radius obeys `R² = R₀² − c·t` — so the flow law can
//! be checked directly rather than by eyeballing that the mesh got rounder.

use morsel::algo::smooth::{
    cotangent_smooth, laplacian_smooth, mean_curvature_flow, taubin_smooth, CurvatureFlowOptions,
    SmoothOptions,
};
use morsel::algo::subdivide::{catmull_clark_subdivide, loop_subdivide, SubdivideOptions};
use morsel::mesh::HalfEdgeMesh;

fn load(path: &str) -> HalfEdgeMesh {
    morsel::io::load(path).unwrap_or_else(|e| panic!("{path}: {e}"))
}

/// Signed volume by the divergence theorem. Positive for outward winding.
fn volume(mesh: &HalfEdgeMesh) -> f64 {
    mesh.face_ids()
        .map(|f| {
            let [a, b, c] = mesh.face_positions(f);
            a.coords.cross(&b.coords).dot(&c.coords)
        })
        .sum::<f64>()
        / 6.0
}

fn total_area(mesh: &HalfEdgeMesh) -> f64 {
    mesh.face_ids().map(|f| mesh.face_area(f)).sum()
}

fn euler_characteristic(mesh: &HalfEdgeMesh) -> isize {
    mesh.num_vertices() as isize - (mesh.num_halfedges() / 2) as isize + mesh.num_faces() as isize
}

fn mean_radius(mesh: &HalfEdgeMesh) -> f64 {
    let sum: f64 = mesh
        .vertex_ids()
        .map(|v| mesh.position(v).coords.norm())
        .sum();
    sum / mesh.num_vertices() as f64
}

fn options(iterations: usize) -> SmoothOptions {
    SmoothOptions {
        iterations,
        lambda: 0.5,
        preserve_boundary: true,
        parallel: true,
    }
}

/// `taubin_smooth` is documented as shrinkage-resistant. Test that against
/// `laplacian_smooth`, which is not: on a closed surface, Laplacian smoothing
/// contracts toward the centroid, while Taubin's second pass pushes back out.
#[test]
fn taubin_resists_shrinkage_where_laplacian_does_not() {
    // Meshes big enough to have something to smooth. An 8-vertex cube collapses
    // under either scheme, which says nothing about shrinkage resistance.
    for name in ["sphere.obj", "torus.obj"] {
        let base = load(&format!("examples/{name}"));
        let v0 = volume(&base);
        assert!(
            v0 > 0.0,
            "{name}: expected outward winding, got volume {v0}"
        );

        let mut lap = base.clone();
        laplacian_smooth(&mut lap, &options(20));
        let lap_change = (volume(&lap) - v0) / v0;

        let mut tau = base.clone();
        taubin_smooth(&mut tau, &options(20));
        let tau_change = (volume(&tau) - v0) / v0;

        println!("{name}: laplacian {lap_change:+.3}, taubin {tau_change:+.3}");

        assert!(
            lap_change < -0.20,
            "{name}: Laplacian smoothing should visibly shrink, got {lap_change:+.3}"
        );
        assert!(
            tau_change.abs() < 0.10,
            "{name}: Taubin is documented as shrinkage-resistant, got {tau_change:+.3}"
        );
        assert!(
            tau_change.abs() < lap_change.abs() / 4.0,
            "{name}: Taubin should preserve volume far better than Laplacian: \
             {tau_change:+.3} vs {lap_change:+.3}"
        );
    }
}

/// Cotangent-weighted smoothing shrinks too — it is a Laplacian scheme with better
/// weights, not a volume-preserving one. Recorded so the distinction from Taubin
/// stays explicit.
#[test]
fn cotangent_smoothing_also_shrinks() {
    let base = load("examples/sphere.obj");
    let v0 = volume(&base);
    let mut m = base.clone();
    cotangent_smooth(&mut m, &options(20));
    let change = (volume(&m) - v0) / v0;
    println!("sphere cotangent: {change:+.3}");
    assert!(
        change < -0.20,
        "cotangent smoothing is not shrinkage-resistant either, got {change:+.3}"
    );
}

/// Mean curvature flow of a sphere has an exact solution: the radius satisfies
/// `R² = R₀² − c·t`, so `R²` falls *linearly* in time. That is the property to
/// check — it holds regardless of which curvature convention sets `c`.
///
/// Measured `c ≈ 2.00`, i.e. `dR/dt = −1/R`, so this implementation flows by the
/// *mean* curvature `1/R`. Note the geometry literature often writes mean curvature
/// flow with `H = κ₁ + κ₂`, which would give `c = 4`; halve or double `time_step`
/// accordingly when comparing against a published rate.
#[test]
fn mean_curvature_flow_follows_the_analytic_law() {
    let base = load("examples/sphere.obj");
    let r0 = mean_radius(&base);
    let dt = 0.001;

    let mut slopes = Vec::new();
    for iterations in [5usize, 10, 20, 40] {
        let mut m = base.clone();
        mean_curvature_flow(
            &mut m,
            &CurvatureFlowOptions {
                iterations,
                time_step: dt,
                preserve_boundary: true,
                implicit: false,
                parallel: true,
            },
        );
        let r = mean_radius(&m);
        let t = iterations as f64 * dt;
        let slope = (r0 * r0 - r * r) / t;
        println!("  {iterations:>3} iters: R {r:.6}, (R0^2 - R^2)/t = {slope:.4}");
        slopes.push(slope);
    }

    // Linear in t means the slope is the same at every horizon.
    let first = slopes[0];
    for &s in &slopes {
        assert!(
            (s - first).abs() / first < 0.02,
            "R^2 should fall linearly in t; slopes vary: {slopes:?}"
        );
    }
    // And the constant identifies the convention.
    assert!(
        (first - 2.0).abs() < 0.05,
        "expected dR/dt = -1/R, i.e. slope 2, got {first:.4}"
    );
}

/// Subdivision refines geometry; it must not change topology. Loop also quadruples
/// the face count exactly, and its limit surface lies inside the control mesh, so
/// area and volume decrease toward a limit rather than wandering.
#[test]
fn loop_subdivision_preserves_topology_and_converges() {
    for name in ["cube-closed.obj", "sphere.obj"] {
        let mut mesh = load(&format!("examples/{name}"));
        let chi0 = euler_characteristic(&mesh);
        let mut faces = mesh.num_faces();
        let mut areas = vec![total_area(&mesh)];
        let mut volumes = vec![volume(&mesh)];

        for step in 1..=3 {
            loop_subdivide(&mut mesh, &SubdivideOptions::new(1));

            assert_eq!(
                euler_characteristic(&mesh),
                chi0,
                "{name} step {step}: subdivision must not change the Euler characteristic"
            );
            assert_eq!(
                mesh.num_faces(),
                faces * 4,
                "{name} step {step}: Loop should split each triangle into four"
            );
            assert!(
                mesh.is_triangle_mesh(),
                "{name} step {step}: output should still be triangles"
            );
            assert!(
                volume(&mesh) > 0.0,
                "{name} step {step}: winding should stay outward"
            );

            faces = mesh.num_faces();
            areas.push(total_area(&mesh));
            volumes.push(volume(&mesh));
        }

        println!("{name}: areas {areas:.4?}");
        println!("{name}: volumes {volumes:.4?}");

        // Approximating scheme, so it contracts toward the limit surface — but the
        // steps must shrink, or it is not converging to anything.
        for series in [&areas, &volumes] {
            let d1 = (series[1] - series[0]).abs();
            let d2 = (series[2] - series[1]).abs();
            let d3 = (series[3] - series[2]).abs();
            assert!(
                d2 < d1 && d3 < d2,
                "{name}: successive changes should shrink, got {d1:.5}, {d2:.5}, {d3:.5}"
            );
        }
    }
}

/// Catmull-Clark is a quad scheme. Handed a triangle mesh it used to index past the
/// end of its per-face data and panic with the invalid-index sentinel; it now leaves
/// the mesh alone, since its signature gives it no way to report the refusal.
#[test]
fn catmull_clark_requires_a_quad_mesh() {
    let base = load("examples/cube-closed.obj");
    assert!(!base.is_quad_mesh(), "fixture should be triangles");

    let mut mesh = base.clone();
    catmull_clark_subdivide(&mut mesh, &SubdivideOptions::new(1));

    assert_eq!(
        mesh.num_vertices(),
        base.num_vertices(),
        "a triangle mesh should come back untouched, not panicked on"
    );
    assert_eq!(mesh.num_faces(), base.num_faces());
}
