//! Discrete curvature against closed-form answers.
//!
//! The curvature module had unit tests but nothing comparing it to a value known
//! independently, so "is it right" was unanswerable. These meshes were built with
//! that in mind: each has analytic Gaussian and mean curvature, so the estimators
//! can be checked rather than merely exercised.
//!
//! Two conventions to note, both established by measurement rather than assumed:
//!
//! - `gaussian_curvature` and `mean_curvature` return **pointwise** curvature, not
//!   an integrated angle defect. Summing the raw values is meaningless; a
//!   Gauss-Bonnet check would need to weight each by its dual area.
//! - Mean curvature comes back **negative** where the outward-normal convention
//!   would give positive. The sign is consistent across meshes, so it is a
//!   convention rather than a defect, and these tests compare magnitudes.
//!
//! Boundary vertices are excluded throughout: the angle-defect estimator assumes a
//! full ring, and a boundary vertex does not have one.

use morsel::algo::curvature::{gaussian_curvature, mean_curvature};
use morsel::mesh::HalfEdgeMesh;

fn load(path: &str) -> HalfEdgeMesh {
    morsel::io::load(path).unwrap_or_else(|e| panic!("{path}: {e}"))
}

/// Values at interior vertices only, as `(mean, min, max)`.
fn interior_stats(mesh: &HalfEdgeMesh, values: &[f64]) -> (f64, f64, f64) {
    let picked: Vec<f64> = mesh
        .vertex_ids()
        .filter(|&v| !mesh.is_boundary_vertex(v))
        .map(|v| values[v.index()])
        .collect();
    assert!(!picked.is_empty(), "mesh has no interior vertices");
    let mean = picked.iter().sum::<f64>() / picked.len() as f64;
    (
        mean,
        picked.iter().cloned().fold(f64::INFINITY, f64::min),
        picked.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    )
}

fn total_area(mesh: &HalfEdgeMesh) -> f64 {
    mesh.face_ids().map(|f| mesh.face_area(f)).sum()
}

/// A sphere of radius `r` has `K = 1/r²` and `|H| = 1/r` everywhere.
/// `sphere.obj` has diameter 1, so `r = 0.5`.
#[test]
fn sphere_curvature_matches_analytic() {
    let mesh = load("examples/sphere.obj");
    let r = 0.5;
    let (k_mean, k_lo, k_hi) = interior_stats(&mesh, &gaussian_curvature(&mesh));
    let (h_mean, _, _) = interior_stats(&mesh, &mean_curvature(&mesh));

    let k_expected = 1.0 / (r * r);
    let h_expected = 1.0 / r;
    println!("sphere: K {k_mean:.4} in [{k_lo:.4}, {k_hi:.4}] vs {k_expected}, |H| {:.4} vs {h_expected}", h_mean.abs());

    // A 352-face sphere is a coarse approximation, so a few percent is expected.
    assert!(
        (k_mean - k_expected).abs() / k_expected < 0.05,
        "Gaussian curvature {k_mean} should be near {k_expected}"
    );
    assert!(
        (h_mean.abs() - h_expected).abs() / h_expected < 0.05,
        "mean curvature magnitude {} should be near {h_expected}",
        h_mean.abs()
    );
    // Polyhedral approximation inscribes the sphere, so area comes in slightly under.
    let area_expected = 4.0 * std::f64::consts::PI * r * r;
    let area = total_area(&mesh);
    assert!(
        area <= area_expected && area / area_expected > 0.95,
        "area {area} should approach {area_expected} from below"
    );
}

/// A cap cut from the unit sphere: `K = 1`, `|H| = 1`, area `2πRh`.
#[test]
fn spherical_cap_curvature_matches_analytic() {
    let mesh = load("examples/spherical-cap.obj");
    let (k_mean, k_lo, k_hi) = interior_stats(&mesh, &gaussian_curvature(&mesh));
    let (h_mean, _, _) = interior_stats(&mesh, &mean_curvature(&mesh));
    println!(
        "cap: K {k_mean:.4} in [{k_lo:.4}, {k_hi:.4}] vs 1, |H| {:.4} vs 1",
        h_mean.abs()
    );

    assert!((k_mean - 1.0).abs() < 0.02, "K {k_mean} should be near 1");
    assert!(
        (h_mean.abs() - 1.0).abs() < 0.02,
        "|H| {} should be near 1",
        h_mean.abs()
    );
    // Constant curvature, so the spread across the interior should be tight.
    assert!(
        k_hi - k_lo < 0.02,
        "K should be nearly constant, spread {}",
        k_hi - k_lo
    );

    let h = 1.0 - (1.0f64 - 0.8 * 0.8).sqrt();
    let area_expected = 2.0 * std::f64::consts::PI * 1.0 * h;
    let area = total_area(&mesh);
    assert!(
        (area - area_expected).abs() / area_expected < 0.02,
        "area {area} should be near {area_expected}"
    );
}

/// A cylinder is developable: `K = 0` exactly, not merely nearly. This is the
/// sharpest of these tests, since the expected answer has no discretization error
/// — the angle defect at a cylinder vertex is exactly zero.
#[test]
fn cylinder_is_developable() {
    let mesh = load("examples/cylinder.obj");
    let (k_mean, k_lo, k_hi) = interior_stats(&mesh, &gaussian_curvature(&mesh));
    let (h_mean, _, _) = interior_stats(&mesh, &mean_curvature(&mesh));
    println!(
        "cylinder: K {k_mean:.2e} in [{k_lo:.2e}, {k_hi:.2e}] vs 0, |H| {:.4} vs 0.5",
        h_mean.abs()
    );

    assert!(
        k_lo.abs() < 1e-12 && k_hi.abs() < 1e-12,
        "a developable surface must have K = 0 exactly, got [{k_lo:e}, {k_hi:e}]"
    );
    // Radius 1, so the non-zero principal curvature is 1 and |H| = 1/2.
    assert!(
        (h_mean.abs() - 0.5).abs() < 0.02,
        "|H| {} should be near 0.5",
        h_mean.abs()
    );

    let area_expected = 2.0 * std::f64::consts::PI * 1.0 * 2.0;
    let area = total_area(&mesh);
    assert!((area - area_expected).abs() / area_expected < 0.02);
}

/// The torus has curvature of both signs, so it checks the estimator's range
/// rather than a single value: `K(v) = cos v / (r (R + r cos v))`.
#[test]
fn torus_curvature_spans_both_signs() {
    let mesh = load("examples/torus.obj");
    let (big_r, little_r) = (1.0, 0.35);
    let k_outer = 1.0 / (little_r * (big_r + little_r)); //  +2.1164
    let k_inner = -1.0 / (little_r * (big_r - little_r)); // -4.3956

    let (_, k_lo, k_hi) = interior_stats(&mesh, &gaussian_curvature(&mesh));
    println!("torus: K in [{k_lo:.4}, {k_hi:.4}] vs [{k_inner:.4}, {k_outer:.4}]");

    assert!(
        (k_hi - k_outer).abs() / k_outer.abs() < 0.05,
        "max K {k_hi} should approach the outer-equator value {k_outer}"
    );
    assert!(
        (k_lo - k_inner).abs() / k_inner.abs() < 0.05,
        "min K {k_lo} should approach the inner-equator value {k_inner}"
    );

    let area_expected = 4.0 * std::f64::consts::PI.powi(2) * big_r * little_r;
    let area = total_area(&mesh);
    assert!(
        area <= area_expected && area / area_expected > 0.98,
        "area {area} should approach {area_expected} from below"
    );
}
