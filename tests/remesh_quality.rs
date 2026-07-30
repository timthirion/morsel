//! What the remeshers actually do to triangle quality.
//!
//! Remeshing exists to improve triangle quality, and until there was a metric that
//! claim was untested — the only coverage was "did it return a structurally valid
//! mesh". These tests measure it. Some of them record *failures*: where a remesher
//! makes quality worse, that is pinned here so a fix shows up as a deliberate change
//! rather than passing unnoticed.
//!
//! Quality alone is not enough, either. A remesher can score well by discarding the
//! shape, so the last test measures drift off a sphere of known radius.

use morsel::algo::quality::{mesh_quality, QualityReport};
use morsel::algo::remesh::{
    anisotropic_remesh, average_edge_length, cvt_remesh, isotropic_remesh, AnisotropicOptions,
    CvtOptions, RemeshOptions,
};
use morsel::mesh::HalfEdgeMesh;

fn load(name: &str) -> HalfEdgeMesh {
    morsel::io::load(format!("examples/{name}.obj")).expect("example mesh loads")
}

fn quality(mesh: &HalfEdgeMesh) -> QualityReport {
    mesh_quality(mesh).expect("mesh has faces")
}

/// Isotropic remeshing on the meshes that have something to improve. Every aggregate
/// measure must move the right way: rounder triangles on average, and edge lengths
/// closer to uniform, which is the whole point of *isotropic*.
#[test]
fn isotropic_remeshing_improves_aggregate_quality() {
    for name in ["sphere", "spherical-cap", "torus", "stanford-bunny"] {
        let mesh = load(name);
        let before = quality(&mesh);

        let mut after_mesh = mesh.clone();
        let target = average_edge_length(&mesh);
        let _ = isotropic_remesh(&mut after_mesh, &RemeshOptions::with_target_length(target));
        let after = quality(&after_mesh);

        assert!(
            after.mean_min_angle_deg > before.mean_min_angle_deg + 1.0,
            "{name}: mean minimum angle {:.2}° -> {:.2}°, expected a clear improvement",
            before.mean_min_angle_deg,
            after.mean_min_angle_deg
        );
        assert!(
            after.mean_radius_ratio > before.mean_radius_ratio,
            "{name}: mean radius ratio {:.4} -> {:.4}",
            before.mean_radius_ratio,
            after.mean_radius_ratio
        );
        assert!(
            after.edge_length_cv < before.edge_length_cv,
            "{name}: edge lengths got less uniform, cv {:.4} -> {:.4}",
            before.edge_length_cv,
            after.edge_length_cv
        );
    }
}

/// On three of those four the worst triangle improves too. The bunny is the exception
/// and gets its own test.
#[test]
fn isotropic_remeshing_improves_the_worst_triangle_on_smooth_meshes() {
    for name in ["sphere", "spherical-cap", "torus"] {
        let mesh = load(name);
        let before = quality(&mesh);

        let mut after_mesh = mesh.clone();
        let target = average_edge_length(&mesh);
        let _ = isotropic_remesh(&mut after_mesh, &RemeshOptions::with_target_length(target));
        let after = quality(&after_mesh);

        assert!(
            after.min_angle_deg > before.min_angle_deg,
            "{name}: worst angle {:.2}° -> {:.2}°",
            before.min_angle_deg,
            after.min_angle_deg
        );
    }
}

/// **Recorded defect.** On the bunny, isotropic remeshing improves every aggregate
/// measure dramatically — mean minimum angle 35.9° to 51.4°, mean radius ratio 0.74 to
/// 0.95 — while driving the *worst* triangle to 5.5e-8 degrees: a face with no usable
/// area at all.
///
/// This is the case for reporting worst and mean side by side. Judged on the mean
/// alone this looks like the remesher's best result of the five, and a degenerate face
/// will break anything downstream that divides by an area — the cotangent Laplacian,
/// for one.
#[test]
fn isotropic_remeshing_emits_a_degenerate_face_on_the_bunny() {
    let mesh = load("stanford-bunny");
    let before = quality(&mesh);
    assert!(
        before.min_angle_deg > 1.0,
        "the input's worst angle is not degenerate"
    );

    let mut after_mesh = mesh.clone();
    let target = average_edge_length(&mesh);
    let _ = isotropic_remesh(&mut after_mesh, &RemeshOptions::with_target_length(target));
    let after = quality(&after_mesh);

    assert!(
        after.mean_min_angle_deg > before.mean_min_angle_deg,
        "the aggregate improvement is the other half of this finding"
    );
    // 5.5e-8 degrees, stable across runs. Not literally zero, but a triangle whose
    // area no downstream calculation can divide by.
    assert!(
        after.min_angle_deg < 1e-6,
        "the recorded defect is a degenerate face; worst angle is now {:.3e}°, so if it \
         has genuinely been fixed this test should be updated deliberately",
        after.min_angle_deg
    );
}

/// The cylinder is already uniformly triangulated, so there is nothing for isotropic
/// remeshing to do and it correctly does nothing. Pinned because churning an
/// already-good mesh would be a regression that no quality assertion would catch.
#[test]
fn isotropic_remeshing_leaves_an_already_uniform_mesh_alone() {
    let mesh = load("cylinder");
    let before = quality(&mesh);

    let mut after_mesh = mesh.clone();
    let target = average_edge_length(&mesh);
    let _ = isotropic_remesh(&mut after_mesh, &RemeshOptions::with_target_length(target));
    let after = quality(&after_mesh);

    // Not bit-identical: the mesh makes a round trip through face-vertex arrays and
    // back, which perturbs the angles at the 1e-7 degree level. Nothing is remeshed.
    assert_eq!(after.num_faces, before.num_faces);
    assert!(
        (after.min_angle_deg - before.min_angle_deg).abs() < 1e-6,
        "worst angle moved: {:.12} -> {:.12}",
        before.min_angle_deg,
        after.min_angle_deg
    );
    assert!((after.edge_length_cv - before.edge_length_cv).abs() < 1e-9);
}

/// Anisotropic remeshing must terminate and must say whether it converged.
///
/// It did not always terminate: its split pass would settle into adding one vertex and
/// two faces per pass without bound, so `anisotropic_remesh` on
/// `examples/spherical-cap.obj` with the default five iterations never returned. The
/// passes are bounded now, and a pass that hits its bound reports `converged: false`
/// instead of the mesh silently being whatever the algorithm last managed.
#[test]
fn anisotropic_remeshing_terminates_and_reports_convergence() {
    for name in [
        "sphere",
        "spherical-cap",
        "cylinder",
        "torus",
        "stanford-bunny",
    ] {
        let mesh = load(name);
        let target = average_edge_length(&mesh);

        let mut after_mesh = mesh.clone();
        let report = anisotropic_remesh(
            &mut after_mesh,
            &AnisotropicOptions::new(0.5 * target, 2.0 * target),
        );

        assert_eq!(report.faces_before, mesh.num_faces(), "{name}");
        assert_eq!(report.faces_after, after_mesh.num_faces(), "{name}");
        assert!(report.iterations_run >= 1, "{name}: no iterations ran");
        assert!(
            after_mesh.num_faces() > 0,
            "{name}: remeshing consumed the whole mesh"
        );
        // Stopping early and claiming five iterations would defeat the point.
        if !report.converged {
            assert!(
                report.iterations_run < 5,
                "{name}: reported non-convergence but claims all iterations ran"
            );
        }
    }
}

/// **Recorded defect.** Anisotropic remeshing does not reliably improve quality: on the
/// cylinder it takes the worst angle from 43.7° down to about 10°, and reports having
/// converged while doing so. Isotropic remeshing is the one to reach for.
///
/// The bound is loose because the pass order interacts with rayon, so the exact figure
/// moves between runs; the direction does not.
#[test]
fn anisotropic_remeshing_degrades_the_cylinder() {
    let mesh = load("cylinder");
    let before = quality(&mesh);

    let mut after_mesh = mesh.clone();
    let target = average_edge_length(&mesh);
    let report = anisotropic_remesh(
        &mut after_mesh,
        &AnisotropicOptions::new(0.5 * target, 2.0 * target),
    );
    let after = quality(&after_mesh);

    assert!(
        report.converged,
        "the claim of convergence is part of the finding"
    );
    assert!(
        after.min_angle_deg < 0.6 * before.min_angle_deg,
        "expected the recorded degradation, got {:.2}° -> {:.2}°",
        before.min_angle_deg,
        after.min_angle_deg
    );
}

/// CVT remeshing is a *resampling* operator, and its default is degenerate: with
/// `target_vertices: None` the seed count equals the vertex count, so each Voronoi cell
/// holds about one vertex, whose centroid is that vertex, and Lloyd's iteration has
/// nothing to move. Given a target below the vertex count it works as intended.
#[test]
fn cvt_remeshing_needs_a_target_below_the_vertex_count() {
    let mesh = load("sphere");
    let before = quality(&mesh);

    let mut defaulted = mesh.clone();
    let _ = cvt_remesh(&mut defaulted, &CvtOptions::default());
    let after_default = quality(&defaulted);
    assert_eq!(
        after_default.num_faces, before.num_faces,
        "the degenerate default should leave the face count alone"
    );
    assert!(
        after_default.min_angle_deg <= before.min_angle_deg,
        "the default does not improve the worst angle: {:.2}° -> {:.2}°",
        before.min_angle_deg,
        after_default.min_angle_deg
    );

    let mut resampled = mesh.clone();
    let target = mesh.num_vertices() * 2 / 3;
    let _ = cvt_remesh(
        &mut resampled,
        &CvtOptions {
            target_vertices: Some(target),
            ..Default::default()
        },
    );
    assert_eq!(resampled.num_vertices(), target, "target was not honoured");

    let after = quality(&resampled);
    assert!(
        after.min_angle_deg > before.min_angle_deg
            && after.mean_radius_ratio > before.mean_radius_ratio,
        "with a real target CVT should improve quality: worst {:.2}° -> {:.2}°, \
         radius ratio {:.4} -> {:.4}",
        before.min_angle_deg,
        after.min_angle_deg,
        before.mean_radius_ratio,
        after.mean_radius_ratio
    );
}

/// **Recorded defect.** None of the three remeshers project their smoothed vertices
/// back onto the input surface, so all of them shrink it. `examples/sphere.obj` has
/// radius 0.5 to within `1e-6`, which makes the drift directly measurable, and
/// anisotropic remeshing is the worst offender by an order of magnitude — every one of
/// its vertices ends up inside the sphere.
///
/// This is why quality numbers alone cannot settle a remeshing claim.
#[test]
fn remeshing_drifts_off_a_sphere_of_known_radius() {
    let mesh = load("sphere");
    let radius = 0.5;
    let drift = |m: &HalfEdgeMesh| {
        m.vertex_ids()
            .map(|v| (m.position(v).coords.norm() - radius).abs() / radius)
            .fold(0.0_f64, f64::max)
    };
    assert!(drift(&mesh) < 1e-5, "the input is not a unit-radius sphere");

    let target = average_edge_length(&mesh);

    let mut iso = mesh.clone();
    let _ = isotropic_remesh(&mut iso, &RemeshOptions::with_target_length(target));
    let iso_drift = drift(&iso);

    let mut cvt = mesh.clone();
    let _ = cvt_remesh(
        &mut cvt,
        &CvtOptions {
            target_vertices: Some(120),
            ..Default::default()
        },
    );
    let cvt_drift = drift(&cvt);

    let mut aniso = mesh.clone();
    let _ = anisotropic_remesh(
        &mut aniso,
        &AnisotropicOptions::new(0.5 * target, 2.0 * target),
    );
    let aniso_drift = drift(&aniso);

    // Bounds recorded from measurement, loose enough to absorb run-to-run variation but
    // tight enough that a projection step being added would break them.
    assert!(iso_drift < 0.05, "isotropic drift {iso_drift:.4}");
    assert!(cvt_drift < 0.05, "cvt drift {cvt_drift:.4}");
    assert!(
        aniso_drift > 0.05,
        "anisotropic drift {aniso_drift:.4}: the recorded shrinkage is much larger than \
         the others, so if this now passes the projection was fixed"
    );
}
