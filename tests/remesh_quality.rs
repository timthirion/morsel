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

/// The worst triangle improves too, where there is room to improve it.
///
/// The torus is deliberately absent, and the reason is worth stating: its worst angle goes
/// from 27.5° to about 16.8°. That is isotropic remeshing doing its job — trading the worst
/// case for uniform edge lengths, with the mean rising 35.6° to 49.2° and the edge-length
/// cv falling 0.29 to 0.16. It used to *appear* to improve the worst angle only because its
/// collapse pass produced a face list the half-edge representation rejected, so the pass
/// was discarded and never applied at all.
#[test]
fn isotropic_remeshing_improves_the_worst_triangle_on_smooth_meshes() {
    for name in ["sphere", "spherical-cap"] {
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

/// **Fixed defect, kept as a regression test.** Isotropic remeshing used to drive the
/// bunny's worst triangle to 5.5e-8 degrees while improving every aggregate measure — its
/// best-looking result of the five meshes, and a face with no usable area, which put
/// curvature values of 6e7 on the vertices around it.
///
/// The cause was the split pass. Halving an edge does opposite things to a thin triangle
/// depending on which edge it is: splitting the base of a sliver halves the base and
/// *doubles* the minimum angle, while splitting a side halves the triangle's *height* and
/// halves the angle. Both sides of a sliver exceed the length threshold, so both were
/// split, and the pass loops up to twenty times — 1.5° / 2²⁰ is about 1.4e-6°.
///
/// Splits now decline when they would produce a triangle below a 1° floor, so the worst
/// triangle comes out *better* than the input's rather than eight orders of magnitude
/// worse.
#[test]
fn isotropic_remeshing_no_longer_degenerates_the_bunny() {
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
        after.min_angle_deg > before.min_angle_deg,
        "worst angle {:.4}° -> {:.4}°, expected an improvement",
        before.min_angle_deg,
        after.min_angle_deg
    );
    assert!(
        after.min_angle_deg > 1.0,
        "worst angle {:.3e}° is below the 1° floor splits are supposed to respect",
        after.min_angle_deg
    );
    assert!(
        after.min_radius_ratio > 1e-3,
        "radius ratio {:.3e} still indicates a near-degenerate face",
        after.min_radius_ratio
    );
    // The aggregate gains are the other half: this was never a tradeoff.
    assert!(after.mean_min_angle_deg > before.mean_min_angle_deg + 10.0);
    assert!(after.mean_radius_ratio > before.mean_radius_ratio + 0.1);
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

/// **Fixed defect, and a correction to how it was described.**
///
/// Anisotropic remeshing was shrinking the surface badly: on `examples/sphere.obj`, radius
/// 0.5, every one of its vertices ended up inside the sphere, as much as 14.8% of the radius
/// inward. It projects smoothed vertices, split midpoints and collapse targets back onto the
/// input surface now, and lands at 2.7%.
///
/// The correction is what 2.7% *means*. The earlier version of this test recorded isotropic
/// remeshing as drifting 2.7% too and called that shrinkage. It is not: a polyhedral sphere's
/// faces are chords, so its surface dips inside the true sphere by the sagitta, and any vertex
/// placed on that surface inherits the same deviation. Measured directly by projecting points
/// of the true sphere onto the input mesh, that floor is about 2.4%. Isotropic was sitting on
/// it, not drifting past it, and neither it nor CVT was ever shrinking the mesh.
///
/// So the test calibrates the floor from the input rather than hard-coding a number, and
/// asserts that no remesher drifts materially past it.
#[test]
fn no_remesher_drifts_past_the_input_surface() {
    use morsel::algo::distance::SurfaceIndex;
    use morsel::algo::remesh::{anisotropic_remesh, cvt_remesh, AnisotropicOptions, CvtOptions};
    use nalgebra::Point3;

    let mesh = load("sphere");
    let radius = 0.5;
    let target = average_edge_length(&mesh);

    let worst_vertex_drift = |m: &HalfEdgeMesh| {
        m.vertex_ids()
            .map(|v| (m.position(v).coords.norm() - radius).abs() / radius)
            .fold(0.0_f64, f64::max)
    };

    // The floor: the deepest the input mesh's own surface dips inside the true sphere. Any
    // vertex lying on that surface is at least this far off the true sphere, so no amount of
    // projection can do better.
    let index = SurfaceIndex::new(&mesh);
    let mut floor = 0.0_f64;
    for i in 0..400 {
        let a = i as f64 * 0.37;
        let dir = Point3::new(a.sin() * a.cos(), a.sin() * a.sin(), a.cos());
        let on_sphere = Point3::from(dir.coords.normalize() * radius);
        let projected = index.project(&on_sphere);
        floor = floor.max((projected.coords.norm() - radius).abs() / radius);
    }
    assert!(
        floor > 0.01 && floor < 0.05,
        "the calibration itself looks wrong: floor {floor:.4}"
    );

    for (label, remeshed) in [
        ("isotropic", {
            let mut m = mesh.clone();
            let _ = isotropic_remesh(&mut m, &RemeshOptions::with_target_length(target));
            m
        }),
        ("anisotropic", {
            let mut m = mesh.clone();
            let _ =
                anisotropic_remesh(&mut m, &AnisotropicOptions::new(0.5 * target, 2.0 * target));
            m
        }),
        ("cvt", {
            let mut m = mesh.clone();
            let _ = cvt_remesh(
                &mut m,
                &CvtOptions {
                    target_vertices: Some(120),
                    ..Default::default()
                },
            );
            m
        }),
    ] {
        let drift = worst_vertex_drift(&remeshed);
        assert!(
            drift < floor * 1.15,
            "{label} drifted {:.4} against a floor of {:.4}",
            drift,
            floor
        );
    }
}

/// The other half of the same finding: turning projection off puts anisotropic remeshing
/// back where it was, several times past the faceting floor. Without this, the assertion
/// above could pass for a reason unrelated to projection.
#[test]
fn without_projection_anisotropic_shrinks_the_sphere() {
    use morsel::algo::remesh::{anisotropic_remesh, AnisotropicOptions};

    let mesh = load("sphere");
    let radius = 0.5;
    let target = average_edge_length(&mesh);
    let worst = |m: &HalfEdgeMesh| {
        m.vertex_ids()
            .map(|v| (m.position(v).coords.norm() - radius).abs() / radius)
            .fold(0.0_f64, f64::max)
    };

    let mut with = mesh.clone();
    let _ = anisotropic_remesh(
        &mut with,
        &AnisotropicOptions::new(0.5 * target, 2.0 * target),
    );

    let mut without = mesh.clone();
    let _ = anisotropic_remesh(
        &mut without,
        &AnisotropicOptions::new(0.5 * target, 2.0 * target).with_project_to_surface(false),
    );

    assert!(
        worst(&without) > 4.0 * worst(&with),
        "projection should make a large difference here: {:.4} without, {:.4} with",
        worst(&without),
        worst(&with)
    );
}
