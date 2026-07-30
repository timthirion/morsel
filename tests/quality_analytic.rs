//! Triangle-quality measures against triangles whose quality is known exactly.
//!
//! These test the *ruler*, not the algorithms measured with it. A quality metric that
//! is subtly wrong is worse than none, because it would certify remeshing improvements
//! that never happened — so every value here is a closed form, not a recorded output.

use morsel::algo::quality::{mesh_quality, triangle_quality};
use morsel::mesh::{build_from_triangles, FaceId, HalfEdgeMesh};
use nalgebra::Point3;

/// A single-triangle mesh from three points.
fn triangle(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> HalfEdgeMesh {
    let verts = vec![
        Point3::new(a[0], a[1], a[2]),
        Point3::new(b[0], b[1], b[2]),
        Point3::new(c[0], c[1], c[2]),
    ];
    build_from_triangles(&verts, &[[0, 1, 2]]).expect("a single triangle is a valid mesh")
}

fn load(name: &str) -> HalfEdgeMesh {
    morsel::io::load(format!("examples/{name}.obj")).expect("example mesh loads")
}

const EXAMPLES: [&str; 6] = [
    "sphere",
    "torus",
    "cylinder",
    "spherical-cap",
    "stanford-bunny",
    "cube-closed",
];

/// The reference case in both directions: an equilateral triangle has every angle at
/// 60° and, by Euler's `R = 2r` holding with equality, a radius ratio of exactly 1.
#[test]
fn an_equilateral_triangle_is_perfect() {
    let h = 3.0_f64.sqrt() / 2.0;
    let mesh = triangle([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, h, 0.0]);
    let q = triangle_quality(&mesh, FaceId::new(0));

    assert!((q.min_angle.to_degrees() - 60.0).abs() < 1e-12, "{q:?}");
    assert!((q.max_angle.to_degrees() - 60.0).abs() < 1e-12, "{q:?}");
    assert!((q.radius_ratio - 1.0).abs() < 1e-12, "{q:?}");
    assert!((q.area - h / 2.0).abs() < 1e-15, "{q:?}");
}

/// A right isoceles triangle with legs 1: angles 45°, 45°, 90°, and a radius ratio of
/// `2√2 − 2`. Derivation: `r = A/s = ½/(1 + √2/2)` and `R = abc/(4A) = √2/2`, so
/// `2r/R = 4/(2√2 + 2) = 2√2 − 2`.
#[test]
fn a_right_isoceles_triangle_matches_its_closed_form() {
    let mesh = triangle([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    let q = triangle_quality(&mesh, FaceId::new(0));

    assert!((q.min_angle.to_degrees() - 45.0).abs() < 1e-12, "{q:?}");
    assert!((q.max_angle.to_degrees() - 90.0).abs() < 1e-12, "{q:?}");

    let expected = 2.0 * 2.0_f64.sqrt() - 2.0;
    assert!(
        (q.radius_ratio - expected).abs() < 1e-12,
        "radius ratio {}, expected {expected}",
        q.radius_ratio
    );
}

/// An isoceles triangle of base 1 and height `h` has base angles `atan(2h)`, which for
/// small `h` is the minimum angle. This checks the metric stays accurate exactly where
/// it matters most and where a law-of-cosines formulation would lose precision.
#[test]
fn a_sliver_reports_its_exact_minimum_angle() {
    for h in [1e-2, 1e-4, 1e-6, 1e-8] {
        let mesh = triangle([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, h, 0.0]);
        let q = triangle_quality(&mesh, FaceId::new(0));

        let expected = (2.0 * h).atan();
        assert!(
            (q.min_angle - expected).abs() < 1e-14,
            "h={h}: min angle {} rad, expected {expected}",
            q.min_angle
        );
        // The apex angle takes up essentially all of π, and the ratios collapse.
        assert!(
            (q.max_angle - (std::f64::consts::PI - 2.0 * expected)).abs() < 1e-13,
            "h={h}: max angle {}",
            q.max_angle
        );
        assert!(q.radius_ratio < 8.0 * h, "h={h}: ratio {}", q.radius_ratio);
    }
}

/// Three collinear points have no shape to measure, and must not produce a NaN that
/// then poisons every aggregate built on it.
#[test]
fn a_degenerate_triangle_scores_zero_rather_than_nan() {
    let mesh = triangle([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]);
    let q = triangle_quality(&mesh, FaceId::new(0));

    assert_eq!(q.radius_ratio, 0.0);
    assert_eq!(q.min_angle, 0.0);
    assert_eq!(q.area, 0.0);
    assert!(q.max_angle.is_finite());
}

/// Angles and ratios are shape measures, so scaling a mesh must leave them untouched
/// while the area scales as the square. A metric that failed this would rank meshes by
/// how large they happen to be.
#[test]
fn quality_is_invariant_under_scale() {
    for name in EXAMPLES {
        let mesh = load(name);
        let base = mesh_quality(&mesh).expect("has faces");

        let (verts, faces) = morsel::mesh::to_face_vertex(&mesh);
        let scaled: Vec<Point3<f64>> = verts.iter().map(|p| p * 1000.0).collect();
        let scaled = build_from_triangles::<u32>(&scaled, &faces).expect("rebuilds");
        let after = mesh_quality(&scaled).expect("has faces");

        assert!(
            (base.min_angle_deg - after.min_angle_deg).abs() < 1e-9,
            "{name}: worst angle changed under scaling, {} -> {}",
            base.min_angle_deg,
            after.min_angle_deg
        );
        assert!(
            (base.mean_radius_ratio - after.mean_radius_ratio).abs() < 1e-12,
            "{name}: mean radius ratio changed under scaling"
        );
        // The coefficient of variation is a ratio too, so it is also scale-free.
        assert!(
            (base.edge_length_cv - after.edge_length_cv).abs() < 1e-12,
            "{name}: edge-length cv changed under scaling"
        );
        assert!(
            (base.edge_length.1 * 1000.0 - after.edge_length.1).abs() < 1e-6 * after.edge_length.1,
            "{name}: mean edge length did not scale"
        );
    }
}

/// A triangle's interior angles sum to π. Checking this on real meshes catches an
/// angle assigned to the wrong corner, which no single hand-built triangle would.
#[test]
fn interior_angles_sum_to_pi_on_every_face() {
    for name in EXAMPLES {
        let mesh = load(name);
        for face in mesh.face_ids() {
            let q = triangle_quality(&mesh, face);
            // Only the extremes are reported, so recover the third from the identity
            // and check it is a plausible angle rather than a residue.
            let third = std::f64::consts::PI - q.min_angle - q.max_angle;
            assert!(
                third >= q.min_angle - 1e-9 && third <= q.max_angle + 1e-9,
                "{name} face {}: min {:.6} max {:.6} leaves {:.6}, which is not between them",
                face.index(),
                q.min_angle,
                q.max_angle,
                third
            );
        }
    }
}

/// Aggregates have to agree with the per-face numbers they summarise, including the
/// histogram, which is easy to get wrong at a bin edge.
#[test]
fn aggregates_agree_with_the_per_face_measures() {
    for name in EXAMPLES {
        let mesh = load(name);
        let report = mesh_quality(&mesh).expect("has faces");

        let per_face: Vec<_> = mesh
            .face_ids()
            .map(|f| triangle_quality(&mesh, f))
            .collect();
        assert_eq!(report.num_faces, per_face.len());

        let worst = per_face
            .iter()
            .map(|q| q.min_angle.to_degrees())
            .fold(f64::INFINITY, f64::min);
        assert!((report.min_angle_deg - worst).abs() < 1e-12, "{name}");

        let under_30 = per_face
            .iter()
            .filter(|q| q.min_angle.to_degrees() < 30.0)
            .count();
        assert_eq!(report.faces_under_30_deg, under_30, "{name}");

        assert_eq!(
            report.min_angle_histogram.iter().sum::<usize>(),
            per_face.len(),
            "{name}: histogram does not account for every face"
        );

        // A minimum angle cannot exceed 60°, so nothing may land above the 60-70 bin.
        assert_eq!(
            report.min_angle_histogram[7..].iter().sum::<usize>(),
            0,
            "{name}: a face reported a minimum angle above 70°, which is impossible"
        );
    }
}

/// An empty mesh has no quality rather than a quality of zero, and saying so keeps
/// callers from averaging over nothing.
///
/// `build_from_triangles` will not produce one — it rejects empty input outright — so
/// the only route in is a directly constructed mesh, which is exactly the case a
/// caller assembling a mesh by hand could hit.
#[test]
fn a_mesh_with_no_faces_has_no_report() {
    assert!(mesh_quality(&HalfEdgeMesh::<u32>::new()).is_none());

    let refused = build_from_triangles::<u32>(&[], &[]);
    assert!(refused.is_err(), "empty input should be rejected at build");
}
