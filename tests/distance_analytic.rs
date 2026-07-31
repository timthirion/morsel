//! Point-to-surface and Hausdorff distances against answers known in closed form.
//!
//! This is the ruler for "is it still the same shape", so it is checked the same way the
//! quality metric was: against geometry whose answer can be written down, not against
//! recorded output. A distance that is quietly wrong would certify shape preservation that
//! never happened, which is worse than having no measure at all.

use morsel::algo::distance::{
    closest_point_on_triangle, hausdorff_distance, point_triangle_distance, HausdorffOptions,
    SurfaceIndex,
};
use morsel::mesh::{build_from_triangles, HalfEdgeMesh};
use nalgebra::Point3;

fn p(x: f64, y: f64, z: f64) -> Point3<f64> {
    Point3::new(x, y, z)
}

fn load(name: &str) -> HalfEdgeMesh {
    morsel::io::load(format!("examples/{name}.obj")).expect("example mesh loads")
}

/// The unit right triangle in the z = 0 plane, with corners at the origin and the two axes.
fn unit_triangle() -> (Point3<f64>, Point3<f64>, Point3<f64>) {
    (p(0.0, 0.0, 0.0), p(1.0, 0.0, 0.0), p(0.0, 1.0, 0.0))
}

/// A point above the interior projects straight down: the distance is its height.
#[test]
fn a_point_over_the_interior_measures_its_height() {
    let (a, b, c) = unit_triangle();
    for h in [1e-9, 0.5, 1.0, 1e6] {
        let q = p(0.25, 0.25, h);
        assert!(
            (point_triangle_distance(&q, &a, &b, &c) - h).abs() < 1e-12 * h.max(1.0),
            "h={h}"
        );
        let foot = closest_point_on_triangle(&q, &a, &b, &c);
        assert!(
            (foot - p(0.25, 0.25, 0.0)).norm() < 1e-12,
            "h={h}: {foot:?}"
        );
    }
}

/// Each of the three vertex regions returns that vertex exactly.
#[test]
fn points_beyond_a_corner_return_that_corner() {
    let (a, b, c) = unit_triangle();
    for (q, expected) in [
        (p(-1.0, -1.0, 0.0), a),
        (p(3.0, -1.0, 0.0), b),
        (p(-1.0, 3.0, 0.0), c),
    ] {
        let got = closest_point_on_triangle(&q, &a, &b, &c);
        assert!((got - expected).norm() < 1e-15, "{q:?} -> {got:?}");
        assert!((point_triangle_distance(&q, &a, &b, &c) - (q - expected).norm()).abs() < 1e-15);
    }
}

/// Each edge region projects onto that edge. For the hypotenuse of this triangle the answer
/// is the classic one: the point `(1, 1, 0)` is `√2/2` from the line `x + y = 1`.
#[test]
fn points_beside_an_edge_project_onto_it() {
    let (a, b, c) = unit_triangle();

    // Beyond the hypotenuse.
    let q = p(1.0, 1.0, 0.0);
    let got = closest_point_on_triangle(&q, &a, &b, &c);
    assert!((got - p(0.5, 0.5, 0.0)).norm() < 1e-15, "{got:?}");
    let expected = 2.0_f64.sqrt() / 2.0;
    assert!((point_triangle_distance(&q, &a, &b, &c) - expected).abs() < 1e-15);

    // Beside the leg along the x axis, and beside the leg along the y axis.
    assert!(
        (closest_point_on_triangle(&p(0.5, -2.0, 0.0), &a, &b, &c) - p(0.5, 0.0, 0.0)).norm()
            < 1e-15
    );
    assert!(
        (closest_point_on_triangle(&p(-2.0, 0.5, 0.0), &a, &b, &c) - p(0.0, 0.5, 0.0)).norm()
            < 1e-15
    );
}

/// The case that rules out the naive "project onto the plane and clamp" implementation.
///
/// For a very obtuse triangle, the perpendicular foot of a nearby point can land outside the
/// triangle across an edge that is *not* the nearest one, so clamping barycentric
/// coordinates gives the wrong edge. Here the apex is nearly collinear with the base.
#[test]
fn an_obtuse_triangle_picks_the_right_edge() {
    let a = p(0.0, 0.0, 0.0);
    let b = p(10.0, 0.0, 0.0);
    let c = p(5.0, 0.01, 0.0);

    // A point beyond `b`, just off the axis. The nearest feature is the vertex `b`.
    let q = p(11.0, 0.005, 0.0);
    let got = closest_point_on_triangle(&q, &a, &b, &c);
    assert!(
        (got - b).norm() < 1e-12,
        "expected the corner b, got {got:?}"
    );

    // Directly below the middle of the base: the nearest feature is the base itself.
    let q = p(5.0, -3.0, 0.0);
    let got = closest_point_on_triangle(&q, &a, &b, &c);
    assert!((got - p(5.0, 0.0, 0.0)).norm() < 1e-12, "{got:?}");
    assert!((point_triangle_distance(&q, &a, &b, &c) - 3.0).abs() < 1e-12);
}

/// A degenerate triangle still has a well-defined nearest point, and must not produce NaN.
#[test]
fn degenerate_triangles_do_not_produce_nan() {
    // Three collinear points reduce to a segment.
    let (a, b, c) = (p(0.0, 0.0, 0.0), p(2.0, 0.0, 0.0), p(1.0, 0.0, 0.0));
    let d = point_triangle_distance(&p(1.0, 5.0, 0.0), &a, &b, &c);
    assert!((d - 5.0).abs() < 1e-12, "collinear: {d}");

    // A single point.
    let z = p(3.0, 4.0, 0.0);
    let d = point_triangle_distance(&p(0.0, 0.0, 0.0), &z, &z, &z);
    assert!((d - 5.0).abs() < 1e-12, "point: {d}");
}

/// The index must agree with checking every triangle. Brute force is obviously correct and
/// hopelessly slow, which is exactly what a spatial index is for — so it is the reference.
#[test]
fn the_index_agrees_with_brute_force() {
    for name in ["sphere", "spherical-cap", "torus", "stanford-bunny"] {
        let mesh = load(name);
        let index = SurfaceIndex::new(&mesh);
        let triangles: Vec<[Point3<f64>; 3]> = mesh
            .face_ids()
            .map(|f| {
                let [a, b, c] = mesh.face_triangle(f);
                [*mesh.position(a), *mesh.position(b), *mesh.position(c)]
            })
            .collect();

        // Query points on a lattice through the mesh's bounding box, so some fall inside the
        // surface, some outside, and some far away.
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    let q = p(
                        -1.0 + 0.7 * i as f64,
                        -1.0 + 0.7 * j as f64,
                        -1.0 + 0.7 * k as f64,
                    );
                    let brute = triangles
                        .iter()
                        .map(|t| point_triangle_distance(&q, &t[0], &t[1], &t[2]))
                        .fold(f64::INFINITY, f64::min);
                    let fast = index.distance(&q).expect("non-empty");
                    assert!(
                        (fast - brute).abs() < 1e-12 * brute.max(1.0),
                        "{name} at {q:?}: index {fast}, brute force {brute}"
                    );
                }
            }
        }
    }
}

/// A mesh is zero distance from itself, in both directions and however it is sampled.
#[test]
fn a_mesh_is_zero_distance_from_itself() {
    for name in ["sphere", "cylinder", "stanford-bunny"] {
        let mesh = load(name);
        for options in [
            HausdorffOptions::vertices_only(),
            HausdorffOptions::default(),
        ] {
            let report = hausdorff_distance(&mesh, &mesh, &options);
            assert!(
                report.distance < 1e-12,
                "{name}: {} with {} samples",
                report.distance,
                report.samples
            );
            assert!(
                report.forward_rms < 1e-12 && report.backward_rms < 1e-12,
                "{name}"
            );
        }
    }
}

/// Two parallel planes offset by `d` are exactly `d` apart, and this is the case where the
/// answer is exact rather than a lower bound: every point of one plane is `d` from the other,
/// so sampling cannot understate it.
#[test]
fn parallel_planes_are_exactly_their_offset_apart() {
    let plane = |z: f64| {
        let verts = vec![
            p(0.0, 0.0, z),
            p(1.0, 0.0, z),
            p(1.0, 1.0, z),
            p(0.0, 1.0, z),
        ];
        build_from_triangles::<u32>(&verts, &[[0, 1, 2], [0, 2, 3]]).expect("builds")
    };

    for d in [1e-6, 0.01, 0.5, 3.0] {
        let report = hausdorff_distance(&plane(0.0), &plane(d), &HausdorffOptions::default());
        assert!(
            (report.distance - d).abs() < 1e-12 * d.max(1.0),
            "offset {d}: measured {}",
            report.distance
        );
        // Every sample is the same distance away, so the RMS equals the maximum.
        assert!(
            (report.forward_rms - d).abs() < 1e-12 * d.max(1.0),
            "offset {d}"
        );
    }
}

/// Scaling a sphere of radius `R` to `R + d` moves every point of it `d` outward, so the
/// distance between the two is `d` — up to the faceting of a polyhedral sphere, which is
/// why the tolerance is a few percent rather than machine epsilon.
#[test]
fn concentric_spheres_are_their_radius_difference_apart() {
    let mesh = load("sphere");
    let radius = 0.5;
    let (verts, faces) = morsel::mesh::to_face_vertex(&mesh);

    for d in [0.01, 0.05, 0.2] {
        let scale = (radius + d) / radius;
        let scaled: Vec<Point3<f64>> = verts
            .iter()
            .map(|v| Point3::from(v.coords * scale))
            .collect();
        let bigger = build_from_triangles::<u32>(&scaled, &faces).expect("rebuilds");

        let report = hausdorff_distance(&mesh, &bigger, &HausdorffOptions::default());
        assert!(
            (report.distance - d).abs() < 0.05 * d,
            "radius gap {d}: measured {}",
            report.distance
        );
    }
}

/// Sampling only vertices understates the distance, which is why the default is not
/// vertices-only.
///
/// Measured one-sided on purpose. Symmetrically, the construction below has its greatest
/// separation at the tent's apex, and the apex *is* a vertex — so vertex sampling happens to
/// find the true answer and the bias is invisible. Forward only, from the flat sheet to the
/// tent, the flat sheet's four corners are also corners of the tent and so are exactly zero
/// away, while its interior is not. Vertex sampling reports the two surfaces as identical.
#[test]
fn vertex_sampling_understates_the_distance() {
    let corners = [
        p(-1.0, -1.0, 0.0),
        p(1.0, -1.0, 0.0),
        p(1.0, 1.0, 0.0),
        p(-1.0, 1.0, 0.0),
    ];

    // A flat sheet over the square: two triangles, four vertices, no interior vertex.
    let flat = build_from_triangles::<u32>(&corners, &[[0, 1, 2], [0, 2, 3]]).expect("builds");

    // The same square boundary, tented up to an apex. Its four corners coincide with the
    // flat sheet's four vertices.
    let mut tent_verts = corners.to_vec();
    tent_verts.push(p(0.0, 0.0, 0.4));
    let tented =
        build_from_triangles::<u32>(&tent_verts, &[[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
            .expect("builds");

    let one_sided = |samples_per_face| HausdorffOptions {
        samples_per_face,
        symmetric: false,
    };

    let sparse = hausdorff_distance(&flat, &tented, &one_sided(0));
    let dense = hausdorff_distance(&flat, &tented, &one_sided(25));

    assert!(
        sparse.forward < 1e-12,
        "every vertex of the flat sheet is a vertex of the tent, so vertex sampling should \
         see no difference at all; got {}",
        sparse.forward
    );
    assert!(
        dense.forward > 0.05,
        "interior samples should find real separation; got {}",
        dense.forward
    );
    assert!(dense.samples > sparse.samples);
}

/// Building the index twice must give identical answers, or nothing built on it can be
/// reproducible.
#[test]
fn the_index_is_deterministic() {
    let mesh = load("stanford-bunny");
    let queries: Vec<Point3<f64>> = (0..50)
        .map(|i| {
            let t = i as f64 / 50.0;
            p(-0.1 + 0.3 * t, 0.05 + 0.2 * t, -0.05 + 0.15 * t)
        })
        .collect();

    let first: Vec<f64> = {
        let index = SurfaceIndex::new(&mesh);
        queries.iter().map(|q| index.distance(q).unwrap()).collect()
    };
    for _ in 0..3 {
        let index = SurfaceIndex::new(&mesh);
        let again: Vec<f64> = queries.iter().map(|q| index.distance(q).unwrap()).collect();
        assert_eq!(first, again, "index queries differ between builds");
    }
}
