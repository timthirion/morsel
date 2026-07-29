//! Correctness tests for UV parameterization, built around cases with a known
//! answer rather than around "it returned something".
//!
//! The flat grid is the workhorse: a planar patch is exactly parameterizable, so
//! a conformal map of it must be the identity and its area distortion must be
//! zero at every mesh size. That pins down the solver independently of any
//! downstream algorithm. Both bugs these tests were written for — an LSCM
//! solver tolerance too loose for its penalty-inflated right-hand side, and an
//! OMT step that iterated a transport map as if it were a relaxation — showed up
//! here as a distortion that grew instead of staying at zero.

use morsel::algo::parameterize::{
    arap, compute_area_distortion, lscm, omt, omt_with_report, ARAPOptions, LSCMOptions,
    OMTOptions,
};
use morsel::mesh::{build_from_triangles, HalfEdgeMesh};
use nalgebra::Point3;

/// Grid over `[-1, 1]²` lifted by `height`. Disk topology, so every
/// boundary-requiring method accepts it.
fn height_grid<F: Fn(f64, f64) -> f64>(n: usize, height: F) -> HalfEdgeMesh {
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

fn flat_grid(n: usize) -> HalfEdgeMesh {
    height_grid(n, |_, _| 0.0)
}

fn paraboloid(n: usize) -> HalfEdgeMesh {
    height_grid(n, |x, y| 0.5 * (x * x + y * y))
}

/// A flat patch is exactly parameterizable, so LSCM must return an isometry no
/// matter how many vertices it has. This fails loudly if the CG tolerance is
/// ever loosened back to a value the pin penalty overwhelms: at `1e-8` the
/// 81-vertex case scored `rms = 1.29`.
#[test]
fn lscm_is_isometric_on_flat_grid_at_every_size() {
    for n in [2usize, 3, 4, 5, 6, 8, 12, 16, 24] {
        let mesh = flat_grid(n);
        let uvs = lscm(&mesh, &LSCMOptions::default()).unwrap();
        let (min_r, max_r, rms) = compute_area_distortion(&mesh, &uvs);
        assert!(
            rms < 1e-5,
            "LSCM on a flat {n}x{n} grid ({} vertices) should be isometric, \
             got rms={rms:.3e} (min={min_r:.4}, max={max_r:.4})",
            mesh.num_vertices()
        );
    }
}

/// ARAP reduces to the identity on a flat patch too.
#[test]
fn arap_is_near_isometric_on_flat_grid() {
    for n in [4usize, 8, 12] {
        let mesh = flat_grid(n);
        let uvs = arap(&mesh, &ARAPOptions::default()).unwrap();
        let (_, _, rms) = compute_area_distortion(&mesh, &uvs);
        assert!(
            rms < 1e-3,
            "ARAP on a flat {n}x{n} grid should be near-isometric, got rms={rms:.3e}"
        );
    }
}

/// A conformal map of a paraboloid patch distorts area, but only within the
/// range of its area element `sqrt(1 + x² + y²) ∈ [1, √3]`. An `rms` far above
/// that means the solver is broken, not that the surface is hard — this is the
/// check that would have caught LSCM returning `rms = 1.44` here.
#[test]
fn lscm_area_distortion_stays_within_the_metric() {
    for n in [4usize, 8, 12, 16] {
        let mesh = paraboloid(n);
        let uvs = lscm(&mesh, &LSCMOptions::default()).unwrap();
        let (_, _, rms) = compute_area_distortion(&mesh, &uvs);
        assert!(
            rms < 0.5,
            "conformal map of a paraboloid should distort area only mildly \
             (area element spans [1, 1.73]); got rms={rms:.4} at n={n}"
        );
    }
}

/// OMT must reduce area distortion on curved patches of both curvature signs.
#[test]
fn omt_reduces_area_distortion_on_curved_patches() {
    let cases: Vec<(&str, HalfEdgeMesh)> = vec![
        ("paraboloid", paraboloid(8)),
        ("saddle", height_grid(8, |x, y| 0.6 * (x * x - y * y))),
        // A bump: positive curvature in the middle, negative on the flanks.
        ("bump", height_grid(8, |x, y| (-(x * x + y * y)).exp())),
    ];

    for (name, mesh) in cases {
        let base = lscm(&mesh, &LSCMOptions::default()).unwrap();
        let (_, _, rms_base) = compute_area_distortion(&mesh, &base);
        assert!(
            rms_base > 0.02,
            "{name}: baseline should have distortion to remove, got {rms_base:.6}"
        );

        let (out, report) = omt_with_report(&mesh, &base, &OMTOptions::default()).unwrap();
        let (_, _, rms_omt) = compute_area_distortion(&mesh, &out);

        // Cells are exact polygons, so they must tile the domain regardless of
        // how far the weight ascent got.
        assert!(
            report.domain_area > 0.0,
            "{name}: domain should have positive area"
        );
        assert!(
            rms_omt < 0.6 * rms_base,
            "{name}: OMT should cut area distortion substantially, \
             got {rms_base:.6} -> {rms_omt:.6}"
        );
    }
}

/// The regression guard for the Lloyd bug. On a flat patch the conformal map is
/// already isometric, so the correct area-preserving step is to do nothing. The
/// implementation that iterated the centroid step scored ~1.23 here.
#[test]
fn omt_leaves_an_isometric_map_alone() {
    for n in [4usize, 8] {
        let mesh = flat_grid(n);
        let base = lscm(&mesh, &LSCMOptions::default()).unwrap();
        let (_, _, rms_base) = compute_area_distortion(&mesh, &base);
        assert!(rms_base < 1e-5);

        let out = omt(&mesh, &base, &OMTOptions::default()).unwrap();
        let (_, _, rms_omt) = compute_area_distortion(&mesh, &out);

        assert!(
            rms_omt < 0.02,
            "OMT distorted an already-isometric {n}x{n} grid: \
             {rms_base:.3e} -> {rms_omt:.6}"
        );
    }
}

/// Whatever the method, UVs come back normalized into the unit square and
/// finite. A NaN here would silently poison an exported OBJ.
#[test]
fn all_methods_return_finite_normalized_uvs() {
    let mesh = paraboloid(6);
    let base = lscm(&mesh, &LSCMOptions::default()).unwrap();
    let omt_uvs = omt(&mesh, &base, &OMTOptions::default()).unwrap();
    let arap_uvs = arap(&mesh, &ARAPOptions::default()).unwrap();

    for (name, uvs) in [("lscm", &base), ("omt", &omt_uvs), ("arap", &arap_uvs)] {
        assert_eq!(uvs.len(), mesh.num_vertices(), "{name}: wrong UV count");
        for (vid, uv) in uvs.iter() {
            assert!(
                uv.x.is_finite() && uv.y.is_finite(),
                "{name}: non-finite UV at {vid:?}: {uv:?}"
            );
            assert!(
                uv.x >= -1e-9 && uv.x <= 1.0 + 1e-9 && uv.y >= -1e-9 && uv.y <= 1.0 + 1e-9,
                "{name}: UV outside unit square at {vid:?}: {uv:?}"
            );
        }
    }
}

/// A closed mesh has no boundary, so the boundary-requiring methods must say so
/// rather than returning nonsense.
#[test]
fn boundary_requiring_methods_reject_closed_meshes() {
    // Tetrahedron: closed, no boundary.
    let vertices = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.5, 1.0, 0.0),
        Point3::new(0.5, 0.5, 1.0),
    ];
    let faces = vec![[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]];
    let mesh: HalfEdgeMesh = build_from_triangles(&vertices, &faces).unwrap();

    assert!(lscm(&mesh, &LSCMOptions::default()).is_err());
    assert!(arap(&mesh, &ARAPOptions::default()).is_err());
}

/// Two distinct pins are what remove the conformal energy's similarity kernel.
/// Pinning one vertex twice leaves the reduced system singular, so it must be
/// rejected up front rather than handed to the solver.
#[test]
fn lscm_rejects_degenerate_pins() {
    use morsel::algo::parameterize::PinnedVertex;

    let mesh = flat_grid(4);
    let opts = LSCMOptions::with_pins(
        PinnedVertex::new(0, 0.0, 0.0),
        PinnedVertex::new(0, 1.0, 0.0),
    );
    assert!(
        lscm(&mesh, &opts).is_err(),
        "pinning the same vertex twice should be rejected"
    );

    let out_of_range = LSCMOptions::with_pins(
        PinnedVertex::new(0, 0.0, 0.0),
        PinnedVertex::new(9999, 1.0, 0.0),
    );
    assert!(
        lscm(&mesh, &out_of_range).is_err(),
        "an out-of-range pin index should be rejected"
    );
}

/// The conformal energy is homogeneous, so scaling both pin targets scales the
/// whole solution — and normalized UVs are therefore unchanged. This holds only
/// because the pins are imposed exactly; a penalty term has its own fixed
/// magnitude and would not commute with the scaling.
#[test]
fn lscm_is_invariant_to_pin_scale() {
    use morsel::algo::parameterize::PinnedVertex;

    let mesh = paraboloid(6);
    let solve = |scale: f64| {
        let opts = LSCMOptions::with_pins(
            PinnedVertex::new(0, 0.0, 0.0),
            PinnedVertex::new(mesh.num_vertices() - 1, scale, 0.1 * scale),
        );
        lscm(&mesh, &opts).unwrap()
    };

    let a = solve(1.0);
    let b = solve(50.0);

    for vid in mesh.vertex_ids() {
        let (pa, pb) = (a.get(vid), b.get(vid));
        assert!(
            (pa - pb).norm() < 1e-6,
            "normalized UVs should not depend on pin scale: {pa:?} vs {pb:?}"
        );
    }
}

/// With ARAP's pinned vertex eliminated rather than penalized, the CG tolerance
/// is a meaningful error proxy: accuracy tracks it and does not decay with mesh
/// size. The penalty formulation this replaces scored `rms = 20.5` here.
#[test]
fn arap_accuracy_tracks_tolerance_at_every_size() {
    for n in [4usize, 8, 16] {
        let mesh = flat_grid(n);
        for tol in [1e-8f64, 1e-10] {
            let opts = ARAPOptions::default().with_cg_tolerance(tol);
            let uvs = arap(&mesh, &opts).unwrap();
            let (_, _, rms) = compute_area_distortion(&mesh, &uvs);
            // Generous factor over `tol` itself, but tight enough that a
            // reintroduced penalty term (which decouples the two) would fail.
            assert!(
                rms < 1e4 * tol,
                "ARAP on a flat {n}x{n} grid at tol={tol:.0e} should track the \
                 tolerance, got rms={rms:.3e}"
            );
        }
    }
}
