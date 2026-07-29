//! Why area-preserving transport plateaus, and where.
//!
//! Semi-discrete OMT assigns each *vertex* a target mass and solves for weights
//! so that each vertex's power cell has that area. It converges: the worst
//! per-cell relative area error reaches `1e-9`. But the thing we actually want
//! preserved is **per-triangle** area, and that plateaus at a few percent no
//! matter how long the solve runs.
//!
//! That gap is structural, not numerical. For a triangulated disk, Euler's
//! formula with `3F = 2E − B` gives
//!
//! ```text
//! F = 2V − B − 2
//! ```
//!
//! The transport has `V − 1` weight degrees of freedom (one lost to the dual's
//! invariance under a constant shift), while per-triangle area preservation asks
//! for `F` equalities. Substituting gives the deficit exactly:
//!
//! ```text
//! F − (V − 1) = V − B − 1 = (interior vertices) − 1
//! ```
//!
//! So *this formulation* is overdetermined as soon as a mesh has two interior
//! vertices, and increasingly so with refinement. Exact dual-cell area
//! preservation therefore cannot imply per-triangle area preservation.
//!
//! Note the qualifier. This is a limit on the power-diagram search space, **not**
//! on piecewise-linear maps. With full vertex freedom there are `2V − 4` effective
//! degrees against `F − 1 = 2V − B − 3` constraints — a *surplus* of `B − 1` — and
//! the stretch-energy / authalic-energy literature attains exact per-triangle area
//! preservation that way. See `plans/0002-research-program.md` for the literature
//! check that established this.
//!
//! These tests pin down both halves: the counting identity, and the resulting
//! plateau. If someone later drives per-triangle distortion to zero with a
//! vertex-weighted formulation, `interior_distortion_plateaus_above_zero` will
//! fail — and that failure would be a genuinely interesting result, so it is
//! written to say so.
//!
//! See `plans/0002-research-program.md`.

use morsel::algo::parameterize::{lscm, omt_with_report, LSCMOptions, OMTOptions, UVMap};
use morsel::mesh::{build_from_triangles, HalfEdgeMesh};
use nalgebra::Point3;

fn height_grid<F: Fn(f64, f64) -> f64>(n: usize, h: F) -> HalfEdgeMesh {
    let mut vs = Vec::new();
    let mut fs = Vec::new();
    for j in 0..=n {
        for i in 0..=n {
            let x = -1.0 + 2.0 * (i as f64) / (n as f64);
            let y = -1.0 + 2.0 * (j as f64) / (n as f64);
            vs.push(Point3::new(x, y, h(x, y)));
        }
    }
    for j in 0..n {
        for i in 0..n {
            let v00 = j * (n + 1) + i;
            let v10 = j * (n + 1) + i + 1;
            let v01 = (j + 1) * (n + 1) + i;
            let v11 = (j + 1) * (n + 1) + i + 1;
            fs.push([v00, v10, v11]);
            fs.push([v00, v11, v01]);
        }
    }
    build_from_triangles(&vs, &fs).unwrap()
}

fn paraboloid(n: usize) -> HalfEdgeMesh {
    height_grid(n, |x, y| 0.5 * (x * x + y * y))
}

/// Per-triangle area ratios split by whether the face touches the boundary.
/// Returns `((interior_min, interior_max), (boundary_min, boundary_max))`.
fn split_area_ratios(mesh: &HalfEdgeMesh, uvs: &UVMap) -> ((f64, f64), (f64, f64)) {
    let uv_area = |f| {
        let [v0, v1, v2] = mesh.face_triangle(f);
        let (p0, p1, p2) = (uvs.get(v0), uvs.get(v1), uvs.get(v2));
        0.5 * ((p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y)).abs()
    };

    let total_3d: f64 = mesh.face_ids().map(|f| mesh.face_area(f)).sum();
    let total_uv: f64 = mesh.face_ids().map(uv_area).sum();
    let scale = total_uv / total_3d;

    let mut interior = (f64::INFINITY, 0.0f64);
    let mut boundary = (f64::INFINITY, 0.0f64);
    for f in mesh.face_ids() {
        let ratio = uv_area(f) / (mesh.face_area(f) * scale);
        let [v0, v1, v2] = mesh.face_triangle(f);
        let touches_boundary = [v0, v1, v2].iter().any(|&v| mesh.is_boundary_vertex(v));
        let slot = if touches_boundary {
            &mut boundary
        } else {
            &mut interior
        };
        slot.0 = slot.0.min(ratio);
        slot.1 = slot.1.max(ratio);
    }
    (interior, boundary)
}

/// `F = 2V − B − 2` for a triangulated disk. The counting identity behind the
/// degree-of-freedom deficit, checked against real meshes.
#[test]
fn face_count_identity_for_a_disk() {
    for n in [1usize, 2, 3, 4, 8, 12] {
        let mesh = paraboloid(n);
        let v = mesh.num_vertices();
        let f = mesh.num_faces();
        let b = mesh
            .vertex_ids()
            .filter(|&x| mesh.is_boundary_vertex(x))
            .count();

        assert_eq!(
            f,
            2 * v - b - 2,
            "n={n}: expected F = 2V - B - 2 = {}, got {f} (V={v}, B={b})",
            2 * v - b - 2
        );

        // The deficit follows exactly: substituting F = 2V - B - 2 into
        //   deficit = F - (V - 1)
        // gives V - B - 1, i.e. one less than the number of interior vertices.
        // So it is zero for a mesh with a single interior vertex and grows
        // linearly from there — it is not merely asymptotic.
        let interior = v - b;
        let deficit = f as isize - (v as isize - 1);
        assert_eq!(
            deficit,
            interior as isize - 1,
            "n={n}: {f} per-face constraints vs {} weight DOF is a deficit of \
             {deficit}, expected interior_vertices - 1 = {} (V={v}, B={b})",
            v - 1,
            interior as isize - 1
        );
    }
}

/// The transport really does converge — this is not an unconverged solve.
#[test]
fn transport_converges_on_cell_areas() {
    for n in [8usize, 12] {
        let mesh = paraboloid(n);
        let base = lscm(&mesh, &LSCMOptions::default()).unwrap();
        let opts = OMTOptions {
            max_iterations: 20_000,
            tolerance: 1e-9,
            ..Default::default()
        };
        let (_, report) = omt_with_report(&mesh, &base, &opts).unwrap();

        assert!(
            report.converged,
            "n={n}: transport should reach 1e-9 on cell areas, got {:.2e} after {} iters",
            report.max_relative_error, report.iterations
        );
    }
}

/// Interior per-triangle distortion improves a lot, and then stops — well above
/// zero, and unmoved by two more orders of magnitude of iterations.
#[test]
fn interior_distortion_plateaus_above_zero() {
    for n in [8usize, 16] {
        let mesh = paraboloid(n);
        let base = lscm(&mesh, &LSCMOptions::default()).unwrap();
        let ((base_lo, base_hi), _) = split_area_ratios(&mesh, &base);

        let solve = |iters: usize| {
            let opts = OMTOptions {
                max_iterations: iters,
                tolerance: 1e-9,
                ..Default::default()
            };
            let (uvs, report) = omt_with_report(&mesh, &base, &opts).unwrap();
            (split_area_ratios(&mesh, &uvs), report)
        };

        let (((lo_a, hi_a), _), rep_a) = solve(5_000);
        let (((lo_b, hi_b), _), rep_b) = solve(20_000);

        println!(
            "n={n}: LSCM interior [{base_lo:.4}, {base_hi:.4}] -> \
             OMT [{lo_a:.4}, {hi_a:.4}] ({} iters) / [{lo_b:.4}, {hi_b:.4}] ({} iters)",
            rep_a.iterations, rep_b.iterations
        );

        // It improves substantially.
        let base_spread = base_hi - base_lo;
        let spread = hi_a - lo_a;
        assert!(
            spread < 0.5 * base_spread,
            "n={n}: interior spread should shrink markedly, {base_spread:.4} -> {spread:.4}"
        );

        // Four times the iteration budget changes nothing: this is a plateau.
        assert!(
            (lo_a - lo_b).abs() < 1e-6 && (hi_a - hi_b).abs() < 1e-6,
            "n={n}: 4x the iterations moved the plateau, [{lo_a:.6}, {hi_a:.6}] vs \
             [{lo_b:.6}, {hi_b:.6}]"
        );

        // And the plateau is well above exact. If this ever fails with a
        // vertex-weighted formulation, the degree-of-freedom count above says it
        // should not be possible — investigate rather than relax the bound.
        assert!(
            (hi_a - 1.0).abs() > 0.01 || (1.0 - lo_a).abs() > 0.01,
            "n={n}: per-triangle area preservation reached [{lo_a:.6}, {hi_a:.6}], \
             which the F = 2V - B - 2 deficit says should be unreachable. \
             This would be a real result — see plans/0002."
        );
    }
}

/// A triangle with all three vertices pinned cannot move, so its area ratio is
/// necessarily untouched. Recorded because this trivially explains an invariance
/// that briefly looked like a deep boundary obstruction.
#[test]
fn fully_pinned_triangles_are_frozen() {
    let mesh = paraboloid(8);
    let base = lscm(&mesh, &LSCMOptions::default()).unwrap();
    let opts = OMTOptions {
        max_iterations: 20_000,
        tolerance: 1e-9,
        fix_boundary: true,
        ..Default::default()
    };
    let (out, _) = omt_with_report(&mesh, &base, &opts).unwrap();

    let mut checked = 0;
    for f in mesh.face_ids() {
        let [v0, v1, v2] = mesh.face_triangle(f);
        if ![v0, v1, v2].iter().all(|&v| mesh.is_boundary_vertex(v)) {
            continue;
        }
        checked += 1;
        for v in [v0, v1, v2] {
            let (before, after) = (base.get(v), out.get(v));
            // `normalize()` rescales both maps the same way, so a frozen vertex
            // lands in the same place.
            assert!(
                (before - after).norm() < 1e-9,
                "{v:?} is on the boundary of a fully-pinned face but moved: \
                 {before:?} -> {after:?}"
            );
        }
    }
    assert!(
        checked > 0,
        "a grid should have corner faces with all three vertices on the boundary"
    );
}
