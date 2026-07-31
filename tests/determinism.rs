//! Every mesh-mutating algorithm gives the same answer every time.
//!
//! Two of them did not, and for the same underlying reason each time: work was ordered by
//! the iteration order of a hash container, which Rust reseeds per instance.
//!
//! - **QEM decimation.** Its heap of collapse candidates ordered them on cost alone, so
//!   equal-cost candidates were separated only by the order they were pushed — and they
//!   were pushed while iterating a `HashSet` of vertex neighbours. `control_grid` produced
//!   four distinct results in twelve runs of one process, `cocircular_lattice` five. Fixed
//!   by making `EdgeCandidate`'s ordering total.
//! - **Isotropic and anisotropic remeshing.** Three separate sites. The split pass sorted
//!   long edges by length alone, and since midpoint vertices are appended in that order the
//!   *indices* of every new vertex depended on hash order. The flip pass selected an
//!   independent set from candidates gathered in hash order. And `build_vertex_neighbors`
//!   returned unsorted adjacency, which `tangential_smooth` then *summed* to find a
//!   centroid — floating-point addition is not associative, so the smoothed position
//!   depended on the order too. That last one is also why `parallel` disagreed with
//!   sequential. All four remeshing runs of one input gave four different meshes.
//!
//! CVT remeshing, both subdivision schemes, and all three smoothers were already
//! deterministic; they are covered here so that stays true.
//!
//! Results are compared by floating-point **bit pattern**, not printed decimals: two
//! orderings can agree to fifteen digits and still differ.

mod common;

use std::collections::BTreeSet;

use common::corpus;
use morsel::algo::decimate::{qem_decimate, DecimateOptions, DecimateReport};
use morsel::algo::remesh::{
    anisotropic_remesh, average_edge_length, cvt_remesh, isotropic_remesh, AnisotropicOptions,
    CvtOptions, RemeshOptions,
};
use morsel::algo::smooth::{cotangent_smooth, laplacian_smooth, taubin_smooth, SmoothOptions};
use morsel::algo::subdivide::{catmull_clark_subdivide, loop_subdivide, SubdivideOptions};
use morsel::mesh::{to_face_vertex, HalfEdgeMesh};

/// Everything about the result that a caller could observe: the report, and the exact
/// mesh down to bit-identical coordinates.
fn observable(report: &DecimateReport, mesh: &HalfEdgeMesh) -> String {
    let (vertices, faces) = to_face_vertex(mesh);
    let mut out = format!(
        "{:?}/{}/{}/{} |",
        report.outcome, report.attempts, report.faces_requested, report.faces_after
    );
    for p in &vertices {
        // Bit patterns, not decimals: two collapse orders can agree to fifteen digits
        // and still differ.
        out.push_str(&format!(
            " {:x},{:x},{:x}",
            p.x.to_bits(),
            p.y.to_bits(),
            p.z.to_bits()
        ));
    }
    for f in &faces {
        out.push_str(&format!(" {}-{}-{}", f[0], f[1], f[2]));
    }
    out
}

fn decimate(mesh: &HalfEdgeMesh, ratio: f64, parallel: bool) -> String {
    let mut m = mesh.clone();
    let mut options = DecimateOptions::with_target_ratio(ratio);
    options.parallel = parallel;
    let report = qem_decimate(&mut m, &options);
    observable(&report, &m)
}

/// Repeated runs in one process must agree. This is the direction that catches hash-order
/// dependence, because Rust reseeds each `HashSet` instance from a per-thread counter, so
/// successive runs genuinely get different iteration orders.
#[test]
fn repeated_decimation_gives_identical_results() {
    for case in corpus() {
        let Ok(mesh) = case.mesh.as_ref() else {
            continue;
        };
        for ratio in [0.25, 0.5, 0.75] {
            let results: BTreeSet<String> = (0..12).map(|_| decimate(mesh, ratio, true)).collect();
            assert_eq!(
                results.len(),
                1,
                "{} at ratio {ratio}: {} distinct results in 12 runs",
                case.name,
                results.len()
            );
        }
    }
}

/// The same, on the bundled example meshes, which are large enough to have many
/// equal-cost collapses.
#[test]
fn repeated_decimation_is_identical_on_the_examples() {
    for name in [
        "sphere",
        "torus",
        "cylinder",
        "spherical-cap",
        "stanford-bunny",
    ] {
        let mesh: HalfEdgeMesh = morsel::io::load(format!("examples/{name}.obj")).expect("loads");
        let results: BTreeSet<String> = (0..6).map(|_| decimate(&mesh, 0.5, true)).collect();
        assert_eq!(
            results.len(),
            1,
            "{name}: {} distinct results",
            results.len()
        );
    }
}

/// Threading must not change the answer either. Candidate costs are computed in parallel,
/// and `collect` into a `Vec` preserves order, so the heap gets the same contents in the
/// same sequence — but that is a property worth pinning rather than assuming.
#[test]
fn parallel_and_sequential_decimation_agree() {
    for case in corpus() {
        let Ok(mesh) = case.mesh.as_ref() else {
            continue;
        };
        assert_eq!(
            decimate(mesh, 0.5, true),
            decimate(mesh, 0.5, false),
            "{}: parallel and sequential disagree",
            case.name
        );
    }

    for name in ["sphere", "torus", "stanford-bunny"] {
        let mesh: HalfEdgeMesh = morsel::io::load(format!("examples/{name}.obj")).expect("loads");
        assert_eq!(
            decimate(&mesh, 0.5, true),
            decimate(&mesh, 0.5, false),
            "{name}: parallel and sequential disagree"
        );
    }
}

/// Everything a caller could observe about a mutated mesh: connectivity, and coordinates
/// as bit patterns.
fn fingerprint(mesh: &HalfEdgeMesh) -> String {
    let (vertices, faces) = to_face_vertex(mesh);
    let mut out = format!("{}v {}f |", vertices.len(), faces.len());
    for p in &vertices {
        out.push_str(&format!(
            " {:x},{:x},{:x}",
            p.x.to_bits(),
            p.y.to_bits(),
            p.z.to_bits()
        ));
    }
    for f in &faces {
        out.push_str(&format!(" {}-{}-{}", f[0], f[1], f[2]));
    }
    out
}

fn example(name: &str) -> HalfEdgeMesh {
    morsel::io::load(format!("examples/{name}.obj")).expect("example mesh loads")
}

/// One mutating operation, parameterised on whether it runs threaded.
type Operation = fn(&HalfEdgeMesh, bool) -> HalfEdgeMesh;

fn operations() -> Vec<(&'static str, Operation)> {
    vec![
        ("remesh:isotropic", |m, parallel| {
            let mut c = m.clone();
            let mut o = RemeshOptions::with_target_length(average_edge_length(m));
            o.parallel = parallel;
            let _ = isotropic_remesh(&mut c, &o);
            c
        }),
        ("remesh:anisotropic", |m, parallel| {
            let mut c = m.clone();
            let t = average_edge_length(m);
            let mut o = AnisotropicOptions::new(0.5 * t, 2.0 * t);
            o.parallel = parallel;
            let _ = anisotropic_remesh(&mut c, &o);
            c
        }),
        ("remesh:cvt", |m, _parallel| {
            let mut c = m.clone();
            let _ = cvt_remesh(
                &mut c,
                &CvtOptions {
                    target_vertices: Some((m.num_vertices() * 2 / 3).max(3)),
                    ..Default::default()
                },
            );
            c
        }),
        ("subdivide:loop", |m, parallel| {
            let mut c = m.clone();
            let mut o = SubdivideOptions::new(1);
            o.parallel = parallel;
            let _ = loop_subdivide(&mut c, &o);
            c
        }),
        ("subdivide:cc", |m, parallel| {
            let mut c = m.clone();
            let mut o = SubdivideOptions::new(1);
            o.parallel = parallel;
            // Declines a triangle mesh, which is still a deterministic outcome.
            let _ = catmull_clark_subdivide(&mut c, &o);
            c
        }),
        ("smooth:laplacian", |m, parallel| {
            let mut c = m.clone();
            let o = SmoothOptions {
                parallel,
                ..Default::default()
            };
            laplacian_smooth(&mut c, &o);
            c
        }),
        ("smooth:taubin", |m, parallel| {
            let mut c = m.clone();
            let o = SmoothOptions {
                parallel,
                ..Default::default()
            };
            taubin_smooth(&mut c, &o);
            c
        }),
        ("smooth:cotangent", |m, parallel| {
            let mut c = m.clone();
            let o = SmoothOptions {
                parallel,
                ..Default::default()
            };
            cotangent_smooth(&mut c, &o);
            c
        }),
        ("decimate:qem", |m, parallel| {
            let mut c = m.clone();
            let mut o = DecimateOptions::with_target_ratio(0.5);
            o.parallel = parallel;
            let _ = qem_decimate(&mut c, &o);
            c
        }),
    ]
}

/// Repeated runs in one process must agree. This is the direction that catches hash-order
/// dependence: Rust seeds each `HashSet` and `HashMap` from a per-thread counter, so
/// successive runs genuinely get different iteration orders.
#[test]
fn every_mutating_algorithm_repeats_identically() {
    for name in ["sphere", "spherical-cap", "cylinder"] {
        let mesh = example(name);
        for (label, op) in operations() {
            let results: BTreeSet<String> = (0..4).map(|_| fingerprint(&op(&mesh, true))).collect();
            assert_eq!(
                results.len(),
                1,
                "{name} / {label}: {} distinct results in 4 runs",
                results.len()
            );
        }
    }
}

/// Threading must not change the answer either. This is the direction that catches
/// floating-point accumulation order — the reason `tangential_smooth` needed its adjacency
/// lists sorted rather than merely deduplicated.
#[test]
fn threading_does_not_change_the_answer() {
    for name in ["sphere", "spherical-cap", "cylinder"] {
        let mesh = example(name);
        for (label, op) in operations() {
            assert_eq!(
                fingerprint(&op(&mesh, true)),
                fingerprint(&op(&mesh, false)),
                "{name} / {label}: threaded and single-threaded disagree"
            );
        }
    }
}

/// The bunny is the mesh where these algorithms do the most work, so it gets its own
/// check. Only isotropic remeshing and decimation, which are the two that were actually
/// broken and the two most likely to be used; anisotropic is covered on the smaller meshes
/// above and takes twenty seconds a run here.
#[test]
fn the_bunny_remeshes_identically() {
    let mesh = example("stanford-bunny");
    for (label, op) in operations() {
        if label != "remesh:isotropic" && label != "decimate:qem" {
            continue;
        }
        let a = fingerprint(&op(&mesh, true));
        let b = fingerprint(&op(&mesh, true));
        let c = fingerprint(&op(&mesh, false));
        assert_eq!(a, b, "{label}: two threaded runs differ on the bunny");
        assert_eq!(a, c, "{label}: threaded and sequential differ on the bunny");
    }
}
