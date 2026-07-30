//! QEM decimation gives the same answer every time.
//!
//! It did not. The heap of collapse candidates ordered them on cost alone, so equal-cost
//! candidates were separated only by the order they happened to be pushed — and they were
//! pushed while iterating a `HashSet` of vertex neighbours, whose iteration order Rust
//! randomises per `HashSet` instance. The collapse sequence therefore varied, and with it
//! the output mesh: `control_grid` produced four distinct results in twelve runs of one
//! process, `cocircular_lattice` five.
//!
//! This was visible from two directions. `tests/robustness_sweep.rs` could not pin the
//! `decimate:qem` column, because whether the back-off fired changed between runs; and a
//! caller had no way to reproduce a result they had just seen.

mod common;

use std::collections::BTreeSet;

use common::corpus;
use morsel::algo::decimate::{qem_decimate, DecimateOptions, DecimateReport};
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
