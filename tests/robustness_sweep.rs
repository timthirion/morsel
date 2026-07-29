//! Every algorithm against every awkward input, with the outcome recorded.
//!
//! The question is not whether an algorithm handles bad input *well* — often it
//! cannot — but whether it **reports** that it couldn't. Three outcomes matter,
//! in increasing severity:
//!
//! - `Refused` — returned an error. Correct behaviour on input outside an
//!   algorithm's assumptions.
//! - `Panicked` — a bug. A library should not abort its caller's process.
//! - `Corrupted` — returned success, having broken a mesh that was *valid going
//!   in*: non-finite positions, a broken half-edge invariant, NaNs in a field.
//!   The worst case, because nothing downstream will notice.
//!
//! Crucially, `Corrupted` is only attributed to an algorithm when its input was
//! sound. Five corpus meshes are already structurally invalid the moment
//! `build_from_triangles` returns `Ok` on them, and results on those are recorded
//! as `Inherited` — they say nothing about the algorithm. Without that
//! distinction the table blames the wrong code: vertex smoothing appeared to break
//! half-edge twins, which it cannot possibly do, because it never touches
//! connectivity.
//!
//! This is a **characterization test**: the tables below record what the library
//! does today, and the test asserts reality still matches. So a new panic or a new
//! corruption fails CI, and *fixing* one requires deliberately updating the
//! record. The tables are also the point — they are the robustness baseline, and
//! every non-`Ok` entry is a candidate piece of work.
//!
//! Run with `--nocapture` to see the full matrix printed.
//!
//! See `plans/0002-research-program.md`.

mod common;

use std::collections::BTreeMap;
use std::panic::{catch_unwind, AssertUnwindSafe};

use common::{corpus, field_defect, structural_defect, Difficulty};
use morsel::algo::{curvature, decimate, geodesic, parameterize, remesh, smooth, subdivide};
use morsel::mesh::{HalfEdgeMesh, VertexId};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Outcome {
    /// Succeeded with structurally sound output.
    Ok,
    /// Returned an error, or declined for a stated reason. Acceptable.
    Refused,
    /// Panicked. A bug regardless of how bad the input was.
    Panicked,
    /// Reported success but broke a mesh that was valid going in.
    Corrupted,
    /// The input was already invalid, so this says nothing about the algorithm.
    Inherited,
}

impl Outcome {
    fn tag(self) -> &'static str {
        match self {
            Outcome::Ok => "ok",
            Outcome::Refused => "refused",
            Outcome::Panicked => "PANIC",
            Outcome::Corrupted => "CORRUPT",
            Outcome::Inherited => "(inh)",
        }
    }
}

/// Run one algorithm, converting a panic into an outcome rather than an abort.
fn run<F>(f: F) -> (Outcome, String)
where
    F: FnOnce() -> std::result::Result<(), String>,
{
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => (Outcome::Ok, String::new()),
        Ok(Err(why)) => {
            // Distinguish "the algorithm said no" from "the algorithm said yes
            // and lied": the closures below prefix the latter with `broken:`.
            if let Some(detail) = why.strip_prefix("broken:") {
                (Outcome::Corrupted, detail.trim().to_string())
            } else {
                (Outcome::Refused, why)
            }
        }
        Err(payload) => {
            let msg = payload
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "<non-string panic>".into());
            (
                Outcome::Panicked,
                msg.lines().next().unwrap_or("").to_string(),
            )
        }
    }
}

/// Check a mutated mesh and report breakage in the form `run` understands.
fn check_mesh(mesh: &HalfEdgeMesh) -> std::result::Result<(), String> {
    match structural_defect(mesh) {
        Some(d) => Err(format!("broken: {d}")),
        None => Ok(()),
    }
}

fn check_field(values: &[f64], label: &str) -> std::result::Result<(), String> {
    match field_defect(values, label) {
        Some(d) => Err(format!("broken: {d}")),
        None => Ok(()),
    }
}

/// One algorithm under test: a label and a probe that runs it against a mesh.
type Probe = (&'static str, fn(&HalfEdgeMesh) -> (Outcome, String));

/// Every algorithm under test. Mutating algorithms clone first, so one failure
/// cannot poison the next.
fn algorithms() -> Vec<Probe> {
    vec![
        ("smooth:laplacian", |m| {
            run(|| {
                let mut m = m.clone();
                smooth::laplacian_smooth(&mut m, &smooth::SmoothOptions::default());
                check_mesh(&m)
            })
        }),
        ("smooth:taubin", |m| {
            run(|| {
                let mut m = m.clone();
                smooth::taubin_smooth(&mut m, &smooth::SmoothOptions::default());
                check_mesh(&m)
            })
        }),
        ("smooth:cotangent", |m| {
            run(|| {
                let mut m = m.clone();
                smooth::cotangent_smooth(&mut m, &smooth::SmoothOptions::default());
                check_mesh(&m)
            })
        }),
        ("subdivide:loop", |m| {
            run(|| {
                let mut m = m.clone();
                subdivide::loop_subdivide(&mut m, &subdivide::SubdivideOptions::new(1));
                check_mesh(&m)
            })
        }),
        ("decimate:qem", |m| {
            run(|| {
                let mut m = m.clone();
                decimate::qem_decimate(&mut m, &decimate::DecimateOptions::with_target_ratio(0.5));
                check_mesh(&m)
            })
        }),
        ("remesh:isotropic", |m| {
            run(|| {
                let mut m = m.clone();
                // Target the mesh's own average edge length so the request is
                // scale-appropriate rather than arbitrary.
                let target = remesh::average_edge_length(&m);
                remesh::isotropic_remesh(
                    &mut m,
                    &remesh::RemeshOptions::with_target_length(target),
                );
                check_mesh(&m)
            })
        }),
        ("curvature:gaussian", |m| {
            run(|| check_field(&curvature::gaussian_curvature(m), "gaussian"))
        }),
        ("curvature:mean", |m| {
            run(|| check_field(&curvature::mean_curvature(m), "mean"))
        }),
        ("geodesic:dijkstra", |m| {
            run(|| {
                let r =
                    geodesic::dijkstra(m, VertexId::new(0), &geodesic::DijkstraOptions::default());
                // Unreachable vertices legitimately carry infinity; only NaN is
                // a defect, so filter before checking.
                let finite: Vec<f64> = r
                    .distances()
                    .iter()
                    .copied()
                    .filter(|d| !d.is_infinite())
                    .collect();
                check_field(&finite, "dijkstra")
            })
        }),
        ("geodesic:heat", |m| {
            run(|| {
                let r = geodesic::heat_method(
                    m,
                    VertexId::new(0),
                    &geodesic::HeatMethodOptions::default(),
                )
                .map_err(|e| e.to_string())?;
                check_field(r.distances(), "heat")
            })
        }),
        ("param:cylindrical", |m| {
            run(|| {
                let uvs = parameterize::cylindrical_projection(m);
                let vals: Vec<f64> = uvs.iter().flat_map(|(_, uv)| [uv.x, uv.y]).collect();
                check_field(&vals, "uv")
            })
        }),
        ("param:lscm", |m| {
            run(|| {
                let uvs = parameterize::lscm(m, &parameterize::LSCMOptions::default())
                    .map_err(|e| e.to_string())?;
                let vals: Vec<f64> = uvs.iter().flat_map(|(_, uv)| [uv.x, uv.y]).collect();
                check_field(&vals, "uv")
            })
        }),
        ("param:arap", |m| {
            run(|| {
                let uvs = parameterize::arap(m, &parameterize::ARAPOptions::default())
                    .map_err(|e| e.to_string())?;
                let vals: Vec<f64> = uvs.iter().flat_map(|(_, uv)| [uv.x, uv.y]).collect();
                check_field(&vals, "uv")
            })
        }),
        ("param:omt", |m| {
            run(|| {
                let base = parameterize::lscm(m, &parameterize::LSCMOptions::default())
                    .map_err(|e| e.to_string())?;
                let uvs = parameterize::omt(m, &base, &parameterize::OMTOptions::fast())
                    .map_err(|e| e.to_string())?;
                let vals: Vec<f64> = uvs.iter().flat_map(|(_, uv)| [uv.x, uv.y]).collect();
                check_field(&vals, "uv")
            })
        }),
    ]
}

/// Silence panic output so the matrix stays readable; panics are still caught.
fn with_quiet_panics<T>(f: impl FnOnce() -> T) -> T {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let out = f();
    std::panic::set_hook(previous);
    out
}

/// Meshes that `build_from_triangles` accepts while producing a structurally
/// invalid half-edge mesh. **This is the library's most consequential defect**:
/// it is the entry point, it reports success, and every algorithm downstream
/// inherits the corruption. Fixing it — validate and reject, or repair — would
/// clear most of the matrix at a stroke.
const BORN_BROKEN: &[&str] = &[
    "duplicate_face",
    "duplicate_vertices",
    "inconsistent_winding",
    "nonmanifold_edge",
    "nonmanifold_vertex",
];

/// Pairs whose outcome **varies between runs of the same binary**. QEM decimation
/// is order-dependent and the library uses `std::collections::HashMap`, whose
/// iteration order is randomised per process — so the collapse sequence differs
/// each run and only some sequences corrupt the mesh. Measured at 3 failures in 8
/// runs on `control_grid`, identically with `parallel` true and false, so the
/// parallelism is not the cause.
///
/// For these the check accepts either `Ok` or `Corrupted`; a stable baseline
/// cannot be had until the ordering is made deterministic.
const NONDETERMINISTIC: &[(&str, &str)] = &[
    ("annulus_two_boundary_loops", "decimate:qem"),
    ("cocircular_lattice", "decimate:qem"),
    ("control_grid", "decimate:qem"),
    ("high_valence_fan", "decimate:qem"),
    ("huge_scale", "decimate:qem"),
    ("tiny_scale", "decimate:qem"),
];

/// Recorded per-algorithm behaviour on inputs that *were* valid. Anything absent
/// is expected to be `Ok`. `Refused` is acceptable; `Panicked` and `Corrupted`
/// are debt.
///
/// Regenerate by running with `--nocapture` and pasting the printed block.
const BASELINE: &[(&str, &str, Outcome)] = &[
    // Panics on structurally invalid input. All eight dereference the invalid
    // index sentinel (u32::MAX = 4294967295) without checking validity first.
    // A library should refuse, not abort its caller.
    ("duplicate_face", "remesh:isotropic", Outcome::Panicked),
    (
        "inconsistent_winding",
        "remesh:isotropic",
        Outcome::Panicked,
    ),
    ("inconsistent_winding", "subdivide:loop", Outcome::Panicked),
    ("nonmanifold_edge", "curvature:gaussian", Outcome::Panicked),
    ("nonmanifold_edge", "curvature:mean", Outcome::Panicked),
    ("nonmanifold_edge", "geodesic:dijkstra", Outcome::Panicked),
    ("nonmanifold_edge", "geodesic:heat", Outcome::Panicked),
    ("nonmanifold_edge", "remesh:isotropic", Outcome::Panicked),
    // Absolute thresholds rather than scale-relative ones.
    ("tiny_scale", "geodesic:heat", Outcome::Refused),
    ("tiny_scale", "param:omt", Outcome::Refused),
    // A zero-area face leaves the cotangent Laplacian undefined; refusing is right.
    ("zero_area_face", "geodesic:heat", Outcome::Refused),
    // No boundary, so the disk-topology methods correctly decline.
    ("control_tetrahedron", "param:arap", Outcome::Refused),
    ("control_tetrahedron", "param:lscm", Outcome::Refused),
    ("control_tetrahedron", "param:omt", Outcome::Refused),
];

fn baseline() -> BTreeMap<(&'static str, &'static str), Outcome> {
    let mut b = BTreeMap::new();
    for &(mesh, algo, outcome) in BASELINE {
        b.insert((mesh, algo), outcome);
    }
    b
}

#[test]
fn robustness_matrix_matches_baseline() {
    let cases = corpus();
    let algos = algorithms();
    let expected = baseline();

    let mut observed: BTreeMap<(&'static str, &'static str), (Outcome, String)> = BTreeMap::new();
    let mut rejected: Vec<(&'static str, String)> = Vec::new();
    let mut born_broken: Vec<(&'static str, String)> = Vec::new();

    for c in &cases {
        let mesh = match &c.mesh {
            Err(e) => {
                rejected.push((c.name, e.to_string()));
                continue;
            }
            Ok(m) => m,
        };

        // Attribute honestly: if the input is already invalid, nothing an
        // algorithm reports about it is the algorithm's fault.
        let input_defect = structural_defect(mesh);
        if let Some(d) = &input_defect {
            born_broken.push((c.name, d.clone()));
        }

        for (algo_name, algo) in &algos {
            let (outcome, detail) = with_quiet_panics(|| algo(mesh));
            let attributed = match (&input_defect, outcome) {
                // A panic is a bug however bad the input.
                (_, Outcome::Panicked) => Outcome::Panicked,
                (Some(_), _) => Outcome::Inherited,
                (None, o) => o,
            };
            observed.insert((c.name, algo_name), (attributed, detail));
        }
    }

    // ---- report ---------------------------------------------------------
    println!("\n=== corpus ===");
    for c in &cases {
        let status = match &c.mesh {
            Ok(m) => format!("{}v {}f", m.num_vertices(), m.num_faces()),
            Err(_) => "REJECTED".to_string(),
        };
        println!(
            "  {:<28} {:<12} {:<10} {}",
            c.name,
            format!("{:?}", c.difficulty),
            status,
            c.note
        );
    }

    if !rejected.is_empty() {
        println!("\n=== rejected at construction (a valid outcome) ===");
        for (name, why) in &rejected {
            println!("  {name:<28} {why}");
        }
    }

    if !born_broken.is_empty() {
        println!(
            "\n=== accepted by build_from_triangles but structurally invalid ({}) ===",
            born_broken.len()
        );
        for (name, why) in &born_broken {
            println!("  {name:<28} {why}");
        }
        println!("  ^ the root cause behind most `(inh)` cells below");
    }

    println!("\n=== matrix ===");
    let algo_names: Vec<&str> = algos.iter().map(|(n, _)| *n).collect();
    print!("{:<28}", "mesh");
    for a in &algo_names {
        print!(" {:>9}", a.split(':').next_back().unwrap_or(a));
    }
    println!();
    for c in &cases {
        if c.mesh.is_err() {
            continue;
        }
        print!("{:<28}", c.name);
        for a in &algo_names {
            let tag = observed
                .get(&(c.name, *a))
                .map(|(o, _)| o.tag())
                .unwrap_or("?");
            print!(" {tag:>9}");
        }
        println!();
    }

    let collect = |want: Outcome| -> Vec<(&str, &str, String)> {
        observed
            .iter()
            .filter(|(_, (o, _))| *o == want)
            .map(|((m, a), (_, d))| (*m, *a, d.clone()))
            .collect()
    };
    let panics = collect(Outcome::Panicked);
    let corrupts = collect(Outcome::Corrupted);

    if !panics.is_empty() {
        println!(
            "\n=== panics ({}) — bugs regardless of input ===",
            panics.len()
        );
        for (m, a, d) in &panics {
            println!("  {m:<28} {a:<20} {d}");
        }
    }
    if !corrupts.is_empty() {
        println!(
            "\n=== corruption of valid input ({}) — the real algorithm bugs ===",
            corrupts.len()
        );
        for (m, a, d) in &corrupts {
            println!("  {m:<28} {a:<20} {d}");
        }
    }

    // ---- compare against the record -------------------------------------
    let mut drift = Vec::new();

    let mut broken_names: Vec<&str> = born_broken.iter().map(|(n, _)| *n).collect();
    broken_names.sort_unstable();
    let mut want_broken = BORN_BROKEN.to_vec();
    want_broken.sort_unstable();
    if broken_names != want_broken {
        drift.push(format!(
            "build_from_triangles validity changed: recorded {want_broken:?}, observed {broken_names:?}"
        ));
    }

    for ((mesh, algo), (outcome, detail)) in &observed {
        // `Inherited` is a consequence of BORN_BROKEN, already checked above.
        if *outcome == Outcome::Inherited {
            continue;
        }
        // Known order-dependent pairs may land either way within one run.
        if NONDETERMINISTIC.contains(&(*mesh, *algo))
            && matches!(outcome, Outcome::Ok | Outcome::Corrupted)
        {
            continue;
        }
        let want = expected
            .get(&(*mesh, *algo))
            .copied()
            .unwrap_or(Outcome::Ok);
        if *outcome != want {
            drift.push(format!(
                "{mesh}/{algo}: expected {want:?}, got {outcome:?}  {detail}"
            ));
        }
    }

    if !drift.is_empty() {
        println!("\n=== BASELINE ENTRIES (paste into BASELINE to accept) ===");
        let mut rows: Vec<_> = observed
            .iter()
            .filter(|(_, (o, _))| !matches!(o, Outcome::Ok | Outcome::Inherited))
            .collect();
        rows.sort();
        for ((mesh, algo), (o, _)) in rows {
            println!("    (\"{mesh}\", \"{algo}\", Outcome::{o:?}),");
        }

        println!("\n=== drift from record ({}) ===", drift.len());
        for d in &drift {
            println!("  {d}");
        }
        panic!(
            "{} deviations from the recorded baseline; see above",
            drift.len()
        );
    }
}

/// Whatever the corpus does to the awkward cases, the controls must be clean.
/// If an algorithm fails here it is not the input's fault.
#[test]
fn controls_are_handled_cleanly() {
    let cases = corpus();
    let algos = algorithms();

    let mut failures = Vec::new();
    for c in cases.iter().filter(|c| c.difficulty == Difficulty::Control) {
        let mesh = c
            .mesh
            .as_ref()
            .unwrap_or_else(|e| panic!("control mesh {} should build: {e}", c.name));
        for (algo_name, algo) in &algos {
            let (outcome, detail) = with_quiet_panics(|| algo(mesh));
            let known_flaky = NONDETERMINISTIC.contains(&(c.name, *algo_name));
            if matches!(outcome, Outcome::Panicked)
                || (matches!(outcome, Outcome::Corrupted) && !known_flaky)
            {
                failures.push(format!(
                    "{}/{}: {:?} — {}",
                    c.name, algo_name, outcome, detail
                ));
            }
        }
    }

    assert!(
        failures.is_empty(),
        "algorithms must not panic or silently corrupt on well-formed input:\n  {}",
        failures.join("\n  ")
    );
}
