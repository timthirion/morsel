//! Every mesh-mutating algorithm reports whether it actually did the work.
//!
//! These algorithms all operate by converting to face-vertex arrays, transforming, and
//! rebuilding through `build_from_triangles`. When the rebuild is rejected — a repeated
//! directed edge, an edge with three faces, a bowtie vertex — the mesh is left as it was.
//! That used to happen in silence: nine call sites discarded the rebuild's `Result`, so a
//! caller saw an unchanged, perfectly valid mesh and no indication that nothing had
//! happened. Worse, the failure only became *reachable* when `build_from_triangles`
//! started validating; before that the same calls returned a corrupt mesh.
//!
//! So the property under test is not "the algorithm succeeds" — often it cannot — but
//! "the algorithm's report matches what it did".

use morsel::algo::decimate::{qem_decimate, DecimateOptions, DecimateOutcome};
use morsel::algo::remesh::{
    anisotropic_remesh, average_edge_length, cvt_remesh, isotropic_remesh, AnisotropicOptions,
    CvtOptions, RemeshOptions,
};
use morsel::algo::subdivide::{
    catmull_clark_subdivide, loop_subdivide, SubdivideOptions, SubdivideOutcome,
};
use morsel::mesh::HalfEdgeMesh;

fn load(name: &str) -> HalfEdgeMesh {
    morsel::io::load(format!("examples/{name}.obj")).expect("example mesh loads")
}

/// The counts in a report have to describe the mesh the caller is holding, or the report
/// is worse than nothing.
#[test]
fn reported_face_counts_match_the_mesh() {
    for name in ["sphere", "spherical-cap", "torus"] {
        let mesh = load(name);
        let before = mesh.num_faces();
        let target = average_edge_length(&mesh);

        let mut m = mesh.clone();
        let r = isotropic_remesh(&mut m, &RemeshOptions::with_target_length(target));
        assert_eq!(r.faces_before, before, "{name} isotropic");
        assert_eq!(r.faces_after, m.num_faces(), "{name} isotropic");

        let mut m = mesh.clone();
        let r = anisotropic_remesh(&mut m, &AnisotropicOptions::new(0.5 * target, 2.0 * target));
        assert_eq!(r.faces_before, before, "{name} anisotropic");
        assert_eq!(r.faces_after, m.num_faces(), "{name} anisotropic");

        let mut m = mesh.clone();
        let r = loop_subdivide(&mut m, &SubdivideOptions::new(1));
        assert_eq!(r.faces_before, before, "{name} loop");
        assert_eq!(r.faces_after, m.num_faces(), "{name} loop");

        let mut m = mesh.clone();
        let r = qem_decimate(&mut m, &DecimateOptions::with_target_ratio(0.5));
        assert_eq!(r.faces_before, before, "{name} qem");
        assert_eq!(r.faces_after, m.num_faces(), "{name} qem");
    }
}

/// Loop subdivision quadruples the face count per level, and says it completed.
#[test]
fn loop_subdivision_reports_completing() {
    let mesh = load("sphere");
    let mut m = mesh.clone();

    let report = loop_subdivide(&mut m, &SubdivideOptions::new(2));

    assert_eq!(report.outcome, SubdivideOutcome::Completed);
    assert!(report.completed());
    assert_eq!(report.iterations_run, 2);
    assert_eq!(
        report.faces_after,
        mesh.num_faces() * 16,
        "two levels of 4x"
    );
}

/// Asking for nothing is distinguishable from failing.
#[test]
fn zero_iterations_reports_nothing_requested() {
    let mesh = load("sphere");

    let mut m = mesh.clone();
    let report = loop_subdivide(&mut m, &SubdivideOptions::new(0));
    assert_eq!(report.outcome, SubdivideOutcome::NothingRequested);
    assert!(!report.completed());
    assert_eq!(report.iterations_run, 0);
    assert_eq!(m.num_faces(), mesh.num_faces());
}

/// Catmull-Clark is a quad scheme. Handed triangles it declines — and now names the
/// reason instead of returning the input as though it had subdivided it.
///
/// This was thirteen `ok` cells in the robustness sweep for an algorithm that had never
/// once run.
#[test]
fn catmull_clark_reports_declining_a_triangle_mesh() {
    let mesh = load("sphere");
    assert!(!mesh.is_quad_mesh());

    let mut m = mesh.clone();
    let report = catmull_clark_subdivide(&mut m, &SubdivideOptions::new(1));

    assert_eq!(report.outcome, SubdivideOutcome::NotAQuadMesh);
    assert!(!report.completed());
    assert_eq!(report.iterations_run, 0);
    assert_eq!(
        m.num_faces(),
        mesh.num_faces(),
        "the mesh must be left alone, not partly subdivided"
    );
}

/// QEM's back-off is the case worth reporting: when the requested target produces a
/// non-manifold mesh it retries more mildly, so a caller asking to halve the mesh can be
/// handed a quarter of the reduction. That was previously invisible.
///
/// The outcome varies with collapse order, so this asserts the report is *consistent*
/// with the mesh rather than pinning which branch was taken.
#[test]
fn decimation_reports_backing_off_or_refusing() {
    for name in ["sphere", "stanford-bunny", "torus"] {
        let mesh = load(name);
        let mut m = mesh.clone();
        let report = qem_decimate(&mut m, &DecimateOptions::with_target_ratio(0.5));

        match report.outcome {
            DecimateOutcome::Completed => {
                assert!(report.attempts >= 1, "{name}");
                assert!(
                    m.num_faces() <= report.faces_requested,
                    "{name}: claimed to reach {} faces but has {}",
                    report.faces_requested,
                    m.num_faces()
                );
            }
            DecimateOutcome::BackedOff => {
                assert!(report.attempts > 1, "{name}: backed off without retrying");
                assert!(
                    m.num_faces() > report.faces_requested,
                    "{name}: backing off should leave more faces than requested"
                );
                assert!(
                    m.num_faces() < mesh.num_faces(),
                    "{name}: backing off should still have reduced something"
                );
            }
            DecimateOutcome::Refused => {
                assert_eq!(
                    m.num_faces(),
                    mesh.num_faces(),
                    "{name}: refusing must leave the mesh untouched"
                );
            }
            DecimateOutcome::NothingRequested => {
                assert_eq!(m.num_faces(), mesh.num_faces(), "{name}");
            }
        }
    }
}

/// A target at or above the input vertex count leaves CVT with nothing to move, because
/// each Voronoi cell holds a single vertex whose centroid is that vertex.
///
/// Note what `converged` does and does not mean here: the rebuild is *accepted*, so it
/// reports `true` while changing nothing of substance. The flag is about whether the
/// operation completed, not about whether it was worth doing — the degenerate target is
/// a separate problem, and the CLI refuses it up front.
#[test]
fn cvt_reports_completing_even_when_its_target_is_degenerate() {
    let mesh = load("sphere");
    let mut m = mesh.clone();

    let report = cvt_remesh(&mut m, &CvtOptions::default());

    assert!(report.converged, "the rebuild is accepted; nothing failed");
    assert_eq!(
        report.faces_after, report.faces_before,
        "and nothing of substance changed, which is the actual complaint"
    );
}

/// When a remesh reports not converging, it must not also claim to have run everything.
#[test]
fn a_remesh_that_stops_early_says_how_far_it_got() {
    let mesh = load("spherical-cap");
    let target = average_edge_length(&mesh);
    let requested = 5;

    let mut m = mesh.clone();
    let report = anisotropic_remesh(
        &mut m,
        &AnisotropicOptions::new(0.5 * target, 2.0 * target).with_iterations(requested),
    );

    // This mesh is the one whose split pass would not terminate before the passes were
    // bounded, so it is the case that most needs the report to be truthful.
    assert!(
        !report.converged,
        "spherical-cap is the recorded early-stop case"
    );
    assert!(report.iterations_run >= 1);
    assert!(
        report.iterations_run < requested,
        "stopped early but claims {} of {requested} iterations",
        report.iterations_run
    );
    assert!(m.num_faces() > 0, "the mesh must still be usable");
}
