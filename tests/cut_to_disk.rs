//! Cutting non-disk meshes open, and checking they can then be flattened.
//!
//! The point of the cut generator is to make LSCM and ARAP applicable to meshes that
//! correctly refuse them, so these tests assert on the flattening as well as on the
//! topology.

use morsel::algo::cut::cut_to_disk;
use morsel::algo::parameterize::{arap, lscm, ARAPOptions, LSCMOptions};
use morsel::mesh::HalfEdgeMesh;

fn load(name: &str) -> HalfEdgeMesh {
    morsel::io::load(format!("examples/{name}.obj"))
        .unwrap_or_else(|e| panic!("failed to load {name}: {e}"))
}

/// Total area of all faces, which cutting must not change: it duplicates positions
/// and rewires corners but never moves a vertex or alters a triangle's shape.
fn surface_area(mesh: &HalfEdgeMesh) -> f64 {
    mesh.face_ids().map(|f| mesh.face_area(f)).sum()
}

/// Every mesh with a genus and no handles should come back a disk.
#[test]
fn genus_zero_meshes_cut_to_a_disk() {
    for name in ["cube-closed", "sphere", "cylinder", "stanford-bunny"] {
        let mesh = load(name);
        assert_ne!(
            mesh.boundary_loop_count(),
            1,
            "{name} is already a disk, so it does not exercise cutting"
        );

        let (cut, report) = cut_to_disk(&mesh).unwrap_or_else(|e| panic!("{name}: {e}"));

        assert!(report.is_disk(), "{name}: {report:?}");
        assert_eq!(cut.boundary_loop_count(), 1, "{name} is not a disk");
        assert_eq!(cut.genus(), Some(0), "{name} gained a handle");

        // chi = V - E + F = 1 characterises a disk.
        let chi = cut.num_vertices() as isize - (cut.num_halfedges() / 2) as isize
            + cut.num_faces() as isize;
        assert_eq!(chi, 1, "{name}: Euler characteristic {chi}, expected 1");

        // Cutting is pure surgery: no faces created or destroyed, no geometry moved.
        assert_eq!(
            cut.num_faces(),
            mesh.num_faces(),
            "{name} changed face count"
        );
        assert!(
            (surface_area(&cut) - surface_area(&mesh)).abs() < 1e-12 * surface_area(&mesh),
            "{name} changed surface area"
        );
        assert!(
            cut.num_vertices() > mesh.num_vertices(),
            "{name} duplicated no vertices"
        );
        assert_eq!(
            cut.num_vertices() - mesh.num_vertices(),
            report.vertices_added,
            "{name}: report disagrees with the mesh"
        );
    }
}

/// Each duplicate sits exactly on top of the vertex it came from, so the cut mesh is
/// the same shape — it is only differently connected.
#[test]
fn duplicated_vertices_keep_their_positions() {
    let mesh = load("cylinder");
    let (cut, _) = cut_to_disk(&mesh).unwrap();

    for v in cut.vertex_ids() {
        let p = cut.position(v);
        assert!(
            mesh.vertex_ids()
                .any(|w| (mesh.position(w) - p).norm() < 1e-15),
            "cut vertex {} is not at any original position",
            v.index()
        );
    }
}

/// The payoff: meshes that LSCM and ARAP refuse become meshes they accept.
#[test]
fn cut_meshes_can_be_flattened() {
    for name in ["sphere", "cylinder", "stanford-bunny"] {
        let mesh = load(name);
        assert!(
            lscm(&mesh, &LSCMOptions::default()).is_err(),
            "{name} should be refused before cutting"
        );

        let (cut, _) = cut_to_disk(&mesh).unwrap();

        for (label, uvs) in [
            ("lscm", lscm(&cut, &LSCMOptions::default())),
            ("arap", arap(&cut, &ARAPOptions::default())),
        ] {
            let uvs = uvs.unwrap_or_else(|e| panic!("{label} on cut {name}: {e}"));
            assert_eq!(uvs.len(), cut.num_vertices(), "{label} on {name}");
            assert!(
                uvs.iter().all(|(_, p)| p.x.is_finite() && p.y.is_finite()),
                "{label} on cut {name} produced non-finite UVs"
            );
            let (min, max) = uvs.bounding_box().expect("non-empty");
            assert!(
                (max.x - min.x) > 1e-6 && (max.y - min.y) > 1e-6,
                "{label} on cut {name} collapsed to a line: {min:?}..{max:?}"
            );
        }
    }
}

/// A mesh that is already a disk is returned untouched rather than cut anyway.
#[test]
fn a_disk_is_left_alone() {
    let mesh = load("spherical-cap");
    let (cut, report) = cut_to_disk(&mesh).unwrap();

    assert_eq!(report.paths_cut, 0);
    assert_eq!(report.vertices_added, 0);
    assert_eq!(cut.num_vertices(), mesh.num_vertices());
    assert_eq!(cut.num_faces(), mesh.num_faces());
}

/// Handles need a different algorithm, so a torus is refused with a clear reason
/// rather than mangled into something that merely looks flattenable.
#[test]
fn a_torus_is_refused() {
    let mesh = load("torus");
    assert_eq!(mesh.genus(), Some(1));

    let err = cut_to_disk(&mesh).unwrap_err().to_string();
    assert!(err.contains("genus-1"), "unhelpful message: {err}");
}

/// The genus is a sum over components, so a disconnected mesh has to be split first.
#[test]
fn a_disconnected_mesh_is_refused() {
    let mesh = load("cube"); // six detached quads
    assert_eq!(mesh.genus(), None);

    let err = cut_to_disk(&mesh).unwrap_err().to_string();
    assert!(err.contains("disconnected"), "unhelpful message: {err}");
}
