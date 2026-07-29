//! Round-trip tests for the mesh file formats.
//!
//! "Save it, load it back, and get the same mesh" is the property every format
//! has to satisfy, and none of the IO modules had a test for it. Silent
//! corruption on write or a mis-parsed index on read is invisible until someone
//! notices a mesh looks wrong.
//!
//! The formats do *not* all round-trip equally, and these tests encode the
//! differences rather than hiding them behind a loose tolerance:
//!
//! | format | positions | notes |
//! |--------|-----------|-------|
//! | OBJ    | exact     | needs `tobj`'s `use_f64` feature; without it the reader narrows to `f32` |
//! | PLY    | exact     | needs `property double` in the header, not `float` |
//! | STL    | `f32`     | inherent to the format; also a triangle soup, so vertices are re-welded on load |
//! | glTF   | load-only | saving is explicitly unsupported |
//!
//! The first two rows were *not* true when these tests were written — both
//! formats silently narrowed positions to `f32`, and writing the tests is what
//! surfaced it. STL's `f32` is the format's own specification and cannot be fixed
//! here.
//!
//! Every round trip also re-checks the half-edge invariants, since a format
//! reader is a place where a structurally broken mesh could be introduced.

use morsel::io;
use morsel::mesh::{build_from_triangles, HalfEdgeMesh};
use nalgebra::Point3;
use tempfile::TempDir;

// ---------------------------------------------------------------- fixtures

/// An open mesh with awkward, non-round coordinates, to catch precision loss.
fn open_patch() -> HalfEdgeMesh {
    let vertices = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.234_567_891_234_5, 0.0, 0.25),
        Point3::new(1.0, 0.987_654_321_098_7, -0.125),
        Point3::new(-0.333_333_333_333_33, 1.0, 0.5),
    ];
    build_from_triangles(&vertices, &[[0, 1, 2], [0, 2, 3]]).unwrap()
}

/// A closed mesh, so the topology assertions have a `χ = 2` case.
fn closed_tetra() -> HalfEdgeMesh {
    let vertices = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.5, 1.0, 0.0),
        Point3::new(0.5, 0.5, 1.0),
    ];
    build_from_triangles(&vertices, &[[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]]).unwrap()
}

/// A grid, for a case with many shared interior edges.
fn grid(n: usize) -> HalfEdgeMesh {
    let mut vertices = Vec::new();
    let mut faces = Vec::new();
    for j in 0..=n {
        for i in 0..=n {
            vertices.push(Point3::new(i as f64 * 0.5, j as f64 * 0.5, 0.0));
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

fn fixtures() -> Vec<(&'static str, HalfEdgeMesh)> {
    vec![
        ("open_patch", open_patch()),
        ("closed_tetra", closed_tetra()),
        ("grid(3)", grid(3)),
    ]
}

// ---------------------------------------------------------------- helpers

/// Re-check the half-edge invariants a reader could plausibly violate.
fn assert_structurally_sound(mesh: &HalfEdgeMesh, context: &str) {
    for he in mesh.halfedge_ids() {
        let t = mesh.twin(he);
        assert!(t.is_valid(), "{context}: {he:?} has an invalid twin");
        assert_eq!(mesh.twin(t), he, "{context}: twin is not an involution");
        assert_eq!(
            mesh.origin(t),
            mesh.dest(he),
            "{context}: twin does not reverse the edge"
        );
        assert_eq!(
            mesh.prev(mesh.next(he)),
            he,
            "{context}: next/prev are not inverses"
        );
        assert_eq!(
            mesh.dest(he),
            mesh.origin(mesh.next(he)),
            "{context}: half-edges do not chain"
        );
    }
    for f in mesh.face_ids() {
        let [a, b, c] = mesh.face_triangle(f);
        assert!(
            a != b && b != c && a != c,
            "{context}: {f:?} repeats a vertex"
        );
        assert!(mesh.face_area(f) > 0.0, "{context}: {f:?} is degenerate");
    }
}

/// Sorted position list, so meshes can be compared without assuming the reader
/// preserved vertex ordering.
fn sorted_positions(mesh: &HalfEdgeMesh) -> Vec<[f64; 3]> {
    let mut ps: Vec<[f64; 3]> = mesh
        .vertex_ids()
        .map(|v| {
            let p = mesh.position(v);
            [p.x, p.y, p.z]
        })
        .collect();
    ps.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ps
}

fn total_area(mesh: &HalfEdgeMesh) -> f64 {
    mesh.face_ids().map(|f| mesh.face_area(f)).sum()
}

/// Compare two meshes for equality up to `tol` on positions.
fn assert_round_tripped(original: &HalfEdgeMesh, loaded: &HalfEdgeMesh, tol: f64, context: &str) {
    assert_eq!(
        loaded.num_vertices(),
        original.num_vertices(),
        "{context}: vertex count changed"
    );
    assert_eq!(
        loaded.num_faces(),
        original.num_faces(),
        "{context}: face count changed"
    );
    assert_eq!(
        loaded.num_halfedges(),
        original.num_halfedges(),
        "{context}: half-edge count changed, so the topology differs"
    );

    let (before, after) = (sorted_positions(original), sorted_positions(loaded));
    for (b, a) in before.iter().zip(after.iter()) {
        for k in 0..3 {
            assert!(
                (b[k] - a[k]).abs() <= tol,
                "{context}: position drifted beyond {tol:e}: {b:?} vs {a:?}"
            );
        }
    }

    let (area_before, area_after) = (total_area(original), total_area(loaded));
    let rel = (area_before - area_after).abs() / area_before.max(1e-30);
    assert!(
        rel <= tol.max(1e-12) * 100.0,
        "{context}: total area changed by {rel:e} ({area_before} -> {area_after})"
    );

    let boundary_before = original
        .vertex_ids()
        .filter(|&v| original.is_boundary_vertex(v))
        .count();
    let boundary_after = loaded
        .vertex_ids()
        .filter(|&v| loaded.is_boundary_vertex(v))
        .count();
    assert_eq!(
        boundary_before, boundary_after,
        "{context}: boundary vertex count changed"
    );

    assert_structurally_sound(loaded, context);
}

// ---------------------------------------------------------------- tests

/// OBJ writes `f64` via `Display`, which is shortest-round-trippable, so the
/// round trip should be bit-exact.
#[test]
fn obj_round_trip_is_exact() {
    let dir = TempDir::new().unwrap();
    for (name, mesh) in fixtures() {
        let path = dir.path().join(format!("{name}.obj"));
        io::save(&mesh, &path).unwrap();
        let loaded: HalfEdgeMesh = io::load(&path).unwrap();

        assert_round_tripped(&mesh, &loaded, 0.0, &format!("obj/{name}"));
    }
}

/// PLY declares `property double` and writes full precision, so this round trip
/// should also be exact. It declared `float` until July 2026, which silently
/// halved every coordinate's precision on the way out.
#[test]
fn ply_round_trip_is_exact() {
    let dir = TempDir::new().unwrap();
    for (name, mesh) in fixtures() {
        let path = dir.path().join(format!("{name}.ply"));
        io::save(&mesh, &path).unwrap();
        let loaded: HalfEdgeMesh = io::load(&path).unwrap();

        assert_round_tripped(&mesh, &loaded, 0.0, &format!("ply/{name}"));
    }
}

/// STL is a triangle soup written as `f32`: every triangle carries its own three
/// vertices, so the loader has to re-weld them. Getting the same vertex count
/// back is the real assertion here.
#[test]
fn stl_round_trip_rewelds_to_the_same_topology() {
    let dir = TempDir::new().unwrap();
    for (name, mesh) in fixtures() {
        let path = dir.path().join(format!("{name}.stl"));
        io::save(&mesh, &path).unwrap();
        let loaded: HalfEdgeMesh = io::load(&path).unwrap();

        assert_round_tripped(&mesh, &loaded, 1e-6, &format!("stl/{name}"));
    }
}

/// UVs survive an OBJ round trip through `save_with_uvs` / `load_with_uvs`.
#[test]
fn obj_uv_round_trip_preserves_uvs() {
    use morsel::algo::parameterize::{cylindrical_projection, UVMap};

    let dir = TempDir::new().unwrap();
    let mesh = grid(3);
    let uvs: UVMap = cylindrical_projection(&mesh);

    let path = dir.path().join("uv.obj");
    morsel::io::obj::save_with_uvs(&mesh, &uvs, &path, Some("checker")).unwrap();

    let (loaded, loaded_uvs): (HalfEdgeMesh, Option<UVMap>) =
        morsel::io::obj::load_with_uvs(&path).unwrap();

    assert_round_tripped(&mesh, &loaded, 0.0, "obj-with-uvs");

    let loaded_uvs = loaded_uvs.expect("UVs should come back from a file that has `vt` entries");
    assert_eq!(
        loaded_uvs.len(),
        uvs.len(),
        "UV count should match the vertex count"
    );

    // Compare as sorted pairs: the reader need not preserve vertex order.
    let key = |m: &UVMap| {
        let mut v: Vec<[f64; 2]> = m.iter().map(|(_, uv)| [uv.x, uv.y]).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v
    };
    for (before, after) in key(&uvs).iter().zip(key(&loaded_uvs).iter()) {
        assert!(
            (before[0] - after[0]).abs() < 1e-9 && (before[1] - after[1]).abs() < 1e-9,
            "UV drifted: {before:?} vs {after:?}"
        );
    }
}

/// glTF is load-only. Saving must fail with an error rather than panicking or,
/// worse, writing a file that cannot be read back.
#[test]
fn gltf_save_is_rejected_cleanly() {
    let dir = TempDir::new().unwrap();
    let mesh = closed_tetra();

    for ext in ["gltf", "glb"] {
        let path = dir.path().join(format!("mesh.{ext}"));
        let err = io::save(&mesh, &path)
            .expect_err("glTF saving is unsupported and should report an error");
        let message = err.to_string();
        assert!(
            message.contains("glTF") || message.contains("not yet supported"),
            "error should explain that glTF saving is unsupported, got: {message}"
        );
        assert!(
            !path.exists(),
            "a rejected save should not leave a file behind at {path:?}"
        );
    }
}

/// An unknown extension is an error on both load and save, naming the extension.
#[test]
fn unknown_extensions_are_rejected() {
    let dir = TempDir::new().unwrap();
    let mesh = closed_tetra();

    let path = dir.path().join("mesh.xyzzy");
    let err = io::save(&mesh, &path).expect_err("unknown extension should not be saved");
    assert!(
        err.to_string().contains("xyzzy"),
        "error should name the offending extension, got: {err}"
    );

    std::fs::write(&path, b"not a mesh").unwrap();
    let err = io::load::<_, u32>(&path).expect_err("unknown extension should not be loaded");
    assert!(
        err.to_string().contains("xyzzy"),
        "error should name the offending extension, got: {err}"
    );
}

/// Two consecutive round trips must be a no-op relative to the first, so the
/// formats are stable under repeated processing rather than drifting.
#[test]
fn repeated_round_trips_are_stable() {
    let dir = TempDir::new().unwrap();
    for ext in ["obj", "ply", "stl"] {
        let first = dir.path().join(format!("pass1.{ext}"));
        let second = dir.path().join(format!("pass2.{ext}"));

        let mesh = open_patch();
        io::save(&mesh, &first).unwrap();
        let once: HalfEdgeMesh = io::load(&first).unwrap();
        io::save(&once, &second).unwrap();
        let twice: HalfEdgeMesh = io::load(&second).unwrap();

        // Whatever precision the first pass cost, the second must cost nothing:
        // the values written on pass two already survived a narrowing.
        assert_round_tripped(&once, &twice, 0.0, &format!("{ext} second pass"));
    }
}
