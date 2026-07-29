//! Structural invariants of [`HalfEdgeMesh`].
//!
//! The half-edge structure is what every algorithm in the crate traverses, and
//! it is the least directly tested part of the codebase. These tests check the
//! invariants that make traversal meaningful — `twin` is an involution, `next`
//! and `prev` are mutual inverses, face cycles close, boundary predicates agree
//! with the face pointers — rather than checking any one algorithm's output.
//!
//! Everything here is a *universal* property, asserted over every half-edge or
//! face of every fixture. That makes the fixture list the only thing to grow
//! when a new topology turns out to matter: add the mesh, and every invariant is
//! checked against it.
//!
//! These are also the properties `plans/0001` earmarks for machine-checked
//! proof: they are pure index arithmetic, so they sit in the subset that
//! Charon/Aeneas can extract into Lean.

use morsel::mesh::{build_from_triangles, HalfEdgeMesh};
use nalgebra::Point3;

// ---------------------------------------------------------------- fixtures

/// A single triangle: the smallest mesh with a boundary.
fn single_triangle() -> HalfEdgeMesh {
    let vertices = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.5, 1.0, 0.0),
    ];
    build_from_triangles(&vertices, &[[0, 1, 2]]).unwrap()
}

/// Two triangles sharing an edge — the smallest mesh with an interior edge.
fn two_triangles() -> HalfEdgeMesh {
    let vertices = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(1.0, 1.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
    ];
    build_from_triangles(&vertices, &[[0, 1, 2], [0, 2, 3]]).unwrap()
}

/// A fan disk: one interior vertex ringed by boundary vertices.
fn fan_disk(spokes: usize) -> HalfEdgeMesh {
    let mut vertices = vec![Point3::new(0.0, 0.0, 0.0)];
    for k in 0..spokes {
        let t = 2.0 * std::f64::consts::PI * (k as f64) / (spokes as f64);
        vertices.push(Point3::new(t.cos(), t.sin(), 0.0));
    }
    let faces: Vec<[usize; 3]> = (0..spokes)
        .map(|k| [0, 1 + k, 1 + (k + 1) % spokes])
        .collect();
    build_from_triangles(&vertices, &faces).unwrap()
}

/// A flat `n × n` grid: interior vertices of valence 6, a rectangular boundary.
fn grid(n: usize) -> HalfEdgeMesh {
    let mut vertices = Vec::new();
    let mut faces = Vec::new();
    for j in 0..=n {
        for i in 0..=n {
            vertices.push(Point3::new(i as f64, j as f64, 0.0));
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

/// A closed tetrahedron: no boundary at all.
fn tetrahedron() -> HalfEdgeMesh {
    let vertices = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.5, 1.0, 0.0),
        Point3::new(0.5, 0.5, 1.0),
    ];
    build_from_triangles(&vertices, &[[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]]).unwrap()
}

/// A closed octahedron: a second closed case, all valences equal.
fn octahedron() -> HalfEdgeMesh {
    let vertices = vec![
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(-1.0, 0.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
        Point3::new(0.0, -1.0, 0.0),
        Point3::new(0.0, 0.0, 1.0),
        Point3::new(0.0, 0.0, -1.0),
    ];
    let faces = [
        [0, 2, 4],
        [2, 1, 4],
        [1, 3, 4],
        [3, 0, 4],
        [2, 0, 5],
        [1, 2, 5],
        [3, 1, 5],
        [0, 3, 5],
    ];
    build_from_triangles(&vertices, &faces).unwrap()
}

/// Every fixture, with a label for assertion messages, and whether it is closed.
fn fixtures() -> Vec<(&'static str, HalfEdgeMesh, bool)> {
    vec![
        ("single_triangle", single_triangle(), false),
        ("two_triangles", two_triangles(), false),
        ("fan_disk(3)", fan_disk(3), false),
        ("fan_disk(6)", fan_disk(6), false),
        ("fan_disk(12)", fan_disk(12), false),
        ("grid(1)", grid(1), false),
        ("grid(4)", grid(4), false),
        ("grid(7)", grid(7), false),
        ("tetrahedron", tetrahedron(), true),
        ("octahedron", octahedron(), true),
    ]
}

// ---------------------------------------------------------------- invariants

/// `twin` is an involution with no fixed points, and it reverses the edge.
#[test]
fn twin_is_a_fixed_point_free_involution() {
    for (name, mesh, _) in fixtures() {
        for he in mesh.halfedge_ids() {
            let t = mesh.twin(he);
            assert!(t.is_valid(), "{name}: {he:?} has an invalid twin");
            assert_ne!(t, he, "{name}: {he:?} is its own twin");
            assert_eq!(
                mesh.twin(t),
                he,
                "{name}: twin(twin({he:?})) should be {he:?}"
            );
            assert_eq!(
                mesh.origin(t),
                mesh.dest(he),
                "{name}: twin should reverse {he:?}"
            );
            assert_eq!(
                mesh.dest(t),
                mesh.origin(he),
                "{name}: twin should reverse {he:?}"
            );
        }
    }
}

/// `next` and `prev` are mutual inverses, and every half-edge chains onto the
/// next one head-to-tail.
#[test]
fn next_and_prev_are_mutual_inverses() {
    for (name, mesh, _) in fixtures() {
        for he in mesh.halfedge_ids() {
            let n = mesh.next(he);
            let p = mesh.prev(he);
            assert!(
                n.is_valid() && p.is_valid(),
                "{name}: {he:?} has no next/prev"
            );
            assert_eq!(
                mesh.prev(n),
                he,
                "{name}: prev(next({he:?})) should be {he:?}"
            );
            assert_eq!(
                mesh.next(p),
                he,
                "{name}: next(prev({he:?})) should be {he:?}"
            );
            assert_eq!(
                mesh.dest(he),
                mesh.origin(n),
                "{name}: {he:?} should end where its next begins"
            );
        }
    }
}

/// `next` stays within a face (or within the boundary loop), and every face
/// cycle closes after exactly as many steps as the face has vertices.
#[test]
fn face_cycles_close_and_stay_in_their_face() {
    for (name, mesh, _) in fixtures() {
        for f in mesh.face_ids() {
            let start = mesh.face(f).halfedge;
            assert!(start.is_valid(), "{name}: {f:?} has no half-edge");

            let expected = mesh.face_vertex_count(f);
            assert!(expected >= 3, "{name}: {f:?} has {expected} vertices");

            let mut he = start;
            for step in 0..expected {
                assert_eq!(
                    mesh.face_of(he),
                    f,
                    "{name}: {he:?} at step {step} of {f:?} belongs to another face"
                );
                he = mesh.next(he);
            }
            assert_eq!(
                he, start,
                "{name}: {f:?} cycle did not close after {expected} steps"
            );
        }
    }
}

/// A half-edge is a boundary half-edge exactly when it has no face, and the
/// boundary loops close under `next` just as face cycles do.
#[test]
fn boundary_halfedges_agree_with_face_pointers() {
    for (name, mesh, closed) in fixtures() {
        let mut boundary_count = 0;
        for he in mesh.halfedge_ids() {
            let has_face = mesh.face_of(he).is_valid();
            assert_eq!(
                mesh.is_boundary_halfedge(he),
                !has_face,
                "{name}: {he:?} boundary flag disagrees with its face pointer"
            );
            if !has_face {
                boundary_count += 1;
            }
        }

        if closed {
            assert_eq!(
                boundary_count, 0,
                "{name}: a closed mesh should have no boundary half-edges"
            );
        } else {
            assert!(
                boundary_count > 0,
                "{name}: an open mesh should have boundary half-edges"
            );
        }

        // Walking `next` from a boundary half-edge must stay on the boundary and
        // return to the start.
        for he in mesh.halfedge_ids() {
            if !mesh.is_boundary_halfedge(he) {
                continue;
            }
            let mut cur = mesh.next(he);
            let mut steps = 1;
            while cur != he {
                assert!(
                    mesh.is_boundary_halfedge(cur),
                    "{name}: boundary loop from {he:?} left the boundary at {cur:?}"
                );
                cur = mesh.next(cur);
                steps += 1;
                assert!(
                    steps <= mesh.num_halfedges(),
                    "{name}: boundary loop from {he:?} did not close"
                );
            }
        }
    }
}

/// `is_boundary_vertex` must agree with "some incident half-edge is a boundary
/// half-edge", computed independently of the vertex circulator.
#[test]
fn boundary_vertices_agree_with_incident_halfedges() {
    for (name, mesh, closed) in fixtures() {
        // Independent ground truth: scan all half-edges rather than circulating.
        let mut touches_boundary = vec![false; mesh.num_vertices()];
        for he in mesh.halfedge_ids() {
            if mesh.is_boundary_halfedge(he) {
                touches_boundary[mesh.origin(he).index()] = true;
                touches_boundary[mesh.dest(he).index()] = true;
            }
        }

        for v in mesh.vertex_ids() {
            assert_eq!(
                mesh.is_boundary_vertex(v),
                touches_boundary[v.index()],
                "{name}: {v:?} boundary status disagrees with its incident half-edges"
            );
        }

        if closed {
            assert!(
                mesh.vertex_ids().all(|v| !mesh.is_boundary_vertex(v)),
                "{name}: a closed mesh should have no boundary vertices"
            );
        }
    }
}

/// Circulating a vertex visits only half-edges leaving it, visits each once, and
/// terminates.
#[test]
fn vertex_circulator_is_consistent() {
    for (name, mesh, _) in fixtures() {
        for v in mesh.vertex_ids() {
            let outgoing: Vec<_> = mesh.vertex_halfedges(v).collect();
            assert!(
                !outgoing.is_empty(),
                "{name}: {v:?} has no outgoing half-edges"
            );

            for &he in &outgoing {
                assert_eq!(
                    mesh.origin(he),
                    v,
                    "{name}: circulator for {v:?} yielded {he:?}, which starts elsewhere"
                );
            }

            let mut sorted = outgoing.clone();
            sorted.sort_by_key(|he| he.index());
            let before = sorted.len();
            sorted.dedup();
            assert_eq!(
                sorted.len(),
                before,
                "{name}: circulator for {v:?} repeated a half-edge"
            );

            // The circulator and the neighbour iterator must agree in length,
            // and neighbours must be distinct.
            let neighbors: Vec<_> = mesh.vertex_neighbors(v).collect();
            assert_eq!(
                neighbors.len(),
                outgoing.len(),
                "{name}: {v:?} has {} neighbours but {} outgoing half-edges",
                neighbors.len(),
                outgoing.len()
            );
            let mut ns: Vec<usize> = neighbors.iter().map(|n| n.index()).collect();
            ns.sort_unstable();
            let n_before = ns.len();
            ns.dedup();
            assert_eq!(ns.len(), n_before, "{name}: {v:?} has a repeated neighbour");
            assert!(
                !ns.contains(&v.index()),
                "{name}: {v:?} is its own neighbour"
            );
        }
    }
}

/// Faces are non-degenerate: three distinct vertices and a positive area.
#[test]
fn triangles_are_non_degenerate() {
    for (name, mesh, _) in fixtures() {
        assert!(mesh.is_triangle_mesh(), "{name}: expected a triangle mesh");
        for f in mesh.face_ids() {
            let [a, b, c] = mesh.face_triangle(f);
            assert!(
                a != b && b != c && a != c,
                "{name}: {f:?} repeats a vertex: {a:?} {b:?} {c:?}"
            );
            let area = mesh.face_area(f);
            assert!(
                area > 0.0 && area.is_finite(),
                "{name}: {f:?} has area {area}"
            );
        }
    }
}

/// Euler's formula. Every fixture is a disk (`χ = 1`) or a sphere (`χ = 2`), so
/// this pins down the edge count against the vertex and face counts — the kind
/// of global consistency no local check catches.
#[test]
fn euler_characteristic_matches_topology() {
    // Edges are half-edge pairs, boundary half-edges included.
    fn edge_count(mesh: &HalfEdgeMesh) -> usize {
        assert_eq!(
            mesh.num_halfedges() % 2,
            0,
            "half-edges should pair up into edges"
        );
        mesh.num_halfedges() / 2
    }

    for (name, mesh, closed) in fixtures() {
        let v = mesh.num_vertices();
        let e = edge_count(&mesh);
        let f = mesh.num_faces();
        let chi = v as isize - e as isize + f as isize;
        let expected = if closed { 2 } else { 1 };
        assert_eq!(
            chi, expected,
            "{name}: V - E + F = {v} - {e} + {f} = {chi}, expected {expected}"
        );
    }
}

/// Each face's half-edges are distinct, and across the mesh every non-boundary
/// half-edge belongs to exactly one face cycle.
#[test]
fn every_halfedge_belongs_to_exactly_one_cycle() {
    for (name, mesh, _) in fixtures() {
        let mut owner = vec![None::<usize>; mesh.num_halfedges()];

        for f in mesh.face_ids() {
            for he in mesh.face_halfedges(f) {
                let slot = &mut owner[he.index()];
                assert!(
                    slot.is_none(),
                    "{name}: {he:?} appears in two face cycles ({:?} and {f:?})",
                    slot.unwrap()
                );
                *slot = Some(f.index());
            }
        }

        for he in mesh.halfedge_ids() {
            let claimed = owner[he.index()].is_some();
            assert_eq!(
                claimed,
                !mesh.is_boundary_halfedge(he),
                "{name}: {he:?} claimed={claimed} but boundary={}",
                mesh.is_boundary_halfedge(he)
            );
        }
    }
}

/// Face normals of a closed, consistently-wound mesh point outward, so the
/// divergence-theorem volume is positive. This catches an orientation flip that
/// no per-face check would.
#[test]
fn closed_meshes_are_consistently_wound() {
    for (name, mesh, closed) in fixtures() {
        if !closed {
            continue;
        }
        // 6V = Σ (a × b) · c over faces, for a mesh enclosing the origin.
        let six_v: f64 = mesh
            .face_ids()
            .map(|f| {
                let [a, b, c] = mesh.face_positions(f);
                a.coords.cross(&b.coords).dot(&c.coords)
            })
            .sum();
        assert!(
            six_v > 0.0,
            "{name}: signed volume {} suggests inverted winding",
            six_v / 6.0
        );
    }
}
