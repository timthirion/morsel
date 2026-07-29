//! A corpus of deliberately awkward meshes, shared across integration tests.
//!
//! Every algorithm in the library is written against a well-behaved manifold
//! triangle mesh. Real input is not that, and the interesting question is not
//! whether an algorithm handles bad input *well* — often it cannot — but whether
//! it **says so**. An algorithm that errors on a non-manifold mesh is fine. One
//! that panics is a bug. One that returns a plausible-looking mesh full of NaNs
//! is the worst case, because nothing downstream will notice.
//!
//! The corpus is split by what makes each entry hard:
//!
//! - *geometric* — constructible and manifold, but numerically nasty: slivers,
//!   all-obtuse triangles, cocircular lattices, extreme coordinate scales.
//! - *topological* — non-manifold edges and vertices, several components, several
//!   boundary loops, closed surfaces, inconsistent winding.
//! - *malformed* — zero-area faces, duplicate vertices and faces, unreferenced
//!   vertices.
//!
//! Some entries cannot be built at all; `build_from_triangles` rejecting them is
//! itself a recorded outcome, so `mesh` is a `Result`.

#![allow(dead_code)] // Each test binary uses only part of the corpus.

use morsel::error::Result;
use morsel::mesh::{build_from_triangles, HalfEdgeMesh};
use nalgebra::Point3;

/// What makes a corpus entry difficult.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Difficulty {
    /// Manifold and constructible, but numerically awkward.
    Geometric,
    /// Violates the manifold/disk assumptions algorithms are written against.
    Topological,
    /// Contains outright invalid elements.
    Malformed,
    /// Well-behaved. Present as a control: whatever fails here is not the
    /// corpus's fault.
    Control,
}

/// One corpus entry.
pub struct Case {
    pub name: &'static str,
    pub difficulty: Difficulty,
    /// Why this input is expected to be hard, in one line.
    pub note: &'static str,
    pub mesh: Result<HalfEdgeMesh>,
}

fn case(
    name: &'static str,
    difficulty: Difficulty,
    note: &'static str,
    vertices: Vec<Point3<f64>>,
    faces: Vec<[usize; 3]>,
) -> Case {
    Case {
        name,
        difficulty,
        note,
        mesh: build_from_triangles(&vertices, &faces),
    }
}

// ------------------------------------------------------------------ builders

fn grid_verts_faces(n: usize, scale: f64) -> (Vec<Point3<f64>>, Vec<[usize; 3]>) {
    let mut vertices = Vec::new();
    let mut faces = Vec::new();
    for j in 0..=n {
        for i in 0..=n {
            vertices.push(Point3::new(i as f64 * scale, j as f64 * scale, 0.0));
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
    (vertices, faces)
}

/// The full corpus.
pub fn corpus() -> Vec<Case> {
    let mut cases = Vec::new();

    // ---- controls -------------------------------------------------------
    {
        let (v, f) = grid_verts_faces(4, 1.0);
        cases.push(case(
            "control_grid",
            Difficulty::Control,
            "well-shaped open grid; the baseline every algorithm should handle",
            v,
            f,
        ));
    }
    cases.push(case(
        "control_tetrahedron",
        Difficulty::Control,
        "well-shaped closed surface, no boundary",
        vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.5, 1.0, 0.0),
            Point3::new(0.5, 0.5, 1.0),
        ],
        vec![[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
    ));

    // ---- geometric ------------------------------------------------------
    cases.push(case(
        "sliver_triangles",
        Difficulty::Geometric,
        "aspect ratio ~1e6; cotangent weights blow up",
        vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.5, 1e-6, 0.0),
            Point3::new(1.5, 1e-6, 0.0),
        ],
        vec![[0, 1, 2], [1, 3, 2]],
    ));
    cases.push(case(
        "all_obtuse",
        Difficulty::Geometric,
        "every triangle obtuse, so cotangent weights go negative and the \
         Laplacian loses its maximum principle",
        vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(4.0, 0.0, 0.0),
            Point3::new(2.0, 0.3, 0.0),
            Point3::new(6.0, 0.3, 0.0),
        ],
        vec![[0, 1, 2], [1, 3, 2]],
    ));
    {
        // A regular lattice is maximally degenerate for Delaunay/Voronoi: every
        // interior square has four cocircular corners. This broke a polygon
        // clipper in July 2026.
        let (v, f) = grid_verts_faces(4, 1.0);
        cases.push(case(
            "cocircular_lattice",
            Difficulty::Geometric,
            "four-way cocircular sites everywhere; degenerate Delaunay/power diagrams",
            v,
            f,
        ));
    }
    {
        let (v, f) = grid_verts_faces(3, 1e6);
        cases.push(case(
            "huge_scale",
            Difficulty::Geometric,
            "coordinates ~1e6; absolute epsilons become meaningless",
            v,
            f,
        ));
    }
    {
        let (v, f) = grid_verts_faces(3, 1e-6);
        cases.push(case(
            "tiny_scale",
            Difficulty::Geometric,
            "coordinates ~1e-6; areas ~1e-12 collide with hard-coded thresholds",
            v,
            f,
        ));
    }
    {
        // One vertex with very high valence.
        let spokes = 64;
        let mut v = vec![Point3::new(0.0, 0.0, 0.0)];
        for k in 0..spokes {
            let t = 2.0 * std::f64::consts::PI * (k as f64) / (spokes as f64);
            v.push(Point3::new(t.cos(), t.sin(), 0.0));
        }
        let f: Vec<[usize; 3]> = (0..spokes)
            .map(|k| [0, 1 + k, 1 + (k + 1) % spokes])
            .collect();
        cases.push(case(
            "high_valence_fan",
            Difficulty::Geometric,
            "valence-64 interior vertex",
            v,
            f,
        ));
    }

    // ---- topological ----------------------------------------------------
    cases.push(case(
        "nonmanifold_edge",
        Difficulty::Topological,
        "three faces share one edge; a half-edge mesh has no representation for it",
        vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.5, 1.0, 0.0),
            Point3::new(0.5, -1.0, 0.0),
            Point3::new(0.5, 0.0, 1.0),
        ],
        vec![[0, 1, 2], [0, 1, 3], [0, 1, 4]],
    ));
    cases.push(case(
        "nonmanifold_vertex",
        Difficulty::Topological,
        "two triangle fans meeting at a single vertex; the vertex link is two \
         circles, so circulating it cannot reach both",
        vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.5, 1.0, 0.0),
            Point3::new(-1.0, 0.0, 0.0),
            Point3::new(-0.5, -1.0, 0.0),
        ],
        vec![[0, 1, 2], [0, 3, 4]],
    ));
    cases.push(case(
        "two_components",
        Difficulty::Topological,
        "two disjoint triangles; anything assuming connectivity breaks",
        vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.5, 1.0, 0.0),
            Point3::new(5.0, 0.0, 0.0),
            Point3::new(6.0, 0.0, 0.0),
            Point3::new(5.5, 1.0, 0.0),
        ],
        vec![[0, 1, 2], [3, 4, 5]],
    ));
    {
        // Annulus: a square grid with the middle quad removed, giving two
        // boundary loops. Not a disk, so parameterization assumptions fail.
        let (v, mut f) = grid_verts_faces(4, 1.0);
        // Remove both faces of the centre cell. For an n=4 grid, cell (i=1, j=1)
        // occupies faces 2*cell and 2*cell+1 where cell = j*n + i.
        let (n, i, j) = (4usize, 1usize, 1usize);
        let cell = j * n + i;
        f.remove(2 * cell + 1);
        f.remove(2 * cell);
        cases.push(case(
            "annulus_two_boundary_loops",
            Difficulty::Topological,
            "a hole punched in a grid: genus 0 but two boundary loops, so not a disk",
            v,
            f,
        ));
    }
    cases.push(case(
        "inconsistent_winding",
        Difficulty::Topological,
        "one face wound opposite to its neighbour; normals disagree across the \
         shared edge",
        vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ],
        vec![[0, 1, 2], [0, 2, 3], [0, 3, 2]],
    ));

    // ---- malformed ------------------------------------------------------
    cases.push(case(
        "zero_area_face",
        Difficulty::Malformed,
        "three collinear vertices; normal and cotangent weights are undefined",
        vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
        ],
        vec![[0, 1, 2], [0, 3, 1]],
    ));
    cases.push(case(
        "duplicate_vertices",
        Difficulty::Malformed,
        "two vertices at the same position with distinct indices; welding is \
         needed before topology means anything",
        vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.5, 1.0, 0.0),
            Point3::new(1.0, 0.0, 0.0), // duplicate of index 1
            Point3::new(1.5, 1.0, 0.0),
        ],
        vec![[0, 1, 2], [3, 4, 2]],
    ));
    cases.push(case(
        "duplicate_face",
        Difficulty::Malformed,
        "the same triangle listed twice",
        vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.5, 1.0, 0.0),
        ],
        vec![[0, 1, 2], [0, 1, 2]],
    ));
    cases.push(case(
        "unreferenced_vertex",
        Difficulty::Malformed,
        "a vertex in the list that no face uses; isolated, so it has no half-edge",
        vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.5, 1.0, 0.0),
            Point3::new(9.0, 9.0, 9.0), // referenced by nothing
        ],
        vec![[0, 1, 2]],
    ));
    cases.push(case(
        "single_triangle",
        Difficulty::Malformed,
        "the minimal mesh; every vertex is on the boundary and there is no interior",
        vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.5, 1.0, 0.0),
        ],
        vec![[0, 1, 2]],
    ));

    cases
}

// ------------------------------------------------------------- validation

/// Why a mesh is structurally unusable.
pub fn structural_defect(mesh: &HalfEdgeMesh) -> Option<String> {
    if mesh.num_vertices() == 0 {
        return Some("no vertices".into());
    }

    for v in mesh.vertex_ids() {
        let p = mesh.position(v);
        if !p.x.is_finite() || !p.y.is_finite() || !p.z.is_finite() {
            return Some(format!("non-finite position at {v:?}: {p:?}"));
        }
    }

    for he in mesh.halfedge_ids() {
        let t = mesh.twin(he);
        if !t.is_valid() {
            return Some(format!("{he:?} has an invalid twin"));
        }
        if mesh.twin(t) != he {
            return Some(format!("twin is not an involution at {he:?}"));
        }
        if mesh.origin(t) != mesh.dest(he) {
            return Some(format!("twin does not reverse {he:?}"));
        }
        let n = mesh.next(he);
        if !n.is_valid() {
            return Some(format!("{he:?} has no next"));
        }
        if mesh.prev(n) != he {
            return Some(format!("next/prev disagree at {he:?}"));
        }
        if mesh.dest(he) != mesh.origin(n) {
            return Some(format!("half-edges do not chain at {he:?}"));
        }
    }

    for f in mesh.face_ids() {
        let [a, b, c] = mesh.face_triangle(f);
        if a == b || b == c || a == c {
            return Some(format!("{f:?} repeats a vertex"));
        }
        let area = mesh.face_area(f);
        if !area.is_finite() {
            return Some(format!("{f:?} has non-finite area"));
        }
    }

    None
}

/// Non-finite values in a scalar field, e.g. curvature or geodesic distance.
pub fn field_defect(values: &[f64], label: &str) -> Option<String> {
    for (i, v) in values.iter().enumerate() {
        if !v.is_finite() {
            return Some(format!("{label}[{i}] is {v}"));
        }
    }
    None
}
