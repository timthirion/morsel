//! Cutting a surface open so it can be flattened.
//!
//! LSCM, ARAP and OMT all need **disk topology** — exactly one boundary loop — and
//! most real meshes do not have it. The Stanford bunny has four loops around its
//! base, a cylinder has two, and a closed surface has none. This module produces a
//! disk by cutting along shortest paths.
//!
//! # How
//!
//! Two moves, applied until one boundary loop remains:
//!
//! - **Join two loops.** Take every vertex of one loop as a source, run Dijkstra,
//!   and cut along the shortest path to the nearest vertex on another loop. Cutting a
//!   channel between two holes merges them into one, so `b` drops by one each time.
//! - **Slit a closed surface.** With no boundary to start from, cut along the path
//!   between a vertex and the one farthest from it. On a genus-0 surface a single
//!   slit is enough: a sphere cut along an arc is a disk.
//!
//! Cutting itself is done in face-vertex space — duplicate the vertices along the
//! path, hand the faces on one side the duplicates, and rebuild. That is far easier
//! to get right than rewiring half-edges in place, and
//! [`build_from_triangles`] validates the result,
//! so a botched cut is rejected rather than quietly producing a corrupt mesh.
//!
//! # Not implemented: genus > 0
//!
//! A handle needs two additional cuts that no shortest path between boundaries will
//! find, so a torus is refused rather than mangled. Reducing a genus-`g` surface to a
//! disk requires finding `2g` independent handle loops, which is a different
//! algorithm.

use std::collections::{HashMap, HashSet};

use nalgebra::Point3;

use crate::algo::geodesic::{dijkstra_multiple, DijkstraOptions};
use crate::error::{MeshError, Result};
use crate::mesh::{
    build_from_triangles, to_face_vertex, FaceId, HalfEdgeMesh, MeshIndex, VertexId,
};

/// A face's index, or `None` for the invalid face that marks a boundary.
fn face_index<I: MeshIndex>(f: FaceId<I>) -> Option<usize> {
    f.is_valid().then(|| f.index())
}

/// What a cut did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CutReport {
    /// Boundary loops before cutting.
    pub loops_before: usize,
    /// Boundary loops after cutting. One means the result is a disk.
    pub loops_after: usize,
    /// How many paths were cut along.
    pub paths_cut: usize,
    /// Vertices duplicated in the process.
    pub vertices_added: usize,
}

impl CutReport {
    /// Whether the result has disk topology, and so can be flattened.
    pub fn is_disk(&self) -> bool {
        self.loops_after == 1
    }
}

/// Cut a mesh until it has exactly one boundary loop.
///
/// Returns the cut mesh and a report. A mesh that is already a disk comes back
/// unchanged, with `paths_cut == 0`.
///
/// # Errors
///
/// - [`MeshError::InvalidState`] if the mesh is disconnected, since each component
///   would need cutting separately and the genus is not well defined across them.
/// - [`MeshError::InvalidState`] for genus > 0. See the module docs.
/// - Whatever [`build_from_triangles`] reports if
///   a cut produces something unrepresentable, which would be a bug here.
pub fn cut_to_disk<I: MeshIndex>(mesh: &HalfEdgeMesh<I>) -> Result<(HalfEdgeMesh<I>, CutReport)> {
    let loops_before = mesh.boundary_loop_count();

    let genus = mesh.genus().ok_or_else(|| {
        MeshError::InvalidState(
            "cannot cut a disconnected mesh: Euler's formula sums over components, so \
             the genus is not well defined. Split it into components first."
                .to_string(),
        )
    })?;
    if genus > 0 {
        return Err(MeshError::InvalidState(format!(
            "cutting a genus-{genus} surface to a disk needs {} handle loops, which no \
             shortest path between boundaries will find. Only genus 0 is supported.",
            2 * genus
        )));
    }

    if loops_before == 1 {
        return Ok((
            mesh.clone(),
            CutReport {
                loops_before,
                loops_after: 1,
                paths_cut: 0,
                vertices_added: 0,
            },
        ));
    }

    let original_vertices = mesh.num_vertices();
    let mut current = mesh.clone();
    let mut paths_cut = 0;

    // Each cut reduces the loop count by one, or opens a closed surface to one loop,
    // so this terminates. The bound is a backstop against a cut that fails to make
    // progress rather than an expected limit.
    let max_paths = loops_before.max(1) + 2;
    while current.boundary_loop_count() != 1 {
        let path = choose_cut_path(&current)?;
        let (vertices, faces) = cut_along_path(&current, &path)?;
        let next = build_from_triangles::<I>(&vertices, &faces)?;

        if next.boundary_loop_count() >= current.boundary_loop_count() && paths_cut > 0 {
            return Err(MeshError::InvalidState(format!(
                "cutting stopped making progress at {} boundary loops",
                current.boundary_loop_count()
            )));
        }
        current = next;

        paths_cut += 1;
        if paths_cut > max_paths {
            return Err(MeshError::InvalidState(format!(
                "cutting did not reach a disk after {paths_cut} paths"
            )));
        }
    }

    let report = CutReport {
        loops_before,
        loops_after: current.boundary_loop_count(),
        paths_cut,
        vertices_added: current.num_vertices() - original_vertices,
    };
    Ok((current, report))
}

/// Pick the next path to cut along.
fn choose_cut_path<I: MeshIndex>(mesh: &HalfEdgeMesh<I>) -> Result<Vec<VertexId<I>>> {
    let loops = mesh.boundary_loops();
    let options = DijkstraOptions {
        store_predecessors: true,
        ..Default::default()
    };

    if loops.is_empty() {
        // Closed: slit between a vertex and the one farthest from it, which gives a
        // long cut rather than a degenerate one.
        let seed = VertexId::new(0);
        let first = dijkstra_multiple(mesh, &[seed], &options);
        let (far, _) = first.farthest_vertex().ok_or_else(|| {
            MeshError::InvalidState("mesh has no reachable vertices to cut between".to_string())
        })?;
        let mut path = first
            .path_to(far)
            .ok_or_else(|| MeshError::InvalidState("no path to the farthest vertex".to_string()))?;

        // A slit's two endpoints are its tips and stay welded, so only the vertices
        // *between* them get duplicated — a two-vertex path opens nothing. That is not
        // a pathological case: on a tetrahedron every vertex is adjacent to every
        // other, so no shortest path is longer than one edge. Extend away from the
        // end until there is an interior vertex to split.
        while path.len() < 3 {
            let tip = *path.last().expect("non-empty");
            let step = mesh
                .vertex_neighbors(tip)
                .find(|n| !path.contains(n))
                .ok_or_else(|| {
                    MeshError::InvalidState(
                        "mesh is too small to slit: no edge path of length two exists".to_string(),
                    )
                })?;
            path.push(step);
        }
        return Ok(path);
    }

    // Two or more loops: bridge the first to whichever other loop is nearest.
    let sources = &loops[0];
    let others: HashSet<usize> = loops[1..].iter().flatten().map(|v| v.index()).collect();

    let result = dijkstra_multiple(mesh, sources, &options);
    let target = others
        .iter()
        .filter(|&&i| result.distances()[i].is_finite())
        .min_by(|&&a, &&b| {
            result.distances()[a]
                .partial_cmp(&result.distances()[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .copied()
        .ok_or_else(|| {
            MeshError::InvalidState(
                "no path between boundary loops, so the mesh is disconnected".to_string(),
            )
        })?;

    result
        .path_to(VertexId::new(target))
        .ok_or_else(|| MeshError::InvalidState("no path to the chosen loop".to_string()))
}

/// Face-vertex mesh data: vertex positions and triangles indexing into them.
type FaceVertex = (Vec<Point3<f64>>, Vec<[usize; 3]>);

/// The faces around `v`, grouped into arcs by the cut.
///
/// The fan around `v` is a cyclic sequence of rays `h⁰, h¹, …` produced by the vertex
/// circulator, which advances by `next(twin(h))`. Since `face_of(next(twin(h)))` and
/// `face_of(twin(h))` are the same face, the *sector* lying between rays `k` and `k+1`
/// is `face_of(twin(hᵏ))` — not `face_of(hᵏ)`, which is the sector on the other side of
/// ray `k`. Getting that index off by one silently assigns faces across the cut.
///
/// A sector is severed from its predecessor when ray `k` is a cut edge, and an invalid
/// sector is the gap at a boundary vertex, which severs both its neighbours.
fn fan_arcs<I: MeshIndex>(
    mesh: &HalfEdgeMesh<I>,
    v: VertexId<I>,
    cut_neighbours: &[usize],
) -> Vec<Vec<usize>> {
    let fan: Vec<_> = mesh.vertex_halfedges(v).collect();
    let m = fan.len();
    let sector = |k: usize| mesh.face_of(mesh.twin(fan[k % m]));
    let is_cut = |k: usize| cut_neighbours.contains(&mesh.dest(fan[k % m]).index());

    // Start at a severed position so the first arc does not straddle the wrap-around.
    let start = match (0..m).find(|&k| is_cut(k) || !sector((k + m - 1) % m).is_valid()) {
        Some(k) => k,
        None => return vec![(0..m).filter_map(|k| face_index(sector(k))).collect()],
    };

    let mut arcs: Vec<Vec<usize>> = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    for step in 0..m {
        let k = (start + step) % m;
        if step > 0 && is_cut(k) && !current.is_empty() {
            arcs.push(std::mem::take(&mut current));
        }
        match face_index(sector(k)) {
            Some(f) => current.push(f),
            // The boundary gap ends whatever arc was in progress.
            None if !current.is_empty() => arcs.push(std::mem::take(&mut current)),
            None => {}
        }
    }
    if !current.is_empty() {
        arcs.push(current);
    }
    arcs
}

/// Split the surface along `path`, returning face-vertex data for the cut mesh.
///
/// Every path vertex is duplicated except an endpoint that lies in the interior: such
/// an endpoint is the tip of a slit, where the two sides stay joined. Duplicating it
/// would tear the surface apart instead of slitting it.
///
/// # Choosing a side coherently
///
/// Each duplicated vertex has two arcs of faces, and one takes the duplicate. That
/// choice must agree along the whole path — hand the duplicate to the left side at one
/// vertex and the right side at the next and the two sides weld back together, which
/// shows up as a bowtie vertex or a burst of spurious boundary loops rather than as
/// anything local.
///
/// Orienting by the path direction makes it coherent. For a directed path edge
/// `vᵢ → vᵢ₊₁` with half-edge `h`, call `face_of(h)` its left face and
/// `face_of(twin(h))` its right face. At `vᵢ` the arc holding right-of-`eᵢ` also holds
/// right-of-`eᵢ₋₁`: the arc runs forward from ray `h` and ends just before the ray
/// toward `vᵢ₋₁`, whose preceding sector is exactly right-of-`eᵢ₋₁`. So "the arc
/// containing the right face of an incident path edge" picks the same side everywhere,
/// and either incident edge gives the same answer.
fn cut_along_path<I: MeshIndex>(
    mesh: &HalfEdgeMesh<I>,
    path: &[VertexId<I>],
) -> Result<FaceVertex> {
    if path.len() < 2 {
        return Err(MeshError::InvalidState(
            "a cut path needs at least two vertices".to_string(),
        ));
    }

    let (mut vertices, faces) = to_face_vertex(mesh);
    let original_count = vertices.len();

    // Which vertices each path vertex is joined to along the path.
    let mut path_neighbours: HashMap<usize, Vec<usize>> = HashMap::new();
    for pair in path.windows(2) {
        path_neighbours
            .entry(pair[0].index())
            .or_default()
            .push(pair[1].index());
        path_neighbours
            .entry(pair[1].index())
            .or_default()
            .push(pair[0].index());
    }

    // Corner remapping: (face index, corner slot) -> duplicated vertex index.
    let mut remap: HashMap<(usize, usize), usize> = HashMap::new();

    for (position, &v) in path.iter().enumerate() {
        let is_endpoint = position == 0 || position + 1 == path.len();
        if is_endpoint && !mesh.is_boundary_vertex(v) {
            continue; // slit tip: stays a single vertex
        }

        // Orient by the path: use the outgoing edge, or the incoming one at the end.
        let (from, to) = if position + 1 < path.len() {
            (v, path[position + 1])
        } else {
            (path[position - 1], v)
        };
        let ray = mesh
            .vertex_halfedges(from)
            .find(|&he| mesh.dest(he) == to)
            .ok_or_else(|| {
                MeshError::InvalidState(format!(
                    "cut path uses a non-edge between vertices {} and {}",
                    from.index(),
                    to.index()
                ))
            })?;
        let right = face_index(mesh.face_of(mesh.twin(ray))).ok_or_else(|| {
            MeshError::InvalidState(format!(
                "cut path runs along the boundary at edge {}-{}, so it has no interior \
                 side to cut",
                from.index(),
                to.index()
            ))
        })?;

        let neighbours = path_neighbours.get(&v.index()).cloned().unwrap_or_default();
        let arcs = fan_arcs(mesh, v, &neighbours);
        if arcs.len() != 2 {
            return Err(MeshError::InvalidState(format!(
                "cut path is not a simple arc at vertex {}: its fan splits into {} \
                 pieces rather than 2",
                v.index(),
                arcs.len()
            )));
        }
        let side = arcs
            .iter()
            .find(|arc| arc.contains(&right))
            .ok_or_else(|| {
                MeshError::InvalidState(format!(
                    "face {right} borders the cut at vertex {} but is in neither arc",
                    v.index()
                ))
            })?;

        let duplicate = vertices.len();
        vertices.push(*mesh.position(v));

        for &fi in side {
            let slot = faces[fi]
                .iter()
                .position(|&x| x == v.index())
                .ok_or_else(|| {
                    MeshError::InvalidState(format!(
                        "face {fi} is in the fan of vertex {} but does not reference it",
                        v.index()
                    ))
                })?;
            remap.insert((fi, slot), duplicate);
        }
    }

    if vertices.len() == original_count {
        return Err(MeshError::InvalidState(
            "cut path duplicated no vertices, so it would not open the surface".to_string(),
        ));
    }

    let cut_faces: Vec<[usize; 3]> = faces
        .iter()
        .enumerate()
        .map(|(fi, face)| {
            let mut out = *face;
            for (slot, corner) in out.iter_mut().enumerate() {
                if let Some(&dup) = remap.get(&(fi, slot)) {
                    *corner = dup;
                }
            }
            out
        })
        .collect();

    Ok((vertices, cut_faces))
}
