//! Dijkstra's algorithm for geodesic distances.
//!
//! Computes shortest path distances along mesh edges using Dijkstra's algorithm.
//! This gives exact distances on the edge graph, which approximates true geodesic
//! distances on the surface.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::mesh::{HalfEdgeMesh, MeshIndex, VertexId};

use super::GeodesicResult;

/// Options for Dijkstra's algorithm.
#[derive(Debug, Clone)]
pub struct DijkstraOptions {
    /// Whether to store predecessor information for path reconstruction.
    pub store_predecessors: bool,

    /// Maximum distance to explore. Vertices beyond this distance won't be visited.
    /// Set to `None` for no limit.
    pub max_distance: Option<f64>,

    /// Target vertex for early termination.
    /// If set, the algorithm stops once this vertex is reached.
    pub target: Option<usize>,
}

impl Default for DijkstraOptions {
    fn default() -> Self {
        Self {
            store_predecessors: false,
            max_distance: None,
            target: None,
        }
    }
}

impl DijkstraOptions {
    /// Enable predecessor storage for path reconstruction.
    pub fn with_predecessors(mut self, store: bool) -> Self {
        self.store_predecessors = store;
        self
    }

    /// Set maximum distance to explore.
    pub fn with_max_distance(mut self, max_dist: f64) -> Self {
        self.max_distance = Some(max_dist);
        self
    }

    /// Set target vertex for early termination.
    pub fn with_target(mut self, target: usize) -> Self {
        self.target = Some(target);
        self
    }
}

/// Entry in Dijkstra's priority queue.
#[derive(Debug, Clone)]
struct DijkstraEntry {
    /// The vertex index.
    vertex: usize,
    /// Distance from source.
    distance: f64,
}

impl DijkstraEntry {
    fn new(vertex: usize, distance: f64) -> Self {
        Self { vertex, distance }
    }
}

// Implement ordering for min-heap (BinaryHeap is a max-heap by default)
impl PartialEq for DijkstraEntry {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for DijkstraEntry {}

impl PartialOrd for DijkstraEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DijkstraEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap behavior
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
    }
}

/// Compute geodesic distances from a single source vertex using Dijkstra's algorithm.
///
/// This computes shortest path distances along mesh edges, which approximates
/// true geodesic distances on the surface.
///
/// # Arguments
///
/// * `mesh` - The input mesh
/// * `source` - The source vertex
/// * `options` - Algorithm options
///
/// # Returns
///
/// A `GeodesicResult` containing distances from the source to all vertices.
///
/// # Example
///
/// ```no_run
/// use morsel::prelude::*;
/// use morsel::algo::geodesic::{dijkstra, DijkstraOptions};
///
/// let mesh: HalfEdgeMesh = morsel::io::load("mesh.obj").unwrap();
/// let source = VertexId::new(0);
///
/// // Basic usage
/// let result = dijkstra(&mesh, source, &DijkstraOptions::default());
///
/// // With path reconstruction
/// let result = dijkstra(&mesh, source, &DijkstraOptions::default().with_predecessors(true));
/// if let Some(path) = result.path_to(VertexId::new(10)) {
///     println!("Path: {:?}", path);
/// }
/// ```
pub fn dijkstra<I: MeshIndex>(
    mesh: &HalfEdgeMesh<I>,
    source: VertexId<I>,
    options: &DijkstraOptions,
) -> GeodesicResult<I> {
    dijkstra_multiple(mesh, &[source], options)
}

/// Compute geodesic distances from multiple source vertices.
///
/// All source vertices are treated as having distance 0. This is useful for
/// computing distance fields from a set of points.
///
/// # Arguments
///
/// * `mesh` - The input mesh
/// * `sources` - The source vertices
/// * `options` - Algorithm options
///
/// # Returns
///
/// A `GeodesicResult` containing distances from the nearest source to all vertices.
///
/// # Example
///
/// ```no_run
/// use morsel::prelude::*;
/// use morsel::algo::geodesic::{dijkstra_multiple, DijkstraOptions};
///
/// let mesh: HalfEdgeMesh = morsel::io::load("mesh.obj").unwrap();
/// let sources = vec![VertexId::new(0), VertexId::new(10), VertexId::new(20)];
///
/// let result = dijkstra_multiple(&mesh, &sources, &DijkstraOptions::default());
/// ```
pub fn dijkstra_multiple<I: MeshIndex>(
    mesh: &HalfEdgeMesh<I>,
    sources: &[VertexId<I>],
    options: &DijkstraOptions,
) -> GeodesicResult<I> {
    let n = mesh.num_vertices();

    if n == 0 || sources.is_empty() {
        return GeodesicResult::new(vec![f64::INFINITY; n], None);
    }

    // Initialize distances
    let mut distances = vec![f64::INFINITY; n];

    // Initialize predecessors if requested
    let mut predecessors: Option<Vec<Option<usize>>> = if options.store_predecessors {
        Some(vec![None; n])
    } else {
        None
    };

    // Priority queue
    let mut heap = BinaryHeap::new();

    // Initialize sources
    for &source in sources {
        let idx = source.index();
        if idx < n {
            distances[idx] = 0.0;
            heap.push(DijkstraEntry::new(idx, 0.0));
        }
    }

    // Main Dijkstra loop
    while let Some(entry) = heap.pop() {
        let u = entry.vertex;
        let dist_u = entry.distance;

        // Skip if this is a stale entry (we found a shorter path already)
        if dist_u > distances[u] {
            continue;
        }

        // Early termination if we reached the target
        if let Some(target) = options.target {
            if u == target {
                break;
            }
        }

        // Early termination if we exceeded max distance
        if let Some(max_dist) = options.max_distance {
            if dist_u > max_dist {
                continue;
            }
        }

        // Relax all neighbors
        let u_vertex: VertexId<I> = VertexId::new(u);
        for he in mesh.vertex_halfedges(u_vertex) {
            let v_vertex = mesh.dest(he);
            let v = v_vertex.index();

            // Get edge length
            let edge_len = mesh.edge_length(he);
            let new_dist = dist_u + edge_len;

            // Check if this is a shorter path
            if new_dist < distances[v] {
                distances[v] = new_dist;

                // Update predecessor
                if let Some(ref mut preds) = predecessors {
                    preds[v] = Some(u);
                }

                // Add to heap (may create duplicate entries, but that's OK)
                heap.push(DijkstraEntry::new(v, new_dist));
            }
        }
    }

    GeodesicResult::new(distances, predecessors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::build_from_triangles;
    use nalgebra::Point3;

    fn create_single_triangle() -> HalfEdgeMesh {
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.5, 1.0, 0.0),
        ];
        let faces = vec![[0, 1, 2]];
        build_from_triangles(&vertices, &faces).unwrap()
    }

    fn create_grid_mesh(n: usize) -> HalfEdgeMesh {
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

    #[test]
    fn test_dijkstra_single_triangle() {
        let mesh = create_single_triangle();
        let result = dijkstra(&mesh, VertexId::new(0), &DijkstraOptions::default());

        assert_eq!(result.len(), 3);

        // Distance to self is 0
        assert!((result.distance(VertexId::new(0)) - 0.0).abs() < 1e-10);

        // Distance to vertex 1 is 1.0 (edge length)
        assert!((result.distance(VertexId::new(1)) - 1.0).abs() < 1e-10);

        // Distance to vertex 2: can go direct or via vertex 1
        // Direct edge length from (0,0,0) to (0.5,1,0) = sqrt(0.25 + 1) = sqrt(1.25)
        let expected_dist_2 = (0.5_f64.powi(2) + 1.0_f64.powi(2)).sqrt();
        assert!((result.distance(VertexId::new(2)) - expected_dist_2).abs() < 1e-10);
    }

    #[test]
    fn test_dijkstra_grid() {
        let mesh = create_grid_mesh(2);
        let result = dijkstra(&mesh, VertexId::new(0), &DijkstraOptions::default());

        // Vertex 0 is at (0,0), vertex 8 is at (2,2)
        // Shortest path along edges: 0 -> 1 -> 2 -> 5 -> 8 or similar
        // Grid edges have length 1 (horizontal/vertical) or sqrt(2) (diagonal)

        // Distance to self
        assert!((result.distance(VertexId::new(0)) - 0.0).abs() < 1e-10);

        // All vertices should be reachable
        assert_eq!(result.reachable_count(), 9);

        // Distance to corner (2,2) - vertex 8
        // Shortest path is along the diagonal edges
        let dist_to_corner = result.distance(VertexId::new(8));
        assert!(dist_to_corner > 0.0);
        assert!(dist_to_corner.is_finite());
    }

    #[test]
    fn test_dijkstra_path_reconstruction() {
        let mesh = create_single_triangle();
        let options = DijkstraOptions::default().with_predecessors(true);
        let result = dijkstra(&mesh, VertexId::new(0), &options);

        // Path to self should be just [0]
        let path_to_self = result.path_to(VertexId::new(0));
        assert!(path_to_self.is_some());
        assert_eq!(path_to_self.unwrap(), vec![VertexId::new(0)]);

        // Path to vertex 1 should be [0, 1]
        let path_to_1 = result.path_to(VertexId::new(1));
        assert!(path_to_1.is_some());
        assert_eq!(path_to_1.unwrap(), vec![VertexId::new(0), VertexId::new(1)]);
    }

    #[test]
    fn test_dijkstra_max_distance() {
        let mesh = create_grid_mesh(3);
        let options = DijkstraOptions::default().with_max_distance(1.5);
        let result = dijkstra(&mesh, VertexId::new(0), &options);

        // Vertices within distance 1.5 should be reachable
        // Vertices beyond should have infinity

        // Source is reachable
        assert!(result.is_reachable(VertexId::new(0)));

        // Adjacent vertices (distance 1) should be reachable
        assert!(result.is_reachable(VertexId::new(1))); // (1,0)
        assert!(result.is_reachable(VertexId::new(4))); // (0,1)

        // Some distant vertices should not be reachable
        // Vertex 15 is at (3,3), definitely beyond 1.5
        assert!(!result.is_reachable(VertexId::new(15)));
    }

    #[test]
    fn test_dijkstra_target() {
        let mesh = create_grid_mesh(3);
        let target_idx = 5; // Some vertex in the middle
        let options = DijkstraOptions::default().with_target(target_idx);
        let result = dijkstra(&mesh, VertexId::new(0), &options);

        // Target should be reachable
        assert!(result.is_reachable(VertexId::new(target_idx)));

        // Distance to target should be computed
        assert!(result.distance(VertexId::new(target_idx)).is_finite());
    }

    #[test]
    fn test_dijkstra_multiple_sources() {
        let mesh = create_grid_mesh(2);

        // Sources at opposite corners
        let sources = vec![VertexId::new(0), VertexId::new(8)];
        let result = dijkstra_multiple(&mesh, &sources, &DijkstraOptions::default());

        // Both sources should have distance 0
        assert!((result.distance(VertexId::new(0)) - 0.0).abs() < 1e-10);
        assert!((result.distance(VertexId::new(8)) - 0.0).abs() < 1e-10);

        // Middle vertex (4) should be equidistant from both sources
        // Actually it depends on the exact grid structure, but should be finite
        assert!(result.is_reachable(VertexId::new(4)));
    }

    #[test]
    fn test_farthest_vertex() {
        let mesh = create_single_triangle();
        let result = dijkstra(&mesh, VertexId::new(0), &DijkstraOptions::default());

        let (farthest, dist) = result.farthest_vertex().unwrap();

        // Farthest vertex should be vertex 2 (the apex)
        assert_eq!(farthest, VertexId::new(2));
        let expected_dist = (0.5_f64.powi(2) + 1.0_f64.powi(2)).sqrt();
        assert!((dist - expected_dist).abs() < 1e-10);
    }

    #[test]
    fn test_dijkstra_empty_mesh() {
        let mesh: HalfEdgeMesh = HalfEdgeMesh::new();
        let result = dijkstra(&mesh, VertexId::new(0), &DijkstraOptions::default());

        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_dijkstra_no_sources() {
        let mesh = create_single_triangle();
        let result = dijkstra_multiple(&mesh, &[], &DijkstraOptions::default());

        // All vertices should be unreachable
        assert_eq!(result.reachable_count(), 0);
    }

    #[test]
    fn test_dijkstra_preserves_triangle_inequality() {
        let mesh = create_grid_mesh(3);
        let result = dijkstra(&mesh, VertexId::new(0), &DijkstraOptions::default());

        // For any two adjacent vertices, the distance difference should be at most the edge length
        for v in mesh.vertex_ids() {
            let d_v = result.distance(v);
            if !d_v.is_finite() {
                continue;
            }

            for he in mesh.vertex_halfedges(v) {
                let u = mesh.dest(he);
                let d_u = result.distance(u);
                let edge_len = mesh.edge_length(he);

                // Triangle inequality: |d_v - d_u| <= edge_len
                assert!(
                    (d_v - d_u).abs() <= edge_len + 1e-10,
                    "Triangle inequality violated: |{} - {}| > {}",
                    d_v,
                    d_u,
                    edge_len
                );
            }
        }
    }
}
