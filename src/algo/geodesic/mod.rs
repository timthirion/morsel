//! Geodesic distance computation on meshes.
//!
//! This module provides algorithms for computing geodesic distances (shortest
//! paths along the surface) on triangle meshes.
//!
//! # Available Algorithms
//!
//! - [`dijkstra`]: Graph-based shortest path along mesh edges (exact on edge graph)
//! - [`heat_method`]: Heat diffusion based geodesics (smooth approximation)
//!
//! # Example
//!
//! ```no_run
//! use morsel::prelude::*;
//! use morsel::algo::geodesic::{dijkstra, DijkstraOptions, GeodesicResult};
//!
//! let mesh: HalfEdgeMesh = morsel::io::load("mesh.obj").unwrap();
//!
//! // Compute distances from vertex 0
//! let source = VertexId::new(0);
//! let result = dijkstra(&mesh, source, &DijkstraOptions::default());
//!
//! // Get distance to another vertex
//! let target = VertexId::new(10);
//! let dist = result.distance(target);
//! println!("Distance: {}", dist);
//!
//! // Find the farthest vertex
//! if let Some((v, d)) = result.farthest_vertex() {
//!     println!("Farthest vertex: {:?} at distance {}", v, d);
//! }
//! ```

mod dijkstra;
mod heat;

use std::marker::PhantomData;

pub use dijkstra::{dijkstra, dijkstra_multiple, DijkstraOptions};
pub use heat::{heat_method, heat_method_multiple, HeatMethodOptions};

use crate::mesh::{MeshIndex, VertexId};

/// Result of geodesic distance computation.
///
/// Contains distances from source vertex/vertices to all other vertices,
/// and optionally predecessor information for path reconstruction.
#[derive(Debug, Clone)]
pub struct GeodesicResult<I: MeshIndex = u32> {
    /// Distance from source(s) to each vertex.
    /// `f64::INFINITY` if the vertex is unreachable.
    distances: Vec<f64>,

    /// Predecessor vertex for each vertex (for path reconstruction).
    /// `None` if predecessors weren't computed or vertex is a source/unreachable.
    predecessors: Option<Vec<Option<usize>>>,

    /// Phantom data for the index type.
    _marker: PhantomData<I>,
}

impl<I: MeshIndex> GeodesicResult<I> {
    /// Create a new geodesic result.
    pub(crate) fn new(distances: Vec<f64>, predecessors: Option<Vec<Option<usize>>>) -> Self {
        Self {
            distances,
            predecessors,
            _marker: PhantomData,
        }
    }

    /// Get the distance to a vertex.
    ///
    /// Returns `f64::INFINITY` if the vertex is unreachable from the source(s).
    #[inline]
    pub fn distance(&self, v: VertexId<I>) -> f64 {
        self.distances[v.index()]
    }

    /// Get all distances as a slice.
    #[inline]
    pub fn distances(&self) -> &[f64] {
        &self.distances
    }

    /// Get the number of vertices.
    #[inline]
    pub fn len(&self) -> usize {
        self.distances.len()
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.distances.is_empty()
    }

    /// Find the vertex with the maximum finite distance from the source(s).
    ///
    /// Returns `None` if all vertices are either sources or unreachable.
    pub fn farthest_vertex(&self) -> Option<(VertexId<I>, f64)> {
        let mut max_dist = f64::NEG_INFINITY;
        let mut max_vertex = None;

        for (i, &d) in self.distances.iter().enumerate() {
            if d.is_finite() && d > max_dist {
                max_dist = d;
                max_vertex = Some(i);
            }
        }

        max_vertex.map(|i| (VertexId::new(i), max_dist))
    }

    /// Reconstruct the shortest path from a source to the given vertex.
    ///
    /// Returns `None` if:
    /// - Predecessors weren't stored (use `DijkstraOptions::with_predecessors(true)`)
    /// - The vertex is unreachable
    /// - The vertex is a source vertex (path is empty)
    ///
    /// The returned path includes both the source and target vertices.
    pub fn path_to(&self, target: VertexId<I>) -> Option<Vec<VertexId<I>>> {
        let predecessors = self.predecessors.as_ref()?;

        // Check if target is reachable
        if !self.distances[target.index()].is_finite() {
            return None;
        }

        // Reconstruct path by following predecessors
        let mut path = Vec::new();
        let mut current = target.index();

        loop {
            path.push(VertexId::new(current));

            match predecessors[current] {
                Some(pred) => current = pred,
                None => break, // Reached a source
            }

            // Safety check to prevent infinite loops
            if path.len() > self.distances.len() {
                return None;
            }
        }

        path.reverse();
        Some(path)
    }

    /// Check if a vertex is reachable from the source(s).
    #[inline]
    pub fn is_reachable(&self, v: VertexId<I>) -> bool {
        self.distances[v.index()].is_finite()
    }

    /// Count the number of reachable vertices.
    pub fn reachable_count(&self) -> usize {
        self.distances.iter().filter(|d| d.is_finite()).count()
    }

    /// Iterate over all vertices with their distances.
    pub fn iter(&self) -> impl Iterator<Item = (VertexId<I>, f64)> + '_ {
        self.distances
            .iter()
            .enumerate()
            .map(|(i, &d)| (VertexId::new(i), d))
    }

    /// Iterate over only reachable vertices with their distances.
    pub fn reachable_iter(&self) -> impl Iterator<Item = (VertexId<I>, f64)> + '_ {
        self.iter().filter(|(_, d)| d.is_finite())
    }
}
