//! UV coordinate storage.
//!
//! This module provides the [`UVMap`] type for storing 2D parameterization
//! coordinates for mesh vertices.

use std::marker::PhantomData;

use nalgebra::Point2;

use crate::mesh::{MeshIndex, VertexId};

/// UV coordinates for mesh vertices.
///
/// This structure stores the 2D parameterization (UV coordinates) computed
/// for each vertex in a mesh. UV coordinates are typically in the range [0, 1]
/// but may extend outside this range depending on the parameterization method.
///
/// # Example
///
/// ```no_run
/// use morsel::algo::parameterize::UVMap;
/// use morsel::mesh::VertexId;
///
/// // UVMap is typically returned from parameterization algorithms
/// // let uv_map = lscm(&mesh, &options).unwrap();
///
/// // Access UV coordinates by vertex ID
/// // let uv = uv_map.get(VertexId::new(0));
/// // println!("UV: ({}, {})", uv.x, uv.y);
/// ```
#[derive(Debug, Clone)]
pub struct UVMap<I: MeshIndex = u32> {
    /// UV coordinates indexed by vertex ID.
    coords: Vec<Point2<f64>>,
    /// Phantom data for the index type.
    _marker: PhantomData<I>,
}

impl<I: MeshIndex> UVMap<I> {
    /// Create a new UV map with the given coordinates.
    ///
    /// The coordinates should be indexed by vertex ID (index 0 corresponds to
    /// vertex 0, etc.).
    pub fn new(coords: Vec<Point2<f64>>) -> Self {
        Self {
            coords,
            _marker: PhantomData,
        }
    }

    /// Create a UV map filled with zeros.
    pub fn zeros(n: usize) -> Self {
        Self {
            coords: vec![Point2::origin(); n],
            _marker: PhantomData,
        }
    }

    /// Get the UV coordinates for a vertex.
    #[inline]
    pub fn get(&self, v: VertexId<I>) -> Point2<f64> {
        self.coords[v.index()]
    }

    /// Get a mutable reference to UV coordinates for a vertex.
    #[inline]
    pub fn get_mut(&mut self, v: VertexId<I>) -> &mut Point2<f64> {
        &mut self.coords[v.index()]
    }

    /// Set the UV coordinates for a vertex.
    #[inline]
    pub fn set(&mut self, v: VertexId<I>, uv: Point2<f64>) {
        self.coords[v.index()] = uv;
    }

    /// Get the number of UV coordinates.
    #[inline]
    pub fn len(&self) -> usize {
        self.coords.len()
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.coords.is_empty()
    }

    /// Iterate over all UV coordinates with their vertex IDs.
    pub fn iter(&self) -> impl Iterator<Item = (VertexId<I>, Point2<f64>)> + '_ {
        self.coords
            .iter()
            .enumerate()
            .map(|(i, &uv)| (VertexId::new(i), uv))
    }

    /// Get the raw coordinates slice.
    pub fn as_slice(&self) -> &[Point2<f64>] {
        &self.coords
    }

    /// Get a mutable slice of coordinates.
    pub fn as_mut_slice(&mut self) -> &mut [Point2<f64>] {
        &mut self.coords
    }

    /// Compute the bounding box of the UV coordinates.
    ///
    /// Returns `None` if the UV map is empty.
    pub fn bounding_box(&self) -> Option<(Point2<f64>, Point2<f64>)> {
        if self.coords.is_empty() {
            return None;
        }

        let mut min = self.coords[0];
        let mut max = self.coords[0];

        for uv in &self.coords {
            min.x = min.x.min(uv.x);
            min.y = min.y.min(uv.y);
            max.x = max.x.max(uv.x);
            max.y = max.y.max(uv.y);
        }

        Some((min, max))
    }

    /// Normalize UV coordinates to fit within [0, 1] range.
    ///
    /// Maintains aspect ratio by scaling uniformly based on the larger dimension.
    pub fn normalize(&mut self) {
        if let Some((min, max)) = self.bounding_box() {
            let scale_x = max.x - min.x;
            let scale_y = max.y - min.y;
            let scale = scale_x.max(scale_y);

            if scale > 1e-10 {
                for uv in &mut self.coords {
                    uv.x = (uv.x - min.x) / scale;
                    uv.y = (uv.y - min.y) / scale;
                }
            }
        }
    }

    /// Normalize UV coordinates to fit within [0, 1] x [0, 1], stretching to fill.
    ///
    /// Does not maintain aspect ratio.
    pub fn normalize_stretch(&mut self) {
        if let Some((min, max)) = self.bounding_box() {
            let scale_x = max.x - min.x;
            let scale_y = max.y - min.y;

            if scale_x > 1e-10 && scale_y > 1e-10 {
                for uv in &mut self.coords {
                    uv.x = (uv.x - min.x) / scale_x;
                    uv.y = (uv.y - min.y) / scale_y;
                }
            }
        }
    }

    /// Compute the total area in UV space.
    ///
    /// Requires the face indices to compute triangle areas.
    pub fn total_area(&self, faces: &[[usize; 3]]) -> f64 {
        let mut total = 0.0;
        for face in faces {
            let p0 = self.coords[face[0]];
            let p1 = self.coords[face[1]];
            let p2 = self.coords[face[2]];

            // 2D cross product for signed area
            let area = 0.5
                * ((p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y)).abs();
            total += area;
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uv_map_basic() {
        let coords = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(0.5, 1.0),
        ];
        let uv_map: UVMap<u32> = UVMap::new(coords);

        assert_eq!(uv_map.len(), 3);
        assert!(!uv_map.is_empty());

        let v0: VertexId<u32> = VertexId::new(0);
        let v1: VertexId<u32> = VertexId::new(1);
        let v2: VertexId<u32> = VertexId::new(2);

        assert_eq!(uv_map.get(v0), Point2::new(0.0, 0.0));
        assert_eq!(uv_map.get(v1), Point2::new(1.0, 0.0));
        assert_eq!(uv_map.get(v2), Point2::new(0.5, 1.0));
    }

    #[test]
    fn test_uv_map_bounding_box() {
        let coords = vec![
            Point2::new(-1.0, 0.5),
            Point2::new(2.0, -0.5),
            Point2::new(0.5, 3.0),
        ];
        let uv_map: UVMap<u32> = UVMap::new(coords);

        let (min, max) = uv_map.bounding_box().unwrap();
        assert_eq!(min, Point2::new(-1.0, -0.5));
        assert_eq!(max, Point2::new(2.0, 3.0));
    }

    #[test]
    fn test_uv_map_normalize() {
        let coords = vec![
            Point2::new(0.0, 0.0),
            Point2::new(4.0, 0.0),
            Point2::new(2.0, 2.0),
        ];
        let mut uv_map: UVMap<u32> = UVMap::new(coords);
        uv_map.normalize();

        let (min, max) = uv_map.bounding_box().unwrap();
        assert!((min.x - 0.0).abs() < 1e-10);
        assert!((min.y - 0.0).abs() < 1e-10);
        // Max x should be 1.0 (since x range is larger)
        assert!((max.x - 1.0).abs() < 1e-10);
        // Max y should be 0.5 (maintains aspect ratio)
        assert!((max.y - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_uv_map_total_area() {
        let coords = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(0.0, 1.0),
        ];
        let uv_map: UVMap<u32> = UVMap::new(coords);
        let faces = vec![[0, 1, 2]];

        let area = uv_map.total_area(&faces);
        assert!((area - 0.5).abs() < 1e-10);
    }
}
