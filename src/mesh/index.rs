//! Index types for mesh elements.
//!
//! This module provides type-safe index wrappers for vertices, half-edges, and faces.
//! The indices are generic over the underlying integer type to support meshes of
//! different sizes (u16 for small meshes, u32 for typical meshes, u64 for massive meshes).

use std::fmt::{self, Debug};
use std::hash::Hash;

/// Trait for types that can be used as mesh indices.
///
/// This trait is implemented for `u16`, `u32`, and `u64`, allowing users to choose
/// the appropriate index size for their mesh.
pub trait MeshIndex: Copy + Clone + Eq + PartialEq + Ord + PartialOrd + Hash + Debug + Send + Sync + 'static {
    /// The maximum valid index value.
    const MAX: Self;

    /// A sentinel value representing an invalid/null index.
    const INVALID: Self;

    /// Convert from usize to this index type.
    ///
    /// # Panics
    /// Panics if the value is too large for this index type.
    fn from_usize(v: usize) -> Self;

    /// Convert to usize.
    fn to_usize(self) -> usize;

    /// Check if this is a valid (non-sentinel) index.
    fn is_valid(self) -> bool {
        self != Self::INVALID
    }
}

impl MeshIndex for u16 {
    const MAX: Self = u16::MAX - 1;
    const INVALID: Self = u16::MAX;

    #[inline]
    fn from_usize(v: usize) -> Self {
        debug_assert!(v <= Self::MAX as usize, "index {} too large for u16", v);
        v as u16
    }

    #[inline]
    fn to_usize(self) -> usize {
        self as usize
    }
}

impl MeshIndex for u32 {
    const MAX: Self = u32::MAX - 1;
    const INVALID: Self = u32::MAX;

    #[inline]
    fn from_usize(v: usize) -> Self {
        debug_assert!(v <= Self::MAX as usize, "index {} too large for u32", v);
        v as u32
    }

    #[inline]
    fn to_usize(self) -> usize {
        self as usize
    }
}

impl MeshIndex for u64 {
    const MAX: Self = u64::MAX - 1;
    const INVALID: Self = u64::MAX;

    #[inline]
    fn from_usize(v: usize) -> Self {
        v as u64
    }

    #[inline]
    fn to_usize(self) -> usize {
        self as usize
    }
}

/// A type-safe vertex index.
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(transparent)]
pub struct VertexId<I: MeshIndex = u32>(I);

/// A type-safe half-edge index.
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(transparent)]
pub struct HalfEdgeId<I: MeshIndex = u32>(I);

/// A type-safe face index.
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(transparent)]
pub struct FaceId<I: MeshIndex = u32>(I);

/// A type-safe edge index (for full edges, not half-edges).
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(transparent)]
pub struct EdgeId<I: MeshIndex = u32>(I);

macro_rules! impl_index_type {
    ($name:ident, $display:literal) => {
        impl<I: MeshIndex> $name<I> {
            /// Create a new index from a raw value.
            #[inline]
            pub fn new(index: usize) -> Self {
                Self(I::from_usize(index))
            }

            /// Create an invalid/null index.
            #[inline]
            pub fn invalid() -> Self {
                Self(I::INVALID)
            }

            /// Get the raw index value.
            #[inline]
            pub fn index(self) -> usize {
                self.0.to_usize()
            }

            /// Get the raw value of the underlying type.
            #[inline]
            pub fn raw(self) -> I {
                self.0
            }

            /// Check if this is a valid (non-null) index.
            #[inline]
            pub fn is_valid(self) -> bool {
                self.0.is_valid()
            }
        }

        impl<I: MeshIndex> Debug for $name<I> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                if self.is_valid() {
                    write!(f, "{}({})", $display, self.index())
                } else {
                    write!(f, "{}(INVALID)", $display)
                }
            }
        }

        impl<I: MeshIndex> Default for $name<I> {
            fn default() -> Self {
                Self::invalid()
            }
        }

        impl<I: MeshIndex> From<usize> for $name<I> {
            fn from(v: usize) -> Self {
                Self::new(v)
            }
        }
    };
}

impl_index_type!(VertexId, "V");
impl_index_type!(HalfEdgeId, "HE");
impl_index_type!(FaceId, "F");
impl_index_type!(EdgeId, "E");

/// Marker trait for index types that can be used together in a mesh.
///
/// This ensures that all indices in a mesh use the same underlying integer type.
pub trait MeshIndexFamily {
    /// The underlying integer type for indices.
    type Index: MeshIndex;
    /// The vertex index type.
    type Vertex: From<usize> + Copy;
    /// The half-edge index type.
    type HalfEdge: From<usize> + Copy;
    /// The face index type.
    type Face: From<usize> + Copy;
    /// The edge index type.
    type Edge: From<usize> + Copy;
}

/// Default index family using u32 indices.
#[derive(Debug, Clone, Copy)]
pub struct DefaultIndexFamily;

impl MeshIndexFamily for DefaultIndexFamily {
    type Index = u32;
    type Vertex = VertexId<u32>;
    type HalfEdge = HalfEdgeId<u32>;
    type Face = FaceId<u32>;
    type Edge = EdgeId<u32>;
}

/// Small index family using u16 indices (for meshes with < 65k elements).
#[derive(Debug, Clone, Copy)]
pub struct SmallIndexFamily;

impl MeshIndexFamily for SmallIndexFamily {
    type Index = u16;
    type Vertex = VertexId<u16>;
    type HalfEdge = HalfEdgeId<u16>;
    type Face = FaceId<u16>;
    type Edge = EdgeId<u16>;
}

/// Large index family using u64 indices (for massive meshes).
#[derive(Debug, Clone, Copy)]
pub struct LargeIndexFamily;

impl MeshIndexFamily for LargeIndexFamily {
    type Index = u64;
    type Vertex = VertexId<u64>;
    type HalfEdge = HalfEdgeId<u64>;
    type Face = FaceId<u64>;
    type Edge = EdgeId<u64>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vertex_id() {
        let v: VertexId = VertexId::new(42);
        assert_eq!(v.index(), 42);
        assert!(v.is_valid());

        let invalid: VertexId = VertexId::invalid();
        assert!(!invalid.is_valid());
    }

    #[test]
    fn test_type_safety() {
        // These are different types and cannot be mixed
        let v: VertexId = VertexId::new(0);
        let he: HalfEdgeId = HalfEdgeId::new(0);
        let f: FaceId = FaceId::new(0);

        // All have the same raw value but are distinct types
        assert_eq!(v.index(), he.index());
        assert_eq!(he.index(), f.index());
    }

    #[test]
    fn test_small_indices() {
        let v: VertexId<u16> = VertexId::new(1000);
        assert_eq!(v.index(), 1000);
    }

    #[test]
    fn test_debug_format() {
        let v: VertexId = VertexId::new(42);
        assert_eq!(format!("{:?}", v), "V(42)");

        let invalid: VertexId = VertexId::invalid();
        assert_eq!(format!("{:?}", invalid), "V(INVALID)");
    }
}
