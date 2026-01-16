# Morsel - Project Notes

A comprehensive 3D mesh processing library in Rust, designed for geometry processing research with a goal of novel research.

## Dependencies

### Local Path Dependencies (not published to crates.io)

- **exactum** (`../exactum`): Exact computational geometry library using integer/rational arithmetic
  - Pure Rust, zero dependencies, `#![forbid(unsafe_code)]`
  - Provides: `Point2<T>`/`Point3<T>` with integer coordinates (i32, i64, i128)
  - `Rational` type for exact fractional results
  - Geometric predicates: `orient2d`, `orient3d`, `incircle`, `insphere`
  - Algorithms: convex hull, Delaunay triangulation, Voronoi, polygon booleans, sweep line
  - Spatial structures: KdTree, Quadtree, Octree, RTree

- **approxum** (`../approxum`): Floating-point approximation geometry library
  - Generic over `F: Float` (f32/f64)
  - Curves: Bezier, B-spline, NURBS, arcs, Catmull-Rom, Hermite
  - Polygon ops: booleans, offsetting, triangulation, visibility, straight skeleton
  - Simplification: RDP, Visvalingam, topology-preserving
  - Sampling: Poisson disk, Sobol, Halton sequences
  - Spatial: KdTree, BVH
  - Distance: SDFs, distance transforms, Hausdorff/Frechet metrics
  - I/O: SVG parsing/export

### How to Use Both

- **Approxum**: Primary use for mesh vertex coordinates, curves, sampling, distance computations. Meshes live in floating-point space.
- **Exactum**: Robust geometric predicates for *decisions* during algorithms. When determining orientation, incircle tests, etc., convert to exact arithmetic to avoid floating-point failures.

Pattern:
```rust
// Vertices stored as f64
let mesh: HalfEdgeMesh = ...;

// But when Delaunay needs an incircle test, convert to exact:
let a = Point2::new(v0.x as i64, v0.y as i64);
let result = exactum::incircle(a, b, c, d);  // robust decision
```

## Architectural Decisions

### Data Structure: Half-Edge Mesh
- Full half-edge (doubly-connected edge list) data structure
- O(1) adjacency queries
- Stores: vertices, half-edges, faces with full connectivity

### Index Types
- Type-safe newtype wrappers: `VertexId<I>`, `HalfEdgeId<I>`, `FaceId<I>`, `EdgeId<I>`
- Generic over `MeshIndex` trait (u16, u32, u64)
- Default: u32 (handles 4 billion elements)
- Compile-time type safety prevents mixing vertex/face/edge indices

### Floating-Point Precision
- f64 only for now (simplicity, correctness for research)
- Can generalize to `F: Float` later if needed

### Linear Algebra
- **nalgebra** for all linear algebra needs
- Full-featured: sparse matrices, eigendecomposition, SVD (needed for curvature, parameterization, etc.)

### Mesh Types
- Triangle meshes first
- Quad support later
- Tet meshes much later

### Manifold Handling
- Include routines to clean non-manifold meshes and repair them

## File Formats

Supported via standard crates:
- **OBJ** (tobj) - load/save
- **STL** (stl_io) - load/save (binary)
- **PLY** (ply-rs) - load/save
- **glTF** (gltf) - load only

## Algorithm Priority Order

1. I/O + data structure (done)
2. Smoothing (Laplacian, bilateral, mean curvature flow) (done)
3. Remeshing (isotropic, anisotropic, CVT-based) (done)
4. Decimation (QEM, edge collapse) (done)
5. Subdivision (Loop, Catmull-Clark) (done)
6. Parameterization (LSCM, ARAP) (done)
7. Geodesics (Dijkstra, heat method) (done)
8. Curvature computation (done)
9. Mesh booleans

## Tools

10. CLI utility (`morsel` command) - run algorithms from command line
11. 3D mesh viewer - interactive inspection with orbit camera, solid/wireframe toggle

## Project Structure

```
morsel/
├── Cargo.toml
├── src/
│   ├── lib.rs              # Main library exports
│   ├── error.rs            # Error types (MeshError)
│   ├── mesh/
│   │   ├── mod.rs          # Mesh module exports
│   │   ├── index.rs        # VertexId, HalfEdgeId, FaceId, EdgeId, MeshIndex trait
│   │   ├── halfedge.rs     # HalfEdgeMesh, Vertex, HalfEdge, Face
│   │   └── builder.rs      # build_from_triangles, to_face_vertex
│   ├── io/
│   │   ├── mod.rs          # Format detection, load/save dispatch
│   │   ├── obj.rs          # OBJ format
│   │   ├── stl.rs          # STL format
│   │   ├── ply.rs          # PLY format
│   │   └── gltf.rs         # glTF format (load only)
│   └── algo/
│       └── mod.rs          # Algorithm modules (to be added)
└── benches/
    └── mesh_ops.rs         # Benchmarks
```

## Scope Exclusions (for now)

- Point cloud operations (mesh only)
- GPU compute (rendering is OK for viewer)
- Animation / skinning

## Research Goals

Target venue: SGP (Symposium on Geometry Processing)
Timeline: 3-5 years for novel research contribution
Focus: Computational geometry / geometry processing
