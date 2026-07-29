# examples/

Test meshes, with their measured properties. Picking the wrong one wastes time —
`cube.obj` looks like a cube and is six disconnected squares — so the table is
here to be read before use.

| file | V | E | F | χ | components | boundary | closed? |
|------|---|---|---|---|-----------|----------|---------|
| `cube.obj` | 24 | 30 | 12 | 6 | 6 | 24 verts, 6 loops | no |
| `cube-closed.obj` | 8 | 18 | 12 | 2 | 1 | none | yes |
| `sphere.obj` | 178 | 528 | 352 | 2 | 1 | none | yes |
| `stanford-bunny.obj` | 2503 | 7473 | 4968 | −2 | 1 | 42 verts, 4 loops | no |

## What each is for

**`cube.obj`** — texturing and UV work. Its corners are split (24 vertices, not 8)
so that each face can carry the full texture; per-face UVs cannot be expressed
with shared corners. That split makes it six disconnected unit squares rather than
a cube, every vertex on a boundary. Ships with `cube.mtl` and four `vt` corners
reused per face. **Not** suitable for topology: decimation correctly refuses it,
because its only interior edges are the quad diagonals and collapsing those deletes
whole patches.

**`cube-closed.obj`** — the topology counterpart. A genuine closed manifold with
shared corners, consistently wound outward (signed volume `+1`, surface area `6`),
small enough to verify by hand. Reach for this when testing Euler characteristic,
orientation, closed-surface handling, or that a method correctly rejects a mesh
with no boundary. No UVs.

**`sphere.obj`** — a closed, boundary-free surface at a useful size (352 faces).
Good for curvature (compare against `1/r²`), smoothing, decimation, and anything
that wants a closed input bigger than twelve faces.

**`stanford-bunny.obj`** — the realistic case, and the only one with genuine
boundary: four loops around the base, 42 boundary vertices, genus 0. Coordinates
are small (total area `0.057`), which makes it a decent check that thresholds are
scale-relative rather than absolute. Ships with `stanford-bunny.mtl`.

## Gaps worth filling

There is no test asset with genus > 0 (a torus), none with a deliberately
degenerate configuration, and nothing large enough to be a performance case. The
degenerate inputs live in code instead, as `tests/common/mod.rs`, which builds
eighteen awkward meshes programmatically — non-manifold edges and vertices,
slivers, cocircular lattices, extreme coordinate scales — for the robustness sweep.
