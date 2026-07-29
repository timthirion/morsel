# examples/

Test meshes with their measured properties. Read the table before picking one:
`cube.obj` looks like a cube and is six disconnected squares, and `cylinder.obj`
has boundary but is not a disk, so the wrong choice wastes time.

| file | V | E | F | χ | components | boundary loops | topology |
|------|---|---|---|---|-----------|----------------|----------|
| `cube-closed.obj` | 8 | 18 | 12 | 2 | 1 | 0 | closed, genus 0 |
| `cube.obj` | 24 | 30 | 12 | 6 | 6 | 6 | six disjoint disks |
| `sphere.obj` | 178 | 528 | 352 | 2 | 1 | 0 | closed, genus 0 |
| `spherical-cap.obj` | 361 | 1044 | 684 | 1 | 1 | 1 | **disk** |
| `cylinder.obj` | 468 | 1332 | 864 | 0 | 1 | 2 | annulus, genus 0 |
| `torus.obj` | 1152 | 3456 | 2304 | 0 | 1 | 0 | closed, **genus 1** |
| `stanford-bunny.obj` | 2503 | 7473 | 4968 | −2 | 1 | 4 | genus 0, four holes |

Note that `cylinder.obj` and `torus.obj` share `χ = 0` while being quite different
surfaces — an open annulus and a closed handle. Useful for checking that code
computes topology rather than inferring it from the Euler characteristic.

## Closed-form values

These four were built so the right answer is known, which is what makes them worth
having. `tests/curvature_analytic.rs` asserts against every number here.

| mesh | Gaussian K | mean \|H\| | area |
|------|-----------|-----------|------|
| `sphere.obj` (r = 0.5) | 4 | 2 | 3.1416 |
| `spherical-cap.obj` (unit sphere, cut at r = 0.8) | 1 | 1 | 2.5133 |
| `cylinder.obj` (r = 1, h = 2) | **0 exactly** | 0.5 | 12.5664 |
| `torus.obj` (R = 1, r = 0.35) | −4.3956 … +2.1164 | — | 13.8174 |

Two conventions, established by measurement rather than assumption: the curvature
functions return **pointwise** values rather than integrated angle defects, so
summing them raw is meaningless; and mean curvature comes back **negative** where
the outward-normal convention gives positive, consistently, so compare magnitudes.

## What each is for

**`cube-closed.obj`** — the topology workhorse. Genuine closed manifold, shared
corners, wound outward (signed volume `+1`), and small enough to check by hand.
Reach for it when testing Euler characteristic, orientation, closed-surface
handling, or that a method rejects a mesh with no boundary. No UVs.

**`cube.obj`** — texturing and UV work only. Its corners are deliberately split
(24 vertices, not 8) so each face carries the full texture, which per-face UVs
require. That makes it six disconnected unit squares with every vertex on a
boundary. Ships with `cube.mtl`. Decimation correctly refuses it: its only interior
edges are the quad diagonals and collapsing those deletes whole patches.

**`sphere.obj`** — closed and boundary-free at a useful size. Good for curvature,
smoothing, decimation, and anything wanting a closed input bigger than 12 faces.

**`spherical-cap.obj`** — the parameterization input, and the only disk here. LSCM,
ARAP and OMT all require exactly one boundary loop, and nothing else in this set
qualifies. Curved, so a conformal map genuinely distorts area: LSCM scores
`rms = 0.158`, ARAP `0.073`, OMT `0.072`. Constant curvature also makes it a
quantitative curvature asset.

**`cylinder.obj`** — has boundary but is *not* a disk, so it is the mesh for
checking that the disk requirement is enforced. LSCM and ARAP now refuse it by
name. It is also developable — `K = 0` exactly, not approximately — which makes it
the sharpest curvature test here, since the expected answer carries no
discretization error.

**`torus.obj`** — the only asset with a handle. For anything that assumes genus 0
(most parameterization does) or that computes topology rather than trusting it.
Curvature of both signs, integrating to zero.

**`stanford-bunny.obj`** — the realistic case: genus 0 with four boundary loops
around the base. Coordinates are small (total area `0.057`), which makes it a good
check that thresholds are scale-relative rather than absolute. Being non-disk, it
is refused by the parameterizers until a cut generator exists. Ships with
`stanford-bunny.mtl`.

## Gaps

No genus-2 asset, and nothing large enough to be a performance case — a benchmark
mesh is better generated procedurally than committed. Degenerate inputs live in
code rather than here: `tests/common/mod.rs` builds eighteen of them
programmatically — non-manifold edges and vertices, slivers, cocircular lattices,
extreme coordinate scales — for the robustness sweep.
