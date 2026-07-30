# morsel

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="images/morsel_banner_dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="images/morsel_banner_light.svg">
  <img alt="morsel" src="images/morsel_banner_dark.svg">
</picture>

[![CI](https://github.com/timthirion/morsel/actions/workflows/ci.yml/badge.svg)](https://github.com/timthirion/morsel/actions/workflows/ci.yml)

Mesh processing in Rust

> 📐 Direction lives in [`plans/ROADMAP.md`](plans/ROADMAP.md); conventions live in
> [`AGENTS.md`](AGENTS.md).

## Gallery

Every image below is produced by the CLI and the viewer in this repository, from the
meshes in [`examples/`](examples/), with the commands shown. Regenerate any of them
with `morsel-view … --screenshot`.

<p align="center">
  <img src="images/examples/torus-curvature.png" alt="Gaussian curvature on a torus" width="49%">
  <img src="images/examples/bunny-uv.png" alt="Cylindrical UV parameterization on the Stanford bunny" width="49%">
</p>

**Left — Gaussian curvature.** Red is positive, blue negative. The colouring is not
decorative: on this torus (`R = 1`, `r = 0.35`) the analytic curvature runs from
`+2.116` on the outer equator to `−4.396` on the inner, and the estimator lands
within 1% of both, which `tests/curvature_analytic.rs` asserts.

```sh
morsel-view examples/torus.obj --curvature gaussian --screenshot out.png
```

**Right — UV parameterization,** shown with a test grid texture. The ragged seam down
the bunny's left side is where cylindrical projection wraps at `atan2`'s branch cut,
and is exactly the artefact a conformal method avoids. LSCM and ARAP need disk
topology, which this mesh does not have — it has four boundary loops around the base
— so reaching them means cutting it open first, with `--cut`.

```sh
morsel-view examples/stanford-bunny.obj --texture images/UV.png --screenshot out.png
```

### Geodesic distance

<p align="center">
  <img src="images/examples/bunny-geodesic.png" alt="Geodesic distance from a source vertex on the Stanford bunny, shown with isolines" width="66%">
</p>

Distance from a single source vertex on the bunny's head, by the heat method — two
linear solves rather than a graph search, so distance is measured *across* faces
rather than along edges. The bands are the readable part: evenly spaced rings mean
the field really is a distance, and they would bunch or kink where it is not.

`--geodesic-dijkstra` colours by graph distance instead. On this mesh the two look
nearly identical, because the bunny is triangulated finely enough that edges
approximate geodesics well — so there is deliberately no side-by-side here. The gap
opens on coarser or more structured meshes: on `cylinder.obj`, whose geodesics are
helices that follow no edge, Dijkstra overestimates by 15% on average and up to 41%,
which `tests/geodesic_analytic.rs` asserts against the exact unrolled answer.

```sh
morsel-view examples/stanford-bunny.obj --geodesic 0 --screenshot out.png
```

### Smoothing: Laplacian shrinks, Taubin does not

<p align="center">
  <img src="images/examples/smooth-original.png" alt="Original sphere" width="32%">
  <img src="images/examples/smooth-laplacian.png" alt="After Laplacian smoothing" width="32%">
  <img src="images/examples/smooth-taubin.png" alt="After Taubin smoothing" width="32%">
</p>

Original, then 20 iterations of Laplacian and of Taubin smoothing at `λ = 0.5`. All
three use the **same camera**, so the sizes are directly comparable — Laplacian loses
**79%** of the volume while Taubin gains 1.8%, which is the shrinkage resistance it is
named for. Note also that uniform Laplacian weights distort the shape into an egg,
because this sphere's triangulation is denser near the poles; that is what cotangent
weights exist to fix.

```sh
morsel smooth examples/sphere.obj out.obj --method taubin --iterations 20
morsel-view out.obj --screenshot out.png --distance 1.45
```

### Loop subdivision: 12 → 48 → 192 faces

<p align="center">
  <img src="images/examples/subdiv-0.png" alt="Closed cube, 12 faces" width="32%">
  <img src="images/examples/subdiv-1.png" alt="One Loop subdivision, 48 faces" width="32%">
  <img src="images/examples/subdiv-2.png" alt="Two Loop subdivisions, 192 faces" width="32%">
</p>

Each step splits every triangle into four and preserves the Euler characteristic
exactly. The surface also visibly contracts, which is correct rather than a bug: Loop
is an approximating scheme, so the limit surface lies inside its control mesh. Here
the volume goes `1.000 → 0.476 → 0.397`, converging.

```sh
morsel subdivide examples/cube-closed.obj out.obj --method loop --iterations 2
morsel-view out.obj --screenshot out.png --distance 1.8 --wireframe
```

## Tools

### morsel

Command-line tool for mesh processing algorithms.

**Install:**
```bash
cargo install --path . --features cli
```

**Commands:**

| Command | Description |
|---------|-------------|
| `info` | Display mesh information (vertices, faces, area, curvature) |
| `smooth` | Smooth a mesh (laplacian, taubin, cotangent) |
| `subdivide` | Subdivide a mesh (loop, catmull-clark) |
| `decimate` | Simplify a mesh using QEM |
| `remesh` | Remesh to improve triangle quality |
| `parameterize` | Compute UV coordinates (cylindrical, LSCM, ARAP, OMT) |
| `geodesic` | Geodesic distance from a source vertex (heat method, Dijkstra) |

**Examples:**
```bash
# Show mesh info
morsel info model.obj
morsel info model.obj --curvature

# Smooth a mesh
morsel smooth input.obj output.obj --method taubin --iterations 5

# Subdivide a mesh
morsel subdivide input.obj output.obj --method loop --iterations 2

# Decimate to 50% of faces
morsel decimate input.obj output.obj --ratio 0.5

# Decimate to exactly 1000 faces
morsel decimate input.obj output.obj --faces 1000

# Remesh with target edge length
morsel remesh input.obj output.obj --target-length 0.1

# UV-unwrap an open (disk-topology) mesh, angle-preserving
morsel parameterize patch.obj patch_uv.obj --method lscm

# Area-preserving: LSCM followed by an optimal-mass-transport correction
morsel parameterize patch.obj patch_uv.obj --method omt

# Cut a non-disk mesh open first, and flatten in one step
morsel parameterize sphere.obj sphere_uv.obj --method lscm --cut

# Or cut on its own, to inspect the seams
morsel cut examples/stanford-bunny.obj bunny_cut.obj
```

```sh
# Geodesic distance, and the graph distance it improves on
morsel geodesic examples/spherical-cap.obj --source 0 --target 200
morsel geodesic examples/spherical-cap.obj --source 0 --method dijkstra
```

`lscm`, `arap` and `omt` require **disk topology** — exactly one boundary loop. A
closed mesh has none and an annulus has two, and both are refused by name rather
than silently producing a degenerate map. `--cut` gets there first: `cut` joins
boundary loops along the shortest path between them, and slits a closed surface
open, so the sphere, the cylinder and the bunny all become flattenable. It refuses
genus > 0 — a torus needs handle loops that no shortest path between boundaries will
find — and refuses disconnected input, since the genus is a sum over components.
`omt` reports area distortion before and after, so the correction it applies is
visible.

Cutting is pure surgery: it duplicates vertices along the seam and rewires face
corners, leaving every face and every position exactly where it was. It is the seam
*placement* that is unambitious — a shortest path is short, not well placed, and a
short slit on a sphere leaves a lot of distortion behind (LSCM's worst area ratio on
the cut sphere is 8.1). Choosing seams to minimise distortion is a separate problem.

`geodesic` defaults to the heat method, which measures distance across faces.
`--method dijkstra` walks the edge graph instead and can only overestimate — by 15%
on average and up to 41% on `cylinder.obj`, whose geodesics are helices that follow
no edge.

Run `morsel <command> --help` for detailed options.

## Viewer

`morsel-view` renders a mesh interactively, and will also render one frame straight
to a PNG, which is how every image in the gallery above was produced.

```sh
cargo build --release --features viewer

# Interactive: drag to orbit, scroll to zoom, W wireframe, C vertex colours
./target/release/morsel-view examples/torus.obj --curvature gaussian

# Offscreen: no window, no display needed
./target/release/morsel-view examples/torus.obj --curvature gaussian \
    --screenshot out.png --size 880x620
```

| Option | Effect |
|--------|--------|
| `--curvature mean\|gaussian` | Colour vertices by curvature |
| `--geodesic <vertex>` | Colour by geodesic distance from a source, with isolines |
| `--geodesic-dijkstra` | Use graph distance instead of the heat method |
| `--texture <file>` | Apply a texture, using UVs from the mesh |
| `--parameterize` | Compute cylindrical UVs first |
| `--screenshot <file>` | Render one frame to PNG and exit |
| `--size WxH` | Screenshot size, default `1200x900` |
| `--azimuth`, `--elevation` | Camera angles in radians |
| `--distance <d>` | Fixed camera distance |
| `--wireframe` | Draw the wireframe instead of solid shading |

`--distance` matters for comparisons. Without it the camera frames each mesh to fill
the view, so a shrunken result renders at the same apparent size as its input and the
difference becomes invisible — which is why the smoothing row above passes the same
distance to all three.

---

### morsel-view

A 3D mesh viewer for inspecting meshes with optional texture support.

**Install:**
```bash
cargo install --path . --features viewer
```

**Usage:**
```bash
morsel-view path/to/mesh.obj
morsel-view mesh.obj --texture texture.png --parameterize
```

**Options:**

| Option | Description |
|--------|-------------|
| `--texture <file>` | Load a texture image (PNG, JPG, etc.) |
| `--parameterize` | Compute UV coordinates (cylindrical projection) |

Supported mesh formats: `.obj`, `.stl`, `.ply`, `.gltf`, `.glb`

**Controls:**

| Input | Action |
|-------|--------|
| Left mouse drag | Rotate camera |
| Scroll wheel | Zoom in/out |
| `W` | Toggle wireframe |
| `B` | Toggle backface culling |
| `T` | Toggle textured mode |
| `R` | Reset camera |
| `Escape` | Quit |
