# morsel Roadmap

## Mission

Mesh processing in Rust: a library whose algorithms are correct, measured, and
explainable, with a CLI thin enough that the library is obviously the product.

Design bias:

- **Correctness against a known answer.** Every algorithm should have at least one
  input whose exact output is derivable by hand, and a test that asserts it. A
  flat planar patch must come back isometric from any parameterizer; a sphere's
  discrete curvature must approach `1/r²`. Tests that only assert "it returned
  something" are how the parameterization solvers shipped visibly wrong output
  while reporting success (see `plans/0001`).
- **A number, not an adjective.** "Better quality" means a metric moved. Area
  distortion, Hausdorff distance, triangle-quality histograms, iteration counts —
  named, measured, and recorded next to the claim.
- **Say what isn't true yet.** An algorithm that only works below some size, or
  only on some topology, states that in its docs and warns at runtime rather than
  degrading quietly.

## Where we are today

A working half-edge mesh library and CLI.

- **Core** — `HalfEdgeMesh` with index-generic storage, face/vertex traversal,
  boundary queries; builders from face-vertex soup.
- **IO** — OBJ (incl. UVs and MTL), PLY, STL, glTF.
- **Algorithms** — smoothing (Laplacian, Taubin, cotangent), subdivision (Loop,
  Catmull-Clark), decimation (QEM), isotropic remeshing, curvature, and UV
  parameterization (cylindrical, LSCM, ARAP, OMT).
- **Parallelism** — `rayon` throughout the heavier passes.
- **Verification posture** — property-based correctness tests for
  parameterization (`tests/parameterize_correctness.rs`); the rest of the library
  is covered by unit tests of varying strength.

The parameterization stack was substantially repaired in July 2026: LSCM and ARAP
both imposed pin constraints by penalty, which coupled solver accuracy to the
penalty magnitude and produced wrong maps above ~50 vertices while CG reported
convergence. Both now eliminate their pinned degrees of freedom. OMT was rewritten
twice — it had been iterating a transport step as Lloyd relaxation, and then
estimating power-cell areas by grid sampling, whose noise floor sat above the
convergence tolerance it was checked against. Cells are now exact polygons, and
converged area distortion falls to 16–36% of conformal, improving with refinement.

A theme worth naming, since it caused three separate bugs here: in each case a
*measurement was noisier than the threshold it was compared against* — a penalty
inflating `‖b‖` past CG's relative residual test, twice, and a sampled area
estimator's `1/√k` error exceeding the transport tolerance. Whenever a solver
reports success while its output is visibly wrong, suspect that shape first.

## Phases

Phases are roughly ordered; boundaries are soft. Each becomes one or more
`plans/NNNN-*.md` as work starts.

### Formal verification in Lean 4 — `plans/0001`
Machine-checked proofs for the parts of morsel that admit them: the combinatorial
half-edge invariants, and the mathematical specifications the numerical code
implements. Deliberately does **not** attempt f64 numerics. **active**

### Candidate directions — not yet committed

These are the author's to shape; listed so the folder isn't empty of forward
direction, not because they've been decided.

- **Faster OMT transport solve.** Exact power cells landed (July 2026) and moved
  the bottleneck: the measurement is now exact, but the first-order ascent needs
  iterations growing linearly in vertex count, and each iteration scans every
  site pair. Two fixes, both known: a damped Newton step on the Kantorovich dual
  (Kitagawa–Mérigot–Thibert) for `O(tens)` iterations instead of thousands, and
  spatial pruning of rival sites for `O(n·k)` instead of `O(n²)` per iteration.
  Converged quality already improves with refinement — 15.7% of conformal at 1089
  vertices — so this is purely about making that budget affordable.
- **Parameterization quality** — boundary vertices that slide along the boundary
  curve rather than being pinned or freed; a seam/cut generator so closed meshes
  can be unwrapped at all.
- **Measurement harness** — Hausdorff and triangle-quality metrics as first-class
  reportable numbers, so decimation and remeshing claims are defensible the way
  the parameterization ones now are.
- **Viewer** — the existing curvature viewer grown into something that can show a
  UV atlas and distortion heatmaps.
