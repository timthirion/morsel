# Research program: structure-first geometry processing

- **Status:** active
- **Last updated:** 2026-07-29
- **Last touched on:** macOS laptop, Claude Code session — spine rewritten around
  algorithm design; first obstruction measured

## Goal

Produce novel geometric algorithms, using structure preservation as the design
discipline and a proof assistant as the instrument that makes the discipline
rigorous.

The spine, in order:

1. **Design.** State the structural properties an operator must satisfy, then
   *derive* the operator. The specification determines the algorithm.
2. **Impossibility.** When the properties cannot all hold, prove they cannot. A
   sharp obstruction is a novel result and it redirects everyone downstream.
3. **Audit.** What falls out along the way — existing discretizations that do not
   satisfy their assumed properties. Real, publishable, and the honest source of
   early results, but a by-product rather than the thesis.

Time horizon is **2–10 years**, deliberately. There is no publication pressure and
the program is designed around that.

## Context

### Why this spine, and not "we verified a library"

An audit is not an algorithm. "These discretizations do not satisfy their stated
properties" is a critique; SGP reviewers will ask what the method is. A program
whose output is findings about other people's methods is a support activity
dressed as a contribution.

But structure preservation is not only an audit tool — **it is how discrete
operators get invented.** Discrete exterior calculus falls out of demanding a
discrete Stokes theorem. The cotangent Laplacian is not a guess; it is the unique
FEM Laplacian on piecewise-linear functions. Intrinsic Delaunay flipping is
motivated by demanding positive weights. In each case the properties came first
and the operator was derived.

We did a miniature version in July 2026 without noticing. The OMT mass-lumping
fix was not "barycentric is wrong, try something else." It was: *what lumping is
consistent with a dual-cell partition?* — and mixed Voronoi is **forced**. The
specification determined the algorithm. That is the move to scale up.

### The shape a formal instrument is best at: impossibility

The template exists and is well known: Wardetzky, Mathur, Kälberer & Grinspun,
*Discrete Laplace Operators: No Free Lunch* (SGP 2007) — no discrete Laplacian on
general meshes is simultaneously symmetric, local, linearly precise, and
positive-weighted. That is a structure-preservation impossibility, it is highly
cited, and it reframed what everyone after it attempted.

A proof assistant is the right instrument for such results because the value is
in the *generality* of the "no" — not in the two or three special cases one can
check by hand.

### First obstruction, already measured

Not conjecture. On a paraboloid patch with the transport driven to convergence
(worst per-cell area error `9.7e-10`; 20,000 iterations change nothing past 783):

| | LSCM | OMT, converged |
|---|---|---|
| n=8, interior faces | [0.730, 1.178] | **[0.968, 1.053]** |
| n=16, interior faces | [0.725, 1.395] | **[0.970, 1.033]** |

Cell areas are matched to nine digits, and **per-triangle** area distortion still
plateaus at ±3–5%. More iterations do not touch it. The reason is a counting
argument, not a numerical one.

For a triangulated disk, Euler's formula with `3F = 2E − B` gives

```
F = 2V − B − 2
```

The semi-discrete transport has `V − 1` weight degrees of freedom (one lost to
the dual's invariance under a constant shift), while per-triangle area
preservation asks for `F` equalities. Substituting gives the deficit *exactly*:

```
F − (V − 1) = V − B − 1 = (interior vertices) − 1
```

**So exact dual-cell area preservation cannot imply per-triangle area
preservation, for any mesh with two or more interior vertices** — and the gap
grows linearly with refinement. That is provable by counting, it explains the
measured plateau, and it constrains a whole family of vertex-weighted transport
methods rather than one implementation.

The sharp form came from writing the test: an initial "roughly twice as many
faces as vertices" reading failed on a single-quad mesh, where every vertex is on
the boundary and there is no deficit at all. The exact statement is better than
the asymptotic one, and it is now pinned down in
`tests/omt_dof_deficit.rs`.

A caution recorded deliberately: an earlier reading of the same data claimed a
*boundary* obstruction, because the maximum area ratio stayed bit-identical to
LSCM at every iteration count. That has a trivial explanation — with
`fix_boundary = true`, a triangle whose three vertices are all boundary vertices
is frozen, and grid corners are exactly that. The lesson is that an arresting
invariance deserves the boring explanation first.

### Where we start from

- **morsel** — half-edge core with machine-checked invariants, exact IO, a
  repaired parameterization stack (LSCM/ARAP/OMT with exact power cells),
  geodesics (heat method, Dijkstra), curvature, remeshing, decimation,
  subdivision, a wgpu viewer, CI.
- **approxum** — floating-point polygon/curve/sampling/distance geometry; backs
  OMT's exact cells.
- **Praxis** (`~/src/lean`) — Lean 4 + Mathlib with a `prove-goal` skill, a
  `lean-prover` agent, and an automation-closed / agent-closed / open benchmark.
- **A blog** — the exposition channel.

Both halves of this program already exist and have never been connected. Most
attempts would have to build one from scratch; over a decade that head start is
worth more than any single technical choice below.

Nothing in any repository is novel research yet. Fine foundation; not progress.

### Why this suits the constraints

The author is an engineering manager: time is consistent but fragmented.
Structure-first design decomposes well — a property to state, an operator to
derive, an obligation to close — and a proof obligation in particular is a
self-contained evening that holds no state between sessions.

## Design

### The algorithm track (the spine)

All four live in semi-discrete transport, where the infrastructure is already
ahead of what is published.

1. **Per-face transport, or a characterised compromise.** The direct response to
   the obstruction above. If per-triangle area preservation needs per-triangle
   degrees of freedom, then either move to a formulation that has them, or accept
   a least-squares compromise and *characterise the residual* — a bound in terms
   of `F/V`, mesh quality, and the conformal factor. Either outcome is a result;
   the second is probably the more useful one.
2. **Boundary-sliding transport.** Boundary vertices constrained to move *along*
   the boundary curve — a 1D transport problem coupled to the 2D interior.
   Currently the code offers only pinned (freezes a boundary layer) or free
   (contracts the boundary inward and buys nothing). Neither is right and, as far
   as we know, the constrained version is not well solved in the literature.
3. **Exact-Hessian damped Newton.** The dual's Hessian has a closed form in shared
   power-cell edge lengths. Cells are now exact polygons, so those lengths are
   *available exactly* where everyone else estimates them. Replaces an ascent that
   currently needs iterations growing linearly in vertex count.
4. **Area preservation with guaranteed local injectivity.** Transport plus a
   no-flip constraint, with the guarantee proved rather than observed.

Each is stated as a property set first, so that the derivation — not a guess — is
what produces the method.

### Verification's role

**Serving the algorithm track, in three ways:**

- *Deriving.* Formalise the property set, then show the operator satisfying it is
  unique (or that the family is exactly characterised). This is where "the
  specification determined the algorithm" becomes a theorem rather than a story.
- *Obstruction.* Prove the impossibilities. The counting argument above is the
  first candidate and needs only Euler's formula plus linear algebra — well within
  what Mathlib supports today.
- *Confidence.* Machine-checked combinatorial invariants, so the library the
  experiments run on is trustworthy. Genuinely useful, not a research output.

Priority within [`0001`](0001-formal-verification-in-lean.md) is unchanged from
its revision: mathematics first, Rust extraction opportunistic. Extraction can
never reach the numerical code, since Lean's `Float` is opaque.

### The proof-obligation queue

Every "this holds because…" in the codebase becomes a logged obligation. Already
accumulated:

- `Σ ℓᵢ² cot θᵢ = 4 · area` — relied on so lumped masses sum to surface area.
- Mixed Voronoi areas partition a triangle (non-obtuse and obtuse branches).
- Barycentric thirds ≠ dual cell areas — the counterexample.
- **`F = 2V − B − 2` for a triangulated disk, and the resulting DOF deficit.**
- Cotangent Laplacian's kernel is the constants — why ARAP pins one vertex.
- LSCM's conformal energy has the 4-dimensional similarity kernel — why LSCM pins
  two, and why DOF elimination is well-posed.
- `compute_area_distortion` is invariant to uniform UV scaling.
- The Kantorovich dual is concave in the weights.

Each is one evening. Praxis's benchmark measures progress through the queue,
which also makes toolkit improvement a number.

### Keeping the ratio honest

Design and implementation lead; verification follows. Target roughly **2:1
implementation to verification** in the early years. If a quarter passes with no
new geometry implemented, the program has drifted into hygiene.

### Venues

- **SGP / CGF** — the method papers, and the obstruction results. This is the
  target, and the spine above is what makes it a method submission with unusually
  strong guarantees rather than a verification paper hoping to interest geometers.
- **ITP / CPP / CICM** — the formalisation itself, once there is a library worth
  describing.
- **SoCG** — a possible home for an exact-arithmetic robustness strand.

A Mathlib contribution of discrete-geometry foundations may outlast all of them.
**Audit what Mathlib actually has before assuming it has nothing**: smooth
manifolds are well developed; the discrete side is believed absent but unverified.

### The C++ comparison problem

Reviewers will require comparison against libigl / geometry-central / CGAL. The
measurement harness needs to be able to invoke a foreign implementation and
compare numerically. Cheap to design in now, painful to retrofit.

## Steps

### M0 — platform credibility (≈ 3–6 months)

- [x] CI: tests, clippy `-D warnings` per feature set, format gate.
- [x] Half-edge invariants as universal properties over a fixture set.
- [x] Exact IO round trips (found `f32` narrowing in OBJ *and* PLY).
- [x] Measure whether the OMT residual is an obstruction or unconvergence — it is
      an obstruction; the transport converges to `1e-9` on cell areas while
      per-triangle distortion plateaus at ±3–5%.
- [ ] Measurement harness beyond parameterization: Hausdorff / Frechet
      (approxum has them), triangle-quality histograms, volume and area
      preservation. Designed so a C++ baseline drops in as another implementation.
- [ ] A degenerate-input corpus: obtuse triangles, slivers, cocircular lattices,
      non-manifold edges, unreferenced vertices, inconsistent winding.
- [ ] Mesh `repair` and a seam/cut generator — without them
      `morsel parameterize examples/stanford-bunny.obj -m omt` fails, because
      every bundled example is closed and LSCM/ARAP/OMT all need boundary.
- [ ] Expose `geodesic` and the CVT/anisotropic remeshers in the CLI (~3,400
      lines of working code currently unreachable).

### M1 — the first algorithmic result (≈ 6–18 months)

- [ ] Literature check: has the DOF-deficit obstruction for vertex-weighted
      transport been stated? Has boundary-constrained semi-discrete transport been
      solved? **Do this before building anything.**
- [ ] Formalise `F = 2V − B − 2` and `deficit = V_interior − 1` in Lean. Small,
      self-contained, needs only Euler's formula and arithmetic, and it converts a
      measurement into a theorem. Executable form already in
      `tests/omt_dof_deficit.rs`.
- [ ] Derive and implement whichever of per-face transport or the characterised
      least-squares residual the counting argument actually licenses.
- [ ] Exact-Hessian Newton, so experiments stop being iteration-bound.
- [ ] Validate the heat method (`algo/geodesic/heat.rs`, 596 lines, never checked
      against the paper's error curves) — free calibration of what "matching
      published numbers" feels like, and it is not exposed in the CLI either.
- [ ] Write it up on the blog. Exposition is half of what makes a program legible.

### M2 — the formal foundation (concurrent from M1)

- [ ] `morsel-verif` as a Lake project on Praxis's toolchain.
- [ ] Audit Mathlib's discrete-geometry coverage.
- [ ] Close the queue, starting with the barycentric counterexample (explicit
      rationals, `decide`/`norm_num`, no geometry library needed).
- [ ] Pair every landed theorem with a `proptest` asserting the same property
      numerically, with a named tolerance.

### M3 — obstruction results (≈ 2–5 years)

- [ ] Formalise the structure-preservation properties as reusable definitions:
      exactness, symmetry, semidefiniteness, kernel, maximum principle, partition
      of unity, mass consistency, convergence.
- [ ] Establish which sets are mutually satisfiable, for transport maps and then
      more broadly. The no-free-lunch template, done for area-preserving maps.
- [ ] Publish the formalisation side; upstream what Mathlib will take.

### M4 — the SGP submission (≈ 4–10 years)

- [ ] A method paper: the derived algorithm, its guarantee, the obstruction that
      motivates it, full C++ baseline comparison.
- [ ] Code release; pursue the Graphics Replicability Stamp, where a solo program
      can win outright.

## Open questions

- **Is the DOF-deficit obstruction already known?** It is elementary enough that
  it may be folklore. If so, the result is the *consequence* — a sharp bound on
  achievable per-triangle distortion in terms of `V_interior`, mesh quality and
  the conformal factor — rather than the counting itself. The measured plateau
  narrows with refinement (±5% at 81 vertices, ±3% at 289), which suggests such a
  bound exists and is worth deriving.
- **Per-face transport: does it even make sense?** Faces do not carry a natural
  dual cell. Possibly the right move is not per-face transport but accepting the
  deficit and characterising the residual sharply.
- **What does Mathlib have?** Verify before building.
- **Does exact arithmetic deserve a lane?** `exactum` was dropped from morsel as
  unused, but degenerate power diagrams broke a polygon clipper in July 2026, so
  the need is demonstrated. Third lane or distraction — undecided.
- **How does the audit strand get framed constructively?** Always pair a
  counterexample with a corrected construction *and* a measurement showing the
  correction matters.

## Done when

- **M0:** every algorithm has a quality metric, a degenerate corpus exists, no
  capability is unreachable from the CLI, and the harness can call a C++ baseline.
- **M1:** one derived algorithm with a stated guarantee, plus the obstruction
  formalised, written up publicly.
- **M2:** `lake build` green with no `sorry`, the queue's first four closed, each
  paired with a numerical test.
- **M3:** enough mutually-satisfiable/unsatisfiable property sets established to
  be worth a paper; formalisation submitted.
- **M4:** SGP submission, code released, replicability stamp pursued.

Non-goals, restated so they cannot creep: no f64 semantics in Lean, no claim that
the numerical kernels are verified, no infrastructure without a named experiment
behind it, and no verification work that is not serving the algorithm track.
