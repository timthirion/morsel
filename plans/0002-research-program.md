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

### Independent confirmation of the spine, from numerical PDEs

The same discipline is standard practice in a neighbouring field, which is worth
knowing both as validation and as a source of mature theory.

Gorard's *Shock with Confidence* (arXiv:2503.13877, 2025) generates hyperbolic PDE
solvers and proves **L² stability, flux conservation and physical validity** of the
generated code. Those are structure-preservation properties; the framing here is
not novel as a *methodology*, only as an application to geometry processing.

Numerical relativity has treated this as non-negotiable for decades. A scheme that
lets the Hamiltonian and momentum constraints drift in an ADM/BSSN evolution does
not merely lose accuracy — it blows up. Hence constraint damping and
symmetric-hyperbolic reformulations.

The mature mathematical theory underneath both is **finite element exterior
calculus** (Arnold, Falk & Winther) and the mimetic-discretization literature: make
the discrete complex preserve the cohomology and stability follows. That is the
rigorous general form of what this plan reaches for with an ad-hoc property list,
and it is a better foundation to build on than assembling the properties
piecemeal. **Read FEEC before designing the property lattice.**

### The shape a formal instrument is best at: impossibility

The template exists and is well known: Wardetzky, Mathur, Kälberer & Grinspun,
*Discrete Laplace Operators: No Free Lunch* (SGP 2007) — no discrete Laplacian on
general meshes is simultaneously symmetric, local, linearly precise, and
positive-weighted. That is a structure-preservation impossibility, it is highly
cited, and it reframed what everyone after it attempted.

A proof assistant is the right instrument for such results because the value is
in the *generality* of the "no" — not in the two or three special cases one can
check by hand.

### First obstruction, measured — and then largely dissolved by the literature

The measurement stands. On a paraboloid patch with the transport driven to
convergence (worst per-cell area error `9.7e-10`; 20,000 iterations change nothing
past 783):

| | LSCM | OMT, converged |
|---|---|---|
| n=8, interior faces | [0.730, 1.178] | **[0.968, 1.053]** |
| n=16, interior faces | [0.725, 1.395] | **[0.970, 1.033]** |

Cell areas are matched to nine digits while **per-triangle** area distortion
plateaus at ±3–5%, and more iterations do not touch it. For a triangulated disk
Euler's formula with `3F = 2E − B` gives `F = 2V − B − 2`, so against the `V − 1`
weight degrees of freedom (one lost to the dual's shift invariance) the deficit is
exactly

```
F − (V − 1) = V − B − 1 = (interior vertices) − 1
```

So *this* formulation cannot achieve exact per-triangle area preservation once a
mesh has two or more interior vertices. That much is real and is pinned down in
`tests/omt_dof_deficit.rs`.

**What the literature check (2026-07-29) found, and why it matters more:**

1. **Per-triangle is the standard definition.** A simplicial map is *authalic* iff
   each triangle's area is proportional to its image area by a common constant. So
   per-triangle was the right target to measure against.
2. **Exact per-triangle area preservation is achievable.** The stretch-energy and
   authalic-energy line (Yueh; Liu & Yueh) optimises over **vertex positions**, and
   there are necessary-and-sufficient conditions for a minimiser to be
   area-preserving — authalic energy is zero exactly when the map is authalic. The
   counting agrees: full vertex freedom gives `2V − 4` effective degrees against
   `F − 1 = 2V − B − 3` constraints, a *surplus* of `B − 1`. Area-preserving PL maps
   form a `(B−1)`-dimensional family; they are not rare.
3. **So the DOF deficit is not an obstruction to the problem — only to the
   power-diagram search space.** The earlier framing ("cannot imply per-triangle
   area preservation") overreached by dropping the qualifier that matters.
4. **The gap is already documented.** OMT is one of about five known families —
   locally authalic maps, Lie advection, OMT, density-equalizing maps, stretch
   energy minimisation — and SEM is described as outperforming the others,
   explicitly including OMT. Our counting argument would at best *explain* an
   empirical gap others have already reported.
5. **The area is actively theorised.** Convergence results for SEM (R-linear) and
   for OMT (`O(1/m)`, `O(1/m²)` with Nesterov acceleration) are recent. A group is
   publishing here yearly.

**Verdict: the transport lane is not a research direction.** morsel's OMT is now a
correct implementation of a known, dominated method — good library capability, not
a contribution. Pursuing it would mean competing on a crowded problem against an
active theory group, from behind.

Caveats on the check, recorded honestly: it rested on abstracts and search
summaries, not full papers. Three things were *not* resolved and would need the
primary sources before this verdict is final —

- whether published OMT parameterisations place mass at vertices (as ours does) or
  per-face, which decides whether the deficit is intrinsic to OMT or an artefact of
  the discretisation we inherited;
- whether anyone computes power-cell areas *exactly* rather than by sampling, which
  is the one thing morsel's implementation now does that may be unusual;
- whether the DOF explanation for the OMT-vs-SEM gap has been stated anywhere.

The methodological lesson is the useful part: **a literature check before building
killed a lane in an afternoon.** That step stays first in every future lane.

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

### The algorithm track (the spine) — being re-chosen

The transport lane was the intended spine and the literature check retired it (see
above). Nothing is committed in its place yet; that is the open decision, and
picking badly is more costly than picking slowly.

What survives from the transport work as *possible* narrow residue, in decreasing
confidence:

- **Exact power-cell geometry.** If sampled cell areas are the norm, exact polygon
  cells are a genuine implementation advance, though the DOF count says they cannot
  close the gap to SEM. Needs claim (2) in the caveats above checked first.
- **Boundary-sliding transport.** Still unsolved in our code — pinned freezes a
  boundary layer, free contracts it inward and buys nothing. Whether it is open in
  the literature is unchecked.

**Selection criteria for the replacement lane**, learned from this one:

1. **Quiet, not crowded.** Avoid problems with an active group publishing
   convergence theory annually.
2. **Plays to the actual differentiators** — exact predicates, machine-checked
   combinatorial invariants, a verified half-edge core. Those point toward
   robustness of *combinatorial* algorithms rather than toward numerical
   optimisation, where the field is strong and we are not differentiated.
   Rounding-robust structural properties (conservation by telescoping, positivity,
   orientation signs) are also now in scope — see the revision in
   [`0001`](0001-formal-verification-in-lean.md).
3. **Literature check first, always.** Before any implementation.

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
- [x] Validate smoothing and subdivision against checkable properties. Two claims
      the code makes about itself both hold: `taubin_smooth` really is
      shrinkage-resistant (+1.8% volume on a sphere where Laplacian loses 79%), and
      mean curvature flow really follows the analytic law, with `R²` falling
      linearly in `t` at slope `2.00` — i.e. it flows by the *mean* curvature
      `1/R`, where the literature's `H = κ₁+κ₂` convention would give `4`. Loop
      subdivision preserves the Euler characteristic, quadruples faces exactly, and
      converges. Two defects found: **Catmull-Clark panicked on a triangle mesh**
      (invalid-index sentinel again; now declines, and the CLI reports it), and
      **`examples/sphere.obj` was wound inward** — the only such asset, which had
      been silently flipping the sign of everything normal-dependent.
- [ ] Measurement harness beyond parameterization: Hausdorff / Frechet
      (approxum has them), triangle-quality histograms, volume and area
      preservation. Designed so a C++ baseline drops in as another implementation.
- [x] A degenerate-input corpus (`tests/common/mod.rs`, 18 cases) plus a
      robustness sweep over every algorithm (`tests/robustness_sweep.rs`),
      recorded as a characterization baseline. Findings, in severity order:
      - **QEM decimation corrupts valid meshes, non-deterministically** — 3 of 8
        runs on the plain control grid, identically with `parallel` true and
        false, so the cause is not the parallelism. The library uses
        `std::collections::HashMap`, whose iteration order is randomised per
        process, so the collapse sequence varies and only some sequences break.
        `morsel decimate` is a shipping command; this is the top item.
      - **`build_from_triangles` accepts 5 invalid inputs and returns a corrupt
        half-edge structure** with `Ok` — non-manifold edges and vertices,
        inconsistent winding, duplicate vertices, duplicate faces. It is the
        library's entry point, so everything downstream inherits it. Fixing this
        one clears most of the matrix.
      - **8 panics**, every one dereferencing the invalid-index sentinel
        (`u32::MAX`) without a validity check, across curvature, geodesics,
        remeshing and subdivision.
      - Scale-relative thresholds are missing: `tiny_scale` (coords ~1e-6) makes
        the heat method and OMT refuse.
      The sweep separates *inherited* from *caused* corruption, which mattered —
      before that distinction the table blamed vertex smoothing for breaking
      half-edge twins, which it cannot do, since it never touches connectivity.
- [x] Fix the entry point and the decimation defects the sweep found. The matrix is
      now entirely `ok` or `refused`: no panics, no corruption.
      `build_from_triangles` validates and rejects non-manifold input, and QEM
      gained the missing "no interior edge with both endpoints on the boundary"
      rule. Numerical threshold work (scale-relative epsilons in the heat method
      and OMT) deliberately left alone — refusing is a safe failure and epsilon
      tuning is a rabbit hole.
- [x] A seam/cut generator (`src/algo/cut.rs`, July 2026). Joins boundary loops
      along the shortest path between them and slits closed surfaces open, so
      `morsel parameterize examples/stanford-bunny.obj -m omt --cut` now works
      where it used to be refused. Genus > 0 and disconnected input are refused by
      name. The surgery runs in face-vertex space and rebuilds through
      `build_from_triangles`, so a botched cut is rejected rather than producing a
      corrupt half-edge mesh — which is only affordable because that validation
      landed first.

      The bug worth remembering: the fan circulator advances by `next(twin(h))`,
      and `face_of(twin(h)) == face_of(next(twin(h)))`, so the face *between* rays
      `k` and `k+1` is `face_of(twin(hᵏ))`, not `face_of(hᵏ)`. Getting that index
      off by one — together with choosing each vertex's side independently rather
      than orienting by the path direction — assigned faces across the cut. The
      symptom was not local: the cylinder came back with 13 boundary loops and the
      bunny and sphere with bowtie vertices. Neither pointed at the fan.
- [ ] Mesh `repair` — dropping unreferenced vertices, which currently make a mesh
      count as disconnected and so block cutting.
- [ ] Expose `geodesic` and the CVT/anisotropic remeshers in the CLI (~3,400
      lines of working code currently unreachable).

### M1 — choose and open a lane (≈ 6–18 months)

- [x] Literature check on the transport lane. **Result: retired.** Per-triangle
      area preservation is achievable by stretch/authalic energy minimisation, the
      OMT-vs-SEM gap is already reported, and the area has active convergence
      theory.
- [ ] Read the primary sources for the three unresolved caveats above, so the
      verdict rests on papers rather than abstracts.
- [ ] Survey candidate lanes against the selection criteria, with a literature
      check *per candidate* before any code.
- [ ] Formalise `F = 2V − B − 2` and `deficit = V_interior − 1` anyway. It is a
      small self-contained Lean exercise needing only Euler's formula, the
      executable form is already in `tests/omt_dof_deficit.rs`, and it is a good
      first calibration of the toolchain even though it is not a result.
- [x] Validate the heat method against exact geodesics. It had **never produced a
      result on a realistic mesh**: its Poisson stage solved against the raw
      cotangent Laplacian, which is singular, so CG could not reduce the relative
      residual below the null-space contribution and reported `ConvergenceFailed`
      at any budget — 100,000 iterations still failed on a 178-vertex sphere. Four
      defects, in order of severity:
      1. **Singular Poisson system**, fixed by eliminating one degree of freedom;
         the gauge is free because the result is min-shifted anyway.
      2. **Default time step `t = h²` made the method diverge** — error grew from
         −7% at 101 vertices to −27% at 6401, because heat decays as `exp(-d²/4t)`
         and a small `t` pushes the far field below the solver's residual floor.
         Now `10 h²`, which converges monotonically (−5.9%, −2.3%, −0.8%, −0.3%).
      3. **CG defaults too loose**, third instance of this shape after LSCM and
         ARAP: 1000 iterations at `1e-8` left enough solver error at 6401 vertices
         to make convergence look like divergence.
      4. **Silent garbage on disconnected meshes** — vertices in other components
         got plausible finite distances, and their arbitrary values set the minimum
         used to shift the component that *did* contain the source. Now infinite,
         matching Dijkstra.
      Also switched its mass matrix from barycentric thirds to mixed Voronoi, which
      is the lumping that pairs with a cotangent Laplacian — the same inconsistency
      that broke OMT, in a second module. Effect not isolated by these tests.
      `geodesic` is now exposed in the CLI.
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

- **What replaces the transport lane?** The open question, and the one that
  matters. Criteria are recorded above; no candidate is chosen.
- **Do the three unresolved caveats change the verdict?** Vertex vs per-face mass
  placement in published OMT is the one that could — if they are per-face, our
  deficit is inherited from the implementation rather than intrinsic.
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
