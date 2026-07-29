# Research program: machine-checked geometry processing

- **Status:** active
- **Last updated:** 2026-07-29
- **Last touched on:** macOS laptop, Claude Code session — program framing written, no research output yet

## Goal

Build a geometry processing research program around one thesis:

> **Formalization finds a class of error that testing structurally cannot — a
> discrete specification that does not satisfy the property it is assumed to
> have. The corrected mathematics is the research result; the proof assistant is
> the method.**

The deliverables, in increasing durability:

1. A library whose invariants are machine-checked rather than merely tested.
2. A published audit of which discrete differential operators satisfy which
   structural properties under which mesh hypotheses.
3. Foundations for discrete differential geometry in Lean 4, ideally upstreamed.

Time horizon is **2–10 years**, deliberately. There is no publication pressure,
and the program is designed around that rather than in spite of it.

## Context

### Why this thesis, and not "we verified a library"

"We verified our geometry library" is not a geometry paper — the geometry
community values proofs of *mathematics*, not proofs of *code*. But the inverse
framing works: use formalization to discover that standard practice rests on a
false premise, then publish the corrected premise.

There is already one instance, found in July 2026 without any Lean at all. The
OMT parameterization lumped vertex masses barycentrically (`area / 3` per
incident triangle) and fed them to a power diagram that partitions the domain
into *dual cells*. Those disagree — on a flat 4×4 grid a corner vertex is
assigned mass `0.0833` against a `0.0625` cell — so satisfying the mass
constraint *required* moving vertices off an already-isometric map. The test
suite was green throughout. The premise is a two-line proposition in Lean and it
is false.

That is the shape of the result. The program is a machine for producing more of
them.

### Why the space is rich

Discretizations whose stated properties are conditional, under-specified, or
simply wrong are common in this field. Candidates to audit:

- **Cotangent Laplacian.** Loses the maximum principle on obtuse triangles;
  weights go negative. The intrinsic-Delaunay line of work exists because of it.
- **Mass lumping.** Barycentric vs. Voronoi vs. mixed Voronoi are not
  interchangeable, and the choice is often made without stating what it must
  satisfy. Already bitten us once.
- **Discrete mean curvature.** Several inequivalent definitions; not all converge.
- **Angle-defect Gaussian curvature.** Converges, but in a weaker sense than
  usually assumed.
- **Vertex normals.** Area-weighted, angle-weighted, and uniform have different
  convergence behaviour.
- **Non-manifold Laplacians.** Sharp & Crane (SGP 2020) exists precisely because
  the naive construction is wrong.

Each is a formalizable claim with hypotheses that may or may not hold.

### Why formalization fits this program's constraints

The author is an engineering manager; available time is consistent but
fragmented. Verification decomposes better than almost any other research
activity: **a proof obligation is a self-contained evening.** You close one lemma
and stop, holding no state. Contrast debugging a numerical solver, which needs
long contiguous focus. This is a structural fit, not only a preference.

### Where we start from

- **morsel** — half-edge core with machine-independent invariant tests, exact IO,
  a repaired parameterization stack (LSCM/ARAP/OMT), geodesics (heat method and
  Dijkstra), curvature, remeshing, decimation, subdivision, a wgpu viewer, CI.
- **approxum** — floating-point polygon/curve/sampling/distance geometry; backs
  OMT's exact power cells.
- **Praxis** (`~/src/lean`) — a Lean 4 + Mathlib practice ground with a
  `prove-goal` skill, a `lean-prover` agent, and an automation-closed /
  agent-closed / open benchmark.
- **A blog** (`timthirion.github.io`) — the exposition channel, with a
  half-finished piece on envelope geometry.

Two halves of this program already exist and have never been connected. That
connection is the play.

Nothing in any repository is novel research yet. That is a fine foundation and
should not be mistaken for progress toward a result.

## Design

### The structure-preservation audit

The concrete decade-scale target. Discrete differential geometry's value
proposition is that good discretizations *preserve structure*. Each such property
is formalizable:

| property | statement |
|---|---|
| exactness | `d ∘ d = 0` |
| symmetry | `L = Lᵀ` |
| semidefiniteness | `xᵀ L x ≥ 0` |
| kernel | `ker L = constants` |
| maximum principle | no interior extrema for harmonic functions |
| partition of unity | `Σᵢ φᵢ = 1` |
| mass consistency | `Σᵢ mᵢ = total area` |
| convergence | operator → smooth counterpart under refinement |

Cross those against operators (cotangent Laplacian, mass matrices, discrete
curvatures, normals, the conformal energy) and against mesh hypotheses
(manifold, Delaunay, non-obtuse, boundary or not). Each cell of the resulting
table is a theorem or a counterexample. **The table is the research artifact**,
and it is buildable one lemma at a time.

### Priority: mathematics first, extraction second

This inverts the emphasis in [`0001`](0001-formal-verification-in-lean.md).

Rust→Lean extraction via Charon/Aeneas can never reach the numerical code —
Lean's `Float` is opaque and not encoded in its logic — so extraction only ever
buys confidence in the *combinatorial* layer. That is genuinely valuable for the
library and worth doing opportunistically, but it is not a research result.

The specifications over ℝ are where the contributions are. Lead with those.

### The proof-obligation queue

The process mechanism, and it suits fragmented time. Every time the codebase
asserts "this holds because…", the claim is logged as an obligation. This is
already happening organically:

- `Σ ℓᵢ² cot θᵢ = 4 · area` — asserted in an `omt.rs` comment, relied upon to
  guarantee lumped masses sum to surface area.
- Mixed Voronoi areas partition a triangle (both the non-obtuse and obtuse cases).
- Barycentric thirds ≠ dual cell areas — the counterexample above.
- Cotangent Laplacian's kernel is the constants — why ARAP pins exactly one vertex.
- LSCM's conformal energy has the 4-dimensional similarity group as its kernel —
  why LSCM pins exactly two, and why DOF elimination is well-posed.
- `compute_area_distortion` is invariant to uniform UV scaling — claimed in its
  docstring.
- The Kantorovich dual is concave in the weights — why ascent converges.

**The queue is the research pipeline.** Each item is one evening. Praxis's
benchmark (automation-closed / agent-closed / open) measures progress through it,
which also makes the toolkit's improvement a number.

### Keeping the ratio honest

Verification only finds bugs if there is real code doing real work. Target
roughly **2:1 implementation to verification** in the early years. If a quarter
passes with no new geometry implemented, the program has drifted.

### Venues

Expect **two papers to two communities**, not one paper that satisfies both:

- **Formalization paper** — ITP / CPP / CICM. "Foundations of discrete
  differential geometry in Lean 4." Establishes the library.
- **Geometry paper** — SGP, or CGF for something longer. "Discrete operators that
  do not satisfy their assumed properties, and what to use instead." Reports what
  the library found.
- **SoCG** is a third possibility for the exact-arithmetic and robustness strand,
  which has a tradition there.

A Mathlib contribution of DDG foundations may outlast both. Worth auditing what
Mathlib actually has first: smooth manifolds are well developed; as far as we
know there is no discrete exterior calculus and no treatment of piecewise-linear
surfaces as metric objects. **Confirm before assuming.**

### The C++ comparison problem

To publish a *geometry* result, reviewers will require comparison against
libigl / geometry-central / CGAL. That means the measurement harness eventually
needs to invoke C++ baselines and compare numerically. Much easier to design in
than to retrofit — see phase M0.

## Steps

### M0 — platform credibility (≈ 3–6 months)

- [x] CI: tests, clippy `-D warnings` per feature set, format gate.
- [x] Half-edge invariants as universal properties over a fixture set.
- [x] Exact IO round trips (found and fixed `f32` narrowing in OBJ *and* PLY).
- [ ] Measurement harness for algorithms other than parameterization: Hausdorff
      and Frechet (approxum has these), triangle-quality histograms, volume and
      area preservation. Every claim gets a number.
- [ ] A benchmark corpus that deliberately includes degenerate input — obtuse
      triangles, near-degenerate slivers, cocircular lattices, non-manifold
      edges, unreferenced vertices, inconsistent winding.
- [ ] Close the gaps that make the library untestable on real input: mesh
      `repair`, and a seam/cut generator. Right now
      `morsel parameterize examples/stanford-bunny.obj -m omt` fails, because
      every bundled example is closed and LSCM/ARAP/OMT all require boundary.
- [ ] Expose `geodesic` and the CVT/anisotropic remeshers in the CLI — roughly
      3,400 lines of working library code is currently unreachable.
- [ ] Design the harness so a C++ baseline can be dropped in as another
      implementation to compare against.

### M1 — one exact reproduction (≈ 6–12 months)

- [ ] Reproduce a published result and match its numbers. **The obvious
      candidate is already in the repo**: `algo/geodesic/heat.rs` is 596 lines
      implementing the heat method (Crane, Weischedel, Wardetzky) and has never
      been validated against the paper's error curves. Free signal on platform
      quality, and it is not even exposed in the CLI.
- [ ] Write it up on the blog. Exposition is roughly half of what makes this kind
      of program legible, and the muscle needs exercising.

### M2 — the formal foundation begins (runs concurrently from M1)

- [ ] Stand up `morsel-verif` as a Lake project on Praxis's toolchain.
- [ ] Audit what Mathlib already provides for triangulated surfaces.
- [ ] Close the first four obligations from the queue, starting with the
      barycentric counterexample (explicit rationals, `decide`/`norm_num`, no
      geometry library needed).
- [ ] Pair every landed theorem with a `proptest` in morsel asserting the same
      property numerically, with a named tolerance.

### M3 — the audit (≈ 2–4 years)

- [ ] Formalize the structure-preservation properties as reusable Lean
      definitions.
- [ ] Fill the operator × property × hypothesis table. Record counterexamples as
      first-class results, not failures.
- [ ] Publish the formalization side (ITP/CPP), and upstream what Mathlib will take.

### M4 — the geometry result (≈ 4–10 years)

- [ ] Assemble the counterexamples into a geometry contribution: which standard
      discretizations fail which stated property, under what conditions, and what
      the corrected construction is.
- [ ] Full C++ baseline comparison.
- [ ] Submit to SGP with code release; pursue the Graphics Replicability Stamp,
      where a solo program can win outright.

## Open questions

- **What does Mathlib actually have?** The plan assumes no discrete exterior
  calculus and no PL surfaces. Verify before building; if a partial foundation
  exists, extend rather than duplicate.
- **Is the audit framing publishable as geometry, or does it read as criticism?**
  A paper that says "these standard methods are subtly wrong" needs to land as
  constructive. Mitigation: always pair a counterexample with a corrected
  construction and a measurement showing the correction matters.
- **How much does exact arithmetic belong in the story?** `exactum` was dropped
  from morsel as unused. A robustness strand (exact predicates for degenerate
  power diagrams — we hit exactly this in July 2026 when cocircular lattice sites
  degenerated a polygon clipper) could be a third lane, or a distraction.
- **Which single operator is the best first audit target?** The cotangent
  Laplacian is the most consequential and the most studied, so it is both the
  highest-value and the hardest place to say something new. Mass lumping is
  narrower and we already have a result there.

## Done when

Staged, because a decade-scale program has no single finish line.

- **M0 done:** every algorithm in morsel has a quality metric, a degenerate-input
  corpus exists, and no capability is unreachable from the CLI.
- **M1 done:** one published result reproduced with numbers that match, written up.
- **M2 done:** `lake build` green with no `sorry`, at least four queue obligations
  closed, each paired with a numerical test in morsel.
- **M3 done:** the operator × property table has enough filled cells to be worth
  a paper, and the formalization is submitted somewhere.
- **M4 done:** an SGP submission, code released, replicability stamp pursued.

Non-goals, restated so they cannot creep: no f64 semantics in Lean, no claim that
morsel's numerical kernels are verified, and no infrastructure work without a
named experiment behind it.
