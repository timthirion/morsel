# Formal verification in Lean 4

- **Status:** proposed
- **Last updated:** 2026-07-29
- **Last touched on:** macOS laptop, Claude Code session — plan written, no code yet

> **Scope note.** This plan is the *engineering* half of
> [`0002-research-program.md`](0002-research-program.md), which sets the research
> thesis and the decade-scale target. Read that first; it also revises the
> priority below. Originally the two tiers were treated as roughly co-equal. They
> are not: the ℝ-specifications are where every research result lives, and Rust
> extraction can never reach the numerical code at all. **Lead with Tier A. Treat
> Tier B as opportunistic** — valuable for library confidence, not a research
> output.

## Goal

Get machine-checked proofs for the parts of morsel that admit them, and be
explicit about the parts that don't. Two targets, in order of payoff:

1. **The mathematical specifications** the numerical code implements — cotangent
   identities, dual-cell areas, energy kernels, scale invariance. These are
   theorems over ℝ in Mathlib, provable today, and independent of any Rust. *This
   is the priority; it is where the research contributions are.*
2. **The combinatorial layer** — half-edge invariants and index bookkeeping,
   extracted from the actual Rust via Charon/Aeneas into Lean. Integer arithmetic
   over indices, which is exactly the subset those tools handle. Buys confidence
   in the library; does not by itself produce a result.

Explicitly **not** a goal: verifying f64 numerics. See *Out of scope*.

The motivating evidence is that both of these would have caught real bugs. The
July 2026 parameterization repair turned up three defects, and the most
interesting one was a **wrong specification**, not a wrong implementation: OMT
lumped vertex masses barycentrically (`area / 3` per incident triangle) while the
power diagram it fed them to partitions the domain into dual cells. Those two
disagree — on a flat 4×4 grid a corner vertex gets mass `0.0833` against a
`0.0625` cell — so satisfying the mass constraint *required* moving vertices off
an already-isometric map. No amount of testing the Rust would have flagged that as
a specification error; it is a two-line proposition in Lean, and it is false.

## Context

### What morsel looks like from a verification standpoint

The library stratifies cleanly, and the strata have very different prospects:

| Layer | Example | Arithmetic | Prospect |
|---|---|---|---|
| Half-edge topology | `twin`, `next`, `is_boundary_vertex`, `canonical_edge` | integer indices | **good** — Aeneas's sweet spot |
| Index bookkeeping | `reduce_pinned_dofs`, `PinnedReduction` | integer indices | **good** |
| Discrete-geometry specs | cotangent identities, mixed Voronoi areas, energy kernels | ℝ | **good** as ℝ-theorems, decoupled from the code |
| Numerical kernels | CG, power-diagram sampling, QEM | f64 | **out of reach** |

### Tooling, as of July 2026

- **Charon** translates safe Rust into LLBC, a typed IR derived from MIR that
  preserves borrow-checker invariants at the type level. **Aeneas** translates
  LLBC into pure functional Lean 4. Both from the AeneasVerif project; Lean 4 is a
  first-class backend alongside F*, Coq, and HOL4.
- Aeneas targets **safe, sequential** Rust. Documented gaps that matter to us:
  `return` inside nested loops and `break`/`continue` to outer loops aren't
  supported; instantiating a generic type with a mutable reference isn't
  supported. morsel's traversal code uses plenty of nested loops, so expect to
  refactor extraction targets into shapes Aeneas accepts, or hand-transcribe.
- morsel is generic over `MeshIndex`. Generics plus mutable references are the
  known weak spot, so the first extraction target should be a concrete
  monomorphic function, not a generic one.
- There is a 2026 experience report on exactly this pipeline (Rust → Lean via
  Charon/Aeneas, with AI provers closing the obligations) — worth reading before
  committing to a shape, since it is the closest published precedent to what
  Praxis would be doing here.

### Why f64 is out

Charon's LLBC *does* represent floats (`FloatTy` covers F16/F32/F64/F128), so
extraction won't reject the code outright. The obstacle is downstream: **Lean's
`Float` is an opaque type, not encoded in Lean's logic** — the kernel cannot
compute with or reason about float values without additional axioms. So an
extracted `f64` function becomes a term you can state things about but not prove
things about.

`FloatSpec` (a Lean 4 port of Coq's Flocq) is the effort to fix this, and it has
the right modules — rounding, ulp, error bounds, IEEE encode/decode. But its deep
proofs are still largely placeholders, notably in `Core/Generic_fmt.lean`,
`Core/Ulp.lean`, and the error-bound files. Coq + Flocq is where this work is
mature; Lean isn't there yet.

Even with a mature float library, the targets here would be hard: CG convergence
rate, the `O(1/√k)` sampling error on power-cell areas, QEM's conditioning. That's
numerical analysis, not the kind of discrete statement that closes with `omega`
and `simp`.

### Relationship to Praxis

This is a natural client for [`~/src/lean`](../../lean) (Praxis), whose stated
thesis is that the prover toolkit is the artifact and whose stated ambition is to
eventually prove something novel. morsel supplies what a practice ground
otherwise lacks: a **real corpus with real stakes**, where a proof either
documents a genuine invariant or exposes a genuine bug. In return Praxis supplies
the `prove-goal` skill, the `lean-prover` agent, and the automation-closed /
agent-closed / open benchmark, which is the right instrument for measuring
progress here too.

Concretely: goals from this plan become benchmark entries in Praxis. If the
portfolio prover can close the cotangent identity but not the kernel
characterization, that's a measured statement about the toolkit.

## Design

### Where the Lean lives

A **separate Lake project**, `morsel-verif`, rather than a `verification/`
subdirectory of this repo.

Rationale: Mathlib is a heavy dependency (a multi-GB cache and a long cold build).
Coupling `cargo test` to `lake build` would make morsel's CI hostage to Mathlib
version churn for a component that gates none of morsel's behaviour. The two
repos are linked by a committed extraction artifact and a documented Charon
version, not by a build dependency.

Trade-off accepted: the two can drift. Mitigated by pinning the Charon/Aeneas
version in the extraction script and re-running extraction in a scheduled job
rather than on every commit, so drift surfaces as a failing scheduled run.

### Tier A — specifications over ℝ (start here)

No Rust involved, so no tooling risk, and it attacks the failure mode that
actually bit. Each item is a proposition about the *design*; several are
restatements of claims currently asserted only in doc comments.

- **`Σ ℓᵢ² · cot θᵢ = 4 · area`** for a Euclidean triangle. This is the identity
  the mixed-Voronoi mass lumping relies on to guarantee total mass equals total
  surface area — currently asserted in a comment in `omt.rs` and tested only
  numerically on one mesh.
- **Mixed Voronoi areas partition the triangle.** The non-obtuse cotangent split
  and the obtuse half/quarter fallback each sum to the triangle's area.
- **Barycentric thirds ≠ dual cell areas.** The formal statement of the OMT bug:
  exhibit a triangulation where `area/3` lumping and the Voronoi dual disagree at
  a boundary vertex. A counterexample, so it should close by `decide`/`norm_num` on
  explicit rationals — a good first-week target.
- **Cotangent Laplacian: symmetric, positive semidefinite, kernel = constants.**
  The kernel claim is what makes ARAP's single pinned vertex exactly the right
  number, and hence what makes the DOF elimination well-posed rather than
  singular.
- **LSCM's conformal energy has the 4-dimensional similarity group as its
  kernel.** Translation (2) + rotation (1) + scale (1). This is why LSCM pins
  *two* vertices — four constraints for four kernel dimensions. Currently a prose
  justification in `lscm.rs`.
- **`compute_area_distortion` is invariant under uniform scaling of the UVs.** The
  docstring claims this; it follows from `det` of the linear map factoring out of
  every triangle ratio identically.
- **The Kantorovich dual is concave in the weights**, with gradient
  `ν_i − |Pow_i(w) ∩ D|`. Justifies ascent converging at all, and is the reason
  the line search is legitimate.

These are stated over ℝ with exact triangles, so they say nothing about the f64
implementation. That gap is bridged in Tier C, not pretended away.

### Tier B — combinatorial extraction via Aeneas

Once Tier A is moving, take the smallest genuinely load-bearing integer function
through the full pipeline. Ordered by increasing risk:

1. **`canonical_edge`** — two lines, no loops, monomorphic if specialized.
   Properties: symmetric (`canonical_edge a b = canonical_edge b a`) and
   idempotent. This is the pipeline smoke test; the point is to prove the
   toolchain works end to end, not to learn anything about morsel.
2. **`PinnedReduction`'s index map** — that `reduced_index` restricted to
   non-pinned vertices is a **bijection onto `0..n_free`**, and that
   `scatter ∘ reduce` is the identity on free entries. This one is worth real
   proof effort: it is new code from July 2026, and if it were wrong LSCM and ARAP
   would silently scramble coordinates rather than fail, which is precisely the
   failure mode that went undetected for months in this module.
3. **Half-edge invariants** — `twin(twin h) = h`; `next` orbits have length 3;
   `face_triangle` yields three distinct vertices; `is_boundary_vertex v` iff some
   incident half-edge's twin has no face. Requires modelling the mesh
   representation, so expect this to need hand-transcription rather than clean
   extraction.

   These now exist as executable properties in
   `tests/halfedge_invariants.rs`, asserted over ten fixtures. That is the
   cheapest possible version of this work and it should be treated as the
   specification to formalize against: the Lean statements and the Rust
   assertions should say the same thing, so a divergence between them is itself
   informative.
4. **Euler characteristic** — `V − E + F = 2 − 2g − b`, checked for the disk case
   (`= 1`) the parameterizers require. The most valuable and the hardest; genus
   needs a real definition.

Where Aeneas can't chew the Rust, **hand-transcribe and say so**. A Lean model
that provably has a property, plus a differential test showing the Rust agrees
with the model, is weaker than extraction but much better than nothing — and it
is honest as long as the plan doesn't claim the Rust was verified.

### Tier C — the bridge for the float layer

Verified specification and shipped code stay connected by **executable
specifications**:

- Every Tier A theorem gets a matching `proptest` in morsel asserting the same
  property numerically with an explicit tolerance. `Σ ℓ² cot θ = 4·area` becomes a
  randomized triangle test to `1e-12`.
- Where the Lean model is hand-transcribed (Tier B, items 3–4), add a differential
  test: run the Rust and a Lean-derived reference on the same inputs and compare.
- Name the tolerance in each test and treat an unexplained tolerance as a bug. A
  test that needs `1e-3` to pass on exact-arithmetic-provable math is reporting a
  conditioning problem.

This is not a proof about the Rust, and the plan should never describe it as one.
It converts "the spec is right" into evidence that "the code implements the spec."

### A better first target than morsel: `exactum`

Worth stating plainly, because it may reorder the whole plan:
[`~/src/exactum`](../../exactum) is exact computational geometry over `i64` with
**no floating point at all**. Everything that makes morsel hard to verify is
absent, and its predicates have crisp specifications: `orientation` returns the
sign of a determinant; `graham_scan` returns the convex hull; Delaunay satisfies
the empty-circumcircle property. Those are end-to-end verifiable *algorithm*
results, not just specification lemmas.

It is `#![forbid(unsafe_code)]` with zero dependencies — about the friendliest
input a Rust-extraction toolchain could be handed.

One clarification, since it affects how much this belongs in *morsel's* plans:
morsel does **not** depend on `exactum`. It was a declared-but-uncalled path
dependency and was dropped in July 2026 (see `AGENTS.md` for why re-adopting it
is a real project rather than a switch). So verifying it discharges no risk here.
The argument for spiking there is purely that it is the easiest place to learn
whether the toolchain works — which is still a good argument, just not a morsel
one. If the spike succeeds, the verification work itself belongs in that repo's
own `plans/`.

Recommendation: run the Aeneas pipeline spike (Tier B item 1) against `exactum`
rather than morsel. If the toolchain works there, port the technique here for the
combinatorial layer. If it doesn't work on the easiest realistic target in the
whole `~/src` tree — no unsafe, no dependencies, no floats — that is decisive
information about the approach, obtained cheaply.

## Steps

- [ ] Read the 2026 Rust-to-Lean experience report; note which obligations their
      AI provers closed and which needed hand proofs.
- [ ] Stand up `morsel-verif` as a Lake project on Lean `v4.30.0` + Mathlib,
      matching Praxis's toolchain so the `prove-goal` skill and `lean-prover`
      agent work unchanged. Green `lake build` with one trivial theorem.
- [ ] **Tier A first proof:** the barycentric-vs-dual-cell counterexample. Explicit
      rational coordinates, closed by `decide`/`norm_num`. Chosen first because it
      is the bug that actually happened and it needs no geometry library.
- [ ] Tier A: `Σ ℓᵢ² cot θᵢ = 4·area`, then the mixed-Voronoi partition result that
      depends on it.
- [ ] Tier A: cotangent Laplacian symmetric + PSD + kernel = constants.
- [ ] Add the matching `proptest`s in morsel for each landed Tier A theorem
      (Tier C), each with a named tolerance.
- [ ] **Aeneas spike, against `exactum`:** extract one predicate, prove one
      property, record every place the toolchain fought back. Time-box it — the
      output is a go/no-go, not a verified library.
- [ ] If the spike lands: extract `canonical_edge` from morsel, then the
      `PinnedReduction` index bijection.
- [ ] Register the goals as Praxis benchmark entries; record automation-closed vs
      agent-closed vs open.
- [ ] Write it up. The specification-error framing — that a green test suite can
      coexist with a false premise, and that the false premise is a one-line Lean
      proposition — is the interesting part and is worth a post.

## Open questions

- **Does Aeneas handle morsel's `MeshIndex` generics at all, or must extraction
  targets be monomorphized first?** Suspect monomorphization is required, since
  generics instantiated with mutable references are a documented gap. Resolve in
  the spike.
- **Hand-transcribe or refactor?** When Aeneas rejects a nested-loop traversal,
  is it better to restructure the Rust into an extractable shape (risking worse
  code for verification's sake) or to hand-transcribe (risking model drift)?
  Leaning transcribe-and-differential-test, because contorting the shipped library
  to suit a tool is the wrong trade — but revisit once we know how often it comes up.
- **Is the Euler characteristic worth it?** It needs a real notion of genus for
  the discrete surface. Possibly better sourced from Mathlib's combinatorics than
  built from scratch — check what exists before starting.
- **Should `exactum` get its own plan?** If the spike goes well, verified exact
  predicates are a substantial piece of work in their own right and probably
  belong in that repo's `plans/`, not this one.

## Done when

Staged, because "verified" is not a single state:

- **Milestone 1 (Tier A).** At least four specification theorems proved with a
  green `lake build`, no `sorry`, including the barycentric counterexample and the
  cotangent identity. Each has a matching `proptest` in morsel. Each Lean file
  cites the morsel function whose doc comment it upgrades from a claim to a proof.
- **Milestone 2 (spike).** A written go/no-go on Charon/Aeneas for this codebase,
  backed by one property proved about one extracted function and a list of the
  places the toolchain refused.
- **Milestone 3 (Tier B).** The `PinnedReduction` index bijection proved about
  extracted-or-transcribed Lean, with the transcription status stated explicitly
  in the file header.

Non-goals restated so they can't quietly creep in: no f64 semantics, no CG
convergence, no claim that morsel's numerical kernels are verified.

## References

- Aeneas — <https://aeneasverif.github.io/>, and the POPL'22 paper *Aeneas: Rust
  verification by functional translation*.
- Charon — <https://arxiv.org/html/2410.18042v2>, *Charon: An Analysis Framework
  for Rust*.
- *A Rust-to-Lean Verification Pipeline with AI Provers: An Experience Report* —
  <https://arxiv.org/abs/2605.30106>.
- Lean's opaque `Float` —
  <https://lean-lang.org/doc/reference/latest/Basic-Types/Floating-Point-Numbers/>.
- FloatSpec, the Flocq port to Lean 4 —
  <https://github.com/Beneficial-AI-Foundation/FloatSpec>.
- Meyer, M., Desbrun, M., Schröder, P., Barr, A. (2003). *Discrete
  Differential-Geometry Operators for Triangulated 2-Manifolds* — the mixed
  Voronoi area this plan's first proofs concern.
