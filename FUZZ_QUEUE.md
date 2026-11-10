# Proof-fuzzing queue

Candidate statements about H-WavJEPA, written down **before** anyone tries to
prove them. Each is then attempted in Lean and ends as a proof or a refutation.
The expected verdict is a guess; record the actual outcome when it differs.

This file is a work list, not evidence. Only results in the reviewed library
(`HJepa/`) may be cited in [README.md](README.md) or [THEORY.md](THEORY.md).

## Two tiers

| Tier | Location | Lean-checked and axiom-audited | Statement read by a person |
| --- | --- | --- | --- |
| Reviewed | `HJepa/`, root [`HJepa.lean`](HJepa.lean) | Yes | Yes |
| Frontier | `HJepaFrontier/`, root [`HJepaFrontier.lean`](HJepaFrontier.lean) | Yes | **No** |

A Frontier proof is valid Lean. What is unverified is whether its statement and
hypotheses mean what the queue entry says. The split follows the
`StacksCore`/`StacksFrontier` layout of the
[Lean Stacks formalization](https://github.com/AxiomMath/Stacks).

## Lifecycle

1. **Queued.** Add a row: the claim in words, the expected verdict, what it
   depends on, and its source.
2. **Frontier.** Add a module under `HJepaFrontier/` in the namespace
   `HJepa.Frontier`, import it from `HJepaFrontier.lean`, and name the queue
   entry in its module comment. `scripts/check_frontier.sh` enforces the
   namespace and the import.
3. **Reviewed.** A person checks that the hypotheses can be satisfied, that the
   statement matches the row, and that its limits are written down. Then move
   the module into `HJepa/`, rename its namespace, import it from `HJepa.lean`,
   and add it to the ledger in `THEORY.md`.
4. **Dropped.** Keep the row and say why.

## View-selection queue

Motivation: Balestriero and LeCun ([arXiv:2610.39302](https://arxiv.org/abs/2610.39302))
argue that a JEPA learns what its two views share. Views with independently
drawn nuisance yield a function of the world state; views that are corruptions
of one observation share its nuisance and may learn it. Every H-WavJEPA positive
pair currently comes from one recording and one rendered scene.

Unless a row says otherwise, entries concern scalar normalized encoders on the
two-sign model of [TEMPORAL_PREDICTION.md](TEMPORAL_PREDICTION.md), with signal
persistence ρ_S and nuisance persistence ρ_N. None is a claim about a trained network.

| ID | Claim to test | Expected | Depends on | Source | Status |
| --- | --- | --- | --- | --- | --- |
| V1 | Redrawing the nuisance with probability r between the two views gives nuisance persistence 1 − r: a shared scene gives 1, an independently drawn scene gives 0. | Prove | — | Paper's good-view and bad-view definitions | Queued |
| V2 | With ρ_N = 0 and ρ_S > 0, every global optimum has task risk 0. | Prove | V1 for the link to views | Paper: good views ignore nuisance | Frontier, partial: `HJepa.Frontier.Views.zero_nuisance_persistence_task_zero` covers the `Dynamics`-level statement only |
| V3 | A longer prediction horizon changes which factor the optimum keeps. | Refute: for a strict order of nonnegative persistences every horizon h ≥ 1 selects the same factor; only the margin, the gap between ρ_S^(2h) and ρ_N^(2h), changes | — | H-WavJEPA horizons 1, 2, 4 | Queued |
| V4 | Two encoders trained separately on good views agree up to sign. If one is trained where the nuisance dominates, the best-sign alignment error is 2. | Prove | V2 | Paper: representations agree up to rotation; bad views cost alignment | Queued |
| V5 | A within-recording variance floor removes the nuisance optimum. | Refute when the nuisance varies slowly (ρ_S < ρ_N < 1) | A definition of within-recording variance in the model | H-WavJEPA `temporal_variance_weight` | Queued |
| V6 | Independent redrawing helps every downstream task. | Refute: a task equal to the redrawn factor has risk 1 at every global optimum | V2 | WavJEPA-Nat spatial cues | Queued |
| V7 | Any finite world: for mean-zero, unit-variance scalar encoders on paired observations of one state with independent nuisances, the best-sign alignment error is at least 2(1 − min ρ), where ρ² is the second moment of the encoder's conditional mean given the state. | Prove | — | Paper: nuisance costs alignment | Queued |
| V8 | Any finite world, good views: a normalized encoder that depends on the nuisance and has positive view correlation has strictly larger loss than its normalized conditional mean given the state. | Prove | — | Paper: good views ignore nuisance | Queued |
| V9 | If the minimizers of a world-level objective are unique up to a group action, the minimizers for every injective rendering equal one common embedding up to that action. | Prove | — | Paper's main theorem, taking uniqueness as a hypothesis | Queued |

## Already in the reviewed library

- **Persistence threshold.** `signal_global_task_zero` and `nuisance_global_task_one`
  in [`Comparison/Optima.lean`](HJepa/TemporalPrediction/Comparison/Optima.lean):
  every global optimum keeps the factor with the larger squared persistence.
- **Extreme case.** `every_global_optimum_task_risk_at_least_one` in
  [`Objective.lean`](HJepa/Objective.lean): a static background is kept and an
  independently redrawn task bit is discarded.

## Not queued

The paper's multi-dimensional results (spectral uniqueness, the kernel-mismatch
bound, the general alignment bound) rest on the Ky Fan principle, a Davis–Kahan
bound and nuclear-norm inequalities. These are believed absent from the pinned
mathlib; that has not been checked. They are a library project, not a queue entry.
