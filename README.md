# H-JEPA and H-WavJEPA gains and losses in Lean 4

[![Lean proof check](https://github.com/quangng2000/hjepa-lean/actions/workflows/lean.yml/badge.svg)](https://github.com/quangng2000/hjepa-lean/actions/workflows/lean.yml)

Machine-checked, conditional comparisons of hierarchical and flat prediction
and planning, motivated by
[H-JEPA: End-to-End Learning of Hierarchical World Models for Visual Planning](https://arxiv.org/abs/2610.06805v1).

The main result makes the tradeoff precise: **abstraction helps prediction only
when its reduction in excess prediction error exceeds the information it loses.**
The project also proves a composed-planner regret bound, a temporal-grouping
identity, algebraic regression thresholds, and explicit counterexamples.
The H-WavJEPA extension adds **three-level telescoping, finite-dimensional
vector targets, and retained-low-level readout guarantees**.

This is a formal mathematical companion—not a verification of the H-JEPA
implementation, a replication of its experiments, or a proof that one trained
neural architecture universally outperforms another. Novelty is not claimed.

## Start here: the exact gain–loss identity

Fix a finite probability space, a real-valued target $`Y`$, lower features $`Z`$
with a finite label type, and upper features $`U=g(Z)`$. Both predictors are evaluated against the **same
target and loss**. The code constructs conditional means directly from weighted
finite sums, including zero-probability fibers, and proves their optimality.

Write $`R^{*}(V)`$ for the optimal squared-loss prediction risk from features $`V`$.
The information penalty is

```math
\begin{aligned}
B &= R^{*}(U)-R^{*}(Z) \\
  &= \mathbb{E}[(\mathbb{E}[Y\mid Z]-\mathbb{E}[Y\mid U])^2] \geq 0.
\end{aligned}
```

For chosen predictors $`f_Z`$ and $`f_U`$, let
$`A_V=R(f_V)-R^{*}(V)`$ be excess risk. Then

```math
\begin{aligned}
R(f_U)-R(f_Z) &= B+A_U-A_Z, \\
R(f_U)\lt R(f_Z) &\iff A_Z-A_U\gt B.
\end{aligned}
```

In plain language: the upper representation cannot beat the information
available to an optimal lower-level predictor. It can beat an imperfect
predictor if its error reduction is large enough. The identity does not promise
that training will produce that reduction.

Read [`Abstraction.lean`](HJepa/Abstraction.lean), especially
`nested_risk_identity`, `learning_gain_loss_identity`, and
`abstraction_improves_iff`. The zero-gap characterization is conditional-mean
sufficiency for this target and squared loss—not preservation of all task
information or of the full conditional distribution.

## H-WavJEPA: three levels, explicit gains and losses

For the proposed low/medium/high audio hierarchy, freeze the encoders and use
the same target, squared loss, probability law, and accessible input context:

```math
Z_1=Z,\qquad Z_2=g(Z_1),\qquad Z_3=h(Z_2).
```

Let $`\mu_\ell=\mathbb{E}[Y\mid Z_\ell]`$, define
$`B_{12}=\mathbb{E}\lVert \mu_1-\mu_2\rVert^2`$ and
$`B_{23}=\mathbb{E}\lVert \mu_2-\mu_3\rVert^2`$, and let
$`A_\ell=R(f_\ell)-R^{*}(Z_\ell)`$ for each chosen predictor. Lean proves:

```math
R^{*}(Z_3)-R^{*}(Z_1)=B_{12}+B_{23},\qquad B_{12},B_{23}\geq0,
```

```math
R(f_3)-R(f_1)=B_{12}+B_{23}+A_3-A_1.
```

Therefore, the highest-level predictor is better **if and only if**
$`A_1-A_3\gt B_{12}+B_{23}`$. The reverse strict inequality makes it worse;
equality gives equal risk. The cumulative information gap is zero exactly
when both adjacent gaps are zero. A later deterministic abstraction cannot
recover information already lost by an earlier one.

These results are checked for both scalar targets and finite-dimensional
real vectors, using the **sum of coordinatewise squared errors**. Vector
coordinates belong to one common target; this is not a comparison of JEPA
training losses from different learned target spaces.

### Keeping all three levels

For the uncompressed bundle $`C=(Z_1,Z_2,Z_3)`$, Lean also proves

```math
R^{*}(C)=R^{*}(Z_1).
```

Every lower-level predictor lifts to a combined predictor with exactly the
same risk, and every combined predictor pulls back to a function of $`Z_1`$.
Thus both have the same achievable risks when **all functions** are allowed.
The bundle adds no population information, but may change how easily a
restricted learner uses the information. This equality is about access to
the original bundle, not an arbitrary learned bottleneck applied afterwards.
It does not promise that a particular linear head, optimizer, or jointly
trained encoder preserves the baseline's performance.

### What is actually formalized

| Module | Main role |
| --- | --- |
| [`ThreeLevel.lean`](HJepa/ThreeLevel.lean) | Scalar cumulative-gap identity, monotone Bayes risks, strict improvement/worsening conditions, and zero/positive-gap criteria. |
| [`Vector.lean`](HJepa/Vector.lean) | Coordinatewise finite-vector extension of prediction risk, Bayes optimality, and the two-/three-level tradeoffs. |
| [`Readout.lean`](HJepa/Readout.lean) | Recoverable deterministic augmentation, three-level feature bundles, and risk-preserving predictor transformations. |
| [`AudioExamples.lean`](HJepa/AudioExamples.lean) | Exact finite witnesses: both adjacent steps lose signal, yet an upper predictor can beat a poorly chosen lower predictor. |

The finite witnesses use two independent fair signs and target
$`Y=S_1+2S_2`$. The low level keeps both signs, the middle keeps only $`S_1`$,
and the high level is constant. The checked optimal risks are **0, 4, 5**,
with adjacent gaps **4 and 1**. A high-level constant-zero predictor has risk
**5**, while a deliberately poor low-level constant-three predictor has risk
**14**. Its excess-risk reduction is 14, offset by information loss 5, for a
risk improvement of 9. These are toy mathematical witnesses, **not audio
experiments or outcomes of JEPA training**.
Additional checked examples confirm that the combined bundle has risk zero,
duplicating target coordinates doubles the summed loss, empty-coordinate loss
is zero, and disagreement confined to a zero-probability outcome incurs no loss.

**Scope:** finite probability spaces; fixed deterministic nested encoders;
finite low/middle feature-label types (and finite bundled labels for the
readout result); one common scalar or finite-dimensional vector target.
Zero-probability outcomes are allowed. The names low/medium/high do not prove
that the levels learn particular time scales or semantics. If an upper
feature pools a time window, $`Z_1`$ must include that **entire same-context
window**, not a single low-level token. Independently trained flat and
hierarchical encoders are not ordered by these theorems. Continuous-population
generalization, training dynamics, and passive-audio planning guarantees
remain outside the formalization.

## Formalization coverage

The accompanying [research note](main.tex) contains broader analytical arguments.
Its full contents are **not** all formalized. This is the exact boundary:

| Result | Lean coverage | Assumptions / boundary |
| --- | --- | --- |
| Conditional mean and Bayes optimality | [`Abstraction.lean`](HJepa/Abstraction.lean): `fiber_residual_zero`, `residual_orthogonality`, `risk_pythagorean`, `bayes_optimal` | Finite probability law and feature-label type, scalar real target, fixed encoder; zero weights allowed. |
| Information loss and learning-error tradeoff | `nested_risk_identity`, `abstraction_no_loss_iff`, `abstraction_improves_iff` | Nested deterministic features; fixed predictors. No theorem about a training algorithm or its sample complexity. |
| Three-level H-WavJEPA tradeoff | [`ThreeLevel.lean`](HJepa/ThreeLevel.lean): `informationGap_telescope`, `three_level_gain_loss_identity`, `three_level_improves_iff`, `cumulative_gap_eq_zero_iff` | Finite law and low/middle label types; scalar target; high labels need not be finite. |
| Finite-dimensional vector targets | [`Vector.lean`](HJepa/Vector.lean) | Sum of coordinatewise real squared losses on a finite law; not a general measure-theoretic or Hilbert-space formalization. |
| Retained-low-level combined readout | [`Readout.lean`](HJepa/Readout.lean): `three_level_bundle_bayes_risk_eq`, `three_level_achievable_risk_iff` | Deterministic bundle with the original low-level feature retained; finite labels; unrestricted predictors, not guaranteed training behavior. |
| Three-level signal-loss and imperfect-predictor witnesses | [`AudioExamples.lean`](HJepa/AudioExamples.lean): `three_level_bayes_risks`, `adjacent_information_gaps`, `exact_gain_loss_budget` | Explicit uniform four-outcome law; no neural or audio experiment. |
| OLS gain/loss threshold | [`OLS.lean`](HJepa/OLS.lean): `risk_difference`, `reduced_better_iff`, `risks_equal_iff`, `reduced_worse_iff` | Algebra of explicit rational risk expressions. Gaussian-design / inverse-Wishart expectation formulas are **not** formalized. |
| Composed-planner regret | [`Planning.lean`](HJepa/Planning.lean): `composite_planner_regret`, `composite_planner_regret_of_infimum`, `flat_planner_regret`, `exact_calibration_optimality` | True cost bounded below; a candidate exists; uniform task-cost calibration and global approximate surrogate optimization. Infima need not be attained. |
| Rollout error and cost calibration | [`Rollout.lean`](HJepa/Rollout.lean): `dynamics_error_recurrence`, `rollout_error_bound`, `task_cost_calibration` | Explicit local-error, Lipschitz, and task-calibration assumptions; deterministic fixed-horizon statements. |
| No improvement from regrouping inherited error bounds | `geometric_blocking`, `inherited_full_bound_equality` | Exact algebraic identity; includes zero horizons and multiplier one. Not a lower bound on actual learned-model error. |
| Perfect latent prediction can fail control | [`Counterexamples.lean`](HJepa/Counterexamples.lean): `abstract_rollout_exact`, `abstract_policy_success_half`, `full_policy_success_one` | Actual finite four-state average; arbitrary deterministic background-only policy, compared with full-state access. |
| Sign-discarding prediction loss | `squared_sign_risk_at_least_one`, `zero_predictor_attains_one`, `retained_sign_risk_zero` | Fair finite $`\pm1`$ target: optimal squared-loss risk one versus zero. |
| Gaussian AR / spectral feature-selection argument | Analytical note only | No Lean proof of its stochastic process or matrix optimization result yet. |
| Trained nonlinear H-JEPA vs independently trained flat JEPA | Not proved | No verification of neural training, generalization, SIGReg, experimental scores, or closed-loop control. |

## Planning: where gains and losses enter

Let $`J(a)`$ be true primitive-plan cost, $`Q(b)`$ the macro-plan surrogate, and
$`\mathcal{R}(b)`$ the **actual refinement procedure**. Define

```math
\rho=\inf_b J(\mathcal{R}(b))-\inf_a J(a)\geq0.
```

Assume uniform calibration
$`|Q(b)-J(\mathcal{R}(b))|\leq\delta`$ and a returned macro plan whose surrogate
cost is within $`\eta`$ of every competitor. Lean proves

```math
J(\mathcal{R}(\hat{b}))-\inf_a J(a)\leq2\delta+\eta+\rho.
```

Here $`\delta`$ is cost-model error, $`\eta`$ is optimization error of the **whole
composed planner**, and $`\rho`$ is the cost of restricting available plans.
The flat planner is identity refinement and has zero restriction gap.
Lower values of these bounds do not, by themselves, order actual achieved costs.
Accurate latent prediction alone does not establish task-cost calibration.

For rollout error, Lean also proves

```math
e_{t+1}\leq\epsilon+Le_t
\implies
e_N\leq L^Ne_0+\epsilon S_N(L),
\qquad S_N(L)=\sum_{i=0}^{N-1}L^i,
```

with $`L\geq0`$. Grouping $`k`$ inherited elementary steps gives
$`S_k(L)S_N(L^k)=S_{kN}(L)`$: a shorter macro horizon alone does not improve this
bound. A better learned predictor or optimization procedure must supply a gain.

## Concrete checked examples

- Fair-sign prediction: retaining the sign permits risk **0**; squaring it
  gives minimum risk **1**, over every predictor of the squared sign.
- Imperfect predictors: on the same fair-sign target, a chosen constant-zero
  upper predictor has risk **1**, versus **5** for a chosen constant-two lower
  predictor. This is an existence example, not a trained-model result.
- Control: a nonconstant background representation has exact dynamics at every
  finite horizon, yet every deterministic policy using only it has one-step
  success **1/2**. A policy that also observes the hidden mode succeeds with
  probability **1**. This does not show that the full hierarchy fails: lower
  levels may retain the mode.
- OLS expressions: with $`n=30`$, retained dimension $`d=2`$, discarded dimension
  $`q=8`$, and noise variance $`1`$, the threshold is **$`8/19`$**. Discarded signal
  variance $`1/10`$ improves the expression; variance $`1`$ worsens it. These are
  algebraic examples, not measured H-JEPA scores.

The regression comparison uses

```math
R_{\mathrm{full}}=\sigma^2\frac{n-1}{n-d-q-1},\qquad
R_{\mathrm{reduced}}=(\sigma^2+B)\frac{n-1}{n-d-1}.
```

For $`d,q\geq0`$ and $`n\gt d+q+1`$, Lean verifies
$`R_{\mathrm{reduced}}\lt R_{\mathrm{full}}\iff B\lt \sigma^2q/(n-d-q-1)`$.
Connecting these expressions to expected test error requires the Gaussian OLS
assumptions and probability argument in the note, outside the current Lean proof.

## Reproduce the proof check

Install [elan](https://github.com/leanprover/elan), then:

```sh
git clone https://github.com/quangng2000/hjepa-lean.git
cd hjepa-lean
lake exe cache get
lake build
lake env lean Audit.lean
```

The repository pins Lean **4.24.0** and mathlib commit
`f897ebcf72cd16f89ab4577d0c826cd14afaafc7` (the mathlib v4.24.0 release).
`lake-manifest.json` pins transitive dependencies. The first setup needs network
access and several GB of disk space; generated dependencies are not committed.
Run commands from this directory so elan uses the pinned toolchain.

For a smaller mathlib cache, fetch only this project's imports:

```sh
lake exe cache get \
  Mathlib.Algebra.BigOperators.Field \
  Mathlib.Algebra.Order.BigOperators.Group.Finset \
  Mathlib.Algebra.Ring.GeomSum \
  Mathlib.Data.Fintype.BigOperators \
  Mathlib.Data.Real.CompleteField \
  Mathlib.Order.ConditionallyCompleteLattice.Basic \
  Mathlib.Topology.MetricSpace.Basic \
  Mathlib.Tactic.FieldSimp Mathlib.Tactic.Linarith \
  Mathlib.Tactic.NormNum Mathlib.Tactic.Ring
```

`lake build` checks all nine modules through [`HJepa.lean`](HJepa.lean).
`lake env lean Audit.lean` additionally rejects transitive axiom dependencies
outside `propext`, `Classical.choice`, and `Quot.sound`. It scans public and
private declarations in the `HJepa` namespace, including definitions, and
prints theorem dependency reports. It fails if the namespace is empty.

There are no unfinished proofs, custom axioms, or `native_decide` proofs in the
development. Ordinary classical mathematics still uses the three standard Lean
axioms above; this is not an axiom-free or foundationally constructive project.
The audit checks the trust boundary, not the usefulness or truth of a model's
empirical assumptions. CI runs both build and audit on GitHub.

## Reading and extending the project

1. Start with `Abstraction.lean` for the central theorem and explicit finite laws.
2. Read `ThreeLevel.lean`, `Vector.lean`, and `Readout.lean` for the H-WavJEPA extension.
3. Read `AudioExamples.lean` and `Counterexamples.lean` to see why the assumptions matter.
4. Read `Planning.lean` and `Rollout.lean` for the composed planning guarantee.
5. Read `OLS.lean` together with its probability-formalization caveat.
6. Read `main.tex` for the broader argument and research directions.

Good next milestones are a general measure-theoretic extension, the
Gaussian OLS expectation proof, the linear spectral theorem, and assumptions
linking representation learning to task calibration. These are open work here,
not hidden premises already verified by this repository.

For contributions, retain explicit assumptions, import new proof modules from
`HJepa.lean`, update the coverage table, and run both verification commands.
Do not replace proof obligations with new axioms or claim neural-architecture
guarantees from a simplified model.

## Attribution

Inspired by Zhang et al., *H-JEPA: End-to-End Learning of Hierarchical World
Models for Visual Planning*, arXiv:2610.06805v1 (2026).
[Paper](https://arxiv.org/abs/2610.06805v1) · [Project](https://h-jepa.com/).
The paper's Appendix L already studies collapse and representation limitations;
those are prior results, not contributions claimed here. This independent
formalization is not affiliated with or endorsed by the paper's authors.

The risk projection identities, regression formulas, and calibration arguments
use standard mathematics. Lean checks the stated formal results; it does not
establish originality. The companion note records the analytical derivations.

MIT licensed; see [LICENSE](LICENSE). Lean and mathlib retain their own licenses.
