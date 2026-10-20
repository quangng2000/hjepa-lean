# H-JEPA gains and losses in Lean 4

[![Lean proof check](https://github.com/quangng2000/hjepa-lean/actions/workflows/lean.yml/badge.svg)](https://github.com/quangng2000/hjepa-lean/actions/workflows/lean.yml)

Machine-checked, conditional comparisons of hierarchical and flat prediction
and planning, motivated by
[H-JEPA: End-to-End Learning of Hierarchical World Models for Visual Planning](https://arxiv.org/abs/2610.06805v1).

The main result makes the tradeoff precise: **abstraction helps prediction only
when its reduction in excess prediction error exceeds the information it loses.**
The project also proves a composed-planner regret bound, a temporal-grouping
identity, algebraic regression thresholds, and explicit counterexamples.

This is a formal mathematical companion—not a verification of the H-JEPA
implementation, a replication of its experiments, or a proof that one trained
neural architecture universally outperforms another. Novelty is not claimed.

## Start here: the exact gain–loss identity

Fix a finite probability space, a real-valued target $Y$, lower features $Z$
with a finite label type, and upper features $U=g(Z)$. Both predictors are evaluated against the **same
target and loss**. The code constructs conditional means directly from weighted
finite sums, including zero-probability fibers, and proves their optimality.

Write $R^*(V)$ for the optimal squared-loss prediction risk from features $V$.
The information penalty is

$$
B = R^*(U)-R^*(Z)
  = \mathbb E[(\mathbb E[Y\mid Z]-\mathbb E[Y\mid U])^2] \geq 0.
$$

For chosen predictors $f_Z$ and $f_U$, let
$A_V=R(f_V)-R^*(V)$ be excess risk. Then

$$
R(f_U)-R(f_Z)=B+A_U-A_Z,
\qquad
R(f_U)<R(f_Z)\iff A_Z-A_U>B.
$$

In plain language: the upper representation cannot beat the information
available to an optimal lower-level predictor. It can beat an imperfect
predictor if its error reduction is large enough. The identity does not promise
that training will produce that reduction.

Read [`Abstraction.lean`](HJepa/Abstraction.lean), especially
`nested_risk_identity`, `learning_gain_loss_identity`, and
`abstraction_improves_iff`. The zero-gap characterization is conditional-mean
sufficiency for this target and squared loss—not preservation of all task
information or of the full conditional distribution.

## Formalization coverage

The accompanying [research note](main.tex) contains broader analytical arguments.
Its full contents are **not** all formalized. This is the exact boundary:

| Result | Lean coverage | Assumptions / boundary |
| --- | --- | --- |
| Conditional mean and Bayes optimality | [`Abstraction.lean`](HJepa/Abstraction.lean): `fiber_residual_zero`, `residual_orthogonality`, `risk_pythagorean`, `bayes_optimal` | Finite probability law and feature-label type, scalar real target, fixed encoder; zero weights allowed. |
| Information loss and learning-error tradeoff | `nested_risk_identity`, `abstraction_no_loss_iff`, `abstraction_improves_iff` | Nested deterministic features; fixed predictors. No theorem about a training algorithm or its sample complexity. |
| OLS gain/loss threshold | [`OLS.lean`](HJepa/OLS.lean): `risk_difference`, `reduced_better_iff`, `risks_equal_iff`, `reduced_worse_iff` | Algebra of explicit rational risk expressions. Gaussian-design / inverse-Wishart expectation formulas are **not** formalized. |
| Composed-planner regret | [`Planning.lean`](HJepa/Planning.lean): `composite_planner_regret`, `composite_planner_regret_of_infimum`, `flat_planner_regret`, `exact_calibration_optimality` | True cost bounded below; a candidate exists; uniform task-cost calibration and global approximate surrogate optimization. Infima need not be attained. |
| Rollout error and cost calibration | [`Rollout.lean`](HJepa/Rollout.lean): `dynamics_error_recurrence`, `rollout_error_bound`, `task_cost_calibration` | Explicit local-error, Lipschitz, and task-calibration assumptions; deterministic fixed-horizon statements. |
| No improvement from regrouping inherited error bounds | `geometric_blocking`, `inherited_full_bound_equality` | Exact algebraic identity; includes zero horizons and multiplier one. Not a lower bound on actual learned-model error. |
| Perfect latent prediction can fail control | [`Counterexamples.lean`](HJepa/Counterexamples.lean): `abstract_rollout_exact`, `abstract_policy_success_half`, `full_policy_success_one` | Actual finite four-state average; arbitrary deterministic background-only policy, compared with full-state access. |
| Sign-discarding prediction loss | `squared_sign_risk_at_least_one`, `zero_predictor_attains_one`, `retained_sign_risk_zero` | Fair finite $\pm1$ target: optimal squared-loss risk one versus zero. |
| Gaussian AR / spectral feature-selection argument | Analytical note only | No Lean proof of its stochastic process or matrix optimization result yet. |
| Trained nonlinear H-JEPA vs independently trained flat JEPA | Not proved | No verification of neural training, generalization, SIGReg, experimental scores, or closed-loop control. |

## Planning: where gains and losses enter

Let $J(a)$ be true primitive-plan cost, $Q(b)$ the macro-plan surrogate, and
$\mathcal R(b)$ the **actual refinement procedure**. Define

$$
\rho=\inf_b J(\mathcal R(b))-\inf_a J(a)\geq0.
$$

Assume uniform calibration
$|Q(b)-J(\mathcal R(b))|\leq\delta$ and a returned macro plan whose surrogate
cost is within $\eta$ of every competitor. Lean proves

$$
J(\mathcal R(\hat b))-\inf_a J(a)\leq2\delta+\eta+\rho.
$$

Here $\delta$ is cost-model error, $\eta$ is optimization error of the **whole
composed planner**, and $\rho$ is the cost of restricting available plans.
The flat planner is identity refinement and has zero restriction gap.
Lower values of these bounds do not, by themselves, order actual achieved costs.
Accurate latent prediction alone does not establish task-cost calibration.

For rollout error, Lean also proves

$$
e_{t+1}\leq\epsilon+Le_t
\implies
e_N\leq L^Ne_0+\epsilon S_N(L),
\qquad S_N(L)=\sum_{i=0}^{N-1}L^i,
$$

with $L\geq0$. Grouping $k$ inherited elementary steps gives
$S_k(L)S_N(L^k)=S_{kN}(L)$: a shorter macro horizon alone does not improve this
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
- OLS expressions: with $n=30$, retained dimension $d=2$, discarded dimension
  $q=8$, and noise variance $1$, the threshold is **$8/19$**. Discarded signal
  variance $1/10$ improves the expression; variance $1$ worsens it. These are
  algebraic examples, not measured H-JEPA scores.

The regression comparison uses

$$
R_{\mathrm{full}}=\sigma^2\frac{n-1}{n-d-q-1},\qquad
R_{\mathrm{reduced}}=(\sigma^2+B)\frac{n-1}{n-d-1}.
$$

For $d,q\geq0$ and $n>d+q+1$, Lean verifies
$R_{\mathrm{reduced}}<R_{\mathrm{full}}\iff B<\sigma^2q/(n-d-q-1)$.
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

`lake build` checks the five modules through [`HJepa.lean`](HJepa.lean).
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
2. Read `Counterexamples.lean` to see why the assumptions matter.
3. Read `Planning.lean` and `Rollout.lean` for the composed planning guarantee.
4. Read `OLS.lean` together with its probability-formalization caveat.
5. Read `main.tex` for the broader argument and research directions.

Good next milestones are a measure-theoretic/vector-valued extension, the
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
