# When three-level fusion improves spoof-cue classification

The answer depends on **what the downstream detector is allowed to compute**.
Lean verifies both an impossibility result for unrestricted detectors and a
strict positive example for affine-threshold detectors, on actual 0–1 error.

## The information boundary

Fix the entire fine context $`z_1`$ and deterministic higher encoders:

```math
z_2=g(z_1),\qquad z_3=h(z_2),\qquad Z=(z_1,z_2,z_3).
```

Every classifier on $`z_1`$ can be lifted to $`Z`$ by ignoring the other levels.
Every classifier on $`Z`$ can be composed with $`g,h`$ to become a classifier on
$`z_1`$. Lean proves that their **entire sets of achievable classification risks
are equal**. If unrestricted optima are attained, their errors are equal too.

Thus deterministic fusion cannot beat the unrestricted optimum using the same
complete fine information. This includes the same temporal context: comparing
one fine token against coarse features from a wider window would change the
available information and is outside this equality.

The useful possibility is making an existing cue easier for a **restricted
detector** to access. A nonlinear encoder can expose interactions that an affine
head cannot compute by itself.

## A positive example that actually needs all three branches

Let $`A,B,C`$ be independent fair signed cue coordinates. The constructed nested
feature maps are

```math
z_1=(A,B,C),\qquad z_2=(AB,BC),\qquad
z_3=(AB)(BC)=AC.
```

The fixed balanced label is the sign of $`A+AB+AC`$. Equivalently, it is the sign
of $`A`$ except when both $`B`$ and $`C`$ are negative, when it is reversed.
This is a defined cue-interaction task, not an assertion about real spoofed audio.

The full affine score uses a coordinate from each branch:

```math
f(Z)=z_{1,A}+z_{2,AB}+z_3=A+AB+AC.
```

Lean proves sharp population classification comparisons on all eight outcomes:

- **All three levels, affine detector:** minimum error **0%**, attained by $`f`$.
- **Fine level alone, any affine detector:** minimum error **12.5%**.
- **Fine + middle, omitting high, any affine detector:** minimum error **12.5%**.
- **Fine + high, omitting middle, any affine detector:** minimum error **12.5%**.
- **Middle + high, omitting fine, any deterministic detector:** minimum error
  **25%**, even allowing nonlinear heads.

The lower bounds compare entire stated classifier classes. They do not compare
the good fused head with one poorly chosen fine head. The fine affine witness
$`-3/4+A+B/2+C/2`$ attains its 12.5% floor and is also available in the two
fine-containing ablations. The coarse-only floor has its own attaining head.

An unrestricted nonlinear fine-only detector can directly compute the label and
achieves zero. The gain is restricted-head accessibility, not added information.
The hierarchy's overlapping pair products are handcrafted feature maps; they
are not an implementation of the proposed stride-two audio Conv1D or proof that
JEPA training learns these maps. Extra feature computation and head dimensions
are explicit; no matched-compute claim is made.

## When approximate fitting still wins

The general comparison specifies fine and fused classifier families. If every
fine classifier has error at least $`\alpha`$, a feasible fused classifier has
error at most $`\beta`$, and a fitted fused classifier is within $`\varepsilon`$
of the best fused population error, then

```math
\beta+\varepsilon<\alpha
\quad\Longrightarrow\quad
R_{\rm fused}<R_{\rm fine}\ \text{for every classifier in the fine family}.
```

The example instantiates these conditions with $`\alpha=1/8`$ and $`\beta=0`$.
Its fused global optimum is explicitly attained. An optimization allowance
strictly below $`1/8`$ preserves the advantage. These are population-fitting
conditions, not an SGD convergence, finite-sample, or generalization guarantee.
In this particular eight-point uniform model, deterministic classification errors
change in increments of $`1/8`$, so that allowance entails zero misclassified
outcomes. The general gap theorem is not restricted to this discrete example.

The ideal fused score also has a verified signed margin of at least one.
Post-feature score perturbations below that margin preserve zero error. A
sufficient condition for three additive branch-score errors is
$`|e_1|+|e_2|+|e_3|<1`$ on every outcome. This does not derive such an error bound
from a noisy waveform or establish new bounds for a perturbed baseline.

## Which aggregation is theoretically best?

There is no universal ranking of selection, averaging, and differencing.
The [cue counterexamples](SPOOFING.md) give opposite winners, now also packaged
as formal failures of universal domination even with freely chosen classifiers.

There are two conditional answers that Lean verifies:

1. **If unrestricted detection accuracy is the only criterion, a recoverable
   aggregate is never worse than any deterministic reduction of the same input.**
   Its classifier can recover the input and simulate the competing pipeline.
   Sum and difference provide a concrete recoverable two-channel bank. This
   accounts for the two output channels and does not imply free compression,
   computational efficiency, or easy learning of the recovered classifier.
2. **With an affine detector, nonlinear feature construction can help, while
   affine aggregation alone adds no scoring expressiveness.** Lean proves that
   an affine detector after affine aggregation collapses to a raw affine detector.
   It also proves this for affine fusion of all three affine levels. The positive
   cue example uses nonlinear products, which escape that restriction.

A fixed Conv1D without nonlinearities is an affine map, so the latter boundary
applies to it followed only by affine stages. **Conv1D plus a nonlinear $`C_2`$
or $`C_3`$ is outside that restriction.** Input-dependent attention is also
outside it. These proofs do not rank particular nonlinear architectures.

For an audio experiment, this supports retaining a fine branch, using higher
levels to offer complementary nonlinear features, and evaluating both fusion
and fine-only detectors. It does not prove that learned Conv1D is the empirical
winner. Real spoofing evaluation still needs held-out attacks, speakers and
recording conditions; the toy errors above are not EER or min-tDCF.

## Checked source

Entry point: [`HJepa/Spoofing/Fusion.lean`](HJepa/Spoofing/Fusion.lean).

- `General`: unrestricted classification lift/pullback and risk-set equality.
- `Conditions`: explicit classifier families, inclusion, and strict-gap bounds.
- `Example/`: the actual finite hierarchy, all-affine lower bounds, exact
  attaining classifiers and nonlinear coarse-only lower bound.
- `ExampleBridge`: concrete families, attained optima and approximate-fit gain.
- `Margin`, `ExampleMargin`: balanced labels, signed margin and robustness.
- `Aggregation/`: affine collapse, nested affine collapse, recovery, optimality
  among deterministic reductions, and no universal fixed-operator winner.

All results use ordinary Lean proofs, with no custom axioms or unfinished proofs.
Run `lake build` and `lake env lean Audit.lean`. These are standard mathematical
mechanisms formalized under explicit assumptions; no novelty claim is made.
