# H-WavJEPA theoretical evidence and proof obligations

Hierarchy alone does not guarantee better prediction. The checked results
establish a narrower positive claim: **nonlinear composition can make existing
task information accessible to a restricted readout**. They also establish
conditions for learning a good readout and for reducing observation noise.
The protected-objective extension now proves that a specified task-aware loss
selects useful representations at every global optimum in a concrete finite
model, plus conditional guarantees for general finite populations. Whether an
audio optimizer reaches those solutions and generalizes remains open. The
encoder-optimization extension now discharges that learning obligation for
specified linear finite-data models, including an exact IID SGD risk law.

## Claim ledger

All Lean results below concern the stated mathematical models, not a verified
implementation of a neural architecture.

| Claim | Verdict | Evidence and boundary |
| --- | --- | --- |
| Deterministic hierarchy improves optimal prediction from the same context. | Refuted in general | Bayes risk can only increase under abstraction. The finite three-level witness has optimal risks 0, 4, 5. [`ThreeLevel.lean`](HJepa/ThreeLevel.lean), [`AudioExamples.lean`](HJepa/AudioExamples.lean). |
| Composition can strictly help a restricted readout. | Proved by construction | Four-sign parity has optimal affine risks 1, 1, 0 across three fixed levels. All unrestricted Bayes risks are 0. [`Expressivity.lean`](HJepa/Expressivity.lean). |
| A finite training set can transfer this advantage to fitted readouts. | Proved by construction | Two balanced examples give exact empirical/population equality for every high affine readout; fitting tolerance below 1 implies strict superiority. [`LearningExamples.lean`](HJepa/LearningExamples.lean). |
| A specified optimizer reaches the useful fixed-feature readout. | Proved by construction | Residual-based gradient descent with step size 1/2 reaches zero risk from every initialization on the balanced quadratic loss. [`Optimization.lean`](HJepa/Optimization.lean). |
| The readout advantage survives random training samples. | Proved for a specified learner | A label-based affine fitter on the fixed parity feature has expected population risk 4/2^n for n positive IID raw examples; with three examples this is 1/2. [`SampleLearning.lean`](HJepa/SampleLearning.lean). |
| Fitting a readout transfers the representation advantage to actual risk. | Proved under explicit conditions | Approximate empirical minimization plus uniform deviation gives an oracle bound; a certified flat-class lower bound establishes strict superiority. [`Learning.lean`](HJepa/Learning.lean). |
| A longer averaging window always improves prediction. | Refuted in general | Bias can exceed the noise saved; perfectly shared error gives no gain. Exact thresholds are proved. [`Pooling.lean`](HJepa/Pooling.lean). |
| Perfect latent prediction guarantees useful control. | Refuted by construction | A nonconstant background-only representation has exact dynamics, but every policy using only it succeeds with probability 1/2 rather than 1. [`Counterexamples.lean`](HJepa/Counterexamples.lean). |
| Global predictive optimality plus variance normalization guarantees useful features. | Refuted by construction | Every global optimum in a finite shared-encoder next-latent prediction model loses the current task bit, despite an admissible encoder retaining it. [`Objective.lean`](HJepa/Objective.lean). |
| A protected-channel objective can exclude those task-discarding optima. | Proved by construction | A normalized two-channel zero-loss witness exists; every positive-weight global optimum recovers the task. Approximate task risk is at most optimization tolerance divided by protection weight. Extra capacity and supervision are explicit. [`ProtectedChannel.lean`](HJepa/ProtectedChannel.lean). |
| Task-aware optimization controls information lost by the learned encoder. | Proved under explicit conditions | A predictive floor, feasible comparator, and approximate optimization give a conditional-mean information-gap bound. A uniform empirical/population objective bound gives the empirical variant. [`TaskObjective.lean`](HJepa/TaskObjective.lean). |
| A label-free reconstruction term can protect capabilities of a fixed reference. | Proved under explicit conditions | Exact population recovery preserves every reference readout's risk; approximate recovery needs a sensitivity bound. Does not prove teacher quality or joint predictive feasibility. [`Anchor.lean`](HJepa/Anchor.lean). |
| An optimizer actually learns the protected encoder, not just its head. | Proved for a specified full-batch algorithm | Residual-gradient updates on balanced raw labeled inputs learn the task coordinate from any initial linear encoder, with a fixed identity decoder. The normalized endpoint is globally optimal. [`EncoderLearning.lean`](HJepa/EncoderLearning.lean). |
| Stochastic encoder updates have a population-learning guarantee. | Proved for a specified IID model | Exact expected population risk after n updates is 2^-n times initial risk at the chosen learning rate. Some finite sample sequences fail. [`StochasticEncoder.lean`](HJepa/StochasticEncoder.lean). |
| Ordinary simultaneous encoder/head GD succeeds from every initialization. | Refuted by construction | A normalized background encoder with zero decoder is stationary at task risk 1 for every step-size schedule. A different guarded block-gradient method succeeds. [`JointOptimization.lean`](HJepa/JointOptimization.lean). |
| A protected reference encoder can be learned without downstream labels. | Proved for a specified full-batch algorithm | A learned matrix encoder reaches exact reconstruction from any initialization; the fixed reference is the whole two-coordinate raw input and the decoder is identity. No compression or teacher-quality gain. [`ReferenceLearning.lean`](HJepa/ReferenceLearning.lean). |
| Lossy compression can guarantee preservation of every future task. | Refuted on finite support | All-task preservation is equivalent to recovery of the original features on support. Collisions give a binary-task risk lower bound. [`UniversalPreservation.lean`](HJepa/UniversalPreservation.lean). |
| A smaller error upper bound proves better actual performance. | Refuted by construction | Actual risks 2 and 1 can satisfy respective bounds 3 and 4. [`Learning.lean`](HJepa/Learning.lean). |
| The specified H-WavJEPA training procedure beats a matched WavJEPA baseline. | Open | Requires a specified objective, data model, optimizer, baseline class, and resource budget, followed by a theorem about those objects. |

## What the comparison must hold fixed

Use one population law, downstream target, loss, and accessible input context.
For nested features, write

```math
Z_1=Z,\qquad Z_2=g(Z_1),\qquad Z_3=h(Z_2),\qquad
R(f)=\mathbb{E}[(Y-f)^2].
```

If an upper encoder consumes a window, the lower baseline must have access to
that entire window. A single token versus a pooled window is not a
same-information comparison. The displayed composed, frozen encoders satisfy
the nesting assumption; independently trained encoders need not.

The unrestricted Bayes predictor minimizes risk over all functions of its
features. A restricted readout class permits only specified functions, such as
affine maps. Neither means the full neural architecture has the same parameter,
compute, or training budget. Different levels' latent JEPA losses also have
different targets and cannot be substituted for this common downstream risk.

## A constructive three-level advantage

Let four independent fair signs be the input and their product the target:

```math
\begin{aligned}
Y&=S_1S_2S_3S_4, & S_i&\in\{-1,1\},\\
Z_1&=(S_1,S_2,S_3,S_4),\\
Z_2&=(S_1S_2,S_3S_4),\\
Z_3&=(S_1S_2)(S_3S_4).
\end{aligned}
```

The Lean proof expands the exact finite expectation over all sixteen equally
likely inputs. For every real choice of affine coefficients:

```math
\begin{aligned}
R\!\left(b+\sum_{i=1}^{4}a_i S_i\right)
  &=1+b^2+\sum_{i=1}^{4}a_i^2\geq1,\\
R(b+a_1S_1S_2+a_2S_3S_4)
  &=1+b^2+a_1^2+a_2^2\geq1,\\
R(b+aZ_3)&=(1-a)^2+b^2.
\end{aligned}
```

Zero coefficients attain the first two lower bounds; the high readout with
zero bias and unit slope attains risk zero. This proves strict improvement
against **every affine low and middle readout**. Intuitively, the target has
zero correlation with each individual low feature and each middle feature,
but composition exposes it as a single high feature.

The encoders are handcrafted using the task structure. A nonlinear flat
readout can compute the same product, and unrestricted Bayes risk is zero at
every level. Thus the result is an accessibility mechanism, not added
information, a theorem that JEPA learns parity, or a compute-efficiency result.
The proof is `high_strictly_beats_affine_readouts`; the equal-information check
is `all_bayes_risks_zero` in [`Expressivity.lean`](HJepa/Expressivity.lean).

## A concrete fitting guarantee without a deviation assumption

Choose the raw examples $`(+1,+1,+1,+1)`$ and $`(-1,+1,+1,+1)`$ with equal
weights and their actual parity labels. The high feature is respectively
$`+1`$ and $`-1`$, just as in the full population. For every high affine
readout, the empirical squared risk on these two examples is therefore
**exactly equal** to its population risk over all sixteen raw inputs.

The proof calculates that equality from the examples, rather than assuming
it. The identity readout is an exact empirical minimizer. Every
$`\eta`$-approximate empirical minimizer satisfies

```math
R(\hat f_3)\leq\eta.
```

Thus $`\eta\lt1`$ makes the fitted high readout strictly better than every
affine low or middle readout. This supplies an explicit, nonvacuous learning
construction. It does not claim two random examples suffice, that an arbitrary
optimizer converges, or that the handcrafted encoders are learned. See
`balanced_empirical_eq_population` and
`approximate_fit_beats_all_affine_readouts` in
[`LearningExamples.lean`](HJepa/LearningExamples.lean).

## An optimizer with a checked risk trajectory

For the balanced fixed-feature problem, define gradient descent using the
average of the two examples' squared-loss residual gradients. The proof derives
the gradient $`(2b,2(a-1))`$ and certifies its linear term by an exact loss
expansion with quadratic remainder. For step size $`\alpha`$ and any initial
coefficients, the implemented mathematical recurrence satisfies

```math
R_t=\left((1-2\alpha)^2\right)^t R_0.
```

A step of size $`1/2`$ reaches $`(b,a)=(0,1)`$ in one step, hence zero
population risk and strict superiority over every affine low or middle
readout. Steps $`0\lt\alpha\lt1`$ reduce nonzero error; $`\alpha=1`$ preserves
it; $`\alpha\gt1`$ increases it. Thus even this simple fitting guarantee
depends on the update rule and step size.

This discharges the optimizer condition for a particular quadratic readout
problem. It does not analyze gradients through the encoders, random
initialization of a full network, or JEPA's self-supervised objective. See
`exact_loss_increment`, `iterate_population_risk`, and
`half_step_beats_all_affine_readouts` in
[`Optimization.lean`](HJepa/Optimization.lean).

## Random training samples and exact generalization risk

Now draw raw four-sign inputs independently from the original population,
encode each with the fixed high feature, and pair it with its actual target
label. The learner accepts arbitrary real labels. It stores the first label
seen for each of the two feature signs. With both signs observed, it solves
the two affine equations:

```math
b=\frac{y_++y_-}{2},\qquad a=\frac{y_+-y_-}{2}.
```

With only one sign observed, it returns the constant observed label. The
coefficients are calculated from the data; the algorithm is not simply given
the correct parity readout. Repeated contradictory labels are outside this
noiseless guarantee.

The proof explicitly averages IID raw samples, proves that their encoded signs
are fair, and derives the learner's expected population risk:

```math
\mathbb{E}_{\mathrm{training\ sample}}[R(\hat f_n)]
=\frac{4}{2^n},\qquad n\geq1.
```

The learner interpolates every training example, but on a one-sign sample its
population risk is 2; when both signs occur it is 0. Thus zero training error
does not give a per-sample test guarantee.

| Number of IID examples | Expected population risk | Best affine raw-input risk |
| --- | --- | --- |
| 1 | 2 | 1 |
| 2 | 1 | 1 |
| 3 | 1/2 | 1 |

The comparison also covers affine coefficients selected from the same random
training dataset: every such fitted low or middle predictor has population
risk at least 1 for each dataset, hence expected risk at least 1. Thus the
three-example result compares learning rules, not just fixed baseline weights.

This is a concrete finite-sample learning result with a derived sampling law,
not an assumed uniform-deviation bound. It concerns a particular supervised
readout rule on known, fixed nonlinear features. It is not a lower bound on
the samples needed by every learner, and it does not show JEPA discovers the
features. A flat nonlinear predictor can solve the same task. See
`iid_labeled_expected_population_risk`, `learned_raw_training_loss_zero`, and
`three_samples_beat_trained_affine_readouts` in
[`SampleLearning.lean`](HJepa/SampleLearning.lean).

## General conditional learning bound

For a fixed admissible class, suppose empirical risk uniformly approximates
population risk and the returned predictor approximately minimizes empirical
risk:

```math
|R(f)-\widehat R(f)|\leq\varepsilon\quad\text{for every admissible }f,
\qquad
\widehat R(\hat f)\leq\widehat R(f)+\eta
\quad\text{for every admissible }f.
```

Then, for each admissible comparator,

```math
R(\hat f)\leq R(f)+2\varepsilon+\eta.
```

The proof passes from population to empirical risk, uses approximate
minimization, then returns to population risk. It is not a proof that a
particular sample size gives the deviation bound, or that gradient descent
achieves the optimization tolerance.

For three levels, let the information losses be
$`B_{12}=R^{*}(Z_2)-R^{*}(Z_1)`$ and
$`B_{23}=R^{*}(Z_3)-R^{*}(Z_2)`$. Suppose a high-level comparator has excess
risk at most $`a`$ relative to its own Bayes predictor. Lean proves

```math
R(\hat f_3)\leq R^{*}(Z_1)+B_{12}+B_{23}+a+2\varepsilon+\eta.
```

To prove actual superiority, also establish that **every** admissible flat
predictor has risk at least $`R^{*}(Z_1)+\Delta`$. The sufficient condition is

```math
B_{12}+B_{23}+a+2\varepsilon+\eta\lt\Delta.
```

This compares a hierarchical upper bound with a flat lower bound, not two
upper bounds. In the parity construction, both information losses and the
best high approximation error vanish, while the affine flat lower gap is 1.
A justified $`2\varepsilon+\eta\lt1`$ therefore suffices. If the flat class
contains the Bayes predictor, it has no positive lower gap to exploit.
See `three_level_strict_superiority` in [`Learning.lean`](HJepa/Learning.lean).

For data-dependent features or class selection, the deviation bound must
cover that selection too. A bound for a frozen class cannot silently be reused
after adapting the encoder to the same data.
With squared loss over all unbounded affine coefficients, a finite uniform
bound fails if empirical and population quadratic risks differ in any
coefficient-dependent term. The balanced construction above proves exact equality;
ordinary finite-sample applications need suitable class restrictions and a
separate deviation argument.

## Temporal noise reduction and its limits

Consider aligned scalar observations $`X_i=Y+e_i`$ of one common target over
a nonempty window of size $`n`$. The pooled prediction is their arithmetic
mean. Without any independence assumption, Lean proves

```math
R\!\left(\frac1n\sum_i X_i\right)
=\frac{1}{n^2}\sum_{i,j}\mathbb{E}[e_i e_j].
```

If the errors are centered, each has second moment $`v`$, and each distinct
pair has cross moment $`c`$, then adding a constant pooling bias $`b`$ gives

```math
R_{\mathrm{pool}}=b^2+\frac{v}{n}+\frac{n-1}{n}c,
\qquad R_{\mathrm{single}}=v.
```

Consequently,

```math
R_{\mathrm{pool}}\lt R_{\mathrm{single}}
\iff b^2\lt(v-c)\left(1-\frac1n\right).
```

With zero bias and zero cross moments, risk becomes $`v/n`$. If errors are
perfectly shared, $`c=v`$, averaging provides no gain even with zero bias.
Signal variation or misalignment can erase noise savings. The constant-bias
model is one explicit case, not a theorem covering all audio transients.
The moment hypotheses describe actual errors under the finite law; they do
not assert that arbitrary choices of $`v,c`$ are realizable.

These are theorems about averaging, available to a flat predictor too. They
motivate a possible H-WavJEPA design only if its representations actually
implement such an operation. H-JEPA's reported experiments use pointwise
upper state encoders with window size one; the paper does not establish
literal state averaging as the source of its gains.
[H-JEPA Section 2.1](https://arxiv.org/html/2610.06805v1#S2.SS1).

See `pooled_risk_pairwise` and `biased_correlated_pooling_improves_iff` in
[`Pooling.lean`](HJepa/Pooling.lean).

## What a JEPA objective still has to establish

### A checked failure of objective alignment

Let the current input be $`(B,U)`$ and the future input $`(B,V)`$, with all
three bits independent and fair. The background $`B`$ remains fixed; the task
bit is independently resampled. Optimize squared next-latent prediction with
a shared scalar encoder, under zero-mean and unit-second-moment constraints:

```math
\mathcal{L}(E,P)
=\mathbb{E}\big[(E(B,V)-P(E(B,U)))^2\big],
\qquad \mathbb{E}[E]=0,\quad \mathbb{E}[E^2]=1.
```

The background encoder with identity predictor achieves loss zero and obeys
the normalization constraint. Thus the global minimum is zero. Full support
of the finite law forces every zero-loss encoder to satisfy
$`E(b,u)=E(b,\mathrm{false})`$: for the same current input, it must predict both
possible future task bits exactly, so it cannot distinguish them in its own
encoding.

Consequently, **every global optimum** discards the current task bit. For the
downstream target $`Y=\operatorname{sign}(U)`$, every readout of such an encoder
has risk at least 1. The normalized task encoder
$`E(B,U)=\operatorname{sign}(U)`$ instead permits downstream risk zero, but every
predictor paired with it incurs next-latent loss at least 1.

This proves a strict mismatch between the training objective and the specified
downstream task, not merely a failure of one optimizer to find a good optimum.
The example concerns a finite moment-normalized objective, **not** SIGReg, EMA
targets, or the complete H-JEPA loss. The task is the current bit; no method
could infer the independent future bit from the current input. See
`every_global_optimum_task_risk_at_least_one` and
`objective_and_task_rank_oppositely` in [`Objective.lean`](HJepa/Objective.lean).

## A protected objective with a checked positive guarantee

### Design: separate predictive and preservation obligations

An unpredictable current signal need not be useless. Instead of forcing every
retained feature to predict itself in the future, separate the representation:

```math
Z_1=(S_1,R),\qquad Z_2=(g(S_1),R),\qquad
Z_3=(h(g(S_1)),R).
```

The predictive branch may become coarser; the protected coordinate remains
accessible unchanged at all three levels. This is an architectural constraint,
not a claim that an arbitrary learned bottleneck discovers preservation. The
choice of low/medium/high time scales remains an additional modeling decision.

Use the objective

```math
\mathcal{J}(c)=P(c)+\lambda T(c),\qquad \lambda\gt0,
```

where a candidate includes its encoders and heads. For a known task,
$`T(c)=\mathbb{E}[(Y-D_c(R_c))^2]`$. This objective explicitly uses task labels:
it is a supervised or task-augmented extension, not a proof that unlabeled
JEPA alone discovers semantic relevance. A sum of nonnegative predictive
losses at several levels can serve as $`P`$ in the generic theorem. The
concrete checked example below instantiates one predictive loss and three
levels with task-preserving carry maps, not a full multiscale audio-training objective.

### General quantitative theorem, including noisy targets

Fix the finite population, task, lower input context $`X`$, and admissible
candidate family. The learned representation is a deterministic function of
that same context. Let $`r_*=R^{*}(X)`$ and suppose:

- Every admissible candidate has $`P(c)\geq p_*`$.
- A feasible comparator has $`P(c_0)\leq p_*+C`$ and
  $`T(c_0)\leq r_*+\delta`$.
- The selected candidate is $`\eta`$-approximately globally optimal for
  $`\mathcal{J}`$ over that family.

Then

```math
\begin{aligned}
\lambda\big(T(\hat c)-r_*\big)&\leq C+\lambda\delta+\eta,\\
0\leq B_{\mathrm{task}}(\hat c)
&\leq T(\hat c)-r_*
\leq\delta+\frac{C+\eta}{\lambda}.
\end{aligned}
```

The first step compares the weighted objective to the feasible comparator;
the second uses Bayes optimality of the selected representation's unrestricted
task readout. Thus we do not assume the selected encoder already preserves
information. The comparator's costs remain visible when prediction and
preservation compete.

If predictive-floor performance and Bayes task performance are jointly
attainable, $`C=\delta=0`$, giving $`B_{\mathrm{task}}\leq\eta/\lambda`$.
At exact optimality the task's conditional mean is preserved on support.
The Bayes risk itself can be positive: irreducible label noise does not require
zero task loss. This is squared-loss, task-specific sufficiency—not equality
of all conditional distributions or preservation of all semantic attributes.

The proof allows arbitrary ambient representation types, including real
vectors, by restricting the Bayes calculation to the finite realized image.
The population still must be finite; this is not a general measure-theoretic
result. See `risk_pythagorean_finite_support`,
`information_bound_of_approximate_minimum`, and
`exact_minimum_preserves_task_information` in
[`TaskObjective.lean`](HJepa/TaskObjective.lean).

If the fitted objective is empirical, a **uniform bound for the entire
weighted objective** over the admissible class gives instead

```math
B_{\mathrm{task}}(\hat c)
\leq\delta+\frac{C+2\varepsilon+\eta}{\lambda}.
```

Here the uniform-deviation and approximate-fitting hypotheses are explicit,
not derived concentration or optimizer theorems. The deviation budget can
grow with $`\lambda`$; increasing the protection weight is not a free way to
remove statistical error. Learned encoder selection must be covered too.

### Concrete repair of the previous failure case

Keep the same current input $`(B,U)`$, future input $`(B,V)`$, finite law,
and current-bit downstream target. The new representation has two scalar
channels, each constrained to mean zero and second moment one:

```math
\mathcal{J}(S,R,P,D)
=\mathbb{E}[(S(B,V)-P(S(B,U)))^2]
+\lambda\mathbb{E}[(\operatorname{sign}(U)-D(R(B,U)))^2].
```

Choose $`S(B,U)=\operatorname{sign}(B)`$ and
$`R(B,U)=\operatorname{sign}(U)`$, with identity heads. Lean checks
normalization and both losses equal to zero. Nonnegativity then proves the
global minimum is zero, and **every** global optimum has zero task error.
For an approximate global optimum:

```math
\mathcal{L}_{\mathrm{predict}}\leq\eta,\qquad
R_{\mathrm{task}}\leq\frac{\eta}{\lambda}.
```

When $`\eta\lt\lambda`$, its task error is strictly below the lower bound 1
for **every** optimum of the original scalar predictive objective. The task
readout achieves the same risk at all three carry-map levels. Lean also proves
that the protected channel cannot simultaneously have zero own-future
prediction loss while recovering the independent current bit. Excluding it
from that predictive target is therefore substantive.

At protection weight zero, a task-discarding optimum still exists. No theorem
claims a neural optimizer reaches global optimality or that this is a
matched-width, matched-compute improvement. The protected model adds a scalar
channel and task supervision. Nor is scalar dimension alone an information
capacity bound with unrestricted real precision and arbitrary encoders.
See `every_global_minimum_preserves_task`,
`approximate_protection_beats_old_global_optimum`, and
`three_levels_preserve_task_at_global_minimum` in
[`ProtectedChannel.lean`](HJepa/ProtectedChannel.lean).

### Label-free option: reconstruct a genuinely fixed reference

When downstream tasks are not known, replace the task term by

```math
A=\mathbb{E}\lVert\Phi(X)-D(E_R(X))\rVert^2,
\qquad \mathcal{J}=P+\lambda A.
```

The finite-dimensional reference $`\Phi`$ is fixed throughout optimization.
It might be a frozen encoder or the original finite input vector. This uses
reconstruction supervision, not downstream labels; it changes the learning
objective from pure latent prediction. A stop-gradient operation on an EMA
or jointly evolving target is **not** the fixed-reference assumption here.

If $`A=0`$, any reference readout $`q`$ can be reproduced through the decoder:

```math
R(q\circ D\circ E_R)=R(q\circ\Phi).
```

Here $`E_R`$ denotes the protected encoder. Equality holds for arbitrary
reference readouts and arbitrary scalar task targets, including noisy ones, on positive
population support. A constant reference contains no input-varying information;
preserving it is not a guarantee of semantic quality. If reference and decoder are allowed to
co-collapse to a constant, the reconstruction loss is zero; that failure is
also checked.

For approximate recovery, assume
$`|q(a)-q(b)|^2\leq L^2\lVert a-b\rVert^2`$. Lean proves

```math
\begin{aligned}
\mathbb{E}[(q(\Phi(X))-q(D(E_R(X))))^2]&\leq L^2 A,\\
R(q\circ D\circ E_R)&\leq2R(q\circ\Phi)+2L^2 A.
\end{aligned}
```

A population certificate $`\mathcal{J}\leq\eta`$, with $`P\geq0`$ and
$`\lambda\gt0`$, gives $`A\leq\eta/\lambda`$. This certificate is an
**absolute objective budget**, not automatically suboptimality relative to
a nonzero optimum. If exact reconstruction is known, use the stronger exact
risk equality above rather than the factor-two bound.

Copying the whole reference explicitly achieves zero reconstruction error,
but does not alone prove simultaneous predictive optimality under shared
architecture constraints. A global-optimum claim needs that joint feasibility
established separately. The generic anchor interface deliberately leaves the
nonnegative predictive cost abstract. See `exact_reconstruction_preserves_readout`,
`zero_objective_preserves_readout`, `objective_bounds_task_risk`, and
`copy_reference_objective_eq_predictive_cost` in
[`Anchor.lean`](HJepa/Anchor.lean).

### The unavoidable limit for unknown tasks

For original features $`Z`$ and deterministic compression $`e`$, Lean proves
on finite positive-probability support:

```math
\begin{aligned}
&\text{every deterministic scalar task of }Z\text{ is exactly readable from }e(Z)\\
&\quad\iff e\text{ merges no distinct realized feature values}\\
&\quad\iff\text{some decoder recovers }Z\text{ on support}.
\end{aligned}
```

If two distinct features at outcomes $`a,b`$ collide, choose a binary task
assigning them labels 0 and 1. For every readout, the proof gives

```math
R_{\mathrm{task}}\geq
\frac{p(a)p(b)}{p(a)+p(b)}\gt0.
```

Thus universal future-task protection rules out genuinely lossy merging,
not reversible recoding or exploiting redundancy. It does not establish a
minimum neural dimension, bit budget, or efficient decoder. For compression
that really discards distinctions, specify a task family and tolerate an
appropriate information budget instead of promising all future tasks.
See `preserves_every_task_iff_recoverable` and
`collision_binary_risk_lower_bound` in
[`UniversalPreservation.lean`](HJepa/UniversalPreservation.lean).

## An optimizer that learns the protected representation

The objective results above assumed approximate optimization or a population
loss certificate. The following results instead define the parameter updates
and derive their output risk. They close that gap for explicit linear models,
not for a nonlinear H-WavJEPA implementation.

### Full-batch task encoder learning

Use independent fair signed raw inputs $`B,U`$, target $`Y=U`$, and the
trainable scalar encoder $`E_{a,b}(B,U)=aB+bU`$. Its downstream head is frozen
to the identity, so these parameters change the representation itself. The
predictive branch is also frozen at background/identity, with proved zero
predictive loss. The full protected objective restricted to this family is

```math
\mathcal{J}(a,b)=\lambda R(a,b),\qquad
R(a,b)=\mathbb{E}[(aB+bU-U)^2]=a^2+(b-1)^2.
```

All four raw inputs, each with its actual task label, form a balanced batch.
Lean proves that this empirical loss equals the population loss for every
encoder coefficient pair. Averaging the labeled residual gradients yields
$`\nabla R=(2a,2(b-1))`$. Exact first-order polynomial expansions certify
these coefficients, including the weighted objective; they are not assumed
gradient oracles or an update that directly assigns the optimum.

Gradient descent at learning rate $`\alpha`$ gives

```math
\begin{aligned}
a_{t+1}&=(1-2\alpha\lambda)a_t,\\
b_{t+1}-1&=(1-2\alpha\lambda)(b_t-1),\\
R_t&=(1-2\alpha\lambda)^{2t}R_0.
\end{aligned}
```

For $`\lambda\gt0`$, choosing $`\alpha=1/(2\lambda)`$ learns
$`(a,b)=(0,1)`$ in one step from **every** encoder initialization. This
reciprocal-curvature result exploits the whitened quadratic input geometry;
it is not a generally sufficient learning rate for neural networks.

The learned endpoint equals the previously constructed normalized protected
model. Thus the optimizer output is a global minimum even against all
admissible models of `ProtectedChannel`, and its task risk is zero at all
three carried levels. Intermediate iterates are not constrained to be
moment-normalized; this is not projected gradient descent on that constraint.
The identity decoder makes this a convex encoder-fitting problem, not a
claim that arbitrary jointly trained features are equally easy to learn.
See `exact_objective_increment`, `iterate_population_risk`,
`calibrated_step_global_minimum`, and `learned_three_levels_zero` in
[`EncoderLearning.lean`](HJepa/EncoderLearning.lean).

### Actual SGD on raw IID samples

For a raw sample $`x=(B,U)`$, the residual-gradient update of encoder
coefficients $`c=(a,b)`$ uses

```math
g_x(c)=2(aB+bU-U)(B,U),\qquad c'=c-\beta g_x(c).
```

For the weighted objective, the effective step is
$`\beta=\alpha\lambda`$. Lean certifies the sample gradient by an exact
sample-loss expansion and proves the weighted-step equivalence. Averaging
the actual update over the four equally likely raw inputs gives

```math
\mathbb{E}_{x}[R(c')]
=(1-4\beta+8\beta^2)R(c).
```

The code recursively processes each sample using the encoder produced by
the previous update. It also defines the exact IID expectation over ordered
sample lists, rather than assuming a contraction or concentration bound.
Induction then proves, for every finite $`n`$ and every initial encoder:

```math
\mathbb{E}_{x_1,\ldots,x_n}[R(c_n)]
=(1-4\beta+8\beta^2)^n R(c_0).
```

The optimal one-step factor in this constant-step family is $`1/2`$, attained
at $`\beta=1/4`$, or actual weighted-objective learning rate
$`\alpha=1/(4\lambda)`$. Consequently

```math
\mathbb{E}[R(c_n)]=2^{-n}R(c_0).
```

The same expectation holds at all three levels that retain this learned
protected coordinate. This is a derived finite-sample population-learning
identity for a noiseless supervised linear encoder with fixed decoder and
predictive branch. It is not a separately formalized almost-sure,
high-probability, or asymptotic convergence theorem.

The distinction matters: from $`c_0=(1,0)`$, repeating the raw input
$`(B,U)=(+1,+1)`$ gives zero sample gradient forever. The representation stays
background-only and its population task risk remains 2. Thus the positive
IID expectation is not a promise about every realized finite training set.
See `iid_expected_population_risk`, `quarter_step_three_level_expected_risks`,
and `repeated_direction_risk_two` in
[`StochasticEncoder.lean`](HJepa/StochasticEncoder.lean).

### Joint encoder/head training: a refutation and a specified remedy

Now let the decoder have trainable weight $`w`$. With the same finite law,
the exact population loss and gradient are

```math
\begin{aligned}
R(a,b,w)&=(wa)^2+(wb-1)^2,\\
\nabla R&=(2w^2a,\;2w(wb-1),\;2w(a^2+b^2)-2b).
\end{aligned}
```

The module derives these from raw-example residuals and certifies the first
order term by an exact polynomial expansion. At $`(a,b,w)=(1,0,0)`$, the
encoder is a normalized background feature, all three gradients vanish, and
task risk is 1. Lean proves that simultaneous full-gradient descent remains
there for **every step-size schedule and every number of steps**. The
identity-head proof therefore cannot be silently generalized to joint GD.

For $`\lambda\gt0`$, a different, explicitly defined solver does succeed:

1. If $`w=0`$, initialize it to 1. This is a deliberate guard, not a gradient step.
2. Freeze the now nonzero head and update only encoder weights with the actual
   residual gradient, using step $`1/(2\lambda w^2)`$ on the weighted objective.
3. Absorb the head scale into the encoder:
   $`(a,b,w)\mapsto(wa,wb,1)`$. Lean proves this preserves decoded predictions.

The block step learns $`(a,b)=(0,1/w)`$; the final reparameterization gives
$`(0,1,1)`$. Thus this guarded block algorithm learns a normalized zero-risk
protected model from any initial parameter triple. Its success is not a
theorem about ordinary simultaneous GD, SGD, or Adam. Near-zero head weights
also require very large exact steps; finite-precision numerical robustness
has not been proved.
The guard itself need not decrease loss: a checked example moves from risk
1 to 5 before the block update. The theorem concerns the completed solver's
output, not monotone improvement at every internal stage.
See `simultaneous_gd_failure`, `learned_parameters_from_weighted_gradient`,
and `learned_model_global_minimum` in
[`JointOptimization.lean`](HJepa/JointOptimization.lean).

### Learning reference protection without downstream task labels

For the label-free alternative, fix the reference to the two-coordinate raw
input $`\Phi(X)=(B,U)`$. Train a 2-by-2 matrix encoder $`E_W(X)=W\Phi(X)`$,
with a fixed identity decoder. The background predictive branch remains fixed
and perfect. The code derives the actual reconstruction residual gradients
from the four balanced raw inputs and proves empirical/population equality:

```math
A(W)=\mathbb{E}\lVert W\Phi(X)-\Phi(X)\rVert^2
=\lVert W-I\rVert_F^2,\qquad
\mathcal{J}(W)=\lambda A(W).
```

For $`\lambda\gt0`$, one full-batch gradient step of size $`1/(2\lambda)`$ learns $`W=I`$ from
every matrix initialization. Therefore the actual optimizer output has zero
reconstruction loss, zero full objective, and global optimality in this
matrix family. For every reference readout $`q`$ and scalar task target,
including noisy targets, Lean proves

```math
R(q\circ E_{W_{\mathrm{learned}}})=R(q\circ\Phi).
```

The task or its labels need not be specified during reconstruction training.
This closes the exact-recovery premise of the earlier anchor result for this
particular optimizer and reference. It does **not** claim that every target
is predictable with zero risk, that the reference is semantically sufficient,
or that its information has been compressed. The entire two-coordinate
reference is retained, and the special balanced data cover the complete
finite raw-input support. See `scaled_step_learns_identity`,
`learned_global_minimum`, and `learned_preserves_every_reference_readout` in
[`ReferenceLearning.lean`](HJepa/ReferenceLearning.lean).

All four optimizer modules use exact real arithmetic. They are mathematical
update algorithms, not verified floating-point implementations or experiment
results. The predictive encoder, its dynamics model, and nonlinear hierarchy
are not learned by these results; only the specified protected encoders and,
in the guarded construction, their head reparameterization are addressed.

### Connection to JEPA research

Useful features being representable does not imply that an objective selects
them. The existing finite control counterexample retains a nonconstant
background, predicts it perfectly, and discards a hidden mode needed for the
correct action. This refutes a guarantee based on perfect latent prediction
alone, not every objective with additional task-alignment constraints.

The H-JEPA paper's Appendix L provides related population theory: under its
atomless-background assumptions, a background-only encoder can simultaneously
achieve zero prediction loss and a prescribed marginal latent law. Its
predictability bound does not choose task-relevant features among zero-loss
solutions. Those are published results, not newly proved measure-theoretic
claims in this repository.
[H-JEPA Appendix L](https://arxiv.org/html/2610.06805v1).

More broadly, depth-separation results establish representational efficiency
for specified architecture classes. Cohen, Sharir, and Shashua prove such a
separation for arithmetic-circuit networks using hierarchical tensor
factorizations. This supports studying composition, but does not establish
that hierarchical JEPA training beats flat JEPA training.
[Cohen et al. 2016](https://proceedings.mlr.press/v49/cohen16.html).

## Remaining theory before an architecture claim

1. **Specify the model and comparator.** Define the audio data process, target,
   horizons, encoders, loss, optimizer, and a baseline with the same context.
   Include feature construction in any resource comparison.
2. **Instantiate the protected objective for audio.** The finite protected
   model closes its objective-alignment obligation; the general bounds still
   need a justified task family or frozen reference, architecture-specific
   joint feasibility, and explicit preservation/approximation budgets. A
   teacher's semantic quality cannot be inferred from reconstruction alone.
3. **Derive the learning budgets.** Establish an appropriate concentration
   result, approximation bound, and optimization guarantee for nonlinear
   audio training. The new linear full-batch and IID SGD results close these
   particular toy-model optimization obligations, not temporal audio
   dependence, learned predictive branches, or numerical stability.
4. **Establish a real separation.** Prove a lower bound for the specified flat
   learner, or an exact comparison of the two training procedures. A better
   hierarchical upper bound alone is not a separation theorem.
5. **Identify empirical assumptions.** Noise correlations, alignment, and
   retention of transient information need evidence on the intended data.
   Experiments test applicability; they cannot turn an unproved universal
   claim into a theorem.

The broader analytical note's Gaussian OLS expectation and spectral arguments
also remain outside the Lean formalization. Proving their risk formulas would
strengthen a finite-sample mechanism, but would still require connecting their
assumptions to a particular learned H-WavJEPA model.

## Verification

Run from the repository root:

```sh
lake build
lake env lean Audit.lean
```

The audit rejects unfinished proofs and transitive dependencies on additional
axioms. It permits Lean's standard `propext`, `Classical.choice`, and
`Quot.sound`. Passing the audit verifies the formal implications, not that
their statistical assumptions hold for a trained audio model.
