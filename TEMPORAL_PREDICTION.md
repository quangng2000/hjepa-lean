# Masked reconstruction versus autoregressive latent prediction

This extension proves when **masked raw reconstruction** or **shared-encoder
autoregressive latent prediction** gives a better representation for the same
linear downstream task. Both directions have attained examples; neither method
universally wins in this model family.

The earlier [masked-target comparison](MASKED_PREDICTION.md) used a frozen
hidden-target rescaling. Here the target is a **learned future latent**, produced
by the same encoder as the current latent. A finite stochastic process supplies
the future data, and multi-step prediction actually feeds predictions back in.

## Relation to H-JEPA

[H-JEPA §2.1–2.2](https://arxiv.org/html/2610.06805v1) trains next-latent
prediction conditioned on latent context and actions, and uses autoregressive
rollouts for planning. This formalization isolates a scalar, one-context-step
version of that mechanism. Its exact unit-variance constraint is an explicit
anti-collapse assumption, **not an implementation or verification of SIGReg**.
It does not verify the hierarchy, transformers, neural optimization, or planning.

## Population, information, and learner class

Current signal $`S`$ and nuisance $`N`$ are independent fair signs. Independent
sign innovations have means $`\rho_S,\rho_N\in[0,1]`$. A known, fixed action sign
$`A\in\{-1,1\}`$ gives

```math
S'=A E_S S,\qquad N'=A E_N N.
```

Lean constructs the probabilities and expands the actual 16-atom expectation.
For multiple steps it recursively applies this finite transition law with fresh
independent innovations. Actions are exogenous sign flips, not adaptive policies.

Both learners see the same current context $`(S,N)`$ and use the same encoder class:

```math
z=aS+bN,\qquad a^2+b^2=1.
```

Normalization gives zero mean and unit variance to both current and future
latents. Without the constraint, the zero encoder attains zero temporal loss.
The raw reconstruction mask hides the scaled duplicate target $`(sS,nN)`$ while
leaving $`(S,N)`$ visible, as specified in the earlier model. It is a fixed mask;
this construction does not represent all masked-image or masked-audio datasets.

Raw reconstruction uses two freely fitted linear heads. Temporal prediction
uses one freely fitted coefficient $`k`$ and predicts $`kAz`$ against
$`z'=aS'+bN'`$. Global optima jointly vary the normalized encoder and its heads.
No convergence or finite-sample generalization assumption is silently proved.

## What “better” measures

Both methods are evaluated by the **best linear probe of the current signal**:

```math
R=\min_\ell\mathbb E[(S-\ell z)^2]=b^2.
```

The minimizing probe is $`\ell=a`$. This is linear accessibility, not unrestricted
information loss: some mixed scalar encoders distinguish all four input atoms,
so a nonlinear probe could recover the signal exactly.

There is also a common **future-signal** evaluation:

```math
Q=\min_\ell\mathbb E[(S'-\ell Az)^2]
 =1-\rho_S^2+\rho_S^2R.
```

The minimizing probe is $`\ell=\rho_Sa`$. Every strict current-task comparison
transfers to this same future target if $`\rho_S>0`$. If $`\rho_S=0`$, every
future-task score is one; a current-task advantage is then not a forecast advantage.

## Exact objective selection and winning conditions

After fitting reconstruction heads, raw loss is
$`s^2b^2+n^2a^2`$. It retains the coordinate with **greater raw target energy**.
The temporal objective has the exact completed-square form

```math
J=(k-c)^2+1-c^2,\qquad c=\rho_Sa^2+\rho_Nb^2.
```

Its globally optimal encoder retains the coordinate with **greater persistence**.
For strict energy and persistence orderings, Lean proves necessary and sufficient
conditions about **every pair of global optima**:

- Autoregressive prediction wins exactly when $`n^2>s^2`$ and
  $`\rho_S^2>\rho_N^2`$: reconstruction retains nuisance ($`R=1`$), AR retains
  signal ($`R=0`$).
- Reconstruction wins exactly when $`s^2>n^2`$ and
  $`\rho_N^2>\rho_S^2`$: the two risks reverse.
- Matching strict orderings tie: both select signal, or both select nuisance.
- Equal energies or equal persistences can leave task quality unidentified.
  Equal persistence admits both $`R=0`$ and $`R=1`$ at the same optimal temporal loss.

Concrete witnesses use $`(s,n)=(1,2)`$ with persistences $`(3/4,1/4)`$ for an AR
win, and $`(s,n)=(2,1)`$ with persistences $`(1/4,3/4)`$ for a reconstruction win.
In both cases signal persistence is positive, so the future-task comparison agrees.

The methods' training losses have different targets and scales. Their numeric
loss values are **not** compared as evidence of downstream superiority.

## Separate target choice from forecast horizon

The initial comparison changes both the target type and its time index. A second
control holds the future horizon fixed: predict masked raw $`(sS',nN')`$ with
heads $`(pAz,qAz)`$ from the same context, on the same trajectories. Lean derives

```math
L_{\mathrm{futureRaw}}=
s^2(1-\rho_S^2)+n^2(1-\rho_N^2)
+L_{\mathrm{masked}}(s\rho_S,n\rho_N).
```

Future raw reconstruction therefore selects by **predictable raw energy**,
$`s^2\rho_S^2`$ versus $`n^2\rho_N^2`$. Learned-latent prediction still selects
by persistence alone. Exact strict-regime comparisons and ties follow with these
effective scales, isolating target choice at the same forecast horizon.

Both winners remain possible: scales $`(1,4)`$ with persistences $`(3/4,1/4)`$
favor AR; scales $`(4,1)`$ with persistences $`(1/4,3/4)`$ favor future raw
reconstruction. All scales and persistences in these witnesses are nonzero.

## Approximate optimization

The exact temporal decomposition, with $`R=b^2`$, is

```math
J=1-\rho_S^2+(\rho_S^2-\rho_N^2)R
+(\rho_S-\rho_N)^2R(1-R)+(k-c)^2.
```

The last two terms are nonnegative. For a positive signal-persistence gap,
objective within $`\eta`$ of its global minimum implies
$`R\le\eta/(\rho_S^2-\rho_N^2)`$; the reverse gap gives the corresponding lower
bound near one. Reconstruction has analogous bounds with its energy gap.
The two methods' tolerances are normalized by their own gaps. When their sum is
less than one in an opposing-order regime, the strict task comparison survives.

The checked positive-tolerance example allows raw loss at most $`2`$ (minimum
$`1`$), and temporal loss at most $`11/16`$ (minimum $`7/16`$). It yields
AR risk at most $`1/2`$ and raw risk at least $`2/3`$.
These are population suboptimality conditions, not a training-speed guarantee.

## Genuine autoregressive rollout

The recursive predictor consumes its own previous prediction for any fixed
action list of length $`h`$:

```math
\hat z_h=k^h\!\left(\prod_{i=0}^{h-1}A_i\right)z_0.
```

The proof derives lag covariances from the iterated transition probabilities,
then computes the exact expected rollout loss:

```math
a^2(1+k^{2h}-2k^h\rho_S^h)
+b^2(1+k^{2h}-2k^h\rho_N^h)
=(k^h-c_h)^2+1-c_h^2,
\qquad c_h=a^2\rho_S^h+b^2\rho_N^h.
```

The equality on the right assumes normalization. A pure signal encoder with
$`k=\rho_S`$ has error $`1-\rho_S^{2h}`$; the nuisance analogue also holds.
The zero-step loss is zero, and the one-step rollout equals the training loss.

One-step optimality for a **fixed mixed encoder** does not guarantee optimal
recurrent multi-step prediction. At two steps,

```math
c_2-c_1^2=a^2b^2(\rho_S-\rho_N)^2.
```

Lean proves the corresponding squared excess risk and an explicit example:
$`a=3/5,b=4/5,\rho_S=1,\rho_N=0`$. The best one-step coefficient is $`9/25`$,
but $`3/5`$ gives strictly better, globally optimal two-step risk for that fixed
encoder. This is not a contradiction of joint one-step optima selecting a pure
coordinate under strict persistence ordering.

## Proof map and boundaries

Entry point: [`HJepa/TemporalPrediction.lean`](HJepa/TemporalPrediction.lean).

- `Dynamics`, `Model`, `Risk`, `Moments`, `Probe`: actual finite law, shared
  learned target, normalization and common-task expectations.
- `Comparison/Optima`, `Characterization`, `Methods`: all joint optima and exact
  winning regimes. `FutureTask`, `Tolerance`, `Examples`, `Boundary`: future
  scores, approximate guarantees, attained winners and ambiguity.
- `Controls/FutureRaw*`, `Comparison`, `Examples`: matched-horizon raw control.
- `Rollout/Process`, `Quadratic`, `Moments`, `Risk`, `Correlation`, `Corollaries`,
  `Optima`, `Examples`: genuine iteration and its derived finite-population risk.

The claims require this finite process, nonnegative correlations, linear heads,
normalized scalar encoders, exogenous sign actions, and specified population
optimization quality. They do not certify H-JEPA's neural training, SIGReg,
multiple hierarchy levels, general actions, learned attention, sample efficiency,
audio quality, or universal superiority. No novelty claim is made.

Check with `lake build` and `lake env lean Audit.lean`. The audit rejects custom
axioms and unfinished proofs, while allowing the repository's three standard
Lean axioms. It does not validate empirical assumptions.
