# Masked reconstruction versus hidden-layer prediction

This extension proves when either objective yields a better **linear downstream
readout** in an explicit finite model. Neither objective universally wins, even
when the hidden target is an invertible rescaling of the raw target.

Both methods receive the same masked observation and optimize the same encoder
and decoder class. The comparison is on one downstream task, not on the numerical
values of two different pretraining losses. Masking and latent prediction are
compatible: both methods here use masking; their prediction targets differ.

## The model

Let `S` and `N` be independent uniform signs in `{-1, 1}`. The downstream target
is `S`; `N` is a nuisance for that task. Each full example is

```math
X=(S,N,sS,nN).
```

A fixed mask hides the last two coordinates and exposes only `(S,N)`. The raw
reconstruction target is `(sS,nN)`. A frozen diagonal hidden layer changes that
target to `(csS,dnN)`. It acts on the target branch, not on the visible context.

Both learners use the normalized scalar encoder and two linear decoder heads

```math
z=aS+bN,\qquad a^2+b^2=1,\qquad \widehat T=(pz,qz).
```

The normalization gives the actual encoder zero mean and unit second moment.
The four real parameters `(a,b,p,q)` are jointly optimized subject to it. No
separate downstream labels or downstream task-loss term enter either pretraining
objective. Evaluation fits an optimal linear probe `kz` to the common task `S`.

The mask genuinely removes coordinates, but their values are deterministic
functions of the visible context. This isolates allocation of a restricted
linear representation; it does not model unpredictable missing audio.

## Exact loss and every global optimum

For target scales `(u,v)`, Lean derives the population expectation over all four
examples, rather than defining the loss to equal a desired risk formula:

```math
J_{u,v}=\mathbb E[(uS-pz)^2+(vN-qz)^2]
       =u^2b^2+v^2a^2+(p-ua)^2+(q-vb)^2.
```

Every joint global optimum must fit the heads `p=ua` and `q=vb`. The optimal
linear downstream risk is also derived from its population expectation:

```math
\mathbb E[(S-kz)^2]=b^2+(k-a)^2,\qquad R=\min_k\mathbb E[(S-kz)^2]=b^2.
```

- If `u² > v²`, every global optimum has `R=0` and objective value `v²`.
- If `u² < v²`, every global optimum has `R=1` and objective value `u²`.
- If `u² = v²`, every normalized encoder is globally optimal with fitted heads.
  The objective does not select task quality; explicit optima have risks 0 and 1.

The two strict regimes include converses: the stated task risk together with
fitted heads is sufficient for global optimality. Axis encoders explicitly attain
the minima, so the comparisons do not rely on an unattained infimum.

## When each method is better

Raw reconstruction uses `(u,v)=(s,n)`. Hidden-layer prediction uses
`(u,v)=(cs,dn)`. When neither objective has equal coordinate energies:

- **Hidden prediction is strictly better if and only if**
  `s² < n²` and `(cs)² > (dn)²`.
- **Raw reconstruction is strictly better if and only if**
  `s² > n²` and `(cs)² < (dn)²`.
- Matching strict energy orderings give equal downstream risk: both 0 or both 1.

These results quantify over every pair of joint global optima, not selected
favorable solutions. An identity hidden layer also gives exactly the same
pretraining objective as raw reconstruction for every parameter choice.

Two fully checked examples establish that neither method dominates:

- Raw target `(S,2N)`, hidden layer scales `(3,1)`: raw optima have task risk 1;
  hidden optima for `(3S,2N)` have task risk 0.
- Raw target `(2S,N)`, hidden layer scales `(1,3)`: raw optima have task risk 0;
  hidden optima for `(2S,3N)` have task risk 1.

Both target maps are invertible. Thus hidden-target information loss is not
needed for this reversal: changing the squared-error metric changes what a
restricted learner prioritizes. If both target coordinates instead have equal
energy, the equal-energy theorem applies and leaves task quality unidentified.

## Approximate optimization

The bounds include errors in both decoder heads. With positive energy gap
`G=u²-v²`, an objective within `η` of the signal-favoring minimum satisfies

```math
J_{u,v}\le v^2+\eta\quad\Longrightarrow\quad R\le\eta/G.
```

For a nuisance-favoring gap `G=v²-u²`, the corresponding bound is
`R ≥ 1-η/G`. Consequently, in either opposite-ordering regime, the favorable
method still wins whenever its normalized tolerance plus the other method's
normalized tolerance is less than 1:

```math
\frac{\eta_{\rm favorable}}{G_{\rm favorable}}
+\frac{\eta_{\rm unfavorable}}{G_{\rm unfavorable}}<1.
```

For the first example above, allowing each objective an excess loss of 1 still
guarantees a strict hidden-method advantage, because `1/5 + 1/3 < 1`. This
nonzero-tolerance instantiation is checked in Lean too.

## Scope and proof map

These are exact-real, finite-population, fixed-mask results with linear encoders,
linear heads, frozen target layers, and an optimal **linear** downstream probe.
A generic scalar mixture can distinguish all four inputs, so `b²` must not be
called unrestricted Bayes risk or information loss. This is a restricted-function
class comparison, not a bit-rate or parameter-efficiency theorem.

Global optimality or a population optimization tolerance is a hypothesis. No
convergence of SGD/Adam, finite-sample generalization, random-mask distribution,
learned/EMA target, nonlinear network, temporal audio process, or measured speech
quality is established. A target layer's alignment with the task must be
justified separately. The result is an elementary squared-error allocation
mechanism; no novelty claim is made.

- [Model](HJepa/MaskedPrediction/Model.lean): mask, population and actual losses.
- [Risk](HJepa/MaskedPrediction/Risk.lean): expectations, head fitting and probe risk.
- [Predictor](HJepa/MaskedPrediction/Predictor.lean): shared admissible class and evaluation.
- [Optima](HJepa/MaskedPrediction/Optima.lean): attained minima and task-risk regimes.
- [Characterization](HJepa/MaskedPrediction/Characterization.lean): all joint optima, including ties.
- [Comparison](HJepa/MaskedPrediction/Comparison.lean): both strict `iff` comparisons.
- [Tolerance](HJepa/MaskedPrediction/Tolerance.lean): approximate population comparisons.
- [Examples](HJepa/MaskedPrediction/Examples.lean): both winners, tie ambiguity and tolerance witness.

`HJepa.lean` imports the extension. Run `lake build` and
`lake env lean Audit.lean` to check all proofs and their transitive axiom dependencies.
