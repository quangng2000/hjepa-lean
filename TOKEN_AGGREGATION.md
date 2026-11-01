# Subsampling versus averaging lower-level latent tokens

The comparison acts on **latent states before the next encoder**. It does not
average a predictor's outputs or equate aggregation with temporal prediction.
The earlier [temporal objective proof](TEMPORAL_PREDICTION.md) answers a different
question: which features a prediction objective favors.

## Architecture reference

H-JEPA uses window one for upper state encoders. Its single-scene experiments use
stride two; DROID uses stride three. Planning separately lifts predicted lower
states into the upper space. See [§2.1, §2.2 and Appendix C](https://arxiv.org/html/2610.06805v1).
Thus “all experiments use stride two” is incorrect.

Our stride-two specialization compares

```math
z^{(2)}_t=E_2(z^{(1)}_{2t})
\quad\text{and}\quad
z^{(2)}_t=E_2((z^{(1)}_{2t}+z^{(1)}_{2t+1})/2).
```

The finite vector operators map `Fin (2*n) → V` to `Fin n → W`. With `n=4`
and `V = Fin d → ℝ`, the input has eight vector tokens and the output has four.
Selection takes zero-based indices `0,2,4,6` and commutes with pointwise encoding.
Averaging is componentwise before the next encoder and generally does not
commute with a nonlinear encoder; a checked square-function example separates
the two averaging orders.

## Exact same-task linear comparison

Fix one finite population, a downstream target $`Y`$, and an already available
latent pair $`X_0,X_1`$. Selection keeps $`X_0`$; averaging returns
$`M=(X_0+X_1)/2`$. Each has its own independently fitted scalar linear probe.

For $`\mathbb E[X^2]>0`$, Lean derives the actual minimum risk and an attaining head:

```math
R_X=\min_k\mathbb E[(Y-kX)^2]
=\mathbb E[Y^2]-\frac{\mathbb E[YX]^2}{\mathbb E[X^2]},
\qquad k^*=\frac{\mathbb E[YX]}{\mathbb E[X^2]}.
```

Write $`V_i=\mathbb E[X_i^2]`$, $`C_i=\mathbb E[YX_i]`$,
$`C_{01}=\mathbb E[X_0X_1]`$, and $`D=V_0+V_1+2C_{01}`$.
For $`V_0>0,D>0`$, the exact criterion is

```math
R_M<R_{X_0}\iff V_0(C_0+C_1)^2>C_0^2D.
```

The reverse inequality selects subsampling; equality gives a tie. These are raw
second moments, not covariances unless features are centered. The probe has no
intercept. A nonzero scalar linear next encoder preserves this optimum because
the fitted head absorbs its scale. An arbitrary nonlinear next encoder requires
evaluating its actual output; the formula does not promise its training succeeds.

## Aligned signal and noise

The conditional `AlignedMoments` result assumes two aligned tokens with signal
second moment $`v>0`$, equal error second moments $`\nu\ge0`$, error cross moment
$`c`$, and zero signal/error cross moments. The record states the resulting
observable moment identities explicitly; it does not infer them from audio.
Assuming $`\nu+c\ge0`$, it proves

```math
R_{\mathrm{select}}=\frac{v\nu}{v+\nu},\qquad
R_{\mathrm{mean}}=\frac{v(\nu+c)}{2v+\nu+c},\qquad
R_{\mathrm{mean}}<R_{\mathrm{select}}\iff c<\nu.
```

Centered errors that are mutually independent and independent of the signal
give one sufficient scenario with $`c=0`$.
Perfectly shared errors, $`c=\nu`$, give no gain. Misaligned or rapidly changing
signal need not satisfy this model; the general moment criterion still applies.

## Actual finite examples, including nonlinear readouts

The following use independent fair signs $`S,N`$ and the same target $`S`$.
The loss bounds hold for **every deterministic scalar readout**, and are attained.

- Stable signal with opposing nuisance: $`(X_0,X_1)=(S+N,S-N)`$. Averaging
  recovers $`S`$ exactly, with risk zero. Selection has minimum risk $`1/2`$;
  its ambiguous zero observation cannot reveal the signal sign.
- Alternating signal: $`(X_0,X_1)=(S,-S)`$. Selection has risk zero. Averaging
  returns a constant, giving minimum risk one. A subsequent deterministic
  encoder and nonlinear readout cannot recover the discarded distinction.
- Skipped content: $`(X_0,X_1)=(N,S)`$. Selection's minimum risk is one;
  averaging's is $`1/2`$. Selection can miss information at the skipped position.

Thus neither operator universally dominates, even allowing nonlinear readouts.
The constant-average example is handled directly; it is outside the positive
energy assumptions of the linear-probe criterion.

## Timing, capacity, and learning boundaries

The comparison assumes both tokens are already available. A forward pair ending
at `2t+1` is not available at time `2t`; using it there would introduce lookahead.
For a causal alternative at the same anchor, compare `z[2t]` with
`(z[2t-1]+z[2t])/2`. The [spoofing extension](SPOOFING.md) models that arrangement
with an explicit causal convolution and left padding.

Lower-level latents are fixed in these theorems. They may already carry context
in a real encoder. End-to-end training could change which information each token
contains. The proofs do not establish a trained neural winner, equal compute,
or an audio benchmark improvement. They identify information-preservation and
statistical conditions to check. Successful witness heads are attainable choices,
not guaranteed outputs of an optimizer.

Entry point: [`HJepa/TokenAggregation.lean`](HJepa/TokenAggregation.lean).
`Operators` encodes the token maps; `LinearRisk`, `PairMoments`, `Comparison`,
`Aligned`, and `Encoder` give the general conditions; `Examples/` derives the
unrestricted finite-population counterexamples. No novelty claim is made.
