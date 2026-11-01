# Causal aggregation and preservation of detection cues

This extension asks what a hierarchical audio representation **can preserve or
irreversibly discard before a spoof detector sees it**. It does not prove that
Conv1D, JEPA, or a specific audio model achieves better spoof-detection scores.

The population examples are constructed binary cue tasks. A balanced Boolean
label is encoded by a sign; it stands for a class distinction, not a claim that
real authentic and synthetic speech follow these two-point distributions.

## Causal kernel matching the proposed implementation

For scalar latent tokens, the three-tap stride-two operator is

```math
u_t=k_0z_{2t}+k_1z_{2t-1}+k_2z_{2t-2}.
```

Lean proves no-future-dependence and maps the stored oldest-to-current Conv1d
weight order to these taps. Explicit zero-left-padding gives the first window
`[0,0,z[0]]`. These are operations on lower state tokens before the next encoder.

The class includes current-token selection, previous/current averaging, and
previous/current differencing. These are feasible kernels; learning is not
assumed to find them. `Applications.lean` places cue pairs in an actual causal
window anchored at time two and connects the classification theorems to the
convolution outputs. The detection difference uses half the full difference
kernel, so its amplitude is explicitly accounted for.

## What is proved about detection

All errors here are balanced-population **0–1 classification error**. They are
not EER, AUC, calibrated scores, or min-tDCF. Both methods see the same available
previous/current pair and are evaluated against the same label.

Let $`B\in\{-1,1\}`$ encode the balanced label and let amplitude $`a>0`$.

- **Alternating cue:** previous/current tokens are $`(-aB,aB)`$. Averaging
  erases the cue, giving exactly **50% error for every deterministic classifier**,
  even after an arbitrary deterministic next encoder. Half-differencing returns
  $`aB`$ and an explicit sign classifier achieves **0% error**. Current-token
  selection also succeeds in this example.
- **Cue only in the skipped token:** tokens are $`(aB,0)`$. Current-token
  selection gives exactly **50% error**; a previous-token tap or averaging
  permits **0% error**.
- **Shared cue:** tokens are $`(aB,aB)`$. Differencing gives exactly **50% error**;
  averaging permits **0% error**. A fixed high-pass filter is not a universal fix.

The chance results quantify over every deterministic classifier of the retained
output, not a deliberately weak head. The zero-error results supply explicit
heads. Neither method's training loss is used as a surrogate for detection quality.

## Cue strength needs a margin

Exact real arithmetic can distinguish arbitrarily small positive $`a`$. That is
not a noise or finite-precision guarantee. The robustness theorem adds a
sufficient condition: if the **post-filter** perturbation is $`e(B)`$ and
$`|e(B)|<a`$ for both labels, the sign classifier still has zero error.
It does not derive the post-filter error bound from raw audio corruption.

## What a learned convolution can and cannot guarantee

A filter bank retaining both sum and difference reconstructs the available
previous/current pair exactly:

```math
u=z_{2t}+z_{2t-1},\qquad v=z_{2t}-z_{2t-1},\qquad
z_{2t}=(u+v)/2,\qquad z_{2t-1}=(u-v)/2.
```

This uses **two scalar output channels**, not one. Applied independently to
arbitrary `d`-channel pairs, that construction needs `2d` output channels.
It is not a proof that an unrestricted `768 → 768` strided convolution is
lossless across time.

Conversely, every fixed scalar linear projection of an unrestricted real-valued
token pair has a nonzero kernel. Lean constructs distinct colliding inputs and
proves that any deterministic encoder afterward preserves that collision. The
claim is “some distinction is lost,” not “every spoof cue is lost.” A single
projection can still separate a particular finite dataset.

Finite streaming boundaries matter: with left padding two, kernel three, stride
two, an eight-token clip emits anchors `0,2,4,6`. Token seven needs a later or
flush window. Pair reconstruction above is not a promise to retain every token
of a truncated clip without an end-boundary convention.

## Why retaining the fine representation helps

For fixed fine features $`z_1`$ and any coarse features $`z_2`$, every fine-only
classifier can be represented by a fused classifier that ignores $`z_2`$.
Lean therefore proves that a global optimum over unrestricted fused classifiers
is no worse than any specified fine-only classifier. An approximate optimum
carries the same comparison with an explicit error allowance.

This is a capacity guarantee for a class containing the ignore-coarse option.
It does not guarantee that a trained fusion network finds that option, that
regularization leaves it admissible, or that end-to-end updates preserve $`z_1`$.
The result applies recursively when retaining three levels.

## Consequences for an audio experiment

The proofs support a causal strided-convolution baseline **with a retained
fine-level path and downstream evaluation of all levels**. They establish no
universal winner. Fixed averaging can erase changing cues, fixed selection can
miss skipped cues, and fixed differencing can erase shared cues.

The [temporal objective results](TEMPORAL_PREDICTION.md) give a separate warning:
predictable information need not be the task-relevant information. Giving a JEPA
encoder a convolution does not by itself make its objective preserve authenticity
cues. Evaluate detection on held-out attacks, speakers, and recording conditions;
formal operator expressiveness is not a measured generalization result.

Entry point: [`HJepa/Spoofing.lean`](HJepa/Spoofing.lean).
`Filters/` covers kernels, causality, padding, recovery and collisions;
`Detection/` covers classification, opposite winners and cue margins;
`Applications` connects them; `Retention` states the fine-feature guarantees.
Run `lake build` and `lake env lean Audit.lean`. No custom axioms, unfinished
proofs, neural training guarantee, or novelty claim is introduced.
