import Mathlib.Data.Real.Basic
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring

/-!
# Algebraic comparison of Gaussian-design OLS risk formulas

This module formalizes the algebraic implications of the classical expected test
risk formulas for ordinary least squares under an independent Gaussian design.
It does **not** prove those probabilistic formulas, Gaussian independence, or
the inverse-Wishart expectation. The definitions below are explicit rational
expressions, not a formal probabilistic model of training.

`n` is sample size, `d` the retained dimension, `q` the discarded dimension,
`noise` the noise variance (not its standard deviation), and `B` the signal
variance discarded by abstraction. Real-valued parameters make the algebra
more general; an OLS application uses natural-valued sample sizes/dimensions
and nonnegative variances. The sign of the risk comparison itself does not
need nonnegative variances.
-/

namespace HJepa.OLS

/-- Classical full-model OLS test-risk expression, with `d + q` features. -/
noncomputable def fullRisk (n d q noise : ℝ) : ℝ :=
  noise * (n - 1) / (n - d - q - 1)

/-- Classical reduced-model OLS test-risk expression, with `d` features. -/
noncomputable def reducedRisk (n d noise B : ℝ) : ℝ :=
  (noise + B) * (n - 1) / (n - d - 1)

/-- Maximum discarded signal variance for a strict improvement. -/
noncomputable def threshold (n d q noise : ℝ) : ℝ :=
  noise * q / (n - d - q - 1)

private theorem denominator_pos {n d q : ℝ} (hq : 0 ≤ q)
    (hn : d + q + 1 < n) :
    0 < n - d - q - 1 ∧ 0 < n - d - 1 := by
  constructor <;> linarith

/-- Exact reduced-minus-full risk, obtained purely by rational algebra. -/
theorem risk_difference (n d q noise B : ℝ) (hq : 0 ≤ q)
    (hn : d + q + 1 < n) :
    reducedRisk n d noise B - fullRisk n d q noise =
      (n - 1) * (B * (n - d - q - 1) - noise * q) /
        ((n - d - 1) * (n - d - q - 1)) := by
  obtain ⟨hf, hr⟩ := denominator_pos hq hn
  unfold reducedRisk fullRisk
  field_simp [ne_of_gt hf, ne_of_gt hr]
  ring

/-- The risk gap is a positive scaling of discarded variance minus threshold. -/
theorem risk_difference_factor (n d q noise B : ℝ) (hq : 0 ≤ q)
    (hn : d + q + 1 < n) :
    reducedRisk n d noise B - fullRisk n d q noise =
      ((n - 1) / (n - d - 1)) * (B - threshold n d q noise) := by
  obtain ⟨hf, hr⟩ := denominator_pos hq hn
  unfold reducedRisk fullRisk threshold
  field_simp [ne_of_gt hf, ne_of_gt hr]
  ring

private theorem factor_pos {n d q : ℝ} (hd : 0 ≤ d) (hq : 0 ≤ q)
    (hn : d + q + 1 < n) : 0 < (n - 1) / (n - d - 1) := by
  apply div_pos <;> linarith

/-- Abstraction strictly improves these OLS risk formulas exactly below threshold. -/
theorem reduced_better_iff (n d q noise B : ℝ) (hd : 0 ≤ d)
    (hq : 0 ≤ q) (hn : d + q + 1 < n) :
    reducedRisk n d noise B < fullRisk n d q noise ↔
      B < threshold n d q noise := by
  have hp := factor_pos hd hq hn
  have he := risk_difference_factor n d q noise B hq hn
  constructor <;> intro h <;> nlinarith

/-- Equality occurs exactly at the threshold. -/
theorem risks_equal_iff (n d q noise B : ℝ) (hd : 0 ≤ d)
    (hq : 0 ≤ q) (hn : d + q + 1 < n) :
    reducedRisk n d noise B = fullRisk n d q noise ↔
      B = threshold n d q noise := by
  have hp := factor_pos hd hq hn
  have he := risk_difference_factor n d q noise B hq hn
  constructor <;> intro h <;> nlinarith

/-- Discarding more useful signal than the threshold strictly worsens risk. -/
theorem reduced_worse_iff (n d q noise B : ℝ) (hd : 0 ≤ d)
    (hq : 0 ≤ q) (hn : d + q + 1 < n) :
    fullRisk n d q noise < reducedRisk n d noise B ↔
      threshold n d q noise < B := by
  have hp := factor_pos hd hq hn
  have he := risk_difference_factor n d q noise B hq hn
  constructor <;> intro h <;> nlinarith

/-- The example in the note: 30 samples, 2 retained and 8 discarded features. -/
theorem example_threshold : threshold 30 2 8 1 = 8 / 19 := by
  norm_num [threshold]

/-- Losing variance 0.1 is worthwhile in the example. -/
theorem example_small_loss_helps :
    reducedRisk 30 2 1 (1 / 10) < fullRisk 30 2 8 1 := by
  norm_num [reducedRisk, fullRisk]

/-- Losing variance 1 is harmful in the same example. -/
theorem example_large_loss_hurts :
    fullRisk 30 2 8 1 < reducedRisk 30 2 1 1 := by
  norm_num [reducedRisk, fullRisk]

/-- At the exact threshold the two expressions coincide. -/
theorem example_boundary_equal :
    reducedRisk 30 2 1 (8 / 19) = fullRisk 30 2 8 1 := by
  norm_num [reducedRisk, fullRisk]

end HJepa.OLS
