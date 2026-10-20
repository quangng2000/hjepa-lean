import HJepa.LearningExamples

/-!
# An explicit optimizer for the fixed high-feature readout

Two balanced raw parity examples determine the population loss of every high
affine readout exactly. We derive the residual gradient from those examples,
certify it by an exact first-order loss expansion with a quadratic remainder,
and analyze the resulting gradient-descent update from arbitrary coefficients.

A step of size one half reaches the unique zero-risk readout in one step.
Steps strictly between zero and one contract every nonzero error; size one
preserves error, and larger positive steps increase it. This discharges the
optimizer assumption for this special fixed-feature quadratic problem. It is
not convergence of JEPA encoder training, a random-sampling guarantee, or an
advantage over flat nonlinear predictors.
-/

namespace HJepa.Optimization

open HJepa.Abstraction HJepa.Expressivity HJepa.LearningExamples

/-- Prediction minus actual target, evaluated on a raw example. -/
noncomputable def residual (c : Coefficients) (s : FourSigns) : ℝ :=
  highAffine c.1 c.2 (high s) - parityTarget s

/-- The averaged squared-loss residual gradient: bias derivative is twice
the residual; slope derivative is twice residual times the encoded feature. -/
noncomputable def residualGradient (c : Coefficients) : Coefficients :=
  ((2 * residual c positiveExample + 2 * residual c negativeExample) / 2,
   (2 * residual c positiveExample * signedBit (high positiveExample) +
    2 * residual c negativeExample * signedBit (high negativeExample)) / 2)

theorem residual_gradient_formula (c : Coefficients) :
    residualGradient c = (2 * c.1, 2 * (c.2 - 1)) := by
  ext <;>
    norm_num [residualGradient, residual, positiveExample, negativeExample,
      highAffine, parityTarget, high, toHigh, middle, toMiddle, low,
      productBit, signedBit] <;> ring

/-- The coefficient of each first-order increment is exactly the residual
gradient. The remainder is the squared Euclidean length of the increment;
this identifies the true quadratic-loss gradient without assuming it. -/
theorem exact_loss_increment (c δ : Coefficients) :
    balancedEmpiricalRisk (c.1 + δ.1, c.2 + δ.2) =
      balancedEmpiricalRisk c +
        (residualGradient c).1 * δ.1 + (residualGradient c).2 * δ.2 +
        δ.1 ^ 2 + δ.2 ^ 2 := by
  rw [balanced_empirical_formula, balanced_empirical_formula,
    residual_gradient_formula]
  dsimp
  ring

/-- One actual gradient-descent step on the two-example empirical loss. -/
noncomputable def gradientStep (α : ℝ) (c : Coefficients) : Coefficients :=
  (c.1 - α * (residualGradient c).1, c.2 - α * (residualGradient c).2)

theorem gradient_step_formula (α : ℝ) (c : Coefficients) :
    gradientStep α c = ((1 - 2 * α) * c.1,
      1 + (1 - 2 * α) * (c.2 - 1)) := by
  rw [gradientStep, residual_gradient_formula]
  ext <;> dsimp <;> ring

/-- Empirical contraction is also exact population contraction, since the
chosen balanced examples integrate every high affine squared loss exactly. -/
theorem population_risk_contraction (α : ℝ) (c : Coefficients) :
    highPopulationRisk (gradientStep α c) =
      (1 - 2 * α) ^ 2 * highPopulationRisk c := by
  rw [gradient_step_formula]
  simp only [highPopulationRisk, high_affine_risk]
  ring

/-- Repeated updates use a fixed step size and arbitrary initial coefficients. -/
noncomputable def gradientIterate (α : ℝ) (initial : Coefficients) : ℕ → Coefficients
  | 0 => initial
  | n + 1 => gradientStep α (gradientIterate α initial n)

theorem iterate_population_risk (α : ℝ) (initial : Coefficients) (n : ℕ) :
    highPopulationRisk (gradientIterate α initial n) =
      ((1 - 2 * α) ^ 2) ^ n * highPopulationRisk initial := by
  induction n with
  | zero => simp [gradientIterate]
  | succ n ih =>
      rw [gradientIterate, population_risk_contraction, ih, pow_succ]
      ring

/-- The half-step optimizer reaches the unique optimum from any initialization. -/
theorem half_step_reaches_identity (c : Coefficients) :
    gradientStep (1 / 2) c = (0, 1) := by
  rw [gradient_step_formula]
  norm_num

theorem half_step_population_risk_zero (c : Coefficients) :
    highPopulationRisk (gradientStep (1 / 2) c) = 0 := by
  rw [half_step_reaches_identity]
  exact high_identity_risk_zero

/-- There is exactly one zero-risk high affine readout in coefficient space. -/
theorem population_risk_zero_iff_identity (c : Coefficients) :
    highPopulationRisk c = 0 ↔ c = (0, 1) := by
  constructor
  · intro h
    rw [highPopulationRisk, high_affine_risk] at h
    have hb : c.1 ^ 2 = 0 := by nlinarith [sq_nonneg (1 - c.2)]
    have ha : (1 - c.2) ^ 2 = 0 := by nlinarith [sq_nonneg c.1]
    have hb0 : c.1 = 0 := (sq_eq_zero_iff).mp hb
    have ha0 : 1 - c.2 = 0 := (sq_eq_zero_iff).mp ha
    apply Prod.ext
    · exact hb0
    · dsimp
      linarith
  · intro h
    rw [h]
    exact high_identity_risk_zero

/-- This learned readout, not just an exhibited choice of coefficients, beats
every low and middle affine predictor under the common population law. -/
theorem half_step_beats_all_affine_readouts (initial : Coefficients)
    (bias a b c d middleBias middleA middleB : ℝ) :
    highPopulationRisk (gradientStep (1 / 2) initial) <
        risk fairFour parityTarget (lowAffine bias a b c d ∘ low) ∧
      highPopulationRisk (gradientStep (1 / 2) initial) <
        risk fairFour parityTarget (middleAffine middleBias middleA middleB ∘ middle) := by
  rw [half_step_population_risk_zero]
  constructor
  · linarith [low_affine_risk_lower_bound bias a b c d]
  · linarith [middle_affine_risk_lower_bound middleBias middleA middleB]

/-- Nonzero population error is strictly reduced throughout the stable range. -/
theorem stable_step_strictly_improves (α : ℝ) (c : Coefficients)
    (hα : 0 < α) (hα1 : α < 1) (hR : 0 < highPopulationRisk c) :
    highPopulationRisk (gradientStep α c) < highPopulationRisk c := by
  rw [population_risk_contraction]
  have hfactor : (1 - 2 * α) ^ 2 < 1 := by
    nlinarith [mul_pos hα (sub_pos.mpr hα1)]
  have h := mul_lt_mul_of_pos_right hfactor hR
  simpa using h

/-- At step size one the error oscillates without improving. -/
theorem unit_step_preserves_risk (c : Coefficients) :
    highPopulationRisk (gradientStep 1 c) = highPopulationRisk c := by
  rw [population_risk_contraction]
  norm_num

/-- Excessively large positive steps strictly worsen every nonzero error. -/
theorem oversized_step_strictly_worsens (α : ℝ) (c : Coefficients)
    (hα : 1 < α) (hR : 0 < highPopulationRisk c) :
    highPopulationRisk c < highPopulationRisk (gradientStep α c) := by
  rw [population_risk_contraction]
  have hfactor : 1 < (1 - 2 * α) ^ 2 := by
    nlinarith [sq_nonneg (α - 1)]
  have h := mul_lt_mul_of_pos_right hfactor hR
  simpa using h

end HJepa.Optimization
