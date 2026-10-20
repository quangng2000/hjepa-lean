import HJepa.Anchor
import HJepa.Objective

/-!
# Learning a protected reference encoder without downstream labels

The frozen reference is the two signed raw input bits. A learned 2-by-2 linear
encoder and a fixed identity decoder reconstruct it. Averaging residual
gradients on the four balanced raw inputs is exactly population optimization.
We derive those gradients from reconstruction targets, not downstream labels,
and prove that one appropriately scaled full-batch step learns exact recovery
from every matrix initialization. The predictive branch is frozen to the
background encoder and already has zero loss.

This is an unconstrained quadratic encoder-training theorem, not nonlinear
JEPA training, a random-minibatch theorem, or a compression gain. The input and
code both have two real coordinates. Its special one-step result depends on
balanced, whitened finite inputs, identity decoding, and the chosen step size.
-/

namespace HJepa.ReferenceLearning

open HJepa.Abstraction HJepa.Objective
open scoped BigOperators

abbrev Weights := Bool → ℝ × ℝ

/-- Two fixed raw-input coordinates; no downstream task labels are used. -/
def reference (x : Input) (d : Bool) : ℝ :=
  if d then signedBit x.1 else signedBit x.2

/-- The parameters belong to the encoder; the decoder stays the identity. -/
noncomputable def encoder (w : Weights) (x : Input) (d : Bool) : ℝ :=
  (w d).1 * signedBit x.1 + (w d).2 * signedBit x.2

def identityWeights : Weights := fun d => if d then (1, 0) else (0, 1)

noncomputable def reconstructionRisk (w : Weights) : ℝ :=
  Vector.risk fairTrajectory (reference ∘ current) (encoder w ∘ current)

noncomputable def residual (w : Weights) (x : Input) (d : Bool) : ℝ :=
  encoder w x d - reference x d

/-- Average residual gradient of the four equally weighted raw examples. -/
noncomputable def batchGradient (w : Weights) (d : Bool) : ℝ × ℝ :=
  ((∑ b : Bool, ∑ u : Bool, 2 * residual w (b, u) d * signedBit b) / 4,
   (∑ b : Bool, ∑ u : Bool, 2 * residual w (b, u) d * signedBit u) / 4)

/-- Sum over output coordinates of the actual four-example training loss. -/
noncomputable def empiricalRisk (w : Weights) : ℝ :=
  ∑ d : Bool, (∑ b : Bool, ∑ u : Bool, residual w (b, u) d ^ 2) / 4

theorem empirical_eq_population (w : Weights) :
    empiricalRisk w = reconstructionRisk w := by
  norm_num [empiricalRisk, reconstructionRisk, Vector.risk, risk,
    fairTrajectory, residual, reference, encoder, current,
    Function.comp_def, signedBit, Fintype.sum_prod_type, Fintype.sum_bool]
  ring

theorem reconstruction_risk_formula (w : Weights) :
    reconstructionRisk w = ∑ d : Bool,
      (((w d).1 - (identityWeights d).1) ^ 2 +
       ((w d).2 - (identityWeights d).2) ^ 2) := by
  norm_num [reconstructionRisk, Vector.risk, risk, fairTrajectory,
    reference, encoder, current, identityWeights, Function.comp_def,
    signedBit, Fintype.sum_prod_type, Fintype.sum_bool]
  ring

theorem batch_gradient_formula (w : Weights) (d : Bool) :
    batchGradient w d = (2 * ((w d).1 - (identityWeights d).1),
      2 * ((w d).2 - (identityWeights d).2)) := by
  cases d <;> apply Prod.ext <;>
    norm_num [batchGradient, residual, reference, encoder,
      identityWeights, signedBit, Fintype.sum_bool] <;> ring

/-- An exact first-order expansion certifies that the residual average is
the gradient; the remainder consists entirely of quadratic increments. -/
theorem exact_loss_increment (w δ : Weights) :
    reconstructionRisk (fun d => ((w d).1 + (δ d).1, (w d).2 + (δ d).2)) =
      reconstructionRisk w + (∑ d : Bool,
        ((batchGradient w d).1 * (δ d).1 + (batchGradient w d).2 * (δ d).2 +
         (δ d).1 ^ 2 + (δ d).2 ^ 2)) := by
  rw [reconstruction_risk_formula, reconstruction_risk_formula]
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro d _
  rw [batch_gradient_formula]
  dsimp
  ring

/-- Actual gradient descent on weight times the reconstruction loss. -/
noncomputable def gradientStep (rate weight : ℝ) (w : Weights) : Weights :=
  fun d => ((w d).1 - rate * weight * (batchGradient w d).1,
    (w d).2 - rate * weight * (batchGradient w d).2)

theorem scaled_step_learns_identity (weight : ℝ) (hw : 0 < weight) (w : Weights) :
    gradientStep (1 / (2 * weight)) weight w = identityWeights := by
  have hcancel : (1 / (2 * weight)) * (2 * weight) = 1 :=
    div_mul_cancel₀ _ (mul_ne_zero (by norm_num) (ne_of_gt hw))
  have hc : (1 / (2 * weight)) * weight * 2 = 1 := by nlinarith
  have hstep (a b : ℝ) : a - (1 / (2 * weight)) * weight * (2 * (a - b)) = b := by
    calc
      _ = a - ((1 / (2 * weight)) * weight * 2) * (a - b) := by ring
      _ = b := by rw [hc]; ring
  funext d
  rw [gradientStep, batch_gradient_formula]
  exact Prod.ext (hstep _ _) (hstep _ _)

theorem identity_encoder_recovers (x : Input) : encoder identityWeights x = reference x := by
  funext d
  cases d <;> simp [encoder, reference, identityWeights]

/-- The learned encoder, not a supplied readout of fixed features, now
recovers the complete two-coordinate reference on every raw input. -/
theorem learned_encoder_recovers (weight : ℝ) (hw : 0 < weight) (w : Weights)
    (x : Input) :
    encoder (gradientStep (1 / (2 * weight)) weight w) x = reference x := by
  rw [scaled_step_learns_identity weight hw w]
  exact identity_encoder_recovers x

theorem learned_reconstruction_risk_zero (weight : ℝ) (hw : 0 < weight) (w : Weights) :
    reconstructionRisk (gradientStep (1 / (2 * weight)) weight w) = 0 := by
  rw [scaled_step_learns_identity weight hw w, reconstruction_risk_formula]
  simp

/-- Here, unlike an abstract predictive-cost interface, the independent
predictive branch is explicitly chosen and its zero loss has been proved. -/
noncomputable def protectedObjective (weight : ℝ) (w : Weights) : ℝ :=
  Anchor.objective fairTrajectory (reference ∘ current) (encoder w ∘ current) id
    (predictionLoss backgroundEncoder id) weight

theorem protected_objective_formula (weight : ℝ) (w : Weights) :
    protectedObjective weight w = weight * reconstructionRisk w := by
  simp [protectedObjective, Anchor.objective, Anchor.reconstructionLoss,
    reconstructionRisk, background_prediction_zero]

theorem learned_objective_zero (weight : ℝ) (hw : 0 < weight) (w : Weights) :
    protectedObjective weight (gradientStep (1 / (2 * weight)) weight w) = 0 := by
  rw [protected_objective_formula, learned_reconstruction_risk_zero weight hw w, mul_zero]

/-- A trained model is a global minimizer over this actual matrix-encoder
family, without assuming approximate or exact minimization as a hypothesis. -/
theorem learned_global_minimum (weight : ℝ) (hw : 0 < weight) (initial other : Weights) :
    protectedObjective weight (gradientStep (1 / (2 * weight)) weight initial) ≤
      protectedObjective weight other := by
  rw [learned_objective_zero weight hw initial, protected_objective_formula]
  exact mul_nonneg (le_of_lt hw) (Vector.risk_nonneg _ _ _)

/-- The optimizer now discharges the reconstruction premise of the anchor
theorem. Every fixed-reference task readout keeps its exact risk, even on a
noisy target. It does not need to be specified during reconstruction training. -/
theorem learned_preserves_every_reference_readout
    (weight : ℝ) (hw : 0 < weight) (initial : Weights)
    (target : Trajectory → ℝ) (readout : (Bool → ℝ) → ℝ) :
    risk fairTrajectory target
      (readout ∘ encoder (gradientStep (1 / (2 * weight)) weight initial) ∘ current) =
      risk fairTrajectory target (readout ∘ reference ∘ current) := by
  have heq : encoder (gradientStep (1 / (2 * weight)) weight initial) = reference :=
    funext (learned_encoder_recovers weight hw initial)
  rw [heq]

end HJepa.ReferenceLearning
