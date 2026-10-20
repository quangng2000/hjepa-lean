import HJepa.ProtectedChannel

/-!
# Learning the protected encoder from raw labeled inputs

The protected scalar encoder is now trainable: `a * background + b * task`.
The downstream head is fixed to the identity, so changing these coefficients
changes the representation itself, not a readout of a fixed representation.
Four balanced raw inputs with their task labels give exactly the population
loss. We derive their residual gradient and certify it by an exact loss
expansion, then prove the gradient-descent trajectory from any initialization.

The predictive branch is frozen at background/identity (zero predictive loss).
Training is unconstrained in coefficient space; intermediate encoders need not
be moment-normalized. The learned endpoint is normalized and globally minimizes
the earlier two-channel objective even against its unrestricted admissible
models. The result concerns this supervised linear encoder, not optimization
of arbitrary neural encoders, unlabeled JEPA learning, or unknown tasks.
-/

namespace HJepa.EncoderLearning

open HJepa.Abstraction HJepa.Objective
open scoped BigOperators

abbrev Coefficients := ℝ × ℝ

noncomputable def encoder (c : Coefficients) (x : Input) : ℝ :=
  c.1 * signedBit x.1 + c.2 * signedBit x.2

noncomputable def populationRisk (c : Coefficients) : ℝ :=
  downstreamRisk (encoder c) id

noncomputable def residual (c : Coefficients) (x : Input) : ℝ :=
  encoder c x - signedBit x.2

noncomputable def sampleGradient (c : Coefficients) (x : Input) : Coefficients :=
  (2 * residual c x * signedBit x.1, 2 * residual c x * signedBit x.2)

/-- Every raw input appears once; labels are the second signed bit. -/
noncomputable def empiricalRisk (c : Coefficients) : ℝ :=
  (∑ b : Bool, ∑ u : Bool, residual c (b, u) ^ 2) / 4

noncomputable def residualGradient (c : Coefficients) : Coefficients :=
  ((∑ b : Bool, ∑ u : Bool, (sampleGradient c (b, u)).1) / 4,
   (∑ b : Bool, ∑ u : Bool, (sampleGradient c (b, u)).2) / 4)

theorem population_risk_formula (c : Coefficients) :
    populationRisk c = c.1 ^ 2 + (c.2 - 1) ^ 2 := by
  norm_num [populationRisk, downstreamRisk, risk, fairTrajectory, currentTask,
    encoder, current, signedBit, Fintype.sum_prod_type, Fintype.sum_bool,
    Function.comp_def]
  ring

theorem empirical_risk_formula (c : Coefficients) :
    empiricalRisk c = c.1 ^ 2 + (c.2 - 1) ^ 2 := by
  norm_num [empiricalRisk, residual, encoder, signedBit, Fintype.sum_bool]
  ring

theorem empirical_equals_population (c : Coefficients) :
    empiricalRisk c = populationRisk c := by
  rw [empirical_risk_formula, population_risk_formula]

theorem residual_gradient_formula (c : Coefficients) :
    residualGradient c = (2 * c.1, 2 * (c.2 - 1)) := by
  ext <;>
    norm_num [residualGradient, sampleGradient, residual, encoder, signedBit,
      Fintype.sum_bool] <;> ring

/-- Exact first-order expansion identifies the averaged residual gradient as
the gradient of the actual balanced empirical squared loss. -/
theorem exact_loss_increment (c δ : Coefficients) :
    empiricalRisk (c.1 + δ.1, c.2 + δ.2) =
      empiricalRisk c + (residualGradient c).1 * δ.1 +
        (residualGradient c).2 * δ.2 + δ.1 ^ 2 + δ.2 ^ 2 := by
  rw [empirical_risk_formula, empirical_risk_formula, residual_gradient_formula]
  dsimp
  ring

/-- The prediction term is identically zero; this is the full objective on
the chosen trainable family, not a surrogate for a changing prediction loss. -/
noncomputable def model (c : Coefficients) : ProtectedChannel.Model where
  predictiveChannel := backgroundEncoder
  taskChannel := encoder c
  predictor := id
  taskHead := id

theorem model_objective (weight : ℝ) (c : Coefficients) :
    ProtectedChannel.objective weight (model c) = weight * populationRisk c := by
  simp [ProtectedChannel.objective, ProtectedChannel.taskRisk, model,
    populationRisk, background_prediction_zero]

theorem exact_objective_increment (weight : ℝ) (c δ : Coefficients) :
    ProtectedChannel.objective weight (model (c.1 + δ.1, c.2 + δ.2)) =
      ProtectedChannel.objective weight (model c) +
        (weight * (residualGradient c).1) * δ.1 +
        (weight * (residualGradient c).2) * δ.2 +
        weight * (δ.1 ^ 2 + δ.2 ^ 2) := by
  rw [model_objective, model_objective, population_risk_formula,
    population_risk_formula, residual_gradient_formula]
  dsimp
  ring

/-- Gradient descent updates the raw-input encoder, keeping its head fixed. -/
noncomputable def gradientStep (weight α : ℝ) (c : Coefficients) : Coefficients :=
  (c.1 - α * weight * (residualGradient c).1,
   c.2 - α * weight * (residualGradient c).2)

theorem gradient_step_formula (weight α : ℝ) (c : Coefficients) :
    gradientStep weight α c =
      ((1 - 2 * α * weight) * c.1,
       1 + (1 - 2 * α * weight) * (c.2 - 1)) := by
  rw [gradientStep, residual_gradient_formula]
  ext <;> dsimp <;> ring

theorem population_risk_contraction (weight α : ℝ) (c : Coefficients) :
    populationRisk (gradientStep weight α c) =
      (1 - 2 * α * weight) ^ 2 * populationRisk c := by
  rw [gradient_step_formula, population_risk_formula, population_risk_formula]
  dsimp
  ring

noncomputable def gradientIterate (weight α : ℝ) (initial : Coefficients) :
    ℕ → Coefficients
  | 0 => initial
  | n + 1 => gradientStep weight α (gradientIterate weight α initial n)

theorem iterate_population_risk (weight α : ℝ) (initial : Coefficients) (n : ℕ) :
    populationRisk (gradientIterate weight α initial n) =
      ((1 - 2 * α * weight) ^ 2) ^ n * populationRisk initial := by
  induction n with
  | zero => simp [gradientIterate]
  | succ n ih =>
      rw [gradientIterate, population_risk_contraction, ih, pow_succ]
      ring

/-- The step size accounts for the protection weight. No optimum is supplied
to the update; it is computed from the four labeled residual gradients. -/
theorem calibrated_step_learns_task (weight : ℝ) (hweight : 0 < weight)
    (initial : Coefficients) :
    gradientStep weight (1 / (2 * weight)) initial = (0, 1) := by
  have hn : weight ≠ 0 := ne_of_gt hweight
  have hc : 1 - 2 * (1 / (2 * weight)) * weight = 0 := by
    have hcancel : (1 / (2 * weight)) * (2 * weight) = 1 :=
      div_mul_cancel₀ _ (mul_ne_zero (by norm_num) hn)
    nlinarith
  rw [gradient_step_formula, hc]
  simp

theorem learned_encoder_is_task : encoder (0, 1) = taskEncoder := by
  funext x
  simp [encoder, taskEncoder]

theorem calibrated_step_population_risk_zero (weight : ℝ) (hweight : 0 < weight)
    (initial : Coefficients) :
    populationRisk (gradientStep weight (1 / (2 * weight)) initial) = 0 := by
  rw [calibrated_step_learns_task weight hweight, population_risk_formula]
  norm_num

theorem learned_model_is_witness : model (0, 1) = ProtectedChannel.witness := by
  simp only [model, learned_encoder_is_task, ProtectedChannel.witness]

/-- Training was unconstrained; its endpoint satisfies both normalizations. -/
theorem calibrated_step_admissible (weight : ℝ) (hweight : 0 < weight)
    (initial : Coefficients) :
    ProtectedChannel.Admissible (model (gradientStep weight (1 / (2 * weight)) initial)) := by
  rw [calibrated_step_learns_task weight hweight, learned_model_is_witness]
  exact ProtectedChannel.witness_admissible

/-- The learned model minimizes the full protected objective among every
admissible model, not merely among the two-coefficient training family. -/
theorem calibrated_step_global_minimum (weight : ℝ) (hweight : 0 < weight)
    (initial : Coefficients) :
    ProtectedChannel.GlobalMinimum weight
      (model (gradientStep weight (1 / (2 * weight)) initial)) := by
  rw [calibrated_step_learns_task weight hweight, learned_model_is_witness]
  exact ProtectedChannel.witness_global_minimum weight (le_of_lt hweight)

/-- Arbitrary predictive abstractions retain the learned task channel. -/
theorem learned_three_levels_zero (weight : ℝ) (hweight : 0 < weight)
    (initial : Coefficients) (g h : ℝ → ℝ) :
    let m := model (gradientStep weight (1 / (2 * weight)) initial)
    ProtectedChannel.representationTaskRisk (ProtectedChannel.lowRepresentation m)
        (ProtectedChannel.protectedReadout m) = 0 ∧
      ProtectedChannel.representationTaskRisk (ProtectedChannel.middleRepresentation m g)
        (ProtectedChannel.protectedReadout m) = 0 ∧
      ProtectedChannel.representationTaskRisk (ProtectedChannel.highRepresentation m g h)
        (ProtectedChannel.protectedReadout m) = 0 := by
  exact ProtectedChannel.three_levels_preserve_task_at_global_minimum weight hweight _
    (calibrated_step_global_minimum weight hweight initial) g h

theorem stable_step_strictly_improves (weight α : ℝ) (c : Coefficients)
    (hstep : 0 < α * weight) (hstep1 : α * weight < 1)
    (hR : 0 < populationRisk c) :
    populationRisk (gradientStep weight α c) < populationRisk c := by
  rw [population_risk_contraction]
  have hf : (1 - 2 * α * weight) ^ 2 < 1 := by
    nlinarith [mul_pos hstep (sub_pos.mpr hstep1)]
  simpa using mul_lt_mul_of_pos_right hf hR

end HJepa.EncoderLearning
