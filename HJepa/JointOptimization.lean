import HJepa.ProtectedChannel
import Mathlib.Tactic.FieldSimp

/-!
# Joint encoder/head training: a stationary failure and a guarded block solver

The protected scalar encoder is `a B + b U`, and its trainable linear head is
`r ↦ w r`. The target is the independent fair task bit `U`. We derive the
population residual gradient from the finite raw-input distribution and
certify its first-order coefficients by an exact polynomial expansion.

Simultaneous gradient descent can remain forever at a normalized background
encoder with zero head and task risk one. A different, explicitly specified
block algorithm avoids this obstruction: initialize a zero head to one, freeze
the now nonzero head, and take an encoder gradient step with reciprocal block
curvature. This learns an exact task representation from any coefficients.
Finally, a prediction-preserving rescaling produces the normalized task encoder
and the already-defined zero-loss protected model.

This is exact population (equivalently, balanced finite full-batch) optimization
for a bilinear linear encoder/head, with the predictive branch fixed. It is not
a theorem about arbitrary initialization of simultaneous GD, minibatch SGD,
Adam, neural encoders, or norm-constrained intermediate iterates. The guarded
solver is deliberately a different algorithm from simultaneous GD.
-/

namespace HJepa.JointOptimization

open HJepa.Abstraction HJepa.Objective
open scoped BigOperators

structure Parameters where
  background : ℝ
  task : ℝ
  head : ℝ

theorem Parameters.ext {p q : Parameters}
    (ha : p.background = q.background) (hb : p.task = q.task) (hw : p.head = q.head) :
    p = q := by
  cases p
  cases q
  simp_all

noncomputable def encoder (p : Parameters) (s : Input) : ℝ :=
  p.background * signedBit s.1 + p.task * signedBit s.2

noncomputable def readout (p : Parameters) (r : ℝ) : ℝ := p.head * r

noncomputable def loss (p : Parameters) : ℝ := downstreamRisk (encoder p) (readout p)

theorem population_loss_formula (p : Parameters) :
    loss p = (p.head * p.background) ^ 2 + (p.head * p.task - 1) ^ 2 := by
  norm_num [loss, downstreamRisk, risk, fairTrajectory, currentTask, current,
    encoder, readout, signedBit, Fintype.sum_prod_type, Fintype.sum_bool,
    Function.comp_def]
  ring

/-- One balanced full batch contains every raw input, with its actual task label. -/
noncomputable def balancedBatchLoss (p : Parameters) : ℝ :=
  (∑ s : Input, (readout p (encoder p s) - signedBit s.2) ^ 2) / 4

theorem balanced_batch_equals_population (p : Parameters) :
    balancedBatchLoss p = loss p := by
  rw [population_loss_formula]
  norm_num [balancedBatchLoss, encoder, readout, signedBit,
    Fintype.sum_prod_type, Fintype.sum_bool]
  ring

/-- Prediction minus the actual task label on a raw trajectory. -/
noncomputable def residual (p : Parameters) (s : Trajectory) : ℝ :=
  readout p (encoder p (current s)) - currentTask s

/-- Chain-rule squared-loss residual gradient, averaged over the raw examples. -/
noncomputable def residualGradient (p : Parameters) : Parameters where
  background := ∑ s, fairTrajectory.weight s *
    (2 * residual p s * p.head * signedBit (current s).1)
  task := ∑ s, fairTrajectory.weight s *
    (2 * residual p s * p.head * signedBit (current s).2)
  head := ∑ s, fairTrajectory.weight s *
    (2 * residual p s * encoder p (current s))

theorem residual_gradient_formula (p : Parameters) :
    residualGradient p =
      ⟨2 * p.head ^ 2 * p.background,
       2 * p.head * (p.head * p.task - 1),
       2 * p.head * (p.background ^ 2 + p.task ^ 2) - 2 * p.task⟩ := by
  apply Parameters.ext <;>
    norm_num [residualGradient, residual, fairTrajectory, currentTask, current,
      encoder, readout, signedBit, Fintype.sum_prod_type, Fintype.sum_bool] <;>
    ring

noncomputable def perturb (p d : Parameters) (t : ℝ) : Parameters :=
  ⟨p.background + t * d.background, p.task + t * d.task, p.head + t * d.head⟩

/-- All terms after the certified first-order gradient are divisible by `t²`.
This is an exact polynomial identity, not an assumed gradient oracle. -/
theorem exact_first_order_expansion (p d : Parameters) (t : ℝ) :
    loss (perturb p d t) = loss p +
      t * ((residualGradient p).background * d.background +
        (residualGradient p).task * d.task + (residualGradient p).head * d.head) +
      t ^ 2 *
        ((p.head * d.background + d.head * p.background) ^ 2 +
         (p.head * d.task + d.head * p.task) ^ 2 +
         2 * (p.head * p.background) * (d.head * d.background) +
         2 * (p.head * p.task - 1) * (d.head * d.task) +
         2 * t * ((p.head * d.background + d.head * p.background) *
           (d.head * d.background) +
           (p.head * d.task + d.head * p.task) * (d.head * d.task)) +
         t ^ 2 * ((d.head * d.background) ^ 2 + (d.head * d.task) ^ 2)) := by
  rw [population_loss_formula, population_loss_formula, residual_gradient_formula]
  dsimp [perturb]
  ring

/-- Ordinary simultaneous full-gradient descent on all three parameters. -/
noncomputable def gradientStep (rate : ℝ) (p : Parameters) : Parameters :=
  ⟨p.background - rate * (residualGradient p).background,
   p.task - rate * (residualGradient p).task,
   p.head - rate * (residualGradient p).head⟩

def backgroundInitialization : Parameters := ⟨1, 0, 0⟩

theorem background_initialization_normalized :
    MomentNormalized (encoder backgroundInitialization) := by
  have h : encoder backgroundInitialization = backgroundEncoder := by
    funext s
    simp [encoder, backgroundInitialization, backgroundEncoder]
  rw [h]
  exact background_normalized

theorem background_initialization_loss : loss backgroundInitialization = 1 := by
  norm_num [population_loss_formula, backgroundInitialization]

theorem background_initialization_stationary (rate : ℝ) :
    gradientStep rate backgroundInitialization = backgroundInitialization := by
  unfold gradientStep
  rw [residual_gradient_formula]
  simp [backgroundInitialization]

/-- Variable step sizes do not escape an exactly zero gradient. -/
noncomputable def gradientIterate (rates : ℕ → ℝ) (initial : Parameters) : ℕ → Parameters
  | 0 => initial
  | n + 1 => gradientStep (rates n) (gradientIterate rates initial n)

theorem simultaneous_gd_stays_at_background (rates : ℕ → ℝ) (n : ℕ) :
    gradientIterate rates backgroundInitialization n = backgroundInitialization := by
  induction n with
  | zero => rfl
  | succ n ih =>
      rw [gradientIterate, ih, background_initialization_stationary]

theorem simultaneous_gd_failure (rates : ℕ → ℝ) (n : ℕ) :
    MomentNormalized (encoder (gradientIterate rates backgroundInitialization n)) ∧
      loss (gradientIterate rates backgroundInitialization n) = 1 := by
  rw [simultaneous_gd_stays_at_background]
  exact ⟨background_initialization_normalized, background_initialization_loss⟩

/-- Encoder-only block gradient descent, holding the current head fixed. -/
noncomputable def encoderBlockStep (rate : ℝ) (p : Parameters) : Parameters :=
  ⟨p.background - rate * (residualGradient p).background,
   p.task - rate * (residualGradient p).task, p.head⟩

/-- The reciprocal block curvature is valid only for a nonzero head. -/
theorem reciprocal_curvature_step (p : Parameters) (hw : p.head ≠ 0) :
    encoderBlockStep (1 / (2 * p.head ^ 2)) p = ⟨0, 1 / p.head, p.head⟩ := by
  unfold encoderBlockStep
  rw [residual_gradient_formula]
  apply Parameters.ext
  · dsimp
    field_simp [hw]
    ring
  · dsimp
    field_simp [hw]
    ring
  · rfl

theorem reciprocal_curvature_step_loss_zero (p : Parameters) (hw : p.head ≠ 0) :
    loss (encoderBlockStep (1 / (2 * p.head ^ 2)) p) = 0 := by
  rw [reciprocal_curvature_step p hw, population_loss_formula]
  simp [hw]

/-- A specified deterministic initialization rule for the block solver. This
guard is not claimed to be an ordinary gradient step. -/
noncomputable def guardHead (p : Parameters) : Parameters :=
  if p.head = 0 then ⟨p.background, p.task, 1⟩ else p

/-- The initialization guard is not a monotone descent step. Its role is to
enable the subsequent encoder block update, not to reduce loss by itself. -/
theorem guard_can_increase_loss :
    loss (⟨2, 0, 0⟩ : Parameters) = 1 ∧
      loss (guardHead (⟨2, 0, 0⟩ : Parameters)) = 5 := by
  norm_num [guardHead, population_loss_formula]

theorem guarded_head_nonzero (p : Parameters) : (guardHead p).head ≠ 0 := by
  unfold guardHead
  split_ifs with hw
  · norm_num
  · exact hw

/-- Learn the encoder by its actual residual gradient, after the head guard. -/
noncomputable def guardedBlockLearn (p : Parameters) : Parameters :=
  encoderBlockStep (1 / (2 * (guardHead p).head ^ 2)) (guardHead p)

theorem guarded_block_learns_from_every_initialization (p : Parameters) :
    loss (guardedBlockLearn p) = 0 := by
  exact reciprocal_curvature_step_loss_zero (guardHead p) (guarded_head_nonzero p)

/-- Absorb the head scale into the encoder. This preserves decoded predictions,
but is a reparameterization, not a gradient step. -/
noncomputable def canonicalize (p : Parameters) : Parameters :=
  ⟨p.head * p.background, p.head * p.task, 1⟩

theorem canonicalization_preserves_predictions (p : Parameters) (s : Input) :
    readout (canonicalize p) (encoder (canonicalize p) s) =
      readout p (encoder p s) := by
  dsimp [canonicalize, encoder, readout]
  ring

theorem canonicalization_preserves_loss (p : Parameters) :
    loss (canonicalize p) = loss p := by
  rw [population_loss_formula, population_loss_formula]
  simp [canonicalize]

noncomputable def learn (p : Parameters) : Parameters := canonicalize (guardedBlockLearn p)

theorem learned_parameters (p : Parameters) : learn p = ⟨0, 1, 1⟩ := by
  unfold learn guardedBlockLearn
  rw [reciprocal_curvature_step _ (guarded_head_nonzero p)]
  simp [canonicalize, guarded_head_nonzero p]

theorem learned_encoder (p : Parameters) : encoder (learn p) = taskEncoder := by
  rw [learned_parameters]
  funext s
  simp [encoder, taskEncoder]

theorem learned_head (p : Parameters) : readout (learn p) = id := by
  rw [learned_parameters]
  funext r
  simp [readout]

/-- The predictive branch is fixed; only the protected encoder/head are learned. -/
noncomputable def model (p : Parameters) : HJepa.ProtectedChannel.Model where
  predictiveChannel := backgroundEncoder
  taskChannel := encoder p
  predictor := id
  taskHead := readout p

theorem protected_objective_equals_weighted_loss (weight : ℝ) (p : Parameters) :
    HJepa.ProtectedChannel.objective weight (model p) = weight * loss p := by
  simp [HJepa.ProtectedChannel.objective, HJepa.ProtectedChannel.taskRisk,
    model, loss, background_prediction_zero]

/-- Gradient descent for the weighted task block uses the scaled residual
gradient. The fixed predictive branch contributes no parameter gradient. -/
noncomputable def weightedEncoderBlockStep (weight rate : ℝ) (p : Parameters) : Parameters :=
  ⟨p.background - rate * (weight * (residualGradient p).background),
   p.task - rate * (weight * (residualGradient p).task), p.head⟩

theorem weighted_step_rate_equivalence (weight rate : ℝ) (p : Parameters) :
    weightedEncoderBlockStep weight rate p = encoderBlockStep (rate * weight) p := by
  apply Parameters.ext <;> dsimp [weightedEncoderBlockStep, encoderBlockStep] <;> ring

/-- For positive protection weight, the same guarded solver is a block-gradient
method for the proposed objective after dividing its step size by that weight. -/
theorem weighted_reciprocal_step (weight : ℝ) (hweight : 0 < weight)
    (p : Parameters) (hw : p.head ≠ 0) :
    weightedEncoderBlockStep weight (1 / (2 * weight * p.head ^ 2)) p =
      ⟨0, 1 / p.head, p.head⟩ := by
  rw [weighted_step_rate_equivalence]
  have hr : 1 / (2 * weight * p.head ^ 2) * weight = 1 / (2 * p.head ^ 2) := by
    field_simp [hw, ne_of_gt hweight]
  rw [hr, reciprocal_curvature_step p hw]

theorem learned_parameters_from_weighted_gradient (weight : ℝ) (hweight : 0 < weight)
    (p : Parameters) :
    canonicalize (weightedEncoderBlockStep weight
      (1 / (2 * weight * (guardHead p).head ^ 2)) (guardHead p)) = learn p := by
  rw [weighted_reciprocal_step weight hweight _ (guarded_head_nonzero p),
    learned_parameters]
  simp [canonicalize, guarded_head_nonzero p]

theorem learned_model_is_witness (p : Parameters) :
    model (learn p) = HJepa.ProtectedChannel.witness := by
  unfold model HJepa.ProtectedChannel.witness
  rw [learned_encoder, learned_head]

/-- Actual optimizer output, not an assumed global-minimum certificate. The
normalization requirement holds at the learned endpoint, not at every step. -/
theorem learned_model_global_minimum (p : Parameters) (weight : ℝ)
    (hweight : 0 ≤ weight) :
    HJepa.ProtectedChannel.GlobalMinimum weight (model (learn p)) := by
  rw [learned_model_is_witness]
  exact HJepa.ProtectedChannel.witness_global_minimum weight hweight

end HJepa.JointOptimization
