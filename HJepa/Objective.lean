import HJepa.Abstraction
import Mathlib.Data.Fintype.BigOperators

/-!
# Predictive representation learning need not select task-useful features

This finite model has eight equally likely trajectories `(background, now, next)`.
The background is static; the current and future task bits are independent fair
bits. Both scalar encoders below have stationary mean zero and second moment one.
Nevertheless, predicting one's own future latent state strictly favors the
irrelevant background encoder over the current-task encoder, even when the
predictor is an arbitrary function.

The example concerns a precise squared predictive objective with a moment
normalization constraint. That constraint rules out constant encoders, but is
not SIGReg, a Gaussian marginal constraint, an EMA training algorithm, or the
full H-JEPA objective. The downstream task is prediction of the current task
bit, not prediction of the future task bit. This distinction is the point:
predictive latent accuracy alone does not imply utility for a specified task.
-/

namespace HJepa.Objective

open HJepa.Abstraction
open scoped BigOperators

abbrev Input := Bool × Bool
abbrev Trajectory := Bool × (Bool × Bool)

/-- Independent fair background, current-task, and future-task bits. -/
noncomputable def fairTrajectory : FiniteLaw Trajectory where
  weight := fun _ => 1 / 8
  nonneg := by intro _; norm_num
  total_weight := by norm_num [Fintype.sum_prod_type, Fintype.sum_bool]

def current (s : Trajectory) : Input := (s.1, s.2.1)
def future (s : Trajectory) : Input := (s.1, s.2.2)

/-- Mean under the common stationary input distribution. -/
noncomputable def encoderMean (encoder : Input → ℝ) : ℝ :=
  (∑ b : Bool, ∑ u : Bool, encoder (b, u)) / 4

/-- Second moment under that same distribution. -/
noncomputable def encoderSecondMoment (encoder : Input → ℝ) : ℝ :=
  (∑ b : Bool, ∑ u : Bool, encoder (b, u) ^ 2) / 4

/-- A noncollapse surrogate, not the full regularizer of any JEPA system. -/
def MomentNormalized (encoder : Input → ℝ) : Prop :=
  encoderMean encoder = 0 ∧ encoderSecondMoment encoder = 1

/-- The learned target is the same encoder applied to the next input. -/
noncomputable def predictionLoss (encoder : Input → ℝ) (predictor : ℝ → ℝ) : ℝ :=
  risk fairTrajectory (encoder ∘ future) (predictor ∘ encoder ∘ current)

def backgroundEncoder (s : Input) : ℝ := signedBit s.1
def taskEncoder (s : Input) : ℝ := signedBit s.2

/-- A fixed downstream target, independent of the background. -/
def currentTask (s : Trajectory) : ℝ := signedBit s.2.1

noncomputable def downstreamRisk (encoder : Input → ℝ) (readout : ℝ → ℝ) : ℝ :=
  risk fairTrajectory currentTask (readout ∘ encoder ∘ current)

theorem background_normalized : MomentNormalized backgroundEncoder := by
  constructor <;>
    norm_num [encoderMean, encoderSecondMoment, backgroundEncoder, signedBit,
      Fintype.sum_bool]

theorem task_normalized : MomentNormalized taskEncoder := by
  constructor <;>
    norm_num [encoderMean, encoderSecondMoment, taskEncoder, signedBit,
      Fintype.sum_bool]

theorem prediction_loss_nonneg (encoder : Input → ℝ) (predictor : ℝ → ℝ) :
    0 ≤ predictionLoss encoder predictor := risk_nonneg _ _ _

/-- A nonconstant, task-irrelevant feature has exactly zero predictive loss. -/
theorem background_prediction_zero : predictionLoss backgroundEncoder id = 0 := by
  simp [predictionLoss, risk, backgroundEncoder, current, future, Function.comp_def]

/-- Background is a global optimum even among all encoders, hence also among
the moment-normalized encoders. This is not merely a comparison of two fits. -/
theorem background_global_minimum (encoder : Input → ℝ) (predictor : ℝ → ℝ) :
    predictionLoss backgroundEncoder id ≤ predictionLoss encoder predictor := by
  rw [background_prediction_zero]
  exact prediction_loss_nonneg encoder predictor

/-- No predictor, linear or nonlinear, predicts the independent future task bit. -/
theorem task_prediction_exact (predictor : ℝ → ℝ) :
    predictionLoss taskEncoder predictor =
      1 + (predictor 1 ^ 2 + predictor (-1) ^ 2) / 2 := by
  norm_num [predictionLoss, risk, fairTrajectory, taskEncoder, current, future,
    signedBit, Fintype.sum_prod_type, Fintype.sum_bool, Function.comp_def]
  ring

theorem task_prediction_at_least_one (predictor : ℝ → ℝ) :
    1 ≤ predictionLoss taskEncoder predictor := by
  rw [task_prediction_exact]
  nlinarith [sq_nonneg (predictor 1), sq_nonneg (predictor (-1))]

theorem task_prediction_minimum_attained :
    predictionLoss taskEncoder (fun _ => 0) = 1 := by
  norm_num [task_prediction_exact]

/-- Predictive training strictly prefers the background to every predictor
paired with the useful task encoder; both encoders obey the same constraint. -/
theorem objective_strictly_prefers_background (predictor : ℝ → ℝ) :
    MomentNormalized backgroundEncoder ∧ MomentNormalized taskEncoder ∧
      predictionLoss backgroundEncoder id < predictionLoss taskEncoder predictor := by
  refine ⟨background_normalized, task_normalized, ?_⟩
  rw [background_prediction_zero]
  linarith [task_prediction_at_least_one predictor]

/-- Every downstream readout of the globally optimal background feature has
the following exact population risk for the fixed current-task target. -/
theorem background_downstream_exact (readout : ℝ → ℝ) :
    downstreamRisk backgroundEncoder readout =
      1 + (readout 1 ^ 2 + readout (-1) ^ 2) / 2 := by
  norm_num [downstreamRisk, risk, fairTrajectory, currentTask, backgroundEncoder,
    current, signedBit, Fintype.sum_prod_type, Fintype.sum_bool, Function.comp_def]
  ring

theorem background_downstream_at_least_one (readout : ℝ → ℝ) :
    1 ≤ downstreamRisk backgroundEncoder readout := by
  rw [background_downstream_exact]
  nlinarith [sq_nonneg (readout 1), sq_nonneg (readout (-1))]

/-- Exact predictive loss forces the encoder to ignore the independently
resampled task bit. The same current input is paired with both future bits. -/
theorem zero_prediction_ignores_task (encoder : Input → ℝ) (predictor : ℝ → ℝ)
    (hzero : predictionLoss encoder predictor = 0) (b u : Bool) :
    encoder (b, u) = encoder (b, false) := by
  have hpoint := (risk_eq_zero_iff fairTrajectory (encoder ∘ future)
    (predictor ∘ encoder ∘ current)).mp hzero
  have hu := hpoint (b, false, u) (by norm_num [fairTrajectory])
  have hf := hpoint (b, false, false) (by norm_num [fairTrajectory])
  exact hu.trans hf.symm

/-- Every zero-loss encoder, not only the named background encoder, is useless
for the independent current-task target under every deterministic readout. -/
theorem zero_prediction_downstream_exact (encoder : Input → ℝ) (predictor readout : ℝ → ℝ)
    (hzero : predictionLoss encoder predictor = 0) :
    downstreamRisk encoder readout =
      1 + (readout (encoder (true, false)) ^ 2 +
        readout (encoder (false, false)) ^ 2) / 2 := by
  have ht := zero_prediction_ignores_task encoder predictor hzero true true
  have hf := zero_prediction_ignores_task encoder predictor hzero false true
  norm_num [downstreamRisk, risk, fairTrajectory, currentTask, current,
    signedBit, Fintype.sum_prod_type, Fintype.sum_bool, Function.comp_def, ht, hf]
  ring

/-- Global optimality over the admissible encoders implies zero predictive
loss, since the admissible background encoder attains zero. Consequently every
such global optimum has downstream error at least one on the current task. -/
theorem every_global_optimum_task_risk_at_least_one
    (encoder : Input → ℝ) (predictor : ℝ → ℝ)
    (hoptimal : ∀ (alternative : Input → ℝ) (otherPredictor : ℝ → ℝ),
      MomentNormalized alternative →
      predictionLoss encoder predictor ≤ predictionLoss alternative otherPredictor)
    (readout : ℝ → ℝ) :
    1 ≤ downstreamRisk encoder readout := by
  have hle := hoptimal backgroundEncoder id background_normalized
  rw [background_prediction_zero] at hle
  have hzero := le_antisymm hle (prediction_loss_nonneg encoder predictor)
  rw [zero_prediction_downstream_exact encoder predictor readout hzero]
  nlinarith [sq_nonneg (readout (encoder (true, false))),
    sq_nonneg (readout (encoder (false, false)))]

theorem task_downstream_zero : downstreamRisk taskEncoder id = 0 := by
  simp [downstreamRisk, risk, taskEncoder, currentTask, current, Function.comp_def]

/-- The representation preferred by the predictive objective is strictly worse
for this downstream task, even allowing arbitrary downstream readouts. -/
theorem objective_and_task_rank_oppositely (predictor readout : ℝ → ℝ) :
    predictionLoss backgroundEncoder id < predictionLoss taskEncoder predictor ∧
      downstreamRisk taskEncoder id < downstreamRisk backgroundEncoder readout := by
  constructor
  · exact (objective_strictly_prefers_background predictor).2.2
  · rw [task_downstream_zero]
    linarith [background_downstream_at_least_one readout]

/-- A fully quantified counterexample: a normalized global optimum of the
predictive objective is strictly worse for the same downstream target than a
different admissible encoder. Therefore global objective optimality plus this
noncollapse constraint does not imply optimal downstream representations. -/
theorem normalized_global_optimum_can_be_task_suboptimal :
    ∃ (encoder : Input → ℝ) (predictor : ℝ → ℝ),
      MomentNormalized encoder ∧
      (∀ (alternative : Input → ℝ) (otherPredictor : ℝ → ℝ),
        MomentNormalized alternative →
        predictionLoss encoder predictor ≤ predictionLoss alternative otherPredictor) ∧
      (∃ alternative : Input → ℝ, MomentNormalized alternative ∧
        downstreamRisk alternative id = 0) ∧
      (∀ readout : ℝ → ℝ, 1 ≤ downstreamRisk encoder readout) := by
  refine ⟨backgroundEncoder, id, background_normalized, ?_, ?_, ?_⟩
  · intro alternative otherPredictor _
    exact background_global_minimum alternative otherPredictor
  · exact ⟨taskEncoder, task_normalized, task_downstream_zero⟩
  · exact background_downstream_at_least_one

end HJepa.Objective
