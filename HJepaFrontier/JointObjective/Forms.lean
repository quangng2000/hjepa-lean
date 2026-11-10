import HJepaFrontier.JointObjective.Model
import Mathlib.Tactic.LinearCombination

/-!
# Queue entry V10: exact forms of the joint objective

Unreviewed Frontier candidate; see `FUZZ_QUEUE.md`.

The joint objective equals the value of a pure encoder, plus the selection gap
times a risk term, plus a nonnegative remainder. Both parts' existing
decompositions are reused; nothing is recomputed from the finite laws.
-/

namespace HJepa.Frontier.JointObjective

theorem signalLearner_masked (p : TemporalPrediction.Problem) :
    (signalLearner p).masked = MaskedPrediction.signalPredictor p.signalScale := rfl

theorem signalLearner_temporal (p : TemporalPrediction.Problem) :
    (signalLearner p).temporal = TemporalPrediction.signalLearner p.dynamics := rfl

theorem nuisanceLearner_masked (p : TemporalPrediction.Problem) :
    (nuisanceLearner p).masked = MaskedPrediction.nuisancePredictor p.nuisanceScale := rfl

theorem nuisanceLearner_temporal (p : TemporalPrediction.Problem) :
    (nuisanceLearner p).temporal = TemporalPrediction.nuisanceLearner p.dynamics := rfl

theorem jointObjective_signalLearner (w : ℝ) (p : TemporalPrediction.Problem) :
    jointObjective w p (signalLearner p) = signalValue w p := by
  have hmasked := MaskedPrediction.objective_signal_axis p.signalScale p.nuisanceScale
  have htemporal := TemporalPrediction.objective_signal_axis p.dynamics
  unfold jointObjective signalValue
  rw [signalLearner_masked, signalLearner_temporal]
  linear_combination w * hmasked + htemporal

theorem jointObjective_nuisanceLearner (w : ℝ) (p : TemporalPrediction.Problem) :
    jointObjective w p (nuisanceLearner p) = nuisanceValue w p := by
  have hmasked := MaskedPrediction.objective_nuisance_axis p.signalScale p.nuisanceScale
  have htemporal := TemporalPrediction.objective_nuisance_axis p.dynamics
  unfold jointObjective nuisanceValue
  rw [nuisanceLearner_masked, nuisanceLearner_temporal]
  linear_combination w * hmasked + htemporal

/-- Signal form: the selection gap multiplies the risk for the signal. -/
theorem jointObjective_signal_form (w : ℝ) (p : TemporalPrediction.Problem)
    (m : JointLearner) :
    jointObjective w p m =
      signalValue w p + selectionGap w p * taskRisk m + slack w p m := by
  have hmasked :=
    MaskedPrediction.objective_signal_form p.signalScale p.nuisanceScale m.masked
  have htemporal := TemporalPrediction.objective_signal_form p.dynamics m.temporal
  have htask : MaskedPrediction.taskRisk m.masked =
      TemporalPrediction.taskRisk m.temporal := rfl
  rw [htask] at hmasked
  unfold jointObjective signalValue selectionGap taskRisk slack
  linear_combination w * hmasked + htemporal

/-- Nuisance form: the negated gap multiplies the room left below risk one. -/
theorem jointObjective_nuisance_form (w : ℝ) (p : TemporalPrediction.Problem)
    (m : JointLearner) :
    jointObjective w p m =
      nuisanceValue w p + (-selectionGap w p) * (1 - taskRisk m) + slack w p m := by
  rw [jointObjective_signal_form]
  unfold signalValue nuisanceValue selectionGap
  ring

theorem taskRisk_bounds (m : JointLearner) : 0 ≤ taskRisk m ∧ taskRisk m ≤ 1 :=
  TemporalPrediction.taskRisk_bounds m.temporal

theorem slack_nonneg (w : ℝ) (p : TemporalPrediction.Problem) (m : JointLearner)
    (hw : 0 ≤ w) : 0 ≤ slack w p m := by
  unfold slack
  exact add_nonneg
    (add_nonneg (mul_nonneg hw (add_nonneg (sq_nonneg _) (sq_nonneg _)))
      (TemporalPrediction.mixingPenalty_nonneg p.dynamics m.temporal))
    (TemporalPrediction.headPenalty_nonneg p.dynamics m.temporal)

end HJepa.Frontier.JointObjective
