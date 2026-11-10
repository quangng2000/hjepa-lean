import HJepaFrontier.JointObjective.Forms

/-!
# Queue entry V10: the sign of the selection gap decides every joint optimum

Unreviewed Frontier candidate; see `FUZZ_QUEUE.md`.

For a nonnegative masked weight `w`, every global optimum of the joint
objective keeps the signal when

  `w * (signalScale² - nuisanceScale²) + (ρ_S² - ρ_N²) > 0`

and keeps the nuisance when that quantity is negative. At zero the optimum is
not identified. The comparison is over the shared encoder and all three heads.
-/

namespace HJepa.Frontier.JointObjective

/-- A global optimum exists: the signal-only encoder, when the gap is nonnegative. -/
theorem signalLearner_global (w : ℝ) (p : TemporalPrediction.Problem) (hw : 0 ≤ w)
    (hgap : 0 ≤ selectionGap w p) : GlobalMinimum w p (signalLearner p) := by
  intro other
  have hvalue := jointObjective_signalLearner w p
  have hform := jointObjective_signal_form w p other
  have hslack := slack_nonneg w p other hw
  have hproduct := mul_nonneg hgap (taskRisk_bounds other).1
  linarith

/-- A global optimum exists: the nuisance-only encoder, when the gap is nonpositive. -/
theorem nuisanceLearner_global (w : ℝ) (p : TemporalPrediction.Problem) (hw : 0 ≤ w)
    (hgap : selectionGap w p ≤ 0) : GlobalMinimum w p (nuisanceLearner p) := by
  intro other
  have hvalue := jointObjective_nuisanceLearner w p
  have hform := jointObjective_nuisance_form w p other
  have hslack := slack_nonneg w p other hw
  have hproduct := mul_nonneg (neg_nonneg.mpr hgap)
    (sub_nonneg.mpr (taskRisk_bounds other).2)
  linarith

/-- A positive selection gap forces every joint optimum onto the signal. -/
theorem positive_gap_task_zero (w : ℝ) (p : TemporalPrediction.Problem) (hw : 0 ≤ w)
    (hgap : 0 < selectionGap w p) (m : JointLearner) (hm : GlobalMinimum w p m) :
    taskRisk m = 0 := by
  have hupper := hm (signalLearner p)
  have hvalue := jointObjective_signalLearner w p
  have hform := jointObjective_signal_form w p m
  have hslack := slack_nonneg w p m hw
  have hrisk := (taskRisk_bounds m).1
  have hproduct : selectionGap w p * taskRisk m ≤ 0 := by linarith
  have hzero : selectionGap w p * taskRisk m = 0 :=
    le_antisymm hproduct (mul_nonneg hgap.le hrisk)
  exact (mul_eq_zero.mp hzero).resolve_left hgap.ne'

/-- A negative selection gap forces every joint optimum onto the nuisance. -/
theorem negative_gap_task_one (w : ℝ) (p : TemporalPrediction.Problem) (hw : 0 ≤ w)
    (hgap : selectionGap w p < 0) (m : JointLearner) (hm : GlobalMinimum w p m) :
    taskRisk m = 1 := by
  have hupper := hm (nuisanceLearner p)
  have hvalue := jointObjective_nuisanceLearner w p
  have hform := jointObjective_nuisance_form w p m
  have hslack := slack_nonneg w p m hw
  have hroom : 0 ≤ 1 - taskRisk m := sub_nonneg.mpr (taskRisk_bounds m).2
  have hflip : 0 < -selectionGap w p := neg_pos.mpr hgap
  have hproduct : (-selectionGap w p) * (1 - taskRisk m) ≤ 0 := by linarith
  have hzero : (-selectionGap w p) * (1 - taskRisk m) = 0 :=
    le_antisymm hproduct (mul_nonneg hflip.le hroom)
  have hfull := (mul_eq_zero.mp hzero).resolve_left hflip.ne'
  linarith

/-- Away from a zero gap, the signal is kept exactly when the gap is positive. -/
theorem task_zero_iff_positive_gap (w : ℝ) (p : TemporalPrediction.Problem)
    (hw : 0 ≤ w) (hgap : selectionGap w p ≠ 0) (m : JointLearner)
    (hm : GlobalMinimum w p m) : taskRisk m = 0 ↔ 0 < selectionGap w p := by
  rcases lt_or_gt_of_ne hgap with hneg | hpos
  · have hone := negative_gap_task_one w p hw hneg m hm
    constructor
    · intro hzero
      linarith
    · intro hpos
      linarith
  · exact ⟨fun _ => hpos, fun _ => positive_gap_task_zero w p hw hpos m hm⟩

end HJepa.Frontier.JointObjective
