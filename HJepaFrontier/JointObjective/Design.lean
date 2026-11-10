import HJepaFrontier.JointObjective.Optima

/-!
# Queue entry V10: what the selection gap says about the masked weight

Unreviewed Frontier candidate; see `FUZZ_QUEUE.md`.

Four readings of the gap `w * (s² - n²) + (ρ_S² - ρ_N²)`, where `s` and `n` are
the signal and nuisance scales in the masked target:

* without a masked term, persistence alone decides;
* a masked term can outweigh a more persistent nuisance, but only above a weight
  threshold and only if the signal has more masked-target energy;
* if the nuisance has at least as much masked-target energy, no weight helps;
* under good views a masked term is harmless only below a weight bound when the
  nuisance has more masked-target energy.

These are statements about the two-sign model, not about a trained network.
-/

namespace HJepa.Frontier.JointObjective

/-- With no masked term the gap is the persistence gap of the temporal objective. -/
theorem selectionGap_zero_weight (p : TemporalPrediction.Problem) :
    selectionGap 0 p =
      p.dynamics.signalPersistence ^ 2 - p.dynamics.nuisancePersistence ^ 2 := by
  unfold selectionGap
  ring

/-- The masked term rescues the signal from a more persistent nuisance once its
weighted energy advantage exceeds the nuisance's persistence advantage. -/
theorem masked_weight_rescues (w : ℝ) (p : TemporalPrediction.Problem) (hw : 0 ≤ w)
    (hweight : p.dynamics.nuisancePersistence ^ 2 - p.dynamics.signalPersistence ^ 2 <
      w * (p.signalScale ^ 2 - p.nuisanceScale ^ 2))
    (m : JointLearner) (hm : GlobalMinimum w p m) : taskRisk m = 0 := by
  have hgap : 0 < selectionGap w p := by
    unfold selectionGap
    linarith
  exact positive_gap_task_zero w p hw hgap m hm

/-- If the nuisance has at least the signal's masked-target energy and is more
persistent, every optimum keeps the nuisance at every masked weight. -/
theorem masked_weight_cannot_rescue (w : ℝ) (p : TemporalPrediction.Problem)
    (hw : 0 ≤ w) (henergy : p.signalScale ^ 2 ≤ p.nuisanceScale ^ 2)
    (hpersist : p.dynamics.signalPersistence ^ 2 < p.dynamics.nuisancePersistence ^ 2)
    (m : JointLearner) (hm : GlobalMinimum w p m) : taskRisk m = 1 := by
  have hproduct : w * (p.signalScale ^ 2 - p.nuisanceScale ^ 2) ≤ 0 :=
    mul_nonpos_of_nonneg_of_nonpos hw (sub_nonpos.mpr henergy)
  have hgap : selectionGap w p < 0 := by
    unfold selectionGap
    linarith
  exact negative_gap_task_one w p hw hgap m hm

/-- Under good views (nuisance persistence zero) the signal survives a masked
term whose weighted nuisance-energy advantage stays below `ρ_S²`. -/
theorem good_views_tolerate_masked_weight (w : ℝ) (p : TemporalPrediction.Problem)
    (hw : 0 ≤ w) (hnuisance : p.dynamics.nuisancePersistence = 0)
    (hweight : w * (p.nuisanceScale ^ 2 - p.signalScale ^ 2) <
      p.dynamics.signalPersistence ^ 2)
    (m : JointLearner) (hm : GlobalMinimum w p m) : taskRisk m = 0 := by
  have hzero : p.dynamics.nuisancePersistence ^ 2 = 0 := by
    rw [hnuisance, zero_pow (by norm_num : (2 : ℕ) ≠ 0)]
  have hgap : 0 < selectionGap w p := by
    unfold selectionGap
    linarith
  exact positive_gap_task_zero w p hw hgap m hm

/-- Under good views a masked term that is too heavy hands the encoder back to a
nuisance with more masked-target energy. -/
theorem heavy_masked_weight_breaks_good_views (w : ℝ) (p : TemporalPrediction.Problem)
    (hw : 0 ≤ w) (hnuisance : p.dynamics.nuisancePersistence = 0)
    (hweight : p.dynamics.signalPersistence ^ 2 <
      w * (p.nuisanceScale ^ 2 - p.signalScale ^ 2))
    (m : JointLearner) (hm : GlobalMinimum w p m) : taskRisk m = 1 := by
  have hzero : p.dynamics.nuisancePersistence ^ 2 = 0 := by
    rw [hnuisance, zero_pow (by norm_num : (2 : ℕ) ≠ 0)]
  have hgap : selectionGap w p < 0 := by
    unfold selectionGap
    linarith
  exact negative_gap_task_one w p hw hgap m hm

end HJepa.Frontier.JointObjective
