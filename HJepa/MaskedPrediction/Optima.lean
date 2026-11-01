import HJepa.MaskedPrediction.Predictor

namespace HJepa.MaskedPrediction

theorem signal_floor (u v : ℝ) (h : v ^ 2 ≤ u ^ 2) (m : Predictor) :
    v ^ 2 ≤ objective u v m := by
  rw [objective_signal_form]
  have hg := mul_nonneg (sub_nonneg.mpr h) (taskRisk_bounds m).1
  nlinarith [sq_nonneg (m.firstHead - u * m.signal),
    sq_nonneg (m.secondHead - v * m.nuisance)]

theorem nuisance_floor (u v : ℝ) (h : u ^ 2 ≤ v ^ 2) (m : Predictor) :
    u ^ 2 ≤ objective u v m := by
  rw [objective_nuisance_form]
  have hg := mul_nonneg (sub_nonneg.mpr h) (sub_nonneg.mpr (taskRisk_bounds m).2)
  nlinarith [sq_nonneg (m.firstHead - u * m.signal),
    sq_nonneg (m.secondHead - v * m.nuisance)]

theorem signal_axis_global (u v : ℝ) (h : v ^ 2 ≤ u ^ 2) :
    GlobalMinimum u v (signalPredictor u) := by
  intro other
  rw [objective_signal_axis]
  exact signal_floor u v h other

theorem nuisance_axis_global (u v : ℝ) (h : u ^ 2 ≤ v ^ 2) :
    GlobalMinimum u v (nuisancePredictor v) := by
  intro other
  rw [objective_nuisance_axis]
  exact nuisance_floor u v h other

/-- The whole joint objective is optimized, not just heads on a fixed encoder. -/
theorem signal_global_value (u v : ℝ) (h : v ^ 2 ≤ u ^ 2) (m : Predictor) :
    GlobalMinimum u v m ↔ objective u v m = v ^ 2 := by
  constructor
  · intro hm
    have hc := hm (signalPredictor u)
    rw [objective_signal_axis] at hc
    exact le_antisymm hc (signal_floor u v h m)
  · intro hm other
    rw [hm]
    exact signal_floor u v h other

theorem nuisance_global_value (u v : ℝ) (h : u ^ 2 ≤ v ^ 2) (m : Predictor) :
    GlobalMinimum u v m ↔ objective u v m = u ^ 2 := by
  constructor
  · intro hm
    have hc := hm (nuisancePredictor v)
    rw [objective_nuisance_axis] at hc
    exact le_antisymm hc (nuisance_floor u v h m)
  · intro hm other
    rw [hm]
    exact nuisance_floor u v h other

theorem signal_global_task_zero (u v : ℝ) (h : v ^ 2 < u ^ 2)
    (m : Predictor) (hm : GlobalMinimum u v m) : taskRisk m = 0 := by
  have he := (signal_global_value u v (le_of_lt h) m).mp hm
  rw [objective_signal_form] at he
  have hp := mul_nonneg (sub_nonneg.mpr (le_of_lt h)) (taskRisk_bounds m).1
  have hz : (u ^ 2 - v ^ 2) * taskRisk m = 0 := by
    nlinarith [sq_nonneg (m.firstHead - u * m.signal),
      sq_nonneg (m.secondHead - v * m.nuisance)]
  exact (mul_eq_zero.mp hz).resolve_left (ne_of_gt (sub_pos.mpr h))

theorem nuisance_global_task_one (u v : ℝ) (h : u ^ 2 < v ^ 2)
    (m : Predictor) (hm : GlobalMinimum u v m) : taskRisk m = 1 := by
  have he := (nuisance_global_value u v (le_of_lt h) m).mp hm
  rw [objective_nuisance_form] at he
  have hp := mul_nonneg (sub_nonneg.mpr (le_of_lt h))
    (sub_nonneg.mpr (taskRisk_bounds m).2)
  have hz : (v ^ 2 - u ^ 2) * (1 - taskRisk m) = 0 := by
    nlinarith [sq_nonneg (m.firstHead - u * m.signal),
      sq_nonneg (m.secondHead - v * m.nuisance)]
  have ht := (mul_eq_zero.mp hz).resolve_left (ne_of_gt (sub_pos.mpr h))
  linarith

end HJepa.MaskedPrediction
