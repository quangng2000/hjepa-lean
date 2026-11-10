import HJepa.TemporalPrediction.Comparison.Margins

namespace HJepa.TemporalPrediction

theorem signal_axis_global (d : Dynamics)
    (h : d.nuisancePersistence ^ 2 ≤ d.signalPersistence ^ 2) :
    GlobalMinimum d (signalLearner d) := by
  intro other
  rw [objective_signal_axis]
  exact signal_floor d other h

theorem nuisance_axis_global (d : Dynamics)
    (h : d.signalPersistence ^ 2 ≤ d.nuisancePersistence ^ 2) :
    GlobalMinimum d (nuisanceLearner d) := by
  intro other
  rw [objective_nuisance_axis]
  exact nuisance_floor d other h

/-- The lower floor is attained over the joint encoder and predictor class. -/
theorem signal_global_value (d : Dynamics)
    (h : d.nuisancePersistence ^ 2 ≤ d.signalPersistence ^ 2) (m : Learner) :
    GlobalMinimum d m ↔ objective d m = 1 - d.signalPersistence ^ 2 := by
  constructor
  · intro hm
    have hc := hm (signalLearner d)
    rw [objective_signal_axis] at hc
    exact le_antisymm hc (signal_floor d m h)
  · intro hm other
    rw [hm]
    exact signal_floor d other h

theorem nuisance_global_value (d : Dynamics)
    (h : d.signalPersistence ^ 2 ≤ d.nuisancePersistence ^ 2) (m : Learner) :
    GlobalMinimum d m ↔ objective d m = 1 - d.nuisancePersistence ^ 2 := by
  constructor
  · intro hm
    have hc := hm (nuisanceLearner d)
    rw [objective_nuisance_axis] at hc
    exact le_antisymm hc (nuisance_floor d m h)
  · intro hm other
    rw [hm]
    exact nuisance_floor d other h

theorem signal_global_task_zero (d : Dynamics)
    (h : d.nuisancePersistence ^ 2 < d.signalPersistence ^ 2)
    (m : Learner) (hm : GlobalMinimum d m) : taskRisk m = 0 := by
  have he := (signal_global_value d (le_of_lt h) m).mp hm
  rw [objective_signal_form] at he
  have hp := mul_nonneg (sub_nonneg.mpr (le_of_lt h)) (taskRisk_bounds m).1
  have hz : (d.signalPersistence ^ 2 - d.nuisancePersistence ^ 2) * taskRisk m = 0 := by
    linarith [mixingPenalty_nonneg d m, headPenalty_nonneg d m]
  exact (mul_eq_zero.mp hz).resolve_left (ne_of_gt (sub_pos.mpr h))

theorem nuisance_global_task_one (d : Dynamics)
    (h : d.signalPersistence ^ 2 < d.nuisancePersistence ^ 2)
    (m : Learner) (hm : GlobalMinimum d m) : taskRisk m = 1 := by
  have he := (nuisance_global_value d (le_of_lt h) m).mp hm
  rw [objective_nuisance_form] at he
  have hp := mul_nonneg (sub_nonneg.mpr (le_of_lt h))
    (sub_nonneg.mpr (taskRisk_bounds m).2)
  have hz : (d.nuisancePersistence ^ 2 - d.signalPersistence ^ 2) * (1 - taskRisk m) = 0 := by
    linarith [mixingPenalty_nonneg d m, headPenalty_nonneg d m]
  have ht := (mul_eq_zero.mp hz).resolve_left (ne_of_gt (sub_pos.mpr h))
  linarith

end HJepa.TemporalPrediction
