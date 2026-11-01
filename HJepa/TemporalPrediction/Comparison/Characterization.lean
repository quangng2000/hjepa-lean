import HJepa.TemporalPrediction.Comparison.Optima

namespace HJepa.TemporalPrediction

/-- Joint optimization necessarily fits the head for the learned encoder. -/
theorem global_head_fit (d : Dynamics) (m : Learner) (hm : GlobalMinimum d m) :
    m.coefficient = correlation d m.signal m.nuisance := by
  have hc := hm ⟨m.signal, m.nuisance,
    correlation d m.signal m.nuisance, m.normalized⟩
  change loss d true m.signal m.nuisance m.coefficient ≤
    loss d true m.signal m.nuisance (correlation d m.signal m.nuisance) at hc
  rw [loss_optimal_predictor _ _ _ _ m.normalized] at hc
  have he := le_antisymm hc
    (reducedLoss_le_loss d true m.signal m.nuisance m.coefficient m.normalized)
  exact (loss_eq_reducedLoss_iff _ _ _ _ _ m.normalized).mp he

theorem signal_global_iff (d : Dynamics)
    (h : d.nuisancePersistence ^ 2 < d.signalPersistence ^ 2) (m : Learner) :
    GlobalMinimum d m ↔ taskRisk m = 0 ∧
      m.coefficient = correlation d m.signal m.nuisance := by
  constructor
  · intro hm
    exact ⟨signal_global_task_zero d h m hm, global_head_fit d m hm⟩
  · rintro ⟨hr, hk⟩
    apply (signal_global_value d (le_of_lt h) m).mpr
    rw [objective_signal_form]
    simp [mixingPenalty, headPenalty, hr, hk]

theorem nuisance_global_iff (d : Dynamics)
    (h : d.signalPersistence ^ 2 < d.nuisancePersistence ^ 2) (m : Learner) :
    GlobalMinimum d m ↔ taskRisk m = 1 ∧
      m.coefficient = correlation d m.signal m.nuisance := by
  constructor
  · intro hm
    exact ⟨nuisance_global_task_one d h m hm, global_head_fit d m hm⟩
  · rintro ⟨hr, hk⟩
    apply (nuisance_global_value d (le_of_lt h) m).mpr
    rw [objective_nuisance_form]
    simp [mixingPenalty, headPenalty, hr, hk]

theorem equal_persistence_correlation (d : Dynamics) (m : Learner)
    (h : d.signalPersistence = d.nuisancePersistence) :
    correlation d m.signal m.nuisance = d.signalPersistence := by
  rw [correlation_via_task, h]
  simp

/-- Equal persistence leaves every normalized encoder globally admissible. -/
theorem equal_persistence_global_iff (d : Dynamics)
    (h : d.signalPersistence = d.nuisancePersistence) (m : Learner) :
    GlobalMinimum d m ↔ m.coefficient = d.signalPersistence := by
  constructor
  · intro hm
    exact (global_head_fit d m hm).trans (equal_persistence_correlation d m h)
  · intro hk
    apply (signal_global_value d (by rw [h]) m).mpr
    rw [equal_persistence_form d m h]
    simp [headPenalty, equal_persistence_correlation d m h, hk]

end HJepa.TemporalPrediction
