import HJepa.TemporalPrediction.Comparison.Learner

namespace HJepa.TemporalPrediction

noncomputable def mixingPenalty (d : Dynamics) (m : Learner) : ℝ :=
  (d.signalPersistence - d.nuisancePersistence) ^ 2 * taskRisk m * (1 - taskRisk m)

noncomputable def headPenalty (d : Dynamics) (m : Learner) : ℝ :=
  (m.coefficient - correlation d m.signal m.nuisance) ^ 2

theorem mixingPenalty_nonneg (d : Dynamics) (m : Learner) : 0 ≤ mixingPenalty d m :=
  mul_nonneg (mul_nonneg (sq_nonneg _) (taskRisk_bounds m).1)
    (sub_nonneg.mpr (taskRisk_bounds m).2)

theorem headPenalty_nonneg (d : Dynamics) (m : Learner) : 0 ≤ headPenalty d m :=
  sq_nonneg _

/-- The additional mixing penalty distinguishes temporal learning from rescaling. -/
theorem objective_signal_form (d : Dynamics) (m : Learner) :
    objective d m = 1 - d.signalPersistence ^ 2 +
      (d.signalPersistence ^ 2 - d.nuisancePersistence ^ 2) * taskRisk m +
      mixingPenalty d m + headPenalty d m := by
  simp only [objective, loss_complete_square _ _ _ _ _ m.normalized,
    reducedLoss, mixingPenalty, headPenalty, correlation_via_task]
  ring

theorem objective_nuisance_form (d : Dynamics) (m : Learner) :
    objective d m = 1 - d.nuisancePersistence ^ 2 +
      (d.nuisancePersistence ^ 2 - d.signalPersistence ^ 2) * (1 - taskRisk m) +
      mixingPenalty d m + headPenalty d m := by
  rw [objective_signal_form]
  ring

theorem signal_floor (d : Dynamics) (m : Learner)
    (h : d.nuisancePersistence ^ 2 ≤ d.signalPersistence ^ 2) :
    1 - d.signalPersistence ^ 2 ≤ objective d m := by
  rw [objective_signal_form]
  have hg := mul_nonneg (sub_nonneg.mpr h) (taskRisk_bounds m).1
  linarith [mixingPenalty_nonneg d m, headPenalty_nonneg d m]

theorem nuisance_floor (d : Dynamics) (m : Learner)
    (h : d.signalPersistence ^ 2 ≤ d.nuisancePersistence ^ 2) :
    1 - d.nuisancePersistence ^ 2 ≤ objective d m := by
  rw [objective_nuisance_form]
  have hg := mul_nonneg (sub_nonneg.mpr h) (sub_nonneg.mpr (taskRisk_bounds m).2)
  linarith [mixingPenalty_nonneg d m, headPenalty_nonneg d m]

theorem equal_persistence_form (d : Dynamics) (m : Learner)
    (h : d.signalPersistence = d.nuisancePersistence) :
    objective d m = 1 - d.signalPersistence ^ 2 + headPenalty d m := by
  rw [objective_signal_form]
  simp [mixingPenalty, h]

end HJepa.TemporalPrediction
