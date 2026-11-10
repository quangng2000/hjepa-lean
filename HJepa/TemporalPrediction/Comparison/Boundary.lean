import HJepa.TemporalPrediction.Comparison.Characterization

namespace HJepa.TemporalPrediction

/-- Equal persistence can hide a maximal downstream-quality difference. -/
theorem equal_persistence_ambiguous (d : Dynamics)
    (h : d.signalPersistence = d.nuisancePersistence) :
    ∃ good bad, GlobalMinimum d good ∧ GlobalMinimum d bad ∧
      taskRisk good = 0 ∧ taskRisk bad = 1 := by
  refine ⟨signalLearner d, nuisanceLearner d, ?_, ?_, ?_, ?_⟩
  · exact signal_axis_global d (by rw [h])
  · exact nuisance_axis_global d (by rw [h])
  · norm_num [taskRisk_eq, signalLearner]
  · norm_num [taskRisk_eq, nuisanceLearner]

end HJepa.TemporalPrediction
