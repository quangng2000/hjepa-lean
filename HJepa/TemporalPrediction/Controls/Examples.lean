import HJepa.TemporalPrediction.Controls.Comparison

namespace HJepa.TemporalPrediction.Controls

noncomputable def signalPersistent : Dynamics where
  signalPersistence := 3 / 4
  nuisancePersistence := 1 / 4
  signal_nonneg := by norm_num
  signal_le_one := by norm_num
  nuisance_nonneg := by norm_num
  nuisance_le_one := by norm_num

noncomputable def nuisancePersistent : Dynamics where
  signalPersistence := 1 / 4
  nuisancePersistence := 3 / 4
  signal_nonneg := by norm_num
  signal_le_one := by norm_num
  nuisance_nonneg := by norm_num
  nuisance_le_one := by norm_num

/-- With the same forecast horizon, learned-latent prediction can still win. -/
theorem futureLatent_wins_example (action : Bool) :
    ∃ raw : MaskedPrediction.Predictor, ∃ ar : Learner,
      FutureRawGlobal signalPersistent action 1 4 raw ∧
      GlobalMinimum signalPersistent ar ∧
      MaskedPrediction.taskRisk raw = 1 ∧ taskRisk ar = 0 := by
  refine ⟨MaskedPrediction.nuisancePredictor 1, signalLearner signalPersistent,
    ?_, ?_, ?_, ?_⟩
  · apply (futureRawGlobal_iff _ _ _ _ _).mpr
    norm_num [signalPersistent]
    exact MaskedPrediction.nuisance_axis_global (3 / 4) 1 (by norm_num)
  · apply signal_axis_global
    norm_num [signalPersistent]
  · norm_num [MaskedPrediction.taskRisk_eq, MaskedPrediction.nuisancePredictor]
  · norm_num [taskRisk_eq, signalLearner]

/-- Reversing persistence and raw amplitude gives the opposite strict winner. -/
theorem futureRaw_wins_example (action : Bool) :
    ∃ raw : MaskedPrediction.Predictor, ∃ ar : Learner,
      FutureRawGlobal nuisancePersistent action 4 1 raw ∧
      GlobalMinimum nuisancePersistent ar ∧
      MaskedPrediction.taskRisk raw = 0 ∧ taskRisk ar = 1 := by
  refine ⟨MaskedPrediction.signalPredictor 1, nuisanceLearner nuisancePersistent,
    ?_, ?_, ?_, ?_⟩
  · apply (futureRawGlobal_iff _ _ _ _ _).mpr
    norm_num [nuisancePersistent]
    exact MaskedPrediction.signal_axis_global 1 (3 / 4) (by norm_num)
  · apply nuisance_axis_global
    norm_num [nuisancePersistent]
  · norm_num [MaskedPrediction.taskRisk_eq, MaskedPrediction.signalPredictor]
  · norm_num [taskRisk_eq, nuisanceLearner]

end HJepa.TemporalPrediction.Controls
