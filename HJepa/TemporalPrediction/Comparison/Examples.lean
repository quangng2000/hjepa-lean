import HJepa.TemporalPrediction.Comparison.Tolerance

namespace HJepa.TemporalPrediction

noncomputable def signalPersistent : Dynamics := ⟨3 / 4, 1 / 4,
  by norm_num, by norm_num, by norm_num, by norm_num⟩
noncomputable def nuisancePersistent : Dynamics := ⟨1 / 4, 3 / 4,
  by norm_num, by norm_num, by norm_num, by norm_num⟩
noncomputable def arWinningProblem : Problem := ⟨1, 2, signalPersistent⟩
noncomputable def rawWinningProblem : Problem := ⟨2, 1, nuisancePersistent⟩

/-- Both objectives attain their minima, but only temporal prediction retains S. -/
theorem autoregressive_wins_example :
    ∃ raw ar, RawOptimal arWinningProblem raw ∧ AROptimal arWinningProblem ar ∧
      HJepa.MaskedPrediction.taskRisk raw = 1 ∧ taskRisk ar = 0 := by
  refine ⟨HJepa.MaskedPrediction.nuisancePredictor 2, signalLearner signalPersistent,
    ?_, ?_, ?_, ?_⟩
  · exact HJepa.MaskedPrediction.nuisance_axis_global 1 2 (by norm_num)
  · exact signal_axis_global signalPersistent (by norm_num [signalPersistent])
  · norm_num [HJepa.MaskedPrediction.taskRisk_eq, HJepa.MaskedPrediction.nuisancePredictor]
  · norm_num [taskRisk_eq, signalLearner]

/-- A persistent nuisance reverses the comparison on the same downstream task. -/
theorem reconstruction_wins_example :
    ∃ raw ar, RawOptimal rawWinningProblem raw ∧ AROptimal rawWinningProblem ar ∧
      HJepa.MaskedPrediction.taskRisk raw = 0 ∧ taskRisk ar = 1 := by
  refine ⟨HJepa.MaskedPrediction.signalPredictor 2, nuisanceLearner nuisancePersistent,
    ?_, ?_, ?_, ?_⟩
  · exact HJepa.MaskedPrediction.signal_axis_global 2 1 (by norm_num)
  · exact nuisance_axis_global nuisancePersistent (by norm_num [nuisancePersistent])
  · norm_num [HJepa.MaskedPrediction.taskRisk_eq, HJepa.MaskedPrediction.signalPredictor]
  · norm_num [taskRisk_eq, nuisanceLearner]

/-- This family admits neither unconditional weak-dominance claim. -/
theorem neither_method_dominates :
    (∃ p raw ar, RawOptimal p raw ∧ AROptimal p ar ∧
      taskRisk ar < HJepa.MaskedPrediction.taskRisk raw) ∧
    (∃ p raw ar, RawOptimal p raw ∧ AROptimal p ar ∧
      HJepa.MaskedPrediction.taskRisk raw < taskRisk ar) := by
  obtain ⟨r₁, a₁, hr₁, ha₁, er₁, ea₁⟩ := autoregressive_wins_example
  obtain ⟨r₂, a₂, hr₂, ha₂, er₂, ea₂⟩ := reconstruction_wins_example
  constructor
  · exact ⟨arWinningProblem, r₁, a₁, hr₁, ha₁, by rw [er₁, ea₁]; norm_num⟩
  · exact ⟨rawWinningProblem, r₂, a₂, hr₂, ha₂, by rw [er₂, ea₂]; norm_num⟩

/-- Positive suboptimality allowances still certify a strict task advantage. -/
theorem approximate_ar_wins_example (raw : HJepa.MaskedPrediction.Predictor) (ar : Learner)
    (hr : HJepa.MaskedPrediction.objective 1 2 raw ≤ 2)
    (ha : objective signalPersistent ar ≤ 11 / 16) :
    taskRisk ar < HJepa.MaskedPrediction.taskRisk raw := by
  apply approximate_autoregressive_better arWinningProblem raw ar 1 (1 / 4)
  · norm_num [arWinningProblem]
  · norm_num [arWinningProblem, signalPersistent]
  · norm_num [arWinningProblem]; exact hr
  · norm_num [arWinningProblem, signalPersistent] at ha ⊢; exact ha
  · norm_num [arWinningProblem, signalPersistent]

end HJepa.TemporalPrediction
