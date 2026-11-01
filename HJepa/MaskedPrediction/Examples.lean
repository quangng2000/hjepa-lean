import HJepa.MaskedPrediction.Tolerance
import HJepa.MaskedPrediction.Characterization

namespace HJepa.MaskedPrediction

/-- The frozen target map has nonzero scales in both separating examples. -/
def hiddenWinningTargets : Targets := ⟨1, 2, 3, 1⟩
def rawWinningTargets : Targets := ⟨2, 1, 1, 3⟩

theorem hidden_wins_example :
    ∃ raw hidden : Predictor,
      RawOptimal hiddenWinningTargets raw ∧ HiddenOptimal hiddenWinningTargets hidden ∧
      taskRisk raw = 1 ∧ taskRisk hidden = 0 := by
  refine ⟨nuisancePredictor 2, signalPredictor 3, ?_, ?_, ?_, ?_⟩
  · exact nuisance_axis_global 1 2 (by norm_num)
  · simpa [HiddenOptimal, hiddenWinningTargets, Targets.hiddenSignal,
      Targets.hiddenNuisance] using signal_axis_global (3 : ℝ) 2 (by norm_num)
  · norm_num [taskRisk_eq, nuisancePredictor]
  · norm_num [taskRisk_eq, signalPredictor]

theorem raw_wins_example :
    ∃ raw hidden : Predictor,
      RawOptimal rawWinningTargets raw ∧ HiddenOptimal rawWinningTargets hidden ∧
      taskRisk raw = 0 ∧ taskRisk hidden = 1 := by
  refine ⟨signalPredictor 2, nuisancePredictor 3, ?_, ?_, ?_, ?_⟩
  · exact signal_axis_global 2 1 (by norm_num)
  · simpa [HiddenOptimal, rawWinningTargets, Targets.hiddenSignal,
      Targets.hiddenNuisance] using nuisance_axis_global (2 : ℝ) 3 (by norm_num)
  · norm_num [taskRisk_eq, signalPredictor]
  · norm_num [taskRisk_eq, nuisancePredictor]

/-- Identical optimum objective values can leave task quality unidentified. -/
theorem equal_energy_ambiguous :
    ∃ good bad : Predictor, GlobalMinimum 1 1 good ∧ GlobalMinimum 1 1 bad ∧
      taskRisk good = 0 ∧ taskRisk bad = 1 := by
  refine ⟨signalPredictor 1, nuisancePredictor 1, ?_, ?_, ?_, ?_⟩
  · exact signal_axis_global 1 1 (by norm_num)
  · exact nuisance_axis_global 1 1 (by norm_num)
  · norm_num [taskRisk_eq, signalPredictor]
  · norm_num [taskRisk_eq, nuisancePredictor]

/-- A nonzero optimization tolerance still permits a strict comparison. -/
theorem approximate_hidden_wins_example (raw hidden : Predictor)
    (hr : objective 1 2 raw ≤ 2) (hh : objective 3 2 hidden ≤ 5) :
    taskRisk hidden < taskRisk raw := by
  apply approximate_hidden_better hiddenWinningTargets raw hidden 1 1
  · norm_num [hiddenWinningTargets]
  · norm_num [hiddenWinningTargets, Targets.hiddenSignal, Targets.hiddenNuisance]
  · norm_num [hiddenWinningTargets]
    exact hr
  · norm_num [hiddenWinningTargets, Targets.hiddenSignal, Targets.hiddenNuisance]
    exact hh
  · norm_num [hiddenWinningTargets, Targets.hiddenSignal, Targets.hiddenNuisance]

/-- Neither method weakly dominates the other across this same model family. -/
theorem neither_method_dominates :
    (∃ t raw hidden, RawOptimal t raw ∧ HiddenOptimal t hidden ∧ taskRisk hidden < taskRisk raw) ∧
    (∃ t raw hidden, RawOptimal t raw ∧ HiddenOptimal t hidden ∧ taskRisk raw < taskRisk hidden) := by
  obtain ⟨r₁, h₁, hr₁, hh₁, er₁, eh₁⟩ := hidden_wins_example
  obtain ⟨r₂, h₂, hr₂, hh₂, er₂, eh₂⟩ := raw_wins_example
  constructor
  · exact ⟨hiddenWinningTargets, r₁, h₁, hr₁, hh₁, by rw [er₁, eh₁]; norm_num⟩
  · exact ⟨rawWinningTargets, r₂, h₂, hr₂, hh₂, by rw [er₂, eh₂]; norm_num⟩

end HJepa.MaskedPrediction
