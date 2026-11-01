import HJepa.TemporalPrediction.Rollout.Optima

namespace HJepa.TemporalPrediction

/-- A persistent signal and completely refreshed nuisance isolate the horizon mismatch. -/
noncomputable def twoStepExampleDynamics : Dynamics :=
  ⟨1, 0, by norm_num, by norm_num, by norm_num, by norm_num⟩

theorem mixed_example_normalized : Normalized (3 / 5 : ℝ) (4 / 5 : ℝ) := by
  norm_num [Normalized, HJepa.MaskedPrediction.Normalized]

/-- The optimal one-step coefficient is `9/25` for this fixed mixed encoder. -/
theorem mixed_one_step_coefficient :
    correlation twoStepExampleDynamics (3 / 5) (4 / 5) = 9 / 25 := by
  norm_num [correlation, twoStepExampleDynamics]

theorem mixed_one_step_optimal (action : Bool) :
    loss twoStepExampleDynamics action (3 / 5) (4 / 5) (9 / 25) =
      reducedLoss twoStepExampleDynamics (3 / 5) (4 / 5) := by
  rw [← mixed_one_step_coefficient]
  exact loss_optimal_predictor _ action _ _ mixed_example_normalized

/-- The one-step optimum is strictly worse at two steps than another recurrent head. -/
theorem mixed_two_step_improvement (first second : Bool) :
    rolloutRisk twoStepExampleDynamics [first, second] (3 / 5) (4 / 5) (3 / 5) <
      rolloutRisk twoStepExampleDynamics [first, second] (3 / 5) (4 / 5) (9 / 25) := by
  norm_num [rolloutRisk_exact, twoStepExampleDynamics]

/-- Here the attainable best two-step coefficient is `(3/5)^2 = 9/25`. -/
theorem mixed_two_step_optimal (first second : Bool) (k : ℝ) :
    rolloutRisk twoStepExampleDynamics [first, second] (3 / 5) (4 / 5) (3 / 5) ≤
      rolloutRisk twoStepExampleDynamics [first, second] (3 / 5) (4 / 5) k := by
  rw [rolloutRisk_complete_square _ _ _ _ _ mixed_example_normalized,
    rolloutRisk_complete_square _ _ _ _ _ mixed_example_normalized]
  norm_num [horizonCorrelation, twoStepExampleDynamics]
  nlinarith [sq_nonneg (k ^ 2 - (9 / 25 : ℝ))]

/-- Concrete population values make the mismatch directly checkable. -/
theorem mixed_two_step_risks (first second : Bool) :
    rolloutRisk twoStepExampleDynamics [first, second] (3 / 5) (4 / 5) (3 / 5) =
        544 / 625 ∧
      rolloutRisk twoStepExampleDynamics [first, second] (3 / 5) (4 / 5) (9 / 25) =
        360736 / 390625 := by
  norm_num [rolloutRisk_exact, twoStepExampleDynamics]

end HJepa.TemporalPrediction
