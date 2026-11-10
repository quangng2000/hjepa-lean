import HJepa.TokenAggregation.Examples.Population

namespace HJepa.TokenAggregation.Examples

open HJepa.Abstraction HJepa.MaskedPrediction

/-- Two collisions at zero retain an irreducible error for any readout. -/
theorem stable_selected_exact (h : ℝ → ℝ) :
    selectedRisk stablePair h = 1 / 2 + (h 0) ^ 2 / 2 +
      ((1 - h 2) ^ 2 + (-1 - h (-2)) ^ 2) / 4 := by
  norm_num [selectedRisk, stablePair, risk, fairInput, taskSign, nuisanceSign,
    signedBit, Fintype.sum_prod_type, Fintype.sum_bool]
  ring

theorem stable_selected_lower_bound (h : ℝ → ℝ) :
    1 / 2 ≤ selectedRisk stablePair h := by
  rw [stable_selected_exact]
  nlinarith [sq_nonneg (h 0), sq_nonneg (1 - h 2), sq_nonneg (-1 - h (-2))]

theorem stable_selected_attained :
    selectedRisk stablePair (fun x => x / 2) = 1 / 2 := by
  rw [stable_selected_exact]
  norm_num

theorem stable_average_zero : averagedRisk stablePair id = 0 := by
  norm_num [averagedRisk, stablePair, risk, fairInput, taskSign, nuisanceSign,
    signedBit, Fintype.sum_prod_type, Fintype.sum_bool]

/-- Averaging wins even against unrestricted nonlinear selected-token readouts. -/
theorem stable_average_strictly_better (h : ℝ → ℝ) :
    averagedRisk stablePair id < selectedRisk stablePair h := by
  rw [stable_average_zero]
  linarith [stable_selected_lower_bound h]

/-- A deterministic next encoder cannot distinguish the collided input tokens. -/
theorem stable_postencoder_lower_bound {Z : Type*} (E : ℝ → Z) (g : Z → ℝ) :
    1 / 2 ≤ selectedRisk stablePair (fun x => g (E x)) :=
  stable_selected_lower_bound _

end HJepa.TokenAggregation.Examples
