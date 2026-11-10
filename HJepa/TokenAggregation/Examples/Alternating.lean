import HJepa.TokenAggregation.Examples.Population

namespace HJepa.TokenAggregation.Examples

open HJepa.Abstraction HJepa.MaskedPrediction

/-- Averaging two opposite task-bearing tokens destroys their sign. -/
theorem alternating_average_exact (h : ℝ → ℝ) :
    averagedRisk alternatingPair h = 1 + (h 0) ^ 2 := by
  norm_num [averagedRisk, alternatingPair, risk, fairInput, taskSign, nuisanceSign,
    signedBit, Fintype.sum_prod_type, Fintype.sum_bool]
  ring

theorem alternating_average_lower_bound (h : ℝ → ℝ) :
    1 ≤ averagedRisk alternatingPair h := by
  rw [alternating_average_exact]
  exact le_add_of_nonneg_right (sq_nonneg _)

theorem alternating_average_attained :
    averagedRisk alternatingPair (fun _ => 0) = 1 := by
  rw [alternating_average_exact]
  norm_num

theorem alternating_selected_zero : selectedRisk alternatingPair id = 0 := by
  norm_num [selectedRisk, alternatingPair, risk, fairInput, taskSign,
    signedBit, Fintype.sum_prod_type, Fintype.sum_bool]

/-- Selection wins even against unrestricted nonlinear averaged-token readouts. -/
theorem alternating_selected_strictly_better (h : ℝ → ℝ) :
    selectedRisk alternatingPair id < averagedRisk alternatingPair h := by
  rw [alternating_selected_zero]
  linarith [alternating_average_lower_bound h]

/-- Increasing the next encoder's capacity cannot restore this discarded sign. -/
theorem alternating_postencoder_lower_bound {Z : Type*} (E : ℝ → Z) (g : Z → ℝ) :
    1 ≤ averagedRisk alternatingPair (fun x => g (E x)) :=
  alternating_average_lower_bound _

end HJepa.TokenAggregation.Examples
