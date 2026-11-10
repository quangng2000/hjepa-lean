import HJepa.TokenAggregation.Examples.Population

namespace HJepa.TokenAggregation.Examples

open HJepa.Abstraction HJepa.MaskedPrediction

/-- The selected token contains only an independent nuisance. -/
theorem skipped_selected_exact (h : ℝ → ℝ) :
    selectedRisk skippedPair h = 1 + ((h 1) ^ 2 + (h (-1)) ^ 2) / 2 := by
  norm_num [selectedRisk, skippedPair, risk, fairInput, taskSign, nuisanceSign,
    signedBit, Fintype.sum_prod_type, Fintype.sum_bool]
  ring

theorem skipped_selected_lower_bound (h : ℝ → ℝ) :
    1 ≤ selectedRisk skippedPair h := by
  rw [skipped_selected_exact]
  nlinarith [sq_nonneg (h 1), sq_nonneg (h (-1))]

theorem skipped_selected_attained : selectedRisk skippedPair (fun _ => 0) = 1 := by
  rw [skipped_selected_exact]
  norm_num

/-- Averaging retains some task information but also produces collisions. -/
theorem skipped_average_exact (h : ℝ → ℝ) :
    averagedRisk skippedPair h = 1 / 2 + (h 0) ^ 2 / 2 +
      ((1 - h 1) ^ 2 + (-1 - h (-1)) ^ 2) / 4 := by
  norm_num [averagedRisk, skippedPair, risk, fairInput, taskSign, nuisanceSign,
    signedBit, Fintype.sum_prod_type, Fintype.sum_bool]
  ring

theorem skipped_average_lower_bound (h : ℝ → ℝ) :
    1 / 2 ≤ averagedRisk skippedPair h := by
  rw [skipped_average_exact]
  nlinarith [sq_nonneg (h 0), sq_nonneg (1 - h 1), sq_nonneg (-1 - h (-1))]

theorem skipped_average_attained : averagedRisk skippedPair id = 1 / 2 := by
  rw [skipped_average_exact]
  norm_num

/-- Content at the skipped position can make averaging preferable. -/
theorem skipped_average_strictly_better (h : ℝ → ℝ) :
    averagedRisk skippedPair id < selectedRisk skippedPair h := by
  rw [skipped_average_attained]
  linarith [skipped_selected_lower_bound h]

end HJepa.TokenAggregation.Examples
