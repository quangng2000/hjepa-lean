import HJepa.Spoofing.Fusion.Aggregation.Affine

namespace HJepa.Spoofing.Fusion.Aggregation

open scoped BigOperators

variable {I J K : Type*} [Fintype I] [Fintype J] [Fintype K]

omit [Fintype J] [Fintype K] in
theorem affineScore_add (b c : ℝ) (u v x : I → ℝ) :
    affineScore b u x + affineScore c v x =
      affineScore (b + c) (fun i => u i + v i) x := by
  simp only [affineScore, add_mul, Finset.sum_add_distrib]
  ring

/-- Concatenating three affine levels does not add nonlinear scoring capacity. -/
theorem three_affine_levels_collapse (bias : ℝ)
    (fineWeights : I → ℝ) (middleWeights offset₁ : J → ℝ)
    (highWeights offset₂ : K → ℝ) (matrix₁ : J → I → ℝ) (matrix₂ : K → J → ℝ) :
    ∃ rawBias rawWeights, ∀ x : I → ℝ,
      affineScore bias fineWeights x +
        affineScore 0 middleWeights (affineAggregate offset₁ matrix₁ x) +
        affineScore 0 highWeights
          (affineAggregate offset₂ matrix₂ (affineAggregate offset₁ matrix₁ x)) =
        affineScore rawBias rawWeights x := by
  refine ⟨bias + composedBias 0 middleWeights offset₁ +
    composedBias (composedBias 0 highWeights offset₂) (composedWeights highWeights matrix₂) offset₁,
    fun i => fineWeights i + composedWeights middleWeights matrix₁ i +
      composedWeights (composedWeights highWeights matrix₂) matrix₁ i, ?_⟩
  intro x
  rw [affine_after_aggregate, affine_after_aggregate, affine_after_aggregate]
  rw [affineScore_add, affineScore_add]

end HJepa.Spoofing.Fusion.Aggregation
