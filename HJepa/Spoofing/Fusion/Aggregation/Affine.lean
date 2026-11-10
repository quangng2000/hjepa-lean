import HJepa.Spoofing.Detection.Model

namespace HJepa.Spoofing.Fusion.Aggregation

open HJepa.Abstraction
open scoped BigOperators

variable {I J Ω : Type*} [Fintype I] [Fintype J] [Fintype Ω]

def affineScore (bias : ℝ) (weights x : I → ℝ) : ℝ :=
  bias + ∑ i, weights i * x i

def affineAggregate (offset : J → ℝ) (matrix : J → I → ℝ)
    (x : I → ℝ) : J → ℝ := fun j => affineScore (offset j) (matrix j) x

def composedBias (bias : ℝ) (weights offset : J → ℝ) : ℝ :=
  bias + ∑ j, weights j * offset j

def composedWeights (weights : J → ℝ) (matrix : J → I → ℝ) : I → ℝ :=
  fun i => ∑ j, weights j * matrix j i

omit [Fintype Ω] in
/-- Fixed affine aggregation followed by an affine head is still affine in raw features. -/
theorem affine_after_aggregate (bias : ℝ) (weights offset : J → ℝ)
    (matrix : J → I → ℝ) (x : I → ℝ) :
    affineScore bias weights (affineAggregate offset matrix x) =
      affineScore (composedBias bias weights offset) (composedWeights weights matrix) x := by
  have hs : (∑ j, ∑ i, weights j * (matrix j i * x i)) =
      ∑ i, (∑ j, weights j * matrix j i) * x i := by
    rw [Finset.sum_comm]
    simp only [Finset.sum_mul, mul_assoc]
  simp only [affineScore, affineAggregate, composedBias, composedWeights,
    mul_add, Finset.sum_add_distrib, Finset.mul_sum, hs]
  ring

noncomputable def affineClassifier (bias : ℝ) (weights x : I → ℝ) : Bool :=
  positiveClassifier (affineScore bias weights x)

/-- This equivalence concerns actual 0–1 error, not a squared-loss proxy. -/
theorem affine_aggregation_risk_eq (p : FiniteLaw Ω) (labels : Ω → Bool)
    (features : Ω → I → ℝ) (bias : ℝ) (weights offset : J → ℝ)
    (matrix : J → I → ℝ) :
    classificationRisk p labels (affineAggregate offset matrix ∘ features)
        (affineClassifier bias weights) =
      classificationRisk p labels features
        (affineClassifier (composedBias bias weights offset) (composedWeights weights matrix)) := by
  simp only [classificationRisk, affineClassifier, Function.comp_apply, affine_after_aggregate]

end HJepa.Spoofing.Fusion.Aggregation
