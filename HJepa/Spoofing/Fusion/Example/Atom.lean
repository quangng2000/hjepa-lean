import HJepa.Spoofing.Fusion.Example.Model

namespace HJepa.Spoofing.Fusion.Example

open scoped BigOperators

/-- A single misclassified atom already costs one eighth of this population. -/
theorem wrong_atom_lower_bound (head : Sample → Bool) (x : Sample)
    (h : head x ≠ label x) : 1 / 8 ≤ riskOf head := by
  have hs := Finset.single_le_sum
    (f := fun y : Sample => fairThree.weight y *
      if head y = label y then (0 : ℝ) else 1)
    (fun y (_ : y ∈ Finset.univ) => by dsimp; split_ifs <;> norm_num [fairThree])
    (Finset.mem_univ x)
  simpa [riskOf, classificationRisk, fairThree, h] using hs

/-- A linear threshold cannot classify the four-corner product relation exactly. -/
theorem affine_product_has_error (bias a b : ℝ) :
    ∃ u v : Bool, positiveClassifier
      (bias + a * HJepa.Abstraction.signedBit u + b * HJepa.Abstraction.signedBit v) ≠
      HJepa.Expressivity.productBit u v := by
  by_contra h
  simp only [not_exists, not_not] at h
  have h₀₀ := h false false
  have h₀₁ := h false true
  have h₁₀ := h true false
  have h₁₁ := h true true
  norm_num [positiveClassifier, HJepa.Abstraction.signedBit,
    HJepa.Expressivity.productBit] at h₀₀ h₀₁ h₁₀ h₁₁
  linarith

end HJepa.Spoofing.Fusion.Example
