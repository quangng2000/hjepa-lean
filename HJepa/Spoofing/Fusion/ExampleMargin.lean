import HJepa.Spoofing.Fusion.Example.Witnesses
import HJepa.Spoofing.Fusion.Margin

namespace HJepa.Spoofing.Fusion.Example

open HJepa.Abstraction HJepa.Expressivity
open scoped BigOperators

/-- The constructed cue task has balanced labels, not a majority-class shortcut. -/
theorem labels_balanced :
    (∑ x, fairThree.weight x * if label x then (1 : ℝ) else 0) = 1 / 2 := by
  norm_num [fairThree, label, Fintype.sum_prod_type, Fintype.sum_bool]

theorem full_score_margin :
    HasMargin label (fun x => affineScore fullWeights (bundle x)) 1 := by
  rintro ⟨a,b,c⟩
  cases a <;> cases b <;> cases c <;>
    norm_num [affineScore, fullWeights, bundle, toMiddle, toHigh, productBit,
      signedBit, label]

/-- The full detector tolerates any additive score error strictly below its margin. -/
theorem full_fusion_robust (error : Sample → ℝ) (h : ∀ x, |error x| < 1) :
    classificationRisk fairThree label
      (fun x => affineScore fullWeights (bundle x) + error x) positiveClassifier = 0 := by
  exact margin_robust_zero fairThree label _ error 1 full_score_margin h

/-- Errors on the three selected branch coordinates have a sufficient L1 bound. -/
theorem three_branch_robust (e₁ e₂ e₃ : Sample → ℝ)
    (h : ∀ x, |e₁ x| + |e₂ x| + |e₃ x| < 1) :
    classificationRisk fairThree label
      (fun x => affineScore fullWeights (bundle x) + (e₁ x + e₂ x + e₃ x))
      positiveClassifier = 0 := by
  apply full_fusion_robust
  intro x
  exact three_errors_bound _ _ _ _ (h x)

end HJepa.Spoofing.Fusion.Example
