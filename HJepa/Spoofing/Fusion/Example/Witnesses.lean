import HJepa.Spoofing.Fusion.Example.Slices

namespace HJepa.Spoofing.Fusion.Example

open HJepa.Abstraction HJepa.Expressivity

/-- One nonzero feature from each of the three levels. -/
def fullWeights : Weights := ⟨0, 1, 0, 0, 1, 0, 1⟩
noncomputable def fineWeights : Weights := ⟨-3 / 4, 1, 1 / 2, 1 / 2, 0, 0, 0⟩

theorem full_head_correct (x : Sample) : affineHead fullWeights (bundle x) = label x := by
  rcases x with ⟨a,b,c⟩
  cases a <;> cases b <;> cases c <;>
    norm_num [affineHead, fullWeights, affineScore, bundle, toMiddle, toHigh,
      productBit, signedBit, positiveClassifier, label]

theorem full_fusion_zero : affineError fullWeights = 0 := by
  exact classificationRisk_zero_of_correct fairThree label id
    (affineHead fullWeights ∘ bundle) full_head_correct

/-- The one-eighth lower bound for the fine-only affine class is attained. -/
theorem fine_witness_exact : affineError fineWeights = 1 / 8 := by
  norm_num [affineError, riskOf, classificationRisk, fairThree, fineWeights, affineHead,
    affineScore, bundle, toMiddle, toHigh, productBit, signedBit, positiveClassifier,
    label, Fintype.sum_prod_type, Fintype.sum_bool]

theorem full_beats_without_high (w : Weights) (h : w.ac = 0) :
    affineError fullWeights < affineError w := by
  rw [full_fusion_zero]
  linarith [without_high_lower_bound w h]

theorem full_beats_without_middle (w : Weights) (h : w.ab = 0) :
    affineError fullWeights < affineError w := by
  rw [full_fusion_zero]
  linarith [without_middle_lower_bound w h]

/-- Unrestricted access to the fine input already permits the same perfect label. -/
theorem unrestricted_fine_zero : riskOf label = 0 := by
  apply classificationRisk_zero_of_correct
  intro x
  rfl

end HJepa.Spoofing.Fusion.Example
