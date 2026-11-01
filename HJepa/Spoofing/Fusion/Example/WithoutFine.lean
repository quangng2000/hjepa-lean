import HJepa.Spoofing.Fusion.Example.Witnesses

namespace HJepa.Spoofing.Fusion.Example

open HJepa.Abstraction HJepa.Expressivity

abbrev CoarseFeatures := (Bool × Bool) × Bool

noncomputable def coarseError (head : CoarseFeatures → Bool) : ℝ :=
  classificationRisk fairThree label (fun x => (toMiddle x, toHigh (toMiddle x))) head

/-- Two coarse fibers contain balanced opposite labels; no nonlinear head fixes them. -/
theorem coarse_error_exact (head : CoarseFeatures → Bool) :
    coarseError head = 1 / 4 +
      (if head ((true, true), true) = true then 0 else 1) / 4 +
      (if head ((false, true), false) = false then 0 else 1) / 4 := by
  cases h₁ : head ((true, true), true) <;>
    cases h₂ : head ((true, false), false) <;>
    cases h₃ : head ((false, true), false) <;>
    cases h₄ : head ((false, false), true) <;>
    norm_num [coarseError, classificationRisk, fairThree, label, toMiddle, toHigh,
      productBit, Fintype.sum_prod_type, Fintype.sum_bool, h₁, h₂, h₃, h₄]

theorem coarse_error_lower_bound (head : CoarseFeatures → Bool) :
    1 / 4 ≤ coarseError head := by
  rw [coarse_error_exact]
  split_ifs <;> norm_num

theorem coarse_error_attained : coarseError (fun z => z.1.1) = 1 / 4 := by
  rw [coarse_error_exact]
  norm_num

noncomputable def coarseAffineHead (w : Weights) (z : CoarseFeatures) : Bool :=
  positiveClassifier (w.bias + w.ab * signedBit z.1.1 +
    w.bc * signedBit z.1.2 + w.ac * signedBit z.2)

theorem affine_without_fine_eq (w : Weights) (ha : w.a = 0)
    (hb : w.b = 0) (hc : w.c = 0) :
    affineError w = coarseError (coarseAffineHead w) := by
  simp [affineError, riskOf, coarseError, classificationRisk, affineHead, affineScore,
    bundle, coarseAffineHead, ha, hb, hc]

theorem without_fine_lower_bound (w : Weights) (ha : w.a = 0)
    (hb : w.b = 0) (hc : w.c = 0) : 1 / 4 ≤ affineError w := by
  rw [affine_without_fine_eq w ha hb hc]
  exact coarse_error_lower_bound _

theorem full_beats_without_fine (head : CoarseFeatures → Bool) :
    affineError fullWeights < coarseError head := by
  rw [full_fusion_zero]
  linarith [coarse_error_lower_bound head]

end HJepa.Spoofing.Fusion.Example
