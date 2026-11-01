import HJepa.Spoofing.Fusion.Example.Atom

namespace HJepa.Spoofing.Fusion.Example

open HJepa.Abstraction HJepa.Expressivity

theorem label_b_false (a c : Bool) : label (a, false, c) = productBit a c := by
  cases a <;> cases c <;> rfl

theorem label_c_false (a b : Bool) : label (a, b, false) = productBit a b := by
  cases a <;> cases b <;> rfl

theorem score_b_false (w : Weights) (h : w.ac = 0) (a c : Bool) :
    affineScore w (bundle (a, false, c)) =
      (w.bias - w.b) + (w.a - w.ab) * signedBit a + (w.c - w.bc) * signedBit c := by
  cases a <;> cases c <;>
    norm_num [affineScore, bundle, toMiddle, toHigh, productBit, signedBit, h] <;> ring

theorem score_c_false (w : Weights) (h : w.ab = 0) (a b : Bool) :
    affineScore w (bundle (a, b, false)) =
      (w.bias - w.c) + (w.a - w.ac) * signedBit a + (w.b - w.bc) * signedBit b := by
  cases a <;> cases b <;>
    norm_num [affineScore, bundle, toMiddle, toHigh, productBit, signedBit, h] <;> ring

/-- Even with all fine and middle coordinates, an absent high feature costs error. -/
theorem without_high_lower_bound (w : Weights) (h : w.ac = 0) :
    1 / 8 ≤ affineError w := by
  obtain ⟨a,c,he⟩ := affine_product_has_error (w.bias - w.b) (w.a - w.ab) (w.c - w.bc)
  apply wrong_atom_lower_bound (affineHead w ∘ bundle) (a, false, c)
  simpa only [Function.comp_apply, affineHead, score_b_false w h, label_b_false] using he

/-- The AB middle feature is necessary even when the AC high feature is present. -/
theorem without_middle_lower_bound (w : Weights) (h : w.ab = 0) :
    1 / 8 ≤ affineError w := by
  obtain ⟨a,b,he⟩ := affine_product_has_error (w.bias - w.c) (w.a - w.ac) (w.b - w.bc)
  apply wrong_atom_lower_bound (affineHead w ∘ bundle) (a, b, false)
  simpa only [Function.comp_apply, affineHead, score_c_false w h, label_c_false] using he

end HJepa.Spoofing.Fusion.Example
