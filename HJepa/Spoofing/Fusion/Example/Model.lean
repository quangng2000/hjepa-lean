import HJepa.Expressivity
import HJepa.Spoofing.Detection.Model

namespace HJepa.Spoofing.Fusion.Example

open HJepa.Abstraction HJepa.Expressivity

abbrev Sample := Bool × Bool × Bool

noncomputable def fairThree : FiniteLaw Sample where
  weight := fun _ => 1 / 8
  nonneg := by intro _; norm_num
  total_weight := by norm_num [Fintype.sum_prod_type, Fintype.sum_bool]

/-- The label is A, except that the joint B=C=false context reverses it. -/
def label (x : Sample) : Bool := if x.2.1 || x.2.2 then x.1 else !x.1

def toMiddle (x : Sample) : Bool × Bool :=
  (productBit x.1 x.2.1, productBit x.2.1 x.2.2)

def toHigh (x : Bool × Bool) : Bool := productBit x.1 x.2

abbrev Features := Sample × (Bool × Bool) × Bool

def bundle (x : Sample) : Features := (x, toMiddle x, toHigh (toMiddle x))

/-- Bias and affine weights of all six retained signed coordinates. -/
structure Weights where
  bias : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  ab : ℝ
  bc : ℝ
  ac : ℝ

def affineScore (w : Weights) (z : Features) : ℝ :=
  w.bias + w.a * signedBit z.1.1 + w.b * signedBit z.1.2.1 +
    w.c * signedBit z.1.2.2 + w.ab * signedBit z.2.1.1 +
    w.bc * signedBit z.2.1.2 + w.ac * signedBit z.2.2

noncomputable def affineHead (w : Weights) (z : Features) : Bool :=
  positiveClassifier (affineScore w z)

noncomputable def riskOf (head : Sample → Bool) : ℝ :=
  classificationRisk fairThree label id head

noncomputable def affineError (w : Weights) : ℝ := riskOf (affineHead w ∘ bundle)

/-- The upper feature is AC because the repeated B factor has square one. -/
theorem high_is_ac (x : Sample) :
    signedBit (toHigh (toMiddle x)) = signedBit x.1 * signedBit x.2.2 := by
  rcases x with ⟨a,b,c⟩
  cases a <;> cases b <;> cases c <;>
    norm_num [toHigh, toMiddle, productBit, signedBit]

end HJepa.Spoofing.Fusion.Example
