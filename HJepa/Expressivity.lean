import HJepa.Abstraction
import Mathlib.Data.Fintype.BigOperators

/-!
# A three-level compositional expressivity mechanism

Four independent fair signs supply one common input context. The target is
their product. The fixed middle encoder forms two pair products, and the fixed
high encoder multiplies those products. Every affine readout of the raw signs
or of the middle pair products has squared population risk at least one. An
affine readout of the high feature attains zero risk.

This is an exact positive mechanism under a restricted readout class, not a
comparison against an arbitrary flat nonlinear network. All three levels have
zero unrestricted Bayes risk. The feature maps are handcrafted, not learned;
there is no claim here that a JEPA training objective will discover them, that
the full architectures have matched parameter or computation budgets, or that
these finite signs model the distribution of real audio.
-/

namespace HJepa.Expressivity

open HJepa.Abstraction
open scoped BigOperators

/-- The sixteen outcomes of four independent fair Boolean signs. -/
abbrev FourSigns := (Bool × Bool) × (Bool × Bool)

/-- One common probability law for every representation and readout below. -/
noncomputable def fairFour : FiniteLaw FourSigns where
  weight := fun _ => 1 / 16
  nonneg := by intro _; norm_num
  total_weight := by norm_num [Fintype.sum_prod_type, Fintype.sum_bool]

/-- A Boolean encoding of multiplication of two signs, not Boolean XOR. -/
def productBit (a b : Bool) : Bool := a == b

theorem signed_productBit (a b : Bool) :
    signedBit (productBit a b) = signedBit a * signedBit b := by
  cases a <;> cases b <;> norm_num [productBit, signedBit]

/-- Low features retain all four signs. -/
def low : FourSigns → FourSigns := id

/-- Each middle feature is the product of one disjoint pair of low signs. -/
def toMiddle (s : FourSigns) : Bool × Bool :=
  (productBit s.1.1 s.1.2, productBit s.2.1 s.2.2)

/-- The high feature combines the two middle products. -/
def toHigh (s : Bool × Bool) : Bool := productBit s.1 s.2

def middle : FourSigns → Bool × Bool := toMiddle ∘ low

def high : FourSigns → Bool := toHigh ∘ middle

/-- One fixed target is used at every level, with no target or context change. -/
def parityTarget (s : FourSigns) : ℝ :=
  (signedBit s.1.1 * signedBit s.1.2) * (signedBit s.2.1 * signedBit s.2.2)

/-- An arbitrary affine readout of the four raw signed coordinates. -/
def lowAffine (bias a b c d : ℝ) (s : FourSigns) : ℝ :=
  bias + a * signedBit s.1.1 + b * signedBit s.1.2 +
    c * signedBit s.2.1 + d * signedBit s.2.2

/-- An arbitrary affine readout of the two middle signed coordinates. -/
def middleAffine (bias a b : ℝ) (s : Bool × Bool) : ℝ :=
  bias + a * signedBit s.1 + b * signedBit s.2

/-- An arbitrary affine readout of the one high signed coordinate. -/
def highAffine (bias a : ℝ) (s : Bool) : ℝ := bias + a * signedBit s

/-- Exact risk for every raw affine readout, not just one poorly chosen fit. -/
theorem low_affine_risk (bias a b c d : ℝ) :
    risk fairFour parityTarget (lowAffine bias a b c d ∘ low) =
      1 + bias ^ 2 + a ^ 2 + b ^ 2 + c ^ 2 + d ^ 2 := by
  norm_num [risk, fairFour, parityTarget, lowAffine, low, signedBit,
    Fintype.sum_prod_type, Fintype.sum_bool]
  ring

/-- The pair-products remain inaccessible to an affine middle readout of parity. -/
theorem middle_affine_risk (bias a b : ℝ) :
    risk fairFour parityTarget (middleAffine bias a b ∘ middle) =
      1 + bias ^ 2 + a ^ 2 + b ^ 2 := by
  norm_num [risk, fairFour, parityTarget, middleAffine, middle, toMiddle, low,
    productBit, signedBit, Fintype.sum_prod_type, Fintype.sum_bool]
  ring

/-- The high feature makes the parity target linearly accessible. -/
theorem high_affine_risk (bias a : ℝ) :
    risk fairFour parityTarget (highAffine bias a ∘ high) =
      (1 - a) ^ 2 + bias ^ 2 := by
  norm_num [risk, fairFour, parityTarget, highAffine, high, toHigh, middle,
    toMiddle, low, productBit, signedBit, Fintype.sum_prod_type, Fintype.sum_bool]
  ring

/-- No choice of affine coefficients can solve parity from the raw signs. -/
theorem low_affine_risk_lower_bound (bias a b c d : ℝ) :
    1 ≤ risk fairFour parityTarget (lowAffine bias a b c d ∘ low) := by
  rw [low_affine_risk]
  nlinarith [sq_nonneg bias, sq_nonneg a, sq_nonneg b, sq_nonneg c, sq_nonneg d]

/-- Even after the first nonlinear composition, affine middle readout cannot solve parity. -/
theorem middle_affine_risk_lower_bound (bias a b : ℝ) :
    1 ≤ risk fairFour parityTarget (middleAffine bias a b ∘ middle) := by
  rw [middle_affine_risk]
  nlinarith [sq_nonneg bias, sq_nonneg a, sq_nonneg b]

/-- The zero affine predictors attain both lower bounds; they are exact optima. -/
theorem affine_lower_bounds_attained :
    risk fairFour parityTarget (lowAffine 0 0 0 0 0 ∘ low) = 1 ∧
      risk fairFour parityTarget (middleAffine 0 0 0 ∘ middle) = 1 := by
  rw [low_affine_risk, middle_affine_risk]
  norm_num

/-- Identity readout of the high signed feature has exactly zero population risk. -/
theorem high_identity_risk_zero :
    risk fairFour parityTarget (highAffine 0 1 ∘ high) = 0 := by
  rw [high_affine_risk]
  norm_num

/-- The high identity readout strictly beats every affine low or middle readout. -/
theorem high_strictly_beats_affine_readouts
    (bias a b c d middleBias middleA middleB : ℝ) :
    risk fairFour parityTarget (highAffine 0 1 ∘ high) <
        risk fairFour parityTarget (lowAffine bias a b c d ∘ low) ∧
      risk fairFour parityTarget (highAffine 0 1 ∘ high) <
        risk fairFour parityTarget (middleAffine middleBias middleA middleB ∘ middle) := by
  rw [high_identity_risk_zero]
  constructor
  · linarith [low_affine_risk_lower_bound bias a b c d]
  · linarith [middle_affine_risk_lower_bound middleBias middleA middleB]

/-- A nonlinear raw readout computes the same target, so hierarchy has no
unrestricted expressive advantage in this example. -/
theorem nonlinear_low_risk_zero :
    risk fairFour parityTarget (parityTarget ∘ low) = 0 := by
  simp [risk, low]

/-- A nonlinear middle readout also computes the same target. -/
theorem nonlinear_middle_risk_zero :
    risk fairFour parityTarget
      ((fun s : Bool × Bool => signedBit s.1 * signedBit s.2) ∘ middle) = 0 := by
  simp [risk, parityTarget, middle, toMiddle, low, signed_productBit, Function.comp_def]

/-- All levels already retain sufficient information for this target. The
positive affine-readout result is not a decrease in unrestricted Bayes risk. -/
theorem all_bayes_risks_zero :
    risk fairFour parityTarget (bayesPredictor fairFour parityTarget low) = 0 ∧
      risk fairFour parityTarget (bayesPredictor fairFour parityTarget middle) = 0 ∧
      risk fairFour parityTarget (bayesPredictor fairFour parityTarget high) = 0 := by
  refine ⟨?_, ?_, ?_⟩
  · have h := bayes_optimal fairFour parityTarget low parityTarget
    rw [nonlinear_low_risk_zero] at h
    exact le_antisymm h (risk_nonneg _ _ _)
  · have h := bayes_optimal fairFour parityTarget middle
      (fun s : Bool × Bool => signedBit s.1 * signedBit s.2)
    rw [nonlinear_middle_risk_zero] at h
    exact le_antisymm h (risk_nonneg _ _ _)
  · have h := bayes_optimal fairFour parityTarget high (highAffine 0 1)
    rw [high_identity_risk_zero] at h
    exact le_antisymm h (risk_nonneg _ _ _)

/-- There is no information penalty at either transition for the parity target. -/
theorem adjacent_information_gaps_zero :
    informationGap fairFour parityTarget low toMiddle = 0 ∧
      informationGap fairFour parityTarget middle toHigh = 0 := by
  have h12 := nested_risk_identity fairFour parityTarget low toMiddle
  have h23 := nested_risk_identity fairFour parityTarget middle toHigh
  change risk fairFour parityTarget (bayesPredictor fairFour parityTarget middle) -
    risk fairFour parityTarget (bayesPredictor fairFour parityTarget low) = _ at h12
  change risk fairFour parityTarget (bayesPredictor fairFour parityTarget high) -
    risk fairFour parityTarget (bayesPredictor fairFour parityTarget middle) = _ at h23
  rw [all_bayes_risks_zero.1, all_bayes_risks_zero.2.1, sub_self] at h12
  rw [all_bayes_risks_zero.2.1, all_bayes_risks_zero.2.2, sub_self] at h23
  exact ⟨h12.symm, h23.symm⟩

end HJepa.Expressivity
