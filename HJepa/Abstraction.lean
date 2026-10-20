import Mathlib.Algebra.BigOperators.Field
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith

/-!
# Finite scalar prediction under hierarchical abstraction

This module works with a finite probability space and a real-valued target.
An encoder `z : Ω → I` represents the lower level; `g ∘ z` represents an upper
level. The conditional mean is constructed by summing over encoder fibers,
including fibers of probability zero. We prove its orthogonality and optimality
before deriving the exact information-loss and learning-error identities.

These results concern prediction from the upper representation alone. They do
not assert that a trained H-JEPA encoder is sufficient for a downstream task.
-/

namespace HJepa.Abstraction

open scoped BigOperators

attribute [local instance] Classical.propDecidable

variable {Ω I J : Type*} [Fintype Ω]

/-- A finite probability law, including outcomes of probability zero. -/
structure FiniteLaw (Ω : Type*) [Fintype Ω] where
  weight : Ω → ℝ
  nonneg : ∀ ω, 0 ≤ weight ω
  total_weight : ∑ ω, weight ω = 1

/-- Expected squared prediction error for a scalar target. -/
def risk (p : FiniteLaw Ω) (y f : Ω → ℝ) : ℝ :=
  ∑ ω, p.weight ω * (y ω - f ω) ^ 2

/-- Probability mass of one encoder fiber. -/
noncomputable def fiberMass (p : FiniteLaw Ω) (z : Ω → I) (i : I) : ℝ := by
  classical
  exact ∑ ω ∈ Finset.univ.filter (fun ω => z ω = i), p.weight ω

/-- The conditional mean on a fiber; division by zero gives zero on null fibers. -/
noncomputable def conditionalMean (p : FiniteLaw Ω) (y : Ω → ℝ)
    (z : Ω → I) (i : I) : ℝ := by
  classical
  exact (∑ ω ∈ Finset.univ.filter (fun ω => z ω = i), p.weight ω * y ω) /
    fiberMass p z i

/-- The prediction obtained by evaluating the conditional mean at the encoder. -/
noncomputable def bayesPredictor (p : FiniteLaw Ω) (y : Ω → ℝ)
    (z : Ω → I) : Ω → ℝ := fun ω => conditionalMean p y z (z ω)

theorem risk_nonneg (p : FiniteLaw Ω) (y f : Ω → ℝ) : 0 ≤ risk p y f := by
  exact Finset.sum_nonneg fun ω _ => mul_nonneg (p.nonneg ω) (sq_nonneg _)

/-- Zero squared error means equality on the probability support, not on null outcomes. -/
theorem risk_eq_zero_iff (p : FiniteLaw Ω) (y f : Ω → ℝ) :
    risk p y f = 0 ↔ ∀ ω, 0 < p.weight ω → y ω = f ω := by
  constructor
  · intro h ω hpos
    have hzero : p.weight ω * (y ω - f ω) ^ 2 = 0 :=
      (Finset.sum_eq_zero_iff_of_nonneg
        (fun a _ => mul_nonneg (p.nonneg a) (sq_nonneg _))).mp h ω (Finset.mem_univ _)
    have hsquare := (mul_eq_zero.mp hzero).resolve_left (ne_of_gt hpos)
    exact sub_eq_zero.mp (sq_eq_zero_iff.mp hsquare)
  · intro h
    apply Finset.sum_eq_zero
    intro ω _
    by_cases hw : p.weight ω = 0
    · simp only [hw, zero_mul]
    · have heq := h ω (lt_of_le_of_ne (p.nonneg ω) (Ne.symm hw))
      simp only [heq, sub_self, zero_pow (by decide : 2 ≠ 0), mul_zero]

theorem fiberMass_nonneg (p : FiniteLaw Ω) (z : Ω → I) (i : I) :
    0 ≤ fiberMass p z i := by
  classical
  exact Finset.sum_nonneg fun ω _ => p.nonneg ω

/-- Every point in a zero-mass fiber itself has weight zero. -/
theorem weight_eq_zero_of_fiberMass_eq_zero (p : FiniteLaw Ω) (z : Ω → I)
    (i : I) (hzero : fiberMass p z i = 0) (ω : Ω) (hz : z ω = i) :
    p.weight ω = 0 := by
  classical
  have hle : p.weight ω ≤ fiberMass p z i := by
    exact Finset.single_le_sum (fun a _ => p.nonneg a)
      (Finset.mem_filter.mpr ⟨Finset.mem_univ _, hz⟩)
  rw [hzero] at hle
  exact le_antisymm hle (p.nonneg ω)

/-- The weighted residual sums to zero on every fiber, including null fibers. -/
theorem fiber_residual_zero (p : FiniteLaw Ω) (y : Ω → ℝ)
    (z : Ω → I) (i : I) :
    ∑ ω ∈ Finset.univ.filter (fun ω => z ω = i),
      p.weight ω * (y ω - conditionalMean p y z i) = 0 := by
  classical
  by_cases hm : fiberMass p z i = 0
  · apply Finset.sum_eq_zero
    intro ω hω
    rw [weight_eq_zero_of_fiberMass_eq_zero p z i hm ω
      (Finset.mem_filter.mp hω).2, zero_mul]
  · simp_rw [mul_sub]
    rw [Finset.sum_sub_distrib, ← Finset.sum_mul]
    change (∑ ω ∈ Finset.univ.filter (fun ω => z ω = i), p.weight ω * y ω) -
      fiberMass p z i *
        ((∑ ω ∈ Finset.univ.filter (fun ω => z ω = i), p.weight ω * y ω) /
          fiberMass p z i) = 0
    rw [mul_div_cancel₀ _ hm, sub_self]

/-- Summation over all encoder fibers recovers the original expectation. -/
theorem sum_over_fibers [Fintype I] (z : Ω → I) (a : Ω → ℝ) :
    (∑ i, ∑ ω ∈ Finset.univ.filter (fun ω => z ω = i), a ω) = ∑ ω, a ω := by
  classical
  simp_rw [Finset.sum_filter]
  rw [Finset.sum_comm]
  simp

/-- Orthogonality to every scalar function of the encoded representation. -/
theorem residual_orthogonality [Fintype I] (p : FiniteLaw Ω) (y : Ω → ℝ)
    (z : Ω → I) (a : I → ℝ) :
    (∑ ω, p.weight ω * (y ω - bayesPredictor p y z ω) * a (z ω)) = 0 := by
  classical
  rw [← sum_over_fibers z]
  apply Finset.sum_eq_zero
  intro i _
  calc
    (∑ ω ∈ Finset.univ.filter (fun ω => z ω = i),
        p.weight ω * (y ω - bayesPredictor p y z ω) * a (z ω)) =
        ∑ ω ∈ Finset.univ.filter (fun ω => z ω = i),
          p.weight ω * (y ω - conditionalMean p y z i) * a i := by
      apply Finset.sum_congr rfl
      intro ω hω
      simp only [bayesPredictor, (Finset.mem_filter.mp hω).2]
    _ = (∑ ω ∈ Finset.univ.filter (fun ω => z ω = i),
          p.weight ω * (y ω - conditionalMean p y z i)) * a i :=
      (Finset.sum_mul _ _ _).symm
    _ = 0 := by rw [fiber_residual_zero, zero_mul]

/-- Exact Pythagorean identity for every predictor that is a function of `z`. -/
theorem risk_pythagorean [Fintype I] (p : FiniteLaw Ω) (y : Ω → ℝ)
    (z : Ω → I) (a : I → ℝ) :
    risk p y (a ∘ z) = risk p y (bayesPredictor p y z) +
      risk p (bayesPredictor p y z) (a ∘ z) := by
  have horth := residual_orthogonality p y z
    (fun i => conditionalMean p y z i - a i)
  have hexpand : risk p y (a ∘ z) = risk p y (bayesPredictor p y z) +
      risk p (bayesPredictor p y z) (a ∘ z) +
      2 * (∑ ω, p.weight ω * (y ω - bayesPredictor p y z ω) *
        (conditionalMean p y z (z ω) - a (z ω))) := by
    simp only [risk, ← Finset.sum_add_distrib, Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro ω _
    simp only [Function.comp_apply, bayesPredictor]
    ring
  rw [horth, mul_zero, add_zero] at hexpand
  exact hexpand

/-- The constructed conditional mean minimizes risk among all functions of `z`. -/
theorem bayes_optimal [Fintype I] (p : FiniteLaw Ω) (y : Ω → ℝ)
    (z : Ω → I) (a : I → ℝ) :
    risk p y (bayesPredictor p y z) ≤ risk p y (a ∘ z) := by
  rw [risk_pythagorean]
  exact le_add_of_nonneg_right (risk_nonneg _ _ _)

/-- Prediction information lost by using the upper representation alone. -/
noncomputable def informationGap (p : FiniteLaw Ω) (y : Ω → ℝ)
    (z : Ω → I) (g : I → J) : ℝ :=
  risk p (bayesPredictor p y z) (bayesPredictor p y (g ∘ z))

/-- The difference of optimal risks is exactly the squared conditional-mean gap. -/
theorem nested_risk_identity [Fintype I] (p : FiniteLaw Ω) (y : Ω → ℝ)
    (z : Ω → I) (g : I → J) :
    risk p y (bayesPredictor p y (g ∘ z)) - risk p y (bayesPredictor p y z) =
      informationGap p y z g := by
  have h := risk_pythagorean p y z (conditionalMean p y (g ∘ z) ∘ g)
  change risk p y (bayesPredictor p y (g ∘ z)) =
    risk p y (bayesPredictor p y z) + informationGap p y z g at h
  linarith

theorem informationGap_nonneg (p : FiniteLaw Ω) (y : Ω → ℝ)
    (z : Ω → I) (g : I → J) : 0 ≤ informationGap p y z g :=
  risk_nonneg _ _ _

/-- Exact conditional-mean sufficiency for this squared-loss target: the two
conditional means agree on support. This is not full distributional sufficiency. -/
theorem informationGap_eq_zero_iff (p : FiniteLaw Ω) (y : Ω → ℝ)
    (z : Ω → I) (g : I → J) :
    informationGap p y z g = 0 ↔
      ∀ ω, 0 < p.weight ω →
        bayesPredictor p y z ω = bayesPredictor p y (g ∘ z) ω :=
  risk_eq_zero_iff _ _ _

/-- A single positive-probability disagreement already creates strict information loss. -/
theorem informationGap_pos_of_disagreement (p : FiniteLaw Ω) (y : Ω → ℝ)
    (z : Ω → I) (g : I → J) (ω : Ω) (hpos : 0 < p.weight ω)
    (hne : bayesPredictor p y z ω ≠ bayesPredictor p y (g ∘ z) ω) :
    0 < informationGap p y z g := by
  have hzero : informationGap p y z g ≠ 0 := by
    intro h
    exact hne ((informationGap_eq_zero_iff p y z g).mp h ω hpos)
  exact lt_of_le_of_ne (informationGap_nonneg p y z g) (Ne.symm hzero)

/-- Equality of optimal squared-loss risks is equivalent to conditional-mean
agreement on the support for this fixed target. -/
theorem abstraction_no_loss_iff [Fintype I] (p : FiniteLaw Ω) (y : Ω → ℝ)
    (z : Ω → I) (g : I → J) :
    risk p y (bayesPredictor p y (g ∘ z)) = risk p y (bayesPredictor p y z) ↔
      ∀ ω, 0 < p.weight ω →
        bayesPredictor p y z ω = bayesPredictor p y (g ∘ z) ω := by
  rw [← informationGap_eq_zero_iff]
  have h := nested_risk_identity p y z g
  constructor <;> intro hh <;> linarith

/-- Deterministic abstraction cannot lower the unrestricted Bayes prediction risk. -/
theorem abstraction_bayes_risk_monotone [Fintype I] (p : FiniteLaw Ω)
    (y : Ω → ℝ) (z : Ω → I) (g : I → J) :
    risk p y (bayesPredictor p y z) ≤ risk p y (bayesPredictor p y (g ∘ z)) := by
  have h := nested_risk_identity p y z g
  have hn := informationGap_nonneg p y z g
  linarith

/-- Excess risk of a chosen (possibly learned) predictor over the Bayes risk. -/
noncomputable def excessRisk (p : FiniteLaw Ω) (y : Ω → ℝ)
    (z : Ω → I) (a : I → ℝ) : ℝ :=
  risk p y (a ∘ z) - risk p y (bayesPredictor p y z)

theorem excessRisk_nonneg [Fintype I] (p : FiniteLaw Ω) (y : Ω → ℝ)
    (z : Ω → I) (a : I → ℝ) : 0 ≤ excessRisk p y z a := by
  exact sub_nonneg.mpr (bayes_optimal p y z a)

/-- Exact gain/loss decomposition for two arbitrary learned feature predictors. -/
theorem learning_gain_loss_identity [Fintype I] (p : FiniteLaw Ω)
    (y : Ω → ℝ) (z : Ω → I) (g : I → J) (lower : I → ℝ) (upper : J → ℝ) :
    risk p y (upper ∘ (g ∘ z)) - risk p y (lower ∘ z) =
      informationGap p y z g -
        (excessRisk p y z lower - excessRisk p y (g ∘ z) upper) := by
  have h := nested_risk_identity p y z g
  unfold excessRisk
  linarith

/-- Abstraction helps exactly when reduced excess risk outweighs information loss. -/
theorem abstraction_improves_iff [Fintype I] (p : FiniteLaw Ω)
    (y : Ω → ℝ) (z : Ω → I) (g : I → J) (lower : I → ℝ) (upper : J → ℝ) :
    risk p y (upper ∘ (g ∘ z)) < risk p y (lower ∘ z) ↔
      informationGap p y z g <
        excessRisk p y z lower - excessRisk p y (g ∘ z) upper := by
  have h := learning_gain_loss_identity p y z g lower upper
  constructor <;> intro hh <;> linarith

/-! ## Explicit two-outcome witnesses

These examples show that information loss can be strict and that improvement
is possible for particular imperfect predictors. They do not model an actual
training algorithm or establish a finite-sample learning guarantee.
-/

/-- Two equally probable outcomes. -/
noncomputable def fairBit : FiniteLaw Bool where
  weight := fun _ => 1 / 2
  nonneg := by intro _; norm_num
  total_weight := by norm_num [Fintype.univ_bool]

/-- A target taking the values `-1` and `1` with equal probability. -/
def signedBit (b : Bool) : ℝ := if b then 1 else -1

/-- Identity features predict perfectly; a constant upper feature incurs risk one. -/
theorem strict_abstraction_loss_example :
    risk fairBit signedBit (bayesPredictor fairBit signedBit (id : Bool → Bool)) = 0 ∧
      risk fairBit signedBit
        (bayesPredictor fairBit signedBit (fun _ : Bool => ())) = 1 := by
  norm_num [risk, fairBit, signedBit, bayesPredictor, conditionalMean, fiberMass,
    Fintype.univ_bool, Finset.filter_insert, Finset.filter_singleton]

/-- The upper constant-zero predictor beats a poorly learned constant-two lower predictor. -/
theorem imperfect_predictor_gain_example :
    risk fairBit signedBit ((fun _ : Unit => (0 : ℝ)) ∘ (fun _ : Bool => ())) = 1 ∧
      risk fairBit signedBit ((fun _ : Bool => (2 : ℝ)) ∘ id) = 5 := by
  norm_num [risk, fairBit, signedBit, Fintype.univ_bool]

end HJepa.Abstraction
