import HJepa.Pooling

namespace HJepa.TokenAggregation

open HJepa.Abstraction
open scoped BigOperators

variable {Ω : Type*} [Fintype Ω]

def secondMoment (p : FiniteLaw Ω) (x : Ω → ℝ) : ℝ :=
  HJepa.Pooling.expectation p (fun ω => x ω ^ 2)

/-- A raw cross moment; no centering assumption is silently imposed. -/
def crossMoment (p : FiniteLaw Ω) (y x : Ω → ℝ) : ℝ :=
  HJepa.Pooling.expectation p (fun ω => y ω * x ω)

def linearRisk (p : FiniteLaw Ω) (y x : Ω → ℝ) (k : ℝ) : ℝ :=
  risk p y (fun ω => k * x ω)

noncomputable def optimalCoefficient (p : FiniteLaw Ω) (y x : Ω → ℝ) : ℝ :=
  crossMoment p y x / secondMoment p x

noncomputable def bestLinearRisk (p : FiniteLaw Ω) (y x : Ω → ℝ) : ℝ :=
  secondMoment p y - crossMoment p y x ^ 2 / secondMoment p x

theorem secondMoment_nonneg (p : FiniteLaw Ω) (x : Ω → ℝ) :
    0 ≤ secondMoment p x := by
  exact Finset.sum_nonneg fun ω _ => mul_nonneg (p.nonneg ω) (sq_nonneg _)

theorem linearRisk_exact (p : FiniteLaw Ω) (y x : Ω → ℝ) (k : ℝ) :
    linearRisk p y x k = secondMoment p y -
      2 * k * crossMoment p y x + k ^ 2 * secondMoment p x := by
  simp only [linearRisk, risk, secondMoment, crossMoment, HJepa.Pooling.expectation,
    Finset.mul_sum, ← Finset.sum_sub_distrib, ← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro ω _
  ring

theorem linearRisk_complete_square (p : FiniteLaw Ω) (y x : Ω → ℝ)
    (k : ℝ) (h : 0 < secondMoment p x) :
    linearRisk p y x k = bestLinearRisk p y x +
      secondMoment p x * (k - optimalCoefficient p y x) ^ 2 := by
  rw [linearRisk_exact]
  unfold bestLinearRisk optimalCoefficient
  field_simp [ne_of_gt h]
  ring

theorem bestLinearRisk_attained (p : FiniteLaw Ω) (y x : Ω → ℝ)
    (h : 0 < secondMoment p x) :
    linearRisk p y x (optimalCoefficient p y x) = bestLinearRisk p y x := by
  rw [linearRisk_complete_square p y x _ h]
  simp

theorem bestLinearRisk_lower_bound (p : FiniteLaw Ω) (y x : Ω → ℝ)
    (k : ℝ) (h : 0 < secondMoment p x) :
    bestLinearRisk p y x ≤ linearRisk p y x k := by
  rw [linearRisk_complete_square p y x k h]
  exact le_add_of_nonneg_right (mul_nonneg (le_of_lt h) (sq_nonneg _))

end HJepa.TokenAggregation
