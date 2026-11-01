import HJepa.TokenAggregation.LinearRisk

namespace HJepa.TokenAggregation

open HJepa.Abstraction
open scoped BigOperators

variable {Ω : Type*} [Fintype Ω]

/-- Reduce two observed latent tokens before applying a subsequent encoder. -/
noncomputable def pairMean (x₀ x₁ : Ω → ℝ) : Ω → ℝ :=
  fun ω => (x₀ ω + x₁ ω) / 2

def pairEnergy (p : FiniteLaw Ω) (x₀ x₁ : Ω → ℝ) : ℝ :=
  secondMoment p x₀ + secondMoment p x₁ + 2 * crossMoment p x₀ x₁

theorem pairMean_secondMoment (p : FiniteLaw Ω) (x₀ x₁ : Ω → ℝ) :
    secondMoment p (pairMean x₀ x₁) = pairEnergy p x₀ x₁ / 4 := by
  simp only [secondMoment, pairMean, pairEnergy, crossMoment, HJepa.Pooling.expectation,
    Finset.mul_sum, ← Finset.sum_add_distrib, Finset.sum_div]
  apply Finset.sum_congr rfl
  intro ω _
  ring

theorem pairMean_crossMoment (p : FiniteLaw Ω) (y x₀ x₁ : Ω → ℝ) :
    crossMoment p y (pairMean x₀ x₁) =
      (crossMoment p y x₀ + crossMoment p y x₁) / 2 := by
  simp only [crossMoment, pairMean, HJepa.Pooling.expectation,
    ← Finset.sum_add_distrib, Finset.sum_div]
  apply Finset.sum_congr rfl
  intro ω _
  ring

theorem pairMean_positive (p : FiniteLaw Ω) (x₀ x₁ : Ω → ℝ)
    (h : 0 < pairEnergy p x₀ x₁) : 0 < secondMoment p (pairMean x₀ x₁) := by
  rw [pairMean_secondMoment]
  exact div_pos h (by norm_num)

theorem pairMean_bestLinearRisk (p : FiniteLaw Ω) (y x₀ x₁ : Ω → ℝ)
    (h : 0 < pairEnergy p x₀ x₁) :
    bestLinearRisk p y (pairMean x₀ x₁) = secondMoment p y -
      (crossMoment p y x₀ + crossMoment p y x₁) ^ 2 / pairEnergy p x₀ x₁ := by
  unfold bestLinearRisk
  rw [pairMean_secondMoment, pairMean_crossMoment]
  field_simp [ne_of_gt h]
  ring

end HJepa.TokenAggregation
