import HJepa.TokenAggregation.PairMoments

namespace HJepa.TokenAggregation

open HJepa.Abstraction

variable {Ω : Type*} [Fintype Ω]

/-- Compare fitted probes on the same target, not differently scaled pretraining losses. -/
theorem mean_better_iff (p : FiniteLaw Ω) (y x₀ x₁ : Ω → ℝ)
    (h₀ : 0 < secondMoment p x₀) (hm : 0 < pairEnergy p x₀ x₁) :
    bestLinearRisk p y (pairMean x₀ x₁) < bestLinearRisk p y x₀ ↔
      crossMoment p y x₀ ^ 2 * pairEnergy p x₀ x₁ <
        (crossMoment p y x₀ + crossMoment p y x₁) ^ 2 * secondMoment p x₀ := by
  rw [pairMean_bestLinearRisk p y x₀ x₁ hm]
  unfold bestLinearRisk
  rw [sub_lt_sub_iff_left]
  exact div_lt_div_iff₀ h₀ hm

theorem subsample_better_iff (p : FiniteLaw Ω) (y x₀ x₁ : Ω → ℝ)
    (h₀ : 0 < secondMoment p x₀) (hm : 0 < pairEnergy p x₀ x₁) :
    bestLinearRisk p y x₀ < bestLinearRisk p y (pairMean x₀ x₁) ↔
      (crossMoment p y x₀ + crossMoment p y x₁) ^ 2 * secondMoment p x₀ <
        crossMoment p y x₀ ^ 2 * pairEnergy p x₀ x₁ := by
  rw [pairMean_bestLinearRisk p y x₀ x₁ hm]
  unfold bestLinearRisk
  rw [sub_lt_sub_iff_left]
  exact div_lt_div_iff₀ hm h₀

theorem risks_equal_iff (p : FiniteLaw Ω) (y x₀ x₁ : Ω → ℝ)
    (h₀ : 0 < secondMoment p x₀) (hm : 0 < pairEnergy p x₀ x₁) :
    bestLinearRisk p y (pairMean x₀ x₁) = bestLinearRisk p y x₀ ↔
      (crossMoment p y x₀ + crossMoment p y x₁) ^ 2 * secondMoment p x₀ =
        crossMoment p y x₀ ^ 2 * pairEnergy p x₀ x₁ := by
  rw [pairMean_bestLinearRisk p y x₀ x₁ hm]
  unfold bestLinearRisk
  rw [sub_right_inj]
  exact div_eq_div_iff (ne_of_gt hm) (ne_of_gt h₀)

end HJepa.TokenAggregation
