import HJepa.TokenAggregation.Operators
import HJepa.TokenAggregation.PairMoments

namespace HJepa.TokenAggregation

open HJepa.Abstraction
open scoped BigOperators

variable {Ω : Type*} [Fintype Ω]

theorem secondMoment_scale (p : FiniteLaw Ω) (x : Ω → ℝ) (c : ℝ) :
    secondMoment p (fun ω => c * x ω) = c ^ 2 * secondMoment p x := by
  simp only [secondMoment, HJepa.Pooling.expectation, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro ω _
  ring

theorem crossMoment_scale (p : FiniteLaw Ω) (y x : Ω → ℝ) (c : ℝ) :
    crossMoment p y (fun ω => c * x ω) = c * crossMoment p y x := by
  simp only [crossMoment, HJepa.Pooling.expectation, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro ω _
  ring

/-- A nonzero scalar linear next encoder is absorbed by the fitted probe. -/
theorem linear_encoder_preserves_bestRisk (p : FiniteLaw Ω) (y x : Ω → ℝ)
    (c : ℝ) (hc : c ≠ 0) (hx : 0 < secondMoment p x) :
    bestLinearRisk p y (fun ω => c * x ω) = bestLinearRisk p y x := by
  unfold bestLinearRisk
  rw [secondMoment_scale, crossMoment_scale]
  field_simp [hc, ne_of_gt hx]

omit [Fintype Ω] in
/-- The statistical pair average is exactly the sequence operator at one block. -/
theorem average_block_eq (z : Ω → ℕ → ℝ) (t : ℕ) :
    (fun ω => averageAdjacent (z ω) t) =
      pairMean (fun ω => z ω (2 * t)) (fun ω => z ω (2 * t + 1)) := rfl

/-- The positive-energy assumptions remain valid after a nonzero linear encoder. -/
theorem scaled_energy_positive (p : FiniteLaw Ω) (x : Ω → ℝ) (c : ℝ)
    (hc : c ≠ 0) (hx : 0 < secondMoment p x) :
    0 < secondMoment p (fun ω => c * x ω) := by
  rw [secondMoment_scale]
  exact mul_pos (sq_pos_of_ne_zero hc) hx

end HJepa.TokenAggregation
