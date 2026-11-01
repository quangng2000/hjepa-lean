import HJepa.TokenAggregation.Comparison

namespace HJepa.TokenAggregation

open HJepa.Abstraction

variable {Ω : Type*} [Fintype Ω]

/-- Conditional raw-moment model for two aligned noisy tokens. These moment
identities are hypotheses; independence or alignment is not inferred from JEPA. -/
structure AlignedMoments (p : FiniteLaw Ω) (y x₀ x₁ : Ω → ℝ) where
  signal : ℝ
  noise : ℝ
  noiseCross : ℝ
  signal_pos : 0 < signal
  noise_nonneg : 0 ≤ noise
  noise_sum_nonneg : 0 ≤ noise + noiseCross
  target : secondMoment p y = signal
  first : secondMoment p x₀ = signal + noise
  second : secondMoment p x₁ = signal + noise
  target_first : crossMoment p y x₀ = signal
  target_second : crossMoment p y x₁ = signal
  cross : crossMoment p x₀ x₁ = signal + noiseCross

variable {p : FiniteLaw Ω} {y x₀ x₁ : Ω → ℝ}

theorem aligned_pairEnergy (h : AlignedMoments p y x₀ x₁) :
    pairEnergy p x₀ x₁ = 2 * (2 * h.signal + h.noise + h.noiseCross) := by
  unfold pairEnergy
  rw [h.first, h.second, h.cross]
  ring

theorem aligned_first_positive (h : AlignedMoments p y x₀ x₁) :
    0 < secondMoment p x₀ := by
  rw [h.first]
  linarith [h.signal_pos, h.noise_nonneg]

theorem aligned_pair_positive (h : AlignedMoments p y x₀ x₁) :
    0 < pairEnergy p x₀ x₁ := by
  rw [aligned_pairEnergy h]
  linarith [h.signal_pos, h.noise_sum_nonneg]

theorem aligned_subsample_risk (h : AlignedMoments p y x₀ x₁) :
    bestLinearRisk p y x₀ = h.signal * h.noise / (h.signal + h.noise) := by
  have hd : h.signal + h.noise ≠ 0 := by linarith [h.signal_pos, h.noise_nonneg]
  unfold bestLinearRisk
  rw [h.target, h.target_first, h.first]
  field_simp [hd]
  ring

theorem aligned_mean_risk (h : AlignedMoments p y x₀ x₁) :
    bestLinearRisk p y (pairMean x₀ x₁) =
      h.signal * (h.noise + h.noiseCross) / (2 * h.signal + h.noise + h.noiseCross) := by
  have hd : 2 * h.signal + h.noise + h.noiseCross ≠ 0 := by
    linarith [h.signal_pos, h.noise_sum_nonneg]
  rw [pairMean_bestLinearRisk p y x₀ x₁ (aligned_pair_positive h),
    h.target, h.target_first, h.target_second, aligned_pairEnergy h]
  rw [show (h.signal + h.signal) ^ 2 = 2 * (2 * h.signal ^ 2) by ring,
    mul_div_mul_left _ _ (by norm_num : (2 : ℝ) ≠ 0)]
  ring_nf at hd ⊢
  field_simp [hd]
  ring

/-- The aligned gain is strict exactly when error is less than perfectly shared. -/
theorem aligned_mean_better_iff (h : AlignedMoments p y x₀ x₁) :
    bestLinearRisk p y (pairMean x₀ x₁) < bestLinearRisk p y x₀ ↔
      h.noiseCross < h.noise := by
  rw [mean_better_iff p y x₀ x₁ (aligned_first_positive h) (aligned_pair_positive h),
    h.target_first, h.target_second, aligned_pairEnergy h, h.first]
  have hs : 0 < 2 * h.signal ^ 2 := mul_pos (by norm_num) (sq_pos_of_pos h.signal_pos)
  have he : (h.signal + h.signal) ^ 2 * (h.signal + h.noise) -
      h.signal ^ 2 * (2 * (2 * h.signal + h.noise + h.noiseCross)) =
      2 * h.signal ^ 2 * (h.noise - h.noiseCross) := by ring
  rw [← sub_pos, he, mul_pos_iff_of_pos_left hs, sub_pos]

theorem aligned_shared_noise_tie (h : AlignedMoments p y x₀ x₁)
    (hc : h.noiseCross = h.noise) :
    bestLinearRisk p y (pairMean x₀ x₁) = bestLinearRisk p y x₀ := by
  rw [risks_equal_iff p y x₀ x₁ (aligned_first_positive h) (aligned_pair_positive h),
    h.target_first, h.target_second, aligned_pairEnergy h, h.first, hc]
  ring

end HJepa.TokenAggregation
