import HJepa.Spoofing.Filters.Padding

namespace HJepa.Spoofing

/-- Two output channels per selected time: this preserves more capacity than one. -/
def pairFilterBank (tokens : ℤ → ℝ) (t : ℤ) : ℝ × ℝ :=
  (applyKernel sumKernel tokens t, applyKernel differenceKernel tokens t)

noncomputable def recoverPair (features : ℝ × ℝ) : ℝ × ℝ :=
  ((features.1 + features.2) / 2, (features.1 - features.2) / 2)

/-- Sum and difference recover current and previous tokens exactly. -/
theorem filterbank_recovers_pair (tokens : ℤ → ℝ) (t : ℤ) :
    recoverPair (pairFilterBank tokens t) = (tokens (2 * t), tokens (2 * t - 1)) := by
  ext <;> simp only [recoverPair, pairFilterBank, sum_kernel_eq, difference_kernel_eq] <;> ring

theorem padded_filterbank_initial (tokens : ℕ → ℝ) :
    recoverPair (pairFilterBank (zeroLeftPad tokens) 0) = (tokens 0, 0) := by
  rw [filterbank_recovers_pair]
  norm_num [zeroLeftPad]

/-- Every subsequent block recovers the available odd/even pair. -/
theorem padded_filterbank_recovers (tokens : ℕ → ℝ) (t : ℕ) :
    recoverPair (pairFilterBank (zeroLeftPad tokens) (t + 1)) =
      (tokens (2 * t + 2), tokens (2 * t + 1)) := by
  rw [filterbank_recovers_pair]
  have h₀ : 2 * ((t : ℤ) + 1) = ((2 * t + 2 : ℕ) : ℤ) := by omega
  have h₁ : 2 * ((t : ℤ) + 1) - 1 = ((2 * t + 1 : ℕ) : ℤ) := by omega
  rw [h₀, zeroLeftPad_nonnegative, ← h₀, h₁, zeroLeftPad_nonnegative]

end HJepa.Spoofing
