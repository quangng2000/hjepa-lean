import HJepa.Spoofing.Filters.Causal

namespace HJepa.Spoofing

/-- Negative token indices are zero; indices at or after zero read the input. -/
def zeroLeftPad (tokens : ℕ → ℝ) (i : ℤ) : ℝ :=
  if 0 ≤ i then tokens i.toNat else 0

def paddedFilter (kernel : Kernel3) (tokens : ℕ → ℝ) (t : ℕ) : ℝ :=
  applyKernel kernel (zeroLeftPad tokens) t

theorem zeroLeftPad_nonnegative (tokens : ℕ → ℝ) (i : ℕ) :
    zeroLeftPad tokens i = tokens i := by
  simp [zeroLeftPad]

theorem zeroLeftPad_negative (tokens : ℕ → ℝ) (i : ℤ) (h : i < 0) :
    zeroLeftPad tokens i = 0 := by
  simp [zeroLeftPad, not_le.mpr h]

/-- Left padding by two makes the first window [0,0,x0], not a future window. -/
theorem paddedFilter_zero (kernel : Kernel3) (tokens : ℕ → ℝ) :
    paddedFilter kernel tokens 0 = kernel.current * tokens 0 := by
  norm_num [paddedFilter, applyKernel, zeroLeftPad]

/-- Every later stride-two window reads indices 2t,2t+1,2t+2 in temporal order. -/
theorem paddedFilter_succ (kernel : Kernel3) (tokens : ℕ → ℝ) (t : ℕ) :
    paddedFilter kernel tokens (t + 1) = kernel.current * tokens (2 * t + 2) +
      kernel.previous * tokens (2 * t + 1) + kernel.oldest * tokens (2 * t) := by
  unfold paddedFilter applyKernel
  have h₀ : 2 * ((t + 1 : ℕ) : ℤ) = ((2 * t + 2 : ℕ) : ℤ) := by omega
  have h₁ : 2 * ((t + 1 : ℕ) : ℤ) - 1 = ((2 * t + 1 : ℕ) : ℤ) := by omega
  have h₂ : 2 * ((t + 1 : ℕ) : ℤ) - 2 = ((2 * t : ℕ) : ℤ) := by omega
  rw [h₀, zeroLeftPad_nonnegative]
  rw [← h₀, h₁, zeroLeftPad_nonnegative, h₂, zeroLeftPad_nonnegative]

end HJepa.Spoofing
