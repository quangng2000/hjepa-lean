import HJepa.Abstraction
import Mathlib.Tactic.NormNum

namespace HJepa.Spoofing

/-- Scalar three-tap kernel; taps refer to current, previous, and oldest tokens. -/
structure Kernel3 where
  current : ℝ
  previous : ℝ
  oldest : ℝ

def applyKernel (kernel : Kernel3) (tokens : ℤ → ℝ) (t : ℤ) : ℝ :=
  kernel.current * tokens (2 * t) + kernel.previous * tokens (2 * t - 1) +
    kernel.oldest * tokens (2 * t - 2)

/-- Conv1d stores the oldest-to-current weights; this explicitly maps that order. -/
def fromConvWeights (weights : Fin 3 → ℝ) : Kernel3 :=
  ⟨weights 2, weights 1, weights 0⟩

theorem conv_weight_order (weights : Fin 3 → ℝ) (tokens : ℤ → ℝ) (t : ℤ) :
    applyKernel (fromConvWeights weights) tokens t =
      weights 0 * tokens (2 * t - 2) + weights 1 * tokens (2 * t - 1) +
        weights 2 * tokens (2 * t) := by
  simp only [applyKernel, fromConvWeights]
  ring

/-- The output anchored at 2t depends only on tokens at or before 2t. -/
theorem causal_no_future (kernel : Kernel3) (x y : ℤ → ℝ) (t : ℤ)
    (h : ∀ i, i ≤ 2 * t → x i = y i) : applyKernel kernel x t = applyKernel kernel y t := by
  unfold applyKernel
  rw [h _ (by omega), h _ (by omega), h _ (by omega)]

def selectionKernel : Kernel3 := ⟨1, 0, 0⟩
noncomputable def meanKernel : Kernel3 := ⟨1 / 2, 1 / 2, 0⟩
def sumKernel : Kernel3 := ⟨1, 1, 0⟩
def differenceKernel : Kernel3 := ⟨1, -1, 0⟩

theorem selection_kernel_eq (x : ℤ → ℝ) (t : ℤ) :
    applyKernel selectionKernel x t = x (2 * t) := by
  simp [applyKernel, selectionKernel]

theorem mean_kernel_eq (x : ℤ → ℝ) (t : ℤ) :
    applyKernel meanKernel x t = (x (2 * t) + x (2 * t - 1)) / 2 := by
  simp only [applyKernel, meanKernel]
  ring

theorem sum_kernel_eq (x : ℤ → ℝ) (t : ℤ) :
    applyKernel sumKernel x t = x (2 * t) + x (2 * t - 1) := by
  simp [applyKernel, sumKernel]

theorem difference_kernel_eq (x : ℤ → ℝ) (t : ℤ) :
    applyKernel differenceKernel x t = x (2 * t) - x (2 * t - 1) := by
  simp [applyKernel, differenceKernel, sub_eq_add_neg]

end HJepa.Spoofing
