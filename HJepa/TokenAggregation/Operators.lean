import HJepa.Abstraction
import Mathlib.Tactic.NormNum

/-! Operators act on lower-level state tokens, before the next state encoder.
The scalar averaging alternative is distinct from H-JEPA's window-one state map. -/

namespace HJepa.TokenAggregation

def selectStride {V : Type*} (stride : ℕ) (z : ℕ → V) (t : ℕ) : V :=
  z (stride * t)

def encodeSelected {V W : Type*} (stride : ℕ) (encode : V → W)
    (z : ℕ → V) (t : ℕ) : W := encode (selectStride stride z t)

noncomputable def averageAdjacent (z : ℕ → ℝ) (t : ℕ) : ℝ :=
  (z (2 * t) + z (2 * t + 1)) / 2

noncomputable def encodeAveraged {W : Type*} (encode : ℝ → W)
    (z : ℕ → ℝ) (t : ℕ) : W := encode (averageAdjacent z t)

/-- With window one, pointwise encoding commutes with temporal selection. -/
theorem select_encode_commute {V W : Type*} (stride : ℕ) (encode : V → W)
    (z : ℕ → V) :
    encodeSelected stride encode z = selectStride stride (encode ∘ z) := rfl

/-- Exactly two input positions correspond to each output index. -/
def firstOfPair {n : ℕ} (t : Fin n) : Fin (2 * n) := ⟨2 * t.val, by omega⟩
def secondOfPair {n : ℕ} (t : Fin n) : Fin (2 * n) := ⟨2 * t.val + 1, by omega⟩

def selectPairs {V W : Type*} {n : ℕ} (encode : V → W)
    (z : Fin (2 * n) → V) : Fin n → W := fun t => encode (z (firstOfPair t))

/-- This includes vector tokens, for example `V = Fin d → ℝ`. -/
noncomputable def averagePairs {V W : Type*} [Add V] [SMul ℝ V] {n : ℕ}
    (encode : V → W) (z : Fin (2 * n) → V) : Fin n → W :=
  fun t => encode ((1 / 2 : ℝ) • (z (firstOfPair t) + z (secondOfPair t)))

theorem averagePairs_scalar {W : Type*} {n : ℕ} (encode : ℝ → W)
    (z : Fin (2 * n) → ℝ) (t : Fin n) :
    averagePairs encode z t = encode ((z (firstOfPair t) + z (secondOfPair t)) / 2) := by
  unfold averagePairs
  congr 1
  simp only [smul_eq_mul]
  ring

/-- The 8-to-4 example selects zero-based positions 0,2,4,6. -/
theorem selected_eight_indices :
    (List.finRange 4).map (fun t => (firstOfPair t).val) = [0, 2, 4, 6] := by
  decide

/-- Averaging after a nonlinear encoder can give a different representation. -/
theorem averaging_order_matters :
    encodeAveraged (fun x => x ^ 2) (fun t => if t = 0 then 1 else -1) 0 = 0 ∧
    averageAdjacent (fun t => (if t = 0 then (1 : ℝ) else -1) ^ 2) 0 = 1 := by
  norm_num [encodeAveraged, averageAdjacent]

end HJepa.TokenAggregation
