import HJepa.Readout
import HJepa.Spoofing.Retention

namespace HJepa.Spoofing.Fusion

open HJepa.Abstraction
open HJepa.Readout

/-- The entire fine context is fixed; higher levels are deterministic functions of it. -/
structure Hierarchy (Ω I J K : Type*) [Fintype Ω] where
  law : FiniteLaw Ω
  labels : Ω → Bool
  fine : Ω → I
  toMiddle : I → J
  toHigh : J → K

variable {Ω I J K : Type*} [Fintype Ω]

def bundle (H : Hierarchy Ω I J K) : I → I × J × K :=
  threeLevelBundle H.toMiddle H.toHigh

noncomputable def fineRisk (H : Hierarchy Ω I J K) (head : I → Bool) : ℝ :=
  classificationRisk H.law H.labels H.fine head

noncomputable def fusedRisk (H : Hierarchy Ω I J K) (head : I × J × K → Bool) : ℝ :=
  classificationRisk H.law H.labels (bundle H ∘ H.fine) head

theorem fusedRisk_nonneg (H : Hierarchy Ω I J K) (head : I × J × K → Bool) :
    0 ≤ fusedRisk H head := by
  apply Finset.sum_nonneg
  intro ω _
  apply mul_nonneg (H.law.nonneg ω)
  split_ifs <;> norm_num

theorem fine_lift_risk (H : Hierarchy Ω I J K) (head : I → Bool) :
    fusedRisk H (head ∘ Prod.fst) = fineRisk H head := rfl

theorem fused_pullback_risk (H : Hierarchy Ω I J K) (head : I × J × K → Bool) :
    fusedRisk H head = fineRisk H (head ∘ bundle H) := rfl

/-- Three deterministic levels do not enlarge the unrestricted attainable risk set. -/
theorem achievable_risk_iff (H : Hierarchy Ω I J K) (r : ℝ) :
    (∃ head, fusedRisk H head = r) ↔ (∃ head, fineRisk H head = r) := by
  constructor
  · rintro ⟨head, hr⟩
    exact ⟨head ∘ bundle H, hr⟩
  · rintro ⟨head, hr⟩
    exact ⟨head ∘ Prod.fst, hr⟩

/-- A fixed fine-only optimum is no worse than every unrestricted fused head. -/
theorem fine_optimum_le_fused (H : Hierarchy Ω I J K) (fineHead : I → Bool)
    (h : ClassificationOptimal H.law H.labels H.fine fineHead)
    (fusedHead : I × J × K → Bool) : fineRisk H fineHead ≤ fusedRisk H fusedHead := by
  exact h (fusedHead ∘ bundle H)

/-- If both unrestricted optima are attained, their classification errors coincide. -/
theorem unrestricted_optima_equal (H : Hierarchy Ω I J K)
    (fineHead : I → Bool) (fusedHead : I × J × K → Bool)
    (hf : ClassificationOptimal H.law H.labels H.fine fineHead)
    (hu : ClassificationOptimal H.law H.labels (bundle H ∘ H.fine) fusedHead) :
    fusedRisk H fusedHead = fineRisk H fineHead := by
  apply le_antisymm
  · exact hu (fineHead ∘ Prod.fst)
  · exact fine_optimum_le_fused H fineHead hf fusedHead

theorem no_strict_unrestricted_gain (H : Hierarchy Ω I J K) (fineHead : I → Bool)
    (h : ClassificationOptimal H.law H.labels H.fine fineHead)
    (fusedHead : I × J × K → Bool) : ¬ fusedRisk H fusedHead < fineRisk H fineHead := by
  exact not_lt_of_ge (fine_optimum_le_fused H fineHead h fusedHead)

end HJepa.Spoofing.Fusion
