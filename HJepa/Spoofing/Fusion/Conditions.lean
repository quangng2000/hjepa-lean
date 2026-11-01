import HJepa.Spoofing.Fusion.General

namespace HJepa.Spoofing.Fusion

variable {Ω I J K Θ Φ : Type*} [Fintype Ω]

/-- Explicit readout classes; restricting them can make deterministic features useful. -/
structure ReadoutFamilies (H : Hierarchy Ω I J K) (Θ Φ : Type*) where
  fine : Θ → I → Bool
  fused : Φ → I × J × K → Bool

noncomputable def fineError {H : Hierarchy Ω I J K}
    (F : ReadoutFamilies H Θ Φ) (θ : Θ) : ℝ := fineRisk H (F.fine θ)

noncomputable def fusedError {H : Hierarchy Ω I J K}
    (F : ReadoutFamilies H Θ Φ) (φ : Φ) : ℝ := fusedRisk H (F.fused φ)

def FusedOptimal {H : Hierarchy Ω I J K} (F : ReadoutFamilies H Θ Φ) (φ : Φ) : Prop :=
  ∀ other, fusedError F φ ≤ fusedError F other

def FusedApproximate {H : Hierarchy Ω I J K}
    (F : ReadoutFamilies H Θ Φ) (φ : Φ) (ε : ℝ) : Prop :=
  ∀ other, fusedError F φ ≤ fusedError F other + ε

/-- Inclusion is required on realized features, not on unused input values. -/
def ContainsFine {H : Hierarchy Ω I J K} (F : ReadoutFamilies H Θ Φ) : Prop :=
  ∀ θ, ∃ φ, ∀ ω, F.fused φ (bundle H (H.fine ω)) = F.fine θ (H.fine ω)

theorem inclusion_no_worse {H : Hierarchy Ω I J K} (F : ReadoutFamilies H Θ Φ)
    (included : ContainsFine F) (φ : Φ) (optimal : FusedOptimal F φ) (θ : Θ) :
    fusedError F φ ≤ fineError F θ := by
  obtain ⟨ψ, hp⟩ := included θ
  have he : fusedError F ψ = fineError F θ := by
    simp only [fusedError, fineError, fusedRisk, fineRisk, classificationRisk,
      Function.comp_apply, hp]
  exact (optimal ψ).trans_eq he

/-- A certified floor for the entire fine class and a feasible fused witness give a gap. -/
theorem strict_of_comparator {H : Hierarchy Ω I J K} (F : ReadoutFamilies H Θ Φ)
    (α β : ℝ) (floor : ∀ θ, α ≤ fineError F θ)
    (witness : Φ) (feasible : fusedError F witness ≤ β) (gap : β < α)
    (φ : Φ) (optimal : FusedOptimal F φ) (θ : Θ) : fusedError F φ < fineError F θ := by
  exact lt_of_le_of_lt ((optimal witness).trans feasible) (gap.trans_le (floor θ))

/-- Optimization error must be smaller than the certified representational gain. -/
theorem approximate_strict_of_comparator {H : Hierarchy Ω I J K} (F : ReadoutFamilies H Θ Φ)
    (α β ε : ℝ) (floor : ∀ θ, α ≤ fineError F θ)
    (witness : Φ) (feasible : fusedError F witness ≤ β) (gap : β + ε < α)
    (φ : Φ) (optimal : FusedApproximate F φ ε) (θ : Θ) :
    fusedError F φ < fineError F θ := by
  have hc := optimal witness
  have hl := floor θ
  linarith

end HJepa.Spoofing.Fusion
