import HJepa.Spoofing.Fusion.Conditions
import HJepa.Spoofing.Fusion.Example.Witnesses

namespace HJepa.Spoofing.Fusion

open HJepa.Abstraction

noncomputable def cueHierarchy : Hierarchy Example.Sample Example.Sample (Bool × Bool) Bool :=
  ⟨Example.fairThree, Example.label, id, Example.toMiddle, Example.toHigh⟩

/-- All fine affine weights are free; only coefficients of unavailable levels vanish. -/
abbrev FineParameters := {w : Example.Weights // w.ab = 0 ∧ w.bc = 0 ∧ w.ac = 0}

noncomputable def fineHead (w : FineParameters) (x : Example.Sample) : Bool :=
  positiveClassifier (w.val.bias + w.val.a * signedBit x.1 +
    w.val.b * signedBit x.2.1 + w.val.c * signedBit x.2.2)

noncomputable def cueFamilies : ReadoutFamilies cueHierarchy FineParameters Example.Weights :=
  ⟨fineHead, Example.affineHead⟩

theorem cue_fused_error (w : Example.Weights) :
    fusedError cueFamilies w = Example.affineError w := rfl

theorem cue_fine_error (w : FineParameters) :
    fineError cueFamilies w = Example.affineError w.val := by
  have he : fineHead w = Example.affineHead w.val ∘ Example.bundle := by
    funext x
    simp [fineHead, Example.affineHead, Example.affineScore, Example.bundle,
      w.property.1, w.property.2.1, w.property.2.2]
  change Example.riskOf (fineHead w) = Example.affineError w.val
  rw [he]
  rfl

theorem cue_contains_fine : ContainsFine cueFamilies := by
  intro w
  refine ⟨w.val, ?_⟩
  intro x
  simp [cueFamilies, cueHierarchy, bundle, HJepa.Readout.threeLevelBundle,
    fineHead, Example.affineHead, Example.affineScore,
    w.property.1, w.property.2.1, w.property.2.2]

theorem cue_fine_floor (w : FineParameters) : 1 / 8 ≤ fineError cueFamilies w := by
  rw [cue_fine_error]
  exact Example.without_high_lower_bound w.val w.property.2.2

noncomputable def bestFineParameters : FineParameters :=
  ⟨Example.fineWeights, by exact ⟨rfl, rfl, rfl⟩⟩

theorem cue_fine_floor_attained : fineError cueFamilies bestFineParameters = 1 / 8 := by
  rw [cue_fine_error]
  exact Example.fine_witness_exact

/-- The zero-error fused witness is an attained optimum, not just a feasible head. -/
theorem cue_full_optimal : FusedOptimal cueFamilies Example.fullWeights := by
  intro other
  rw [cue_fused_error, Example.full_fusion_zero]
  exact fusedRisk_nonneg cueHierarchy (cueFamilies.fused other)

/-- The general gap theorem applies to a concrete, nonempty fused class. -/
theorem cue_fused_optimum_strict (w : Example.Weights)
    (h : FusedOptimal cueFamilies w) (fine : FineParameters) :
    fusedError cueFamilies w < fineError cueFamilies fine := by
  apply strict_of_comparator cueFamilies (1 / 8) 0 cue_fine_floor
    Example.fullWeights ?_ (by norm_num) w h fine
  rw [cue_fused_error, Example.full_fusion_zero]

/-- Any approximate population optimum with error allowance below 1/8 still wins. -/
theorem cue_approximate_strict (w : Example.Weights) (ε : ℝ)
    (h : FusedApproximate cueFamilies w ε) (gap : ε < 1 / 8) (fine : FineParameters) :
    fusedError cueFamilies w < fineError cueFamilies fine := by
  apply approximate_strict_of_comparator cueFamilies (1 / 8) 0 ε cue_fine_floor
    Example.fullWeights ?_ (by simpa using gap) w h fine
  rw [cue_fused_error, Example.full_fusion_zero]

end HJepa.Spoofing.Fusion
