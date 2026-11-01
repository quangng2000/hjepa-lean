import HJepa.Spoofing.Filters.Filterbank
import HJepa.Spoofing.Detection.Risk

namespace HJepa.Spoofing.Fusion.Aggregation

open HJepa.Abstraction

/-- Two scalar outputs preserve the pair; this is not one-channel compression. -/
def sumDifference (x : ℝ × ℝ) : ℝ × ℝ := (x.1 + x.2, x.1 - x.2)

theorem sumDifference_recover (x : ℝ × ℝ) : recoverPair (sumDifference x) = x := by
  ext <;> simp only [recoverPair, sumDifference] <;> ring

variable {Ω : Type*} [Fintype Ω]

/-- Every raw classifier can be lifted through an information-preserving bank. -/
theorem bank_simulates_raw (p : FiniteLaw Ω) (labels : Ω → Bool)
    (features : Ω → ℝ × ℝ) (head : ℝ × ℝ → Bool) :
    classificationRisk p labels (sumDifference ∘ features) (head ∘ recoverPair) =
      classificationRisk p labels features head := by
  simp only [classificationRisk, Function.comp_apply, sumDifference_recover]

/-- Conversely, a classifier of the bank remains a classifier of the same raw pair. -/
theorem bank_pullback (p : FiniteLaw Ω) (labels : Ω → Bool)
    (features : Ω → ℝ × ℝ) (head : ℝ × ℝ → Bool) :
    classificationRisk p labels (sumDifference ∘ features) head =
      classificationRisk p labels features (head ∘ sumDifference) := rfl

theorem bank_achievable_risk_iff (p : FiniteLaw Ω) (labels : Ω → Bool)
    (features : Ω → ℝ × ℝ) (r : ℝ) :
    (∃ head, classificationRisk p labels (sumDifference ∘ features) head = r) ↔
      (∃ head, classificationRisk p labels features head = r) := by
  constructor
  · rintro ⟨head, hr⟩
    exact ⟨head ∘ sumDifference, hr⟩
  · rintro ⟨head, hr⟩
    exact ⟨head ∘ recoverPair, (bank_simulates_raw p labels features head).trans hr⟩

end HJepa.Spoofing.Fusion.Aggregation
