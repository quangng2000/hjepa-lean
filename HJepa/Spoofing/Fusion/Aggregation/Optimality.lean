import HJepa.Spoofing.Fusion.Aggregation.Recoverable
import HJepa.Spoofing.Retention

namespace HJepa.Spoofing.Fusion.Aggregation

open HJepa.Abstraction

/-- Recovery concerns the entire input, with no claim of compression or cheap decoding. -/
structure RecoverableMap (I J : Type*) where
  encode : I → J
  recover : J → I
  inverse : ∀ x, recover (encode x) = x

variable {Ω I J K : Type*} [Fintype Ω]

/-- A lossless aggregate can reproduce every classifier of any deterministic aggregate. -/
theorem recoverable_simulates_any (p : FiniteLaw Ω) (labels : Ω → Bool)
    (features : Ω → I) (bank : RecoverableMap I J) (other : I → K) (head : K → Bool) :
    classificationRisk p labels (bank.encode ∘ features) (head ∘ other ∘ bank.recover) =
      classificationRisk p labels (other ∘ features) head := by
  simp only [classificationRisk, Function.comp_apply, bank.inverse]

/-- With unrestricted optimal heads, a recoverable aggregate is never worse. -/
theorem recoverable_optimum_no_worse (p : FiniteLaw Ω) (labels : Ω → Bool)
    (features : Ω → I) (bank : RecoverableMap I J) (best : J → Bool)
    (optimal : ClassificationOptimal p labels (bank.encode ∘ features) best)
    (other : I → K) (head : K → Bool) :
    classificationRisk p labels (bank.encode ∘ features) best ≤
      classificationRisk p labels (other ∘ features) head := by
  have h := optimal (head ∘ other ∘ bank.recover)
  rwa [recoverable_simulates_any] at h

/-- A concrete two-channel bank meets the recovery requirement. -/
noncomputable def sumDifferenceMap : RecoverableMap (ℝ × ℝ) (ℝ × ℝ) :=
  ⟨sumDifference, recoverPair, sumDifference_recover⟩

end HJepa.Spoofing.Fusion.Aggregation
