import HJepa.Spoofing.Detection.Risk

namespace HJepa.Spoofing

open HJepa.Abstraction

variable {Ω L U : Type*} [Fintype Ω]

/-- Retaining the fine state permits a downstream head to ignore the coarse one. -/
theorem retained_simulates_fine (p : FiniteLaw Ω) (labels : Ω → Bool)
    (fine : Ω → L) (coarse : Ω → U) (head : L → Bool) :
    classificationRisk p labels (fun ω => (fine ω, coarse ω)) (head ∘ Prod.fst) =
      classificationRisk p labels fine head := rfl

def ClassificationOptimal (p : FiniteLaw Ω) (labels : Ω → Bool)
    (features : Ω → L) (head : L → Bool) : Prop :=
  ∀ other, classificationRisk p labels features head ≤
    classificationRisk p labels features other

/-- A global fused optimum is no worse than any fixed fine-only classifier. -/
theorem retained_optimum_no_worse (p : FiniteLaw Ω) (labels : Ω → Bool)
    (fine : Ω → L) (coarse : Ω → U) (fineHead : L → Bool)
    (fusedHead : L × U → Bool)
    (h : ClassificationOptimal p labels (fun ω => (fine ω, coarse ω)) fusedHead) :
    classificationRisk p labels (fun ω => (fine ω, coarse ω)) fusedHead ≤
      classificationRisk p labels fine fineHead := by
  exact h (fineHead ∘ Prod.fst)

/-- The same comparator carries an explicit optimization allowance. -/
theorem retained_approximate_no_worse (p : FiniteLaw Ω) (labels : Ω → Bool)
    (fine : Ω → L) (coarse : Ω → U) (fineHead : L → Bool)
    (fusedHead : L × U → Bool) (ε : ℝ)
    (h : ∀ other, classificationRisk p labels (fun ω => (fine ω, coarse ω)) fusedHead ≤
      classificationRisk p labels (fun ω => (fine ω, coarse ω)) other + ε) :
    classificationRisk p labels (fun ω => (fine ω, coarse ω)) fusedHead ≤
      classificationRisk p labels fine fineHead + ε := by
  exact h (fineHead ∘ Prod.fst)

end HJepa.Spoofing
