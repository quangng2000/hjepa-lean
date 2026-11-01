import HJepa.Spoofing.Detection.Model

namespace HJepa.Spoofing.Fusion

open HJepa.Abstraction

variable {Ω : Type*}

def HasMargin (labels : Ω → Bool) (score : Ω → ℝ) (margin : ℝ) : Prop :=
  ∀ ω, margin ≤ signedBit (labels ω) * score ω

/-- A positive signed margin tolerates a strictly smaller score perturbation. -/
theorem margin_classifier_correct (labels : Ω → Bool) (score error : Ω → ℝ)
    (margin : ℝ) (h : HasMargin labels score margin)
    (he : ∀ ω, |error ω| < margin) (ω : Ω) :
    positiveClassifier (score ω + error ω) = labels ω := by
  obtain ⟨hlo, hhi⟩ := abs_lt.mp (he ω)
  have hs := h ω
  cases hb : labels ω <;> simp [hb, signedBit] at hs
  · have hn : ¬ (0 : ℝ) < score ω + error ω := by linarith
    simp [positiveClassifier, hn]
  · have hp : 0 < score ω + error ω := by linarith
    simp [positiveClassifier, hp]

/-- The guarantee concerns actual classification risk under any finite law. -/
theorem margin_robust_zero [Fintype Ω] (p : FiniteLaw Ω) (labels : Ω → Bool)
    (score error : Ω → ℝ) (margin : ℝ)
    (h : HasMargin labels score margin) (he : ∀ ω, |error ω| < margin) :
    classificationRisk p labels (fun ω => score ω + error ω) positiveClassifier = 0 := by
  apply classificationRisk_zero_of_correct
  exact margin_classifier_correct labels score error margin h he

/-- Three branch errors can be bounded before they are added to a fused score. -/
theorem three_errors_bound (e₁ e₂ e₃ margin : ℝ)
    (h : |e₁| + |e₂| + |e₃| < margin) : |e₁ + e₂ + e₃| < margin := by
  have h₁ := abs_add_le e₁ e₂
  have h₂ := abs_add_le (e₁ + e₂) e₃
  linarith

end HJepa.Spoofing.Fusion
