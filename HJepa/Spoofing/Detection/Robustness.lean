import HJepa.Spoofing.Detection.Model

namespace HJepa.Spoofing

open HJepa.Abstraction

/-- Classification is guaranteed stable when perturbation stays below the cue margin. -/
theorem positiveClassifier_robust (a error : ℝ) (_ha : 0 < a) (h : |error| < a)
    (label : Bool) : positiveClassifier (a * signedBit label + error) = label := by
  obtain ⟨hl, hu⟩ := abs_lt.mp h
  cases label
  · have hn : ¬ (0 : ℝ) < -a + error := by linarith
    simpa [positiveClassifier, signedBit] using hn
  · have hp : (0 : ℝ) < a + error := by linarith
    simpa [positiveClassifier, signedBit] using hp

/-- This is a bounded-error guarantee; it does not claim arbitrary small cues
survive noise, quantization, or finite-precision model evaluation. -/
theorem noisy_signed_cue_perfect (a : ℝ) (ha : 0 < a) (error : Bool → ℝ)
    (h : ∀ label, |error label| < a) :
    toyError (fun label => a * signedBit label + error label) positiveClassifier = 0 := by
  apply classificationRisk_zero_of_correct
  intro label
  exact positiveClassifier_robust a (error label) ha (h label) label

end HJepa.Spoofing
