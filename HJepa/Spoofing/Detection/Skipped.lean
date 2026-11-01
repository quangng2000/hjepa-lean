import HJepa.Spoofing.Detection.Model

namespace HJepa.Spoofing

open HJepa.Abstraction

theorem skipped_current_zero (a : ℝ) (label : Bool) :
    currentCue (skippedCuePair a) label = 0 := rfl

theorem skipped_average_signal (a : ℝ) (label : Bool) :
    averageCue (skippedCuePair a) label = (a / 2) * signedBit label := by
  unfold averageCue skippedCuePair
  ring

/-- The cue occurs only at the previous position, which current-only selection skips. -/
theorem skipped_current_chance (a : ℝ) (classifier : ℝ → Bool) :
    toyError (currentCue (skippedCuePair a)) classifier = 1 / 2 :=
  toyError_constant _ _

theorem skipped_current_postencoder_chance {Z : Type*} (a : ℝ)
    (encoder : ℝ → Z) (classifier : Z → Bool) :
    toyError (fun label => encoder (currentCue (skippedCuePair a) label))
      classifier = 1 / 2 := by
  apply toyError_postencoder_collapsed
  rfl

theorem skipped_previous_perfect (a : ℝ) (ha : 0 < a) :
    toyError (skippedCuePair a).previous positiveClassifier = 0 := by
  apply classificationRisk_zero_of_correct
  intro label
  exact positiveClassifier_signed a ha label

theorem skipped_average_perfect (a : ℝ) (ha : 0 < a) :
    toyError (averageCue (skippedCuePair a)) positiveClassifier = 0 := by
  apply classificationRisk_zero_of_correct
  intro label
  rw [skipped_average_signal]
  exact positiveClassifier_signed (a / 2) (div_pos ha (by norm_num)) label

theorem skipped_average_better (a : ℝ) (ha : 0 < a) (classifier : ℝ → Bool) :
    toyError (averageCue (skippedCuePair a)) positiveClassifier <
      toyError (currentCue (skippedCuePair a)) classifier := by
  rw [skipped_average_perfect a ha, skipped_current_chance]
  norm_num

end HJepa.Spoofing
