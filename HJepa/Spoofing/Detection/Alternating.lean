import HJepa.Spoofing.Detection.Model

namespace HJepa.Spoofing

open HJepa.Abstraction

theorem alternating_average_zero (a : ℝ) (label : Bool) :
    averageCue (alternatingCuePair a) label = 0 := by
  simp [averageCue, alternatingCuePair]

theorem alternating_difference_signal (a : ℝ) (label : Bool) :
    differenceCue (alternatingCuePair a) label = a * signedBit label := by
  unfold differenceCue alternatingCuePair
  ring

/-- Pairwise averaging erases this binary cue at any amplitude. -/
theorem alternating_average_chance (a : ℝ) (classifier : ℝ → Bool) :
    toyError (averageCue (alternatingCuePair a)) classifier = 1 / 2 := by
  apply toyError_collapsed
  rw [alternating_average_zero, alternating_average_zero]

theorem alternating_average_postencoder_chance {Z : Type*} (a : ℝ)
    (encoder : ℝ → Z) (classifier : Z → Bool) :
    toyError (fun label => encoder (averageCue (alternatingCuePair a) label))
      classifier = 1 / 2 := by
  apply toyError_postencoder_collapsed
  rw [alternating_average_zero, alternating_average_zero]

/-- A difference feature perfectly separates the labels, even for tiny positive a. -/
theorem alternating_difference_perfect (a : ℝ) (ha : 0 < a) :
    toyError (differenceCue (alternatingCuePair a)) positiveClassifier = 0 := by
  apply classificationRisk_zero_of_correct
  intro label
  rw [alternating_difference_signal]
  exact positiveClassifier_signed a ha label

theorem alternating_current_perfect (a : ℝ) (ha : 0 < a) :
    toyError (currentCue (alternatingCuePair a)) positiveClassifier = 0 := by
  apply classificationRisk_zero_of_correct
  intro label
  exact positiveClassifier_signed a ha label

theorem alternating_current_better (a : ℝ) (ha : 0 < a)
    (classifier : ℝ → Bool) :
    toyError (currentCue (alternatingCuePair a)) positiveClassifier <
      toyError (averageCue (alternatingCuePair a)) classifier := by
  rw [alternating_current_perfect a ha, alternating_average_chance]
  norm_num

theorem alternating_difference_better (a : ℝ) (ha : 0 < a)
    (classifier : ℝ → Bool) :
    toyError (differenceCue (alternatingCuePair a)) positiveClassifier <
      toyError (averageCue (alternatingCuePair a)) classifier := by
  rw [alternating_difference_perfect a ha, alternating_average_chance]
  norm_num

end HJepa.Spoofing
