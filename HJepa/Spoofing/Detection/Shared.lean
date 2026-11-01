import HJepa.Spoofing.Detection.Model

namespace HJepa.Spoofing

open HJepa.Abstraction

theorem shared_difference_zero (a : ℝ) (label : Bool) :
    differenceCue (sharedCuePair a) label = 0 := by
  simp [differenceCue, sharedCuePair]

theorem shared_average_signal (a : ℝ) (label : Bool) :
    averageCue (sharedCuePair a) label = a * signedBit label := by
  unfold averageCue sharedCuePair
  ring

/-- A difference-only detector erases a cue shared by both tokens. -/
theorem shared_difference_chance (a : ℝ) (classifier : ℝ → Bool) :
    toyError (differenceCue (sharedCuePair a)) classifier = 1 / 2 := by
  apply toyError_collapsed
  rw [shared_difference_zero, shared_difference_zero]

theorem shared_difference_postencoder_chance {Z : Type*} (a : ℝ)
    (encoder : ℝ → Z) (classifier : Z → Bool) :
    toyError (fun label => encoder (differenceCue (sharedCuePair a) label))
      classifier = 1 / 2 := by
  apply toyError_postencoder_collapsed
  rw [shared_difference_zero, shared_difference_zero]

theorem shared_average_perfect (a : ℝ) (ha : 0 < a) :
    toyError (averageCue (sharedCuePair a)) positiveClassifier = 0 := by
  apply classificationRisk_zero_of_correct
  intro label
  rw [shared_average_signal]
  exact positiveClassifier_signed a ha label

theorem shared_average_better (a : ℝ) (ha : 0 < a) (classifier : ℝ → Bool) :
    toyError (averageCue (sharedCuePair a)) positiveClassifier <
      toyError (differenceCue (sharedCuePair a)) classifier := by
  rw [shared_average_perfect a ha, shared_difference_chance]
  norm_num

end HJepa.Spoofing
