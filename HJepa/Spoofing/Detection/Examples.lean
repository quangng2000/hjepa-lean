import HJepa.Spoofing.Detection.Alternating
import HJepa.Spoofing.Detection.Skipped
import HJepa.Spoofing.Detection.Shared
import HJepa.Spoofing.Detection.Robustness

namespace HJepa.Spoofing

/-- Neither subsampling the current token nor averaging is uniformly best. -/
theorem current_average_opposite_winners (a : ℝ) (ha : 0 < a) :
    (∀ classifier, toyError (currentCue (alternatingCuePair a)) positiveClassifier <
      toyError (averageCue (alternatingCuePair a)) classifier) ∧
    (∀ classifier, toyError (averageCue (skippedCuePair a)) positiveClassifier <
      toyError (currentCue (skippedCuePair a)) classifier) := by
  exact ⟨alternating_current_better a ha, skipped_average_better a ha⟩

/-- A difference channel fixes one erased cue but erases a different cue itself. -/
theorem average_difference_opposite_winners (a : ℝ) (ha : 0 < a) :
    (∀ classifier, toyError (differenceCue (alternatingCuePair a)) positiveClassifier <
      toyError (averageCue (alternatingCuePair a)) classifier) ∧
    (∀ classifier, toyError (averageCue (sharedCuePair a)) positiveClassifier <
      toyError (differenceCue (sharedCuePair a)) classifier) := by
  exact ⟨alternating_difference_better a ha, shared_average_better a ha⟩

end HJepa.Spoofing
