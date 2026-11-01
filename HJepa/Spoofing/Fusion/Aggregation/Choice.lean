import HJepa.Spoofing.Detection.Examples

namespace HJepa.Spoofing.Fusion.Aggregation

abbrev CueOperator := CuePair → Bool → ℝ

/-- Compare attainable classification errors, allowing every deterministic head. -/
def UniformlyNoWorse (candidate reference : CueOperator) : Prop :=
  ∀ pair referenceHead, ∃ candidateHead,
    toyError (candidate pair) candidateHead ≤ toyError (reference pair) referenceHead

theorem selection_not_uniformly_better_than_mean :
    ¬ UniformlyNoWorse currentCue averageCue := by
  intro h
  obtain ⟨head, he⟩ := h (skippedCuePair 1) positiveClassifier
  have hs := skipped_average_better 1 (by norm_num) head
  linarith

theorem mean_not_uniformly_better_than_selection :
    ¬ UniformlyNoWorse averageCue currentCue := by
  intro h
  obtain ⟨head, he⟩ := h (alternatingCuePair 1) positiveClassifier
  have hs := alternating_current_better 1 (by norm_num) head
  linarith

theorem mean_not_uniformly_better_than_difference :
    ¬ UniformlyNoWorse averageCue differenceCue := by
  intro h
  obtain ⟨head, he⟩ := h (alternatingCuePair 1) positiveClassifier
  have hs := alternating_difference_better 1 (by norm_num) head
  linarith

theorem difference_not_uniformly_better_than_mean :
    ¬ UniformlyNoWorse differenceCue averageCue := by
  intro h
  obtain ⟨head, he⟩ := h (sharedCuePair 1) positiveClassifier
  have hs := shared_average_better 1 (by norm_num) head
  linarith

/-- Every listed fixed reduction fails against another on some balanced cue task. -/
theorem no_universal_fixed_choice :
    ¬ UniformlyNoWorse currentCue averageCue ∧
    ¬ UniformlyNoWorse averageCue currentCue ∧
    ¬ UniformlyNoWorse differenceCue averageCue :=
  ⟨selection_not_uniformly_better_than_mean,
    mean_not_uniformly_better_than_selection, difference_not_uniformly_better_than_mean⟩

end HJepa.Spoofing.Fusion.Aggregation
