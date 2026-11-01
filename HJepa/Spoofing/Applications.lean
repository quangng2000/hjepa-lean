import HJepa.Spoofing.Filters.Causal
import HJepa.Spoofing.Detection.Alternating
import HJepa.Spoofing.Detection.Skipped

namespace HJepa.Spoofing

/-- Place both available cue tokens in a causal window anchored at time two. -/
noncomputable def cueSequence (pair : CuePair) (label : Bool) (i : ℤ) : ℝ :=
  if i = 2 then pair.current label else if i = 1 then pair.previous label else 0

noncomputable def halfDifferenceKernel : Kernel3 := ⟨1 / 2, -1 / 2, 0⟩

theorem selected_pair_eq (pair : CuePair) (label : Bool) :
    applyKernel selectionKernel (cueSequence pair label) 1 = currentCue pair label := by
  simp [selection_kernel_eq, cueSequence, currentCue]

theorem mean_pair_eq (pair : CuePair) (label : Bool) :
    applyKernel meanKernel (cueSequence pair label) 1 = averageCue pair label := by
  simp only [mean_kernel_eq, cueSequence, averageCue]
  norm_num
  ring

theorem difference_pair_eq (pair : CuePair) (label : Bool) :
    applyKernel halfDifferenceKernel (cueSequence pair label) 1 = differenceCue pair label := by
  simp only [applyKernel, halfDifferenceKernel, cueSequence, differenceCue]
  norm_num
  ring

/-- An actual causal Conv1d specialization separates the alternating cue. -/
theorem causal_alternating_perfect (a : ℝ) (ha : 0 < a) :
    toyError (fun label => applyKernel halfDifferenceKernel
      (cueSequence (alternatingCuePair a) label) 1) positiveClassifier = 0 := by
  simpa only [difference_pair_eq] using alternating_difference_perfect a ha

theorem causal_alternating_mean_chance (a : ℝ) (head : ℝ → Bool) :
    toyError (fun label => applyKernel meanKernel
      (cueSequence (alternatingCuePair a) label) 1) head = 1 / 2 := by
  simpa only [mean_pair_eq] using alternating_average_chance a head

theorem causal_skipped_selection_chance (a : ℝ) (head : ℝ → Bool) :
    toyError (fun label => applyKernel selectionKernel
      (cueSequence (skippedCuePair a) label) 1) head = 1 / 2 := by
  simpa only [selected_pair_eq] using skipped_current_chance a head

end HJepa.Spoofing
