import HJepa.Spoofing.Detection.Risk

namespace HJepa.Spoofing

open HJepa.Abstraction

/-- Both tokens are available: one previous latent and the current latent. -/
structure CuePair where
  previous : Bool → ℝ
  current : Bool → ℝ

noncomputable def currentCue (tokens : CuePair) : Bool → ℝ := tokens.current

noncomputable def averageCue (tokens : CuePair) (label : Bool) : ℝ :=
  (tokens.previous label + tokens.current label) / 2

noncomputable def differenceCue (tokens : CuePair) (label : Bool) : ℝ :=
  (tokens.current label - tokens.previous label) / 2

noncomputable def alternatingCuePair (amplitude : ℝ) : CuePair :=
  ⟨fun label => -(amplitude * signedBit label), fun label => amplitude * signedBit label⟩

noncomputable def skippedCuePair (amplitude : ℝ) : CuePair :=
  ⟨fun label => amplitude * signedBit label, fun _ => 0⟩

noncomputable def sharedCuePair (amplitude : ℝ) : CuePair :=
  ⟨fun label => amplitude * signedBit label, fun label => amplitude * signedBit label⟩

noncomputable def positiveClassifier (x : ℝ) : Bool := decide (0 < x)

theorem positiveClassifier_signed (amplitude : ℝ) (ha : 0 < amplitude) (label : Bool) :
    positiveClassifier (amplitude * signedBit label) = label := by
  have hn : ¬ (0 : ℝ) < -amplitude := by linarith
  cases label <;> simp [positiveClassifier, signedBit, ha, hn]

end HJepa.Spoofing
