import HJepa.MaskedPrediction.Model

namespace HJepa.TokenAggregation.Examples

open HJepa.Abstraction
open HJepa.MaskedPrediction

/-- Two already encoded, consecutive scalar latent tokens. -/
structure LatentPair where
  first : Input → ℝ
  second : Input → ℝ

/-- The stride-two/window-one state construction retains the first token. -/
noncomputable def selectedRisk (tokens : LatentPair) (h : ℝ → ℝ) : ℝ :=
  risk fairInput taskSign (fun ω => h (tokens.first ω))

/-- Pool the input tokens before applying any subsequent encoder/readout. -/
noncomputable def averagedRisk (tokens : LatentPair) (h : ℝ → ℝ) : ℝ :=
  risk fairInput taskSign (fun ω => h ((tokens.first ω + tokens.second ω) / 2))

noncomputable def stablePair : LatentPair :=
  ⟨fun ω => taskSign ω + nuisanceSign ω, fun ω => taskSign ω - nuisanceSign ω⟩

noncomputable def alternatingPair : LatentPair :=
  ⟨taskSign, fun ω => -taskSign ω⟩

noncomputable def skippedPair : LatentPair := ⟨nuisanceSign, taskSign⟩

end HJepa.TokenAggregation.Examples
