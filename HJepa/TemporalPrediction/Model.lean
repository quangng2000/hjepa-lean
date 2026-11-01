import HJepa.TemporalPrediction.Dynamics
import HJepa.MaskedPrediction.Risk

/-! Temporal prediction uses the same trainable encoder on both time slices.
The target therefore changes with the encoder; it is not a frozen feature map. -/

namespace HJepa.TemporalPrediction

open HJepa.Abstraction
open scoped BigOperators

noncomputable abbrev encoder := HJepa.MaskedPrediction.encoder
abbrev Normalized := HJepa.MaskedPrediction.Normalized
noncomputable abbrev downstreamRisk := HJepa.MaskedPrediction.downstreamRisk

/-- The action-conditioned scalar autoregressive predictor. -/
noncomputable def predictor (action : Bool) (k z : ℝ) : ℝ :=
  k * signedBit action * z

/-- Actual finite expected error of predicting the learned future latent. -/
noncomputable def loss (d : Dynamics) (action : Bool) (a b k : ℝ) : ℝ :=
  ∑ t, (trajectoryLaw d).weight t *
    (encoder a b (future action t.1 t.2) -
      predictor action k (encoder a b (current t.1))) ^ 2

noncomputable def correlation (d : Dynamics) (a b : ℝ) : ℝ :=
  d.signalPersistence * a ^ 2 + d.nuisancePersistence * b ^ 2

/-- Optimal one-step temporal loss for a normalized encoder. -/
noncomputable def reducedLoss (d : Dynamics) (a b : ℝ) : ℝ :=
  1 - correlation d a b ^ 2

/-- Both encoder coefficients vanish at the unconstrained collapsed solution. -/
theorem collapsed_loss (d : Dynamics) (action : Bool) (k : ℝ) :
    loss d action 0 0 k = 0 := by
  simp [loss, encoder, HJepa.MaskedPrediction.encoder, predictor]

theorem collapse_not_normalized : ¬ Normalized 0 0 := by
  norm_num [Normalized, HJepa.MaskedPrediction.Normalized]

end HJepa.TemporalPrediction
