import HJepa.TemporalPrediction.Risk

/-! A matched-horizon control predicts raw future coordinates from the same
current encoder and known action used by learned-latent temporal prediction. -/

namespace HJepa.TemporalPrediction.Controls

open HJepa.Abstraction
open scoped BigOperators

/-- Actual expected raw future error, with two action-conditioned linear heads. -/
noncomputable def futureRawLoss (d : Dynamics) (action : Bool)
    (s n a b p q : ℝ) : ℝ :=
  ∑ t, (trajectoryLaw d).weight t *
    ((s * (future action t.1 t.2).1 -
      predictor action p (encoder a b (current t.1))) ^ 2 +
    (n * (future action t.1 t.2).2 -
      predictor action q (encoder a b (current t.1))) ^ 2)

/-- Irreducible coordinate noise; it does not depend on the encoder or heads. -/
noncomputable def futureRawNoise (d : Dynamics) (s n : ℝ) : ℝ :=
  s ^ 2 * (1 - d.signalPersistence ^ 2) +
    n ^ 2 * (1 - d.nuisancePersistence ^ 2)

/-- Direct expectation over the controlled trajectories, without normalization. -/
theorem futureRawLoss_exact (d : Dynamics) (action : Bool)
    (s n a b p q : ℝ) :
    futureRawLoss d action s n a b p q = s ^ 2 + n ^ 2 +
      (p ^ 2 + q ^ 2) * (a ^ 2 + b ^ 2) -
      2 * s * p * d.signalPersistence * a -
      2 * n * q * d.nuisancePersistence * b := by
  cases action <;>
    norm_num [futureRawLoss, trajectoryLaw, innovationLaw, signWeight, future,
      step, current, predictor, encoder, MaskedPrediction.encoder,
      signedBit, Fintype.sum_prod_type, Fintype.sum_bool] <;> ring

/-- Future raw prediction allocates capacity by predictable coordinate energies.
The identity holds for arbitrary encoder and head coefficients. -/
theorem futureRawLoss_eq_maskedLoss (d : Dynamics) (action : Bool)
    (s n a b p q : ℝ) :
    futureRawLoss d action s n a b p q = futureRawNoise d s n +
      MaskedPrediction.loss (s * d.signalPersistence) (n * d.nuisancePersistence)
        a b p q := by
  rw [futureRawLoss_exact, MaskedPrediction.loss_exact]
  unfold futureRawNoise
  ring

theorem futureRawLoss_action_invariant (d : Dynamics) (s n a b p q : ℝ) :
    futureRawLoss d false s n a b p q = futureRawLoss d true s n a b p q := by
  rw [futureRawLoss_exact, futureRawLoss_exact]

end HJepa.TemporalPrediction.Controls
