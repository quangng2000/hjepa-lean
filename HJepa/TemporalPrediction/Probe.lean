import HJepa.TemporalPrediction.Risk

namespace HJepa.TemporalPrediction

open HJepa.Abstraction
open scoped BigOperators

/-- A common downstream forecast target: the future signal, with known action. -/
noncomputable def futureTaskRisk (d : Dynamics) (action : Bool) (a b ell : ℝ) : ℝ :=
  ∑ t, (trajectoryLaw d).weight t *
    ((future action t.1 t.2).1 -
      predictor action ell (encoder a b (current t.1))) ^ 2

/-- Future task prediction integrates exactly the same controlled trajectories. -/
theorem futureTaskRisk_exact (d : Dynamics) (action : Bool) (a b ell : ℝ) :
    futureTaskRisk d action a b ell =
      1 + ell ^ 2 * (a ^ 2 + b ^ 2) - 2 * ell * d.signalPersistence * a := by
  cases action <;>
    norm_num [futureTaskRisk, trajectoryLaw, innovationLaw, signWeight, future,
      step, current, predictor, encoder, HJepa.MaskedPrediction.encoder,
      signedBit, Fintype.sum_prod_type, Fintype.sum_bool] <;> ring

theorem futureTaskRisk_complete_square (d : Dynamics) (action : Bool)
    (a b ell : ℝ) (h : Normalized a b) :
    futureTaskRisk d action a b ell = 1 - d.signalPersistence ^ 2 +
      d.signalPersistence ^ 2 * b ^ 2 + (ell - d.signalPersistence * a) ^ 2 := by
  rw [futureTaskRisk_exact, h]
  unfold Normalized HJepa.MaskedPrediction.Normalized at h
  nlinarith [congrArg (fun t : ℝ => d.signalPersistence ^ 2 * t) h]

theorem futureTaskRisk_optimal (d : Dynamics) (action : Bool) (a b : ℝ)
    (h : Normalized a b) :
    futureTaskRisk d action a b (d.signalPersistence * a) =
      1 - d.signalPersistence ^ 2 + d.signalPersistence ^ 2 * b ^ 2 := by
  rw [futureTaskRisk_complete_square d action a b _ h]
  simp

theorem futureTaskRisk_lower_bound (d : Dynamics) (action : Bool) (a b ell : ℝ)
    (h : Normalized a b) :
    1 - d.signalPersistence ^ 2 + d.signalPersistence ^ 2 * b ^ 2 ≤
      futureTaskRisk d action a b ell := by
  rw [futureTaskRisk_complete_square d action a b ell h]
  exact le_add_of_nonneg_right (sq_nonneg _)

theorem futureTaskRisk_action_invariant (d : Dynamics) (a b ell : ℝ) :
    futureTaskRisk d false a b ell = futureTaskRisk d true a b ell := by
  rw [futureTaskRisk_exact, futureTaskRisk_exact]

end HJepa.TemporalPrediction
