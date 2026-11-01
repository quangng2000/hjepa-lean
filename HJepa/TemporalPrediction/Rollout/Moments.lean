import HJepa.TemporalPrediction.Rollout.Quadratic

namespace HJepa.TemporalPrediction

open HJepa.Abstraction
open scoped BigOperators

/-- Average the Markov expectation over the independent fair initial signs. -/
noncomputable def populationExpectation (d : Dynamics) (actions : List Bool)
    (f : (ℝ × ℝ) → (ℝ × ℝ) → ℝ) : ℝ :=
  ∑ initial, HJepa.MaskedPrediction.fairInput.weight initial *
    pathExpectation d actions (f (current initial)) (current initial)

theorem pathExpectation_signal (d : Dynamics) (actions : List Bool)
    (c : ℝ) (x : ℝ × ℝ) :
    pathExpectation d actions (fun y => c * y.1) x =
      c * actionProduct actions * d.signalPersistence ^ actions.length * x.1 := by
  have heq : (fun y : ℝ × ℝ => c * y.1) =
      (RolloutQuadratic.mk 0 c 0 0 0 0).eval := by
    funext y
    simp [RolloutQuadratic.eval]
  rw [heq, pathExpectation_quadratic]
  simp [RolloutQuadratic.transport, RolloutQuadratic.eval]

theorem pathExpectation_nuisance (d : Dynamics) (actions : List Bool)
    (c : ℝ) (x : ℝ × ℝ) :
    pathExpectation d actions (fun y => c * y.2) x =
      c * actionProduct actions * d.nuisancePersistence ^ actions.length * x.2 := by
  have heq : (fun y : ℝ × ℝ => c * y.2) =
      (RolloutQuadratic.mk 0 0 c 0 0 0).eval := by
    funext y
    simp [RolloutQuadratic.eval]
  rw [heq, pathExpectation_quadratic]
  simp [RolloutQuadratic.transport, RolloutQuadratic.eval]

/-- Actual signal lag covariance, including the known product of action signs. -/
theorem signal_lag_covariance (d : Dynamics) (actions : List Bool) :
    populationExpectation d actions (fun x y => x.1 * y.1) =
      actionProduct actions * d.signalPersistence ^ actions.length := by
  unfold populationExpectation
  simp only [pathExpectation_signal]
  norm_num [HJepa.MaskedPrediction.fairInput, current, signedBit,
    Fintype.sum_prod_type, Fintype.sum_bool]
  ring

theorem nuisance_lag_covariance (d : Dynamics) (actions : List Bool) :
    populationExpectation d actions (fun x y => x.2 * y.2) =
      actionProduct actions * d.nuisancePersistence ^ actions.length := by
  unfold populationExpectation
  simp only [pathExpectation_nuisance]
  norm_num [HJepa.MaskedPrediction.fairInput, current, signedBit,
    Fintype.sum_prod_type, Fintype.sum_bool]
  ring

/-- Independence of the initial signs removes the cross-coordinate lag moment. -/
theorem cross_lag_covariance (d : Dynamics) (actions : List Bool) :
    populationExpectation d actions (fun x y => x.1 * y.2) = 0 := by
  unfold populationExpectation
  simp only [pathExpectation_nuisance]
  norm_num [HJepa.MaskedPrediction.fairInput, current, signedBit,
    Fintype.sum_prod_type, Fintype.sum_bool]

end HJepa.TemporalPrediction
