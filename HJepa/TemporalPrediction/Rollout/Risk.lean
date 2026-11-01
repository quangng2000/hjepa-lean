import HJepa.TemporalPrediction.Rollout.Moments

namespace HJepa.TemporalPrediction

open HJepa.Abstraction
open scoped BigOperators

/-- The squared error against the learned encoder is a quadratic observable. -/
noncomputable def errorQuadratic (a b prediction : ℝ) : RolloutQuadratic :=
  ⟨prediction ^ 2, -2 * prediction * a, -2 * prediction * b, a ^ 2, b ^ 2, 2 * a * b⟩

theorem errorQuadratic_eval (a b prediction : ℝ) (x : ℝ × ℝ) :
    (errorQuadratic a b prediction).eval x = (encoder a b x - prediction) ^ 2 := by
  simp only [errorQuadratic, RolloutQuadratic.eval, encoder,
    HJepa.MaskedPrediction.encoder]
  ring

theorem pathExpectation_encoder_error (d : Dynamics) (actions : List Bool)
    (a b prediction : ℝ) (x : ℝ × ℝ) :
    pathExpectation d actions (fun y => (encoder a b y - prediction) ^ 2) x =
      ((errorQuadratic a b prediction).transport d actions).eval x := by
  simp_rw [← errorQuadratic_eval]
  exact pathExpectation_quadratic _ d actions x

/-- Population error against the true encoded future after a fully recurrent rollout. -/
noncomputable def rolloutRisk (d : Dynamics) (actions : List Bool) (a b k : ℝ) : ℝ :=
  populationExpectation d actions fun initial final =>
    (encoder a b final - rollout k actions (encoder a b initial)) ^ 2

/-- Closed form derived from the finite Markov process, valid at every horizon. -/
theorem rolloutRisk_exact (d : Dynamics) (actions : List Bool) (a b k : ℝ) :
    rolloutRisk d actions a b k =
      a ^ 2 * (1 + k ^ (2 * actions.length) -
        2 * k ^ actions.length * d.signalPersistence ^ actions.length) +
      b ^ 2 * (1 + k ^ (2 * actions.length) -
        2 * k ^ actions.length * d.nuisancePersistence ^ actions.length) := by
  unfold rolloutRisk populationExpectation
  simp_rw [rollout_closed_form, pathExpectation_encoder_error]
  norm_num [errorQuadratic, RolloutQuadratic.transport, RolloutQuadratic.eval,
    HJepa.MaskedPrediction.fairInput, current, signedBit, encoder,
    HJepa.MaskedPrediction.encoder, Fintype.sum_prod_type, Fintype.sum_bool]
  simp only [mul_pow, actionProduct_sq, mul_one, pow_mul]
  ring_nf
  rw [actionProduct_sq]
  ring

/-- Any two known action plans with the same horizon have the same risk. -/
theorem rolloutRisk_action_invariant (d : Dynamics) (actions other : List Bool)
    (h : actions.length = other.length) (a b k : ℝ) :
    rolloutRisk d actions a b k = rolloutRisk d other a b k := by
  simp only [rolloutRisk_exact, h]

end HJepa.TemporalPrediction
