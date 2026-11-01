import HJepa.TemporalPrediction.Rollout.Risk
import HJepa.TemporalPrediction.Risk

namespace HJepa.TemporalPrediction

/-- The rollout starts from the observed encoded initial state. -/
theorem rolloutRisk_empty (d : Dynamics) (a b k : ℝ) :
    rolloutRisk d [] a b k = 0 := by
  norm_num [rolloutRisk_exact]

/-- The genuine multistep expectation agrees with the original one-step objective. -/
theorem rolloutRisk_singleton (d : Dynamics) (action : Bool) (a b k : ℝ) :
    rolloutRisk d [action] a b k = loss d action a b k := by
  rw [rolloutRisk_exact, loss_exact]
  simp only [List.length_singleton, mul_one, pow_one, correlation]
  ring

/-- With its optimal one-step predictor, a pure signal encoder has this horizon risk. -/
theorem rolloutRisk_pure_signal (d : Dynamics) (actions : List Bool) :
    rolloutRisk d actions 1 0 d.signalPersistence =
      1 - d.signalPersistence ^ (2 * actions.length) := by
  rw [rolloutRisk_exact]
  simp only [one_pow, zero_pow (by norm_num : (2 : ℕ) ≠ 0), zero_mul, add_zero, one_mul]
  rw [mul_comm 2 actions.length, pow_mul]
  ring

/-- The same calculation holds for a pure nuisance encoder, regardless of task utility. -/
theorem rolloutRisk_pure_nuisance (d : Dynamics) (actions : List Bool) :
    rolloutRisk d actions 0 1 d.nuisancePersistence =
      1 - d.nuisancePersistence ^ (2 * actions.length) := by
  rw [rolloutRisk_exact]
  simp only [one_pow, zero_pow (by norm_num : (2 : ℕ) ≠ 0), zero_mul, zero_add, one_mul]
  rw [mul_comm 2 actions.length, pow_mul]
  ring

end HJepa.TemporalPrediction
