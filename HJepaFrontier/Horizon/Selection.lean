import HJepa.TemporalPrediction.Comparison.Optima

/-!
# Queue entry V3: the prediction horizon does not change what is selected

Unreviewed Frontier candidate; see `FUZZ_QUEUE.md`.

A level that predicts `steps` transitions ahead sees each factor with its
persistence raised to the power `steps`. The reviewed library proves that the
lag covariance of a sign factor over `steps` transitions is that power
(`signal_lag_covariance`, `nuisance_lag_covariance`). This module takes the
one-step model with those persistences as the model of direct prediction at
that horizon; it does not re-derive the model from the multi-step process.

Result: for nonnegative persistences in a strict order, every positive horizon
keeps the same factor. Changing the horizon alone cannot make a level prefer
the signal.
-/

namespace HJepa.Frontier.Horizon

noncomputable section

/-- The one-step model whose persistences are those of `steps` transitions. -/
def atHorizon (d : TemporalPrediction.Dynamics) (steps : ℕ) :
    TemporalPrediction.Dynamics where
  signalPersistence := d.signalPersistence ^ steps
  nuisancePersistence := d.nuisancePersistence ^ steps
  signal_nonneg := pow_nonneg d.signal_nonneg steps
  signal_le_one := pow_le_one₀ d.signal_nonneg d.signal_le_one
  nuisance_nonneg := pow_nonneg d.nuisance_nonneg steps
  nuisance_le_one := pow_le_one₀ d.nuisance_nonneg d.nuisance_le_one

end

/-- A strict order of nonnegative reals survives powers and squaring. -/
theorem squared_power_lt (low high : ℝ) (steps : ℕ) (hsteps : steps ≠ 0)
    (hlow : 0 ≤ low) (horder : low < high) :
    (low ^ steps) ^ 2 < (high ^ steps) ^ 2 := by
  have hpower : low ^ steps < high ^ steps := pow_lt_pow_left₀ horder hlow hsteps
  exact pow_lt_pow_left₀ hpower (pow_nonneg hlow steps) (by norm_num : (2 : ℕ) ≠ 0)

/-- If the signal is more persistent, every positive horizon keeps the signal. -/
theorem signal_selected_at_every_horizon (d : TemporalPrediction.Dynamics)
    (steps : ℕ) (hsteps : steps ≠ 0)
    (horder : d.nuisancePersistence < d.signalPersistence)
    (m : TemporalPrediction.Learner)
    (hm : TemporalPrediction.GlobalMinimum (atHorizon d steps) m) :
    TemporalPrediction.taskRisk m = 0 :=
  TemporalPrediction.signal_global_task_zero (atHorizon d steps)
    (squared_power_lt d.nuisancePersistence d.signalPersistence steps hsteps
      d.nuisance_nonneg horder) m hm

/-- If the nuisance is more persistent, every positive horizon keeps the nuisance. -/
theorem nuisance_selected_at_every_horizon (d : TemporalPrediction.Dynamics)
    (steps : ℕ) (hsteps : steps ≠ 0)
    (horder : d.signalPersistence < d.nuisancePersistence)
    (m : TemporalPrediction.Learner)
    (hm : TemporalPrediction.GlobalMinimum (atHorizon d steps) m) :
    TemporalPrediction.taskRisk m = 1 :=
  TemporalPrediction.nuisance_global_task_one (atHorizon d steps)
    (squared_power_lt d.signalPersistence d.nuisancePersistence steps hsteps
      d.signal_nonneg horder) m hm

end HJepa.Frontier.Horizon
