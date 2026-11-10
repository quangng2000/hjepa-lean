import HJepa.TemporalPrediction.Comparison.Optima

/-!
# Queue entry V2: a nuisance that is redrawn across views is not learned

Unreviewed Frontier candidate; see `FUZZ_QUEUE.md`.

In the two-sign temporal model the two views of a positive pair are the current
and the future input. A nuisance drawn independently for the second view keeps
no correlation across the pair, which the model expresses as persistence zero.

This module states only the consequence for a `Dynamics` with that persistence.
Constructing the pair law that produces it is queue entry V1, so nothing here
yet connects the result to a rendering pipeline.
-/

namespace HJepa.Frontier.Views

/-- If the nuisance keeps no correlation across the two views and the signal
keeps some, every joint optimum of encoder and predictor retains the signal. -/
theorem zero_nuisance_persistence_task_zero (d : TemporalPrediction.Dynamics)
    (hnuisance : d.nuisancePersistence = 0) (hsignal : 0 < d.signalPersistence)
    (m : TemporalPrediction.Learner) (hm : TemporalPrediction.GlobalMinimum d m) :
    TemporalPrediction.taskRisk m = 0 := by
  have hgap : d.nuisancePersistence ^ 2 < d.signalPersistence ^ 2 := by
    rw [hnuisance, zero_pow (by norm_num : (2 : ℕ) ≠ 0)]
    exact pow_pos hsignal 2
  exact TemporalPrediction.signal_global_task_zero d hgap m hm

end HJepa.Frontier.Views
