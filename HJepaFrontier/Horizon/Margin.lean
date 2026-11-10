import HJepaFrontier.Horizon.Selection
import HJepa.TemporalPrediction.Comparison.Tolerance

/-!
# Queue entry V3: the horizon changes the margin, not the selection

Unreviewed Frontier candidate; see `FUZZ_QUEUE.md`.

The squared signal persistence at horizon `steps` only shrinks as the horizon
grows. Two readings follow for a learner whose objective is within `η` of the
optimum:

* good views (nuisance persistence zero): task risk is at most
  `η / ρ_S^(2·steps)`, a guarantee that weakens with the horizon;
* a factor constant within a recording (nuisance persistence one): task risk is
  at least `1 - η / (1 - ρ_S^(2·steps))`, a lower bound that tightens with it.

So longer horizons make a good-view level harder to identify and make a
shared-recording level more firmly attached to the constant factor.
-/

namespace HJepa.Frontier.Horizon

/-- Longer horizons shrink the squared signal persistence. -/
theorem signal_power_antitone (d : TemporalPrediction.Dynamics) {near far : ℕ}
    (hle : near ≤ far) :
    (d.signalPersistence ^ far) ^ 2 ≤ (d.signalPersistence ^ near) ^ 2 :=
  pow_le_pow_left₀ (pow_nonneg d.signal_nonneg far)
    (pow_le_pow_of_le_one d.signal_nonneg d.signal_le_one hle) 2

/-- Good views: an `η`-approximate optimum at horizon `steps` has task risk at
most `η` divided by the squared signal persistence at that horizon. -/
theorem good_view_risk_le (d : TemporalPrediction.Dynamics) (steps : ℕ)
    (hsteps : steps ≠ 0) (hnuisance : d.nuisancePersistence = 0)
    (hsignal : 0 < d.signalPersistence) (m : TemporalPrediction.Learner) (η : ℝ)
    (h : TemporalPrediction.objective (atHorizon d steps) m ≤
      1 - (d.signalPersistence ^ steps) ^ 2 + η) :
    TemporalPrediction.taskRisk m ≤ η / (d.signalPersistence ^ steps) ^ 2 := by
  have horder : d.nuisancePersistence < d.signalPersistence := by
    rw [hnuisance]
    exact hsignal
  have hgap : (atHorizon d steps).nuisancePersistence ^ 2 <
      (atHorizon d steps).signalPersistence ^ 2 :=
    squared_power_lt d.nuisancePersistence d.signalPersistence steps hsteps
      d.nuisance_nonneg horder
  have hbound := TemporalPrediction.signal_risk_le (atHorizon d steps) m η hgap h
  have hzero : (atHorizon d steps).nuisancePersistence ^ 2 = 0 := by
    show (d.nuisancePersistence ^ steps) ^ 2 = 0
    rw [hnuisance, zero_pow hsteps, zero_pow (by norm_num : (2 : ℕ) ≠ 0)]
  rw [hzero, sub_zero] at hbound
  exact hbound

/-- A factor constant within a recording: an `η`-approximate optimum at horizon
`steps` has task risk at least one minus `η` over the nuisance's margin. -/
theorem constant_nuisance_risk_ge (d : TemporalPrediction.Dynamics) (steps : ℕ)
    (hsteps : steps ≠ 0) (hnuisance : d.nuisancePersistence = 1)
    (hsignal : d.signalPersistence < 1) (m : TemporalPrediction.Learner) (η : ℝ)
    (h : TemporalPrediction.objective (atHorizon d steps) m ≤
      1 - (d.nuisancePersistence ^ steps) ^ 2 + η) :
    1 - η / (1 - (d.signalPersistence ^ steps) ^ 2) ≤
      TemporalPrediction.taskRisk m := by
  have horder : d.signalPersistence < d.nuisancePersistence := by
    rw [hnuisance]
    exact hsignal
  have hgap : (atHorizon d steps).signalPersistence ^ 2 <
      (atHorizon d steps).nuisancePersistence ^ 2 :=
    squared_power_lt d.signalPersistence d.nuisancePersistence steps hsteps
      d.signal_nonneg horder
  have hbound := TemporalPrediction.nuisance_risk_ge (atHorizon d steps) m η hgap h
  have hone : (atHorizon d steps).nuisancePersistence ^ 2 = 1 := by
    show (d.nuisancePersistence ^ steps) ^ 2 = 1
    rw [hnuisance, one_pow, one_pow]
  rw [hone] at hbound
  exact hbound

end HJepa.Frontier.Horizon
