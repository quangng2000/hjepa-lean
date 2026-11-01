import HJepa.TemporalPrediction.Rollout.Corollaries

namespace HJepa.TemporalPrediction

/-- Correlation after removing the known action signs from a horizon target. -/
noncomputable def horizonCorrelation (d : Dynamics) (horizon : ℕ) (a b : ℝ) : ℝ :=
  a ^ 2 * d.signalPersistence ^ horizon + b ^ 2 * d.nuisancePersistence ^ horizon

/-- The best effective horizon coefficient is the actual horizon correlation.
A recurrent predictor supplies `k^h`, so one-step optimality need not attain it. -/
theorem rolloutRisk_complete_square (d : Dynamics) (actions : List Bool)
    (a b k : ℝ) (h : Normalized a b) :
    rolloutRisk d actions a b k =
      (k ^ actions.length - horizonCorrelation d actions.length a b) ^ 2 +
        1 - horizonCorrelation d actions.length a b ^ 2 := by
  have hn : a ^ 2 + b ^ 2 = 1 := h
  have hs := congrArg (fun t : ℝ => (1 + (k ^ actions.length) ^ 2) * t) hn
  rw [rolloutRisk_exact]
  unfold horizonCorrelation
  rw [mul_comm 2 actions.length, pow_mul]
  nlinarith [hs]

/-- Mixtures of unequal persistence rates do not compose as a single rate. -/
theorem two_step_correlation_gap (d : Dynamics) (a b : ℝ) (h : Normalized a b) :
    horizonCorrelation d 2 a b - correlation d a b ^ 2 =
      a ^ 2 * b ^ 2 * (d.signalPersistence - d.nuisancePersistence) ^ 2 := by
  have hn : a ^ 2 + b ^ 2 = 1 := h
  have hs := congrArg (fun t : ℝ =>
    (a ^ 2 * d.signalPersistence ^ 2 + b ^ 2 * d.nuisancePersistence ^ 2) * t) hn
  unfold horizonCorrelation correlation
  nlinarith [hs]

/-- Exact two-step excess of repeatedly applying the optimal one-step coefficient. -/
theorem two_step_trained_excess (d : Dynamics) (first second : Bool)
    (a b : ℝ) (h : Normalized a b) :
    rolloutRisk d [first, second] a b (correlation d a b) =
      1 - horizonCorrelation d 2 a b ^ 2 +
        (a ^ 2 * b ^ 2 * (d.signalPersistence - d.nuisancePersistence) ^ 2) ^ 2 := by
  rw [rolloutRisk_complete_square d [first, second] a b _ h]
  simp only [List.length_cons, List.length_nil, Nat.zero_add]
  have hg := two_step_correlation_gap d a b h
  have he : correlation d a b ^ 2 - horizonCorrelation d 2 a b =
      -(a ^ 2 * b ^ 2 * (d.signalPersistence - d.nuisancePersistence) ^ 2) := by
    linarith
  rw [he]
  ring

end HJepa.TemporalPrediction
