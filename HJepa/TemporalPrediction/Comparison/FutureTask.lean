import HJepa.TemporalPrediction.Comparison.Methods

namespace HJepa.TemporalPrediction

/-- The optimal linear forecast of the common future signal from a raw learner. -/
noncomputable def rawFutureTaskRisk (d : Dynamics) (raw : HJepa.MaskedPrediction.Predictor) : ℝ :=
  futureTaskRisk d true raw.signal raw.nuisance (d.signalPersistence * raw.signal)

/-- The identical future-task evaluation for an autoregressively learned encoder. -/
noncomputable def arFutureTaskRisk (d : Dynamics) (ar : Learner) : ℝ :=
  futureTaskRisk d true ar.signal ar.nuisance (d.signalPersistence * ar.signal)

theorem rawFutureTaskRisk_eq (d : Dynamics) (raw : HJepa.MaskedPrediction.Predictor) :
    rawFutureTaskRisk d raw = 1 - d.signalPersistence ^ 2 +
      d.signalPersistence ^ 2 * HJepa.MaskedPrediction.taskRisk raw := by
  rw [HJepa.MaskedPrediction.taskRisk_eq]
  exact futureTaskRisk_optimal d true raw.signal raw.nuisance raw.normalized

theorem arFutureTaskRisk_eq (d : Dynamics) (ar : Learner) :
    arFutureTaskRisk d ar = 1 - d.signalPersistence ^ 2 +
      d.signalPersistence ^ 2 * taskRisk ar := by
  rw [taskRisk_eq]
  exact futureTaskRisk_optimal d true ar.signal ar.nuisance ar.normalized

theorem rawFutureTaskRisk_optimal (d : Dynamics) (raw : HJepa.MaskedPrediction.Predictor)
    (ell : ℝ) : rawFutureTaskRisk d raw ≤
      futureTaskRisk d true raw.signal raw.nuisance ell := by
  rw [rawFutureTaskRisk_eq, HJepa.MaskedPrediction.taskRisk_eq]
  exact futureTaskRisk_lower_bound d true raw.signal raw.nuisance ell raw.normalized

theorem arFutureTaskRisk_optimal (d : Dynamics) (ar : Learner) (ell : ℝ) :
    arFutureTaskRisk d ar ≤ futureTaskRisk d true ar.signal ar.nuisance ell := by
  rw [arFutureTaskRisk_eq, taskRisk_eq]
  exact futureTaskRisk_lower_bound d true ar.signal ar.nuisance ell ar.normalized

theorem ar_future_comparison_iff (d : Dynamics) (raw : HJepa.MaskedPrediction.Predictor)
    (ar : Learner) (h : 0 < d.signalPersistence) :
    arFutureTaskRisk d ar < rawFutureTaskRisk d raw ↔
      taskRisk ar < HJepa.MaskedPrediction.taskRisk raw := by
  rw [arFutureTaskRisk_eq, rawFutureTaskRisk_eq, add_lt_add_iff_left]
  exact mul_lt_mul_iff_right₀ (sq_pos_of_pos h)

theorem raw_future_comparison_iff (d : Dynamics) (raw : HJepa.MaskedPrediction.Predictor)
    (ar : Learner) (h : 0 < d.signalPersistence) :
    rawFutureTaskRisk d raw < arFutureTaskRisk d ar ↔
      HJepa.MaskedPrediction.taskRisk raw < taskRisk ar := by
  rw [arFutureTaskRisk_eq, rawFutureTaskRisk_eq, add_lt_add_iff_left]
  exact mul_lt_mul_iff_right₀ (sq_pos_of_pos h)

theorem autoregressive_future_better_iff (p : Problem)
    (raw : HJepa.MaskedPrediction.Predictor) (ar : Learner)
    (hr : RawOptimal p raw) (ha : AROptimal p ar)
    (nr : p.signalScale ^ 2 ≠ p.nuisanceScale ^ 2)
    (na : p.dynamics.signalPersistence ^ 2 ≠ p.dynamics.nuisancePersistence ^ 2)
    (positive : 0 < p.dynamics.signalPersistence) :
    arFutureTaskRisk p.dynamics ar < rawFutureTaskRisk p.dynamics raw ↔
      p.signalScale ^ 2 < p.nuisanceScale ^ 2 ∧
      p.dynamics.nuisancePersistence ^ 2 < p.dynamics.signalPersistence ^ 2 := by
  rw [ar_future_comparison_iff _ _ _ positive]
  exact autoregressive_better_iff p raw ar hr ha nr na

theorem reconstruction_future_better_iff (p : Problem)
    (raw : HJepa.MaskedPrediction.Predictor) (ar : Learner)
    (hr : RawOptimal p raw) (ha : AROptimal p ar)
    (nr : p.signalScale ^ 2 ≠ p.nuisanceScale ^ 2)
    (na : p.dynamics.signalPersistence ^ 2 ≠ p.dynamics.nuisancePersistence ^ 2)
    (positive : 0 < p.dynamics.signalPersistence) :
    rawFutureTaskRisk p.dynamics raw < arFutureTaskRisk p.dynamics ar ↔
      p.nuisanceScale ^ 2 < p.signalScale ^ 2 ∧
      p.dynamics.signalPersistence ^ 2 < p.dynamics.nuisancePersistence ^ 2 := by
  rw [raw_future_comparison_iff _ _ _ positive]
  exact reconstruction_better_iff p raw ar hr ha nr na

/-- An unpredictable future signal makes every encoder tie at risk one. -/
theorem zero_signal_future_tie (d : Dynamics) (raw : HJepa.MaskedPrediction.Predictor)
    (ar : Learner) (h : d.signalPersistence = 0) :
    rawFutureTaskRisk d raw = 1 ∧ arFutureTaskRisk d ar = 1 := by
  simp [rawFutureTaskRisk_eq, arFutureTaskRisk_eq, h]

end HJepa.TemporalPrediction
