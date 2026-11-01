import HJepa.TemporalPrediction.Controls.FutureRawOptima
import HJepa.TemporalPrediction.Comparison.Methods

namespace HJepa.TemporalPrediction.Controls

/-- Future raw prediction weights each coordinate by its predictable amplitude. -/
noncomputable def matchedProblem (d : Dynamics) (s n : ℝ) : Problem :=
  ⟨s * d.signalPersistence, n * d.nuisancePersistence, d⟩

/-- Both objectives predict the same future time slice from the same context. -/
theorem futureLatent_better (d : Dynamics) (action : Bool) (s n : ℝ)
    (raw : MaskedPrediction.Predictor) (ar : Learner)
    (hr : FutureRawGlobal d action s n raw) (ha : GlobalMinimum d ar)
    (gr : (s * d.signalPersistence) ^ 2 < (n * d.nuisancePersistence) ^ 2)
    (ga : d.nuisancePersistence ^ 2 < d.signalPersistence ^ 2) :
    taskRisk ar < MaskedPrediction.taskRisk raw :=
  autoregressive_better (matchedProblem d s n) raw ar
    ((futureRawGlobal_iff d action s n raw).mp hr) ha gr ga

theorem futureRaw_better (d : Dynamics) (action : Bool) (s n : ℝ)
    (raw : MaskedPrediction.Predictor) (ar : Learner)
    (hr : FutureRawGlobal d action s n raw) (ha : GlobalMinimum d ar)
    (gr : (n * d.nuisancePersistence) ^ 2 < (s * d.signalPersistence) ^ 2)
    (ga : d.signalPersistence ^ 2 < d.nuisancePersistence ^ 2) :
    MaskedPrediction.taskRisk raw < taskRisk ar :=
  reconstruction_better (matchedProblem d s n) raw ar
    ((futureRawGlobal_iff d action s n raw).mp hr) ha gr ga

/-- Outside ties, predictable raw energy and latent persistence decide the order. -/
theorem futureLatent_better_iff (d : Dynamics) (action : Bool) (s n : ℝ)
    (raw : MaskedPrediction.Predictor) (ar : Learner)
    (hr : FutureRawGlobal d action s n raw) (ha : GlobalMinimum d ar)
    (nr : (s * d.signalPersistence) ^ 2 ≠ (n * d.nuisancePersistence) ^ 2)
    (na : d.signalPersistence ^ 2 ≠ d.nuisancePersistence ^ 2) :
    taskRisk ar < MaskedPrediction.taskRisk raw ↔
      (s * d.signalPersistence) ^ 2 < (n * d.nuisancePersistence) ^ 2 ∧
      d.nuisancePersistence ^ 2 < d.signalPersistence ^ 2 :=
  autoregressive_better_iff (matchedProblem d s n) raw ar
    ((futureRawGlobal_iff d action s n raw).mp hr) ha nr na

theorem futureRaw_better_iff (d : Dynamics) (action : Bool) (s n : ℝ)
    (raw : MaskedPrediction.Predictor) (ar : Learner)
    (hr : FutureRawGlobal d action s n raw) (ha : GlobalMinimum d ar)
    (nr : (s * d.signalPersistence) ^ 2 ≠ (n * d.nuisancePersistence) ^ 2)
    (na : d.signalPersistence ^ 2 ≠ d.nuisancePersistence ^ 2) :
    MaskedPrediction.taskRisk raw < taskRisk ar ↔
      (n * d.nuisancePersistence) ^ 2 < (s * d.signalPersistence) ^ 2 ∧
      d.signalPersistence ^ 2 < d.nuisancePersistence ^ 2 :=
  reconstruction_better_iff (matchedProblem d s n) raw ar
    ((futureRawGlobal_iff d action s n raw).mp hr) ha nr na

theorem matched_strict_order_ties (d : Dynamics) (action : Bool) (s n : ℝ)
    (raw : MaskedPrediction.Predictor) (ar : Learner)
    (hr : FutureRawGlobal d action s n raw) (ha : GlobalMinimum d ar)
    (same : ((n * d.nuisancePersistence) ^ 2 < (s * d.signalPersistence) ^ 2 ∧
        d.nuisancePersistence ^ 2 < d.signalPersistence ^ 2) ∨
      ((s * d.signalPersistence) ^ 2 < (n * d.nuisancePersistence) ^ 2 ∧
        d.signalPersistence ^ 2 < d.nuisancePersistence ^ 2)) :
    MaskedPrediction.taskRisk raw = taskRisk ar :=
  matching_strict_order_ties (matchedProblem d s n) raw ar
    ((futureRawGlobal_iff d action s n raw).mp hr) ha same

end HJepa.TemporalPrediction.Controls
