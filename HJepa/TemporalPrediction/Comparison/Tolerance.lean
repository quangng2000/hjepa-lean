import HJepa.TemporalPrediction.Comparison.Methods
import HJepa.MaskedPrediction.Tolerance

namespace HJepa.TemporalPrediction

theorem signal_tolerance (d : Dynamics) (m : Learner) (η : ℝ)
    (h : objective d m ≤ 1 - d.signalPersistence ^ 2 + η) :
    (d.signalPersistence ^ 2 - d.nuisancePersistence ^ 2) * taskRisk m ≤ η := by
  rw [objective_signal_form] at h
  linarith [mixingPenalty_nonneg d m, headPenalty_nonneg d m]

theorem nuisance_tolerance (d : Dynamics) (m : Learner) (η : ℝ)
    (h : objective d m ≤ 1 - d.nuisancePersistence ^ 2 + η) :
    (d.nuisancePersistence ^ 2 - d.signalPersistence ^ 2) * (1 - taskRisk m) ≤ η := by
  rw [objective_nuisance_form] at h
  linarith [mixingPenalty_nonneg d m, headPenalty_nonneg d m]

theorem signal_risk_le (d : Dynamics) (m : Learner) (η : ℝ)
    (gap : d.nuisancePersistence ^ 2 < d.signalPersistence ^ 2)
    (h : objective d m ≤ 1 - d.signalPersistence ^ 2 + η) :
    taskRisk m ≤ η / (d.signalPersistence ^ 2 - d.nuisancePersistence ^ 2) := by
  apply (le_div_iff₀ (sub_pos.mpr gap)).mpr
  simpa [mul_comm] using signal_tolerance d m η h

theorem nuisance_risk_ge (d : Dynamics) (m : Learner) (η : ℝ)
    (gap : d.signalPersistence ^ 2 < d.nuisancePersistence ^ 2)
    (h : objective d m ≤ 1 - d.nuisancePersistence ^ 2 + η) :
    1 - η / (d.nuisancePersistence ^ 2 - d.signalPersistence ^ 2) ≤ taskRisk m := by
  have hb : 1 - taskRisk m ≤ η / (d.nuisancePersistence ^ 2 - d.signalPersistence ^ 2) := by
    apply (le_div_iff₀ (sub_pos.mpr gap)).mpr
    simpa [mul_comm] using nuisance_tolerance d m η h
  linarith

theorem approximate_autoregressive_better (p : Problem) (raw : HJepa.MaskedPrediction.Predictor)
    (ar : Learner) (εr εa : ℝ)
    (gr : p.signalScale ^ 2 < p.nuisanceScale ^ 2)
    (ga : p.dynamics.nuisancePersistence ^ 2 < p.dynamics.signalPersistence ^ 2)
    (hr : HJepa.MaskedPrediction.objective p.signalScale p.nuisanceScale raw ≤ p.signalScale ^ 2 + εr)
    (ha : objective p.dynamics ar ≤ 1 - p.dynamics.signalPersistence ^ 2 + εa)
    (small : εa / (p.dynamics.signalPersistence ^ 2 - p.dynamics.nuisancePersistence ^ 2) +
      εr / (p.nuisanceScale ^ 2 - p.signalScale ^ 2) < 1) :
    taskRisk ar < HJepa.MaskedPrediction.taskRisk raw := by
  have har := signal_risk_le _ ar εa ga ha
  have hraw := HJepa.MaskedPrediction.nuisance_risk_ge _ _ εr raw gr hr
  linarith

theorem approximate_reconstruction_better (p : Problem) (raw : HJepa.MaskedPrediction.Predictor)
    (ar : Learner) (εr εa : ℝ)
    (gr : p.nuisanceScale ^ 2 < p.signalScale ^ 2)
    (ga : p.dynamics.signalPersistence ^ 2 < p.dynamics.nuisancePersistence ^ 2)
    (hr : HJepa.MaskedPrediction.objective p.signalScale p.nuisanceScale raw ≤ p.nuisanceScale ^ 2 + εr)
    (ha : objective p.dynamics ar ≤ 1 - p.dynamics.nuisancePersistence ^ 2 + εa)
    (small : εr / (p.signalScale ^ 2 - p.nuisanceScale ^ 2) +
      εa / (p.dynamics.nuisancePersistence ^ 2 - p.dynamics.signalPersistence ^ 2) < 1) :
    HJepa.MaskedPrediction.taskRisk raw < taskRisk ar := by
  have hraw := HJepa.MaskedPrediction.signal_risk_le _ _ εr raw gr hr
  have har := nuisance_risk_ge _ ar εa ga ha
  linarith

end HJepa.TemporalPrediction
