import HJepa.TemporalPrediction.Moments
import HJepa.TemporalPrediction.Probe
import HJepa.MaskedPrediction.Optima

namespace HJepa.TemporalPrediction

/-- Same normalized encoder class as masked reconstruction; one temporal head. -/
structure Learner where
  signal : ℝ
  nuisance : ℝ
  coefficient : ℝ
  normalized : Normalized signal nuisance

/-- Either action gives this value; the known action enters the actual loss. -/
noncomputable def objective (d : Dynamics) (m : Learner) : ℝ :=
  loss d true m.signal m.nuisance m.coefficient

noncomputable def taskRisk (m : Learner) : ℝ :=
  downstreamRisk m.signal m.nuisance m.signal

def GlobalMinimum (d : Dynamics) (m : Learner) : Prop :=
  ∀ other : Learner, objective d m ≤ objective d other

noncomputable def signalLearner (d : Dynamics) : Learner :=
  ⟨1, 0, d.signalPersistence, by norm_num [Normalized, HJepa.MaskedPrediction.Normalized]⟩

noncomputable def nuisanceLearner (d : Dynamics) : Learner :=
  ⟨0, 1, d.nuisancePersistence, by norm_num [Normalized, HJepa.MaskedPrediction.Normalized]⟩

theorem taskRisk_eq (m : Learner) : taskRisk m = m.nuisance ^ 2 :=
  HJepa.MaskedPrediction.downstreamRisk_optimal _ _ m.normalized

theorem taskRisk_bounds (m : Learner) : 0 ≤ taskRisk m ∧ taskRisk m ≤ 1 := by
  rw [taskRisk_eq]
  have h := m.normalized
  change m.signal ^ 2 + m.nuisance ^ 2 = 1 at h
  exact ⟨sq_nonneg _, by nlinarith [sq_nonneg m.signal]⟩

theorem objective_any_action (d : Dynamics) (m : Learner) (action : Bool) :
    loss d action m.signal m.nuisance m.coefficient = objective d m := by
  cases action
  · exact loss_action_invariant _ _ _ _
  · rfl

theorem objective_signal_axis (d : Dynamics) :
    objective d (signalLearner d) = 1 - d.signalPersistence ^ 2 := by
  simp [objective, signalLearner, loss_exact, correlation]
  ring

theorem objective_nuisance_axis (d : Dynamics) :
    objective d (nuisanceLearner d) = 1 - d.nuisancePersistence ^ 2 := by
  simp [objective, nuisanceLearner, loss_exact, correlation]
  ring

theorem correlation_via_task (d : Dynamics) (m : Learner) :
    correlation d m.signal m.nuisance = d.signalPersistence +
      (d.nuisancePersistence - d.signalPersistence) * taskRisk m := by
  rw [taskRisk_eq]
  have h := congrArg (fun x : ℝ => d.signalPersistence * x) m.normalized
  unfold correlation
  nlinarith

end HJepa.TemporalPrediction
