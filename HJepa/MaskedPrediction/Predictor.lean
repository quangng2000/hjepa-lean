import HJepa.MaskedPrediction.Risk

namespace HJepa.MaskedPrediction

/-- Both objectives optimize exactly this encoder/head class. -/
structure Predictor where
  signal : ℝ
  nuisance : ℝ
  firstHead : ℝ
  secondHead : ℝ
  normalized : Normalized signal nuisance

noncomputable def objective (u v : ℝ) (m : Predictor) : ℝ :=
  loss u v m.signal m.nuisance m.firstHead m.secondHead

/-- Common evaluation: minimum linear-probe risk for the visible task sign.
This is not unrestricted Bayes risk of a generic scalar encoder. -/
noncomputable def taskRisk (m : Predictor) : ℝ :=
  downstreamRisk m.signal m.nuisance m.signal

def GlobalMinimum (u v : ℝ) (m : Predictor) : Prop :=
  ∀ other : Predictor, objective u v m ≤ objective u v other

noncomputable def signalPredictor (u : ℝ) : Predictor :=
  ⟨1, 0, u, 0, by norm_num [Normalized]⟩

noncomputable def nuisancePredictor (v : ℝ) : Predictor :=
  ⟨0, 1, 0, v, by norm_num [Normalized]⟩

theorem taskRisk_eq (m : Predictor) : taskRisk m = m.nuisance ^ 2 :=
  downstreamRisk_optimal _ _ m.normalized

theorem taskRisk_bounds (m : Predictor) : 0 ≤ taskRisk m ∧ taskRisk m ≤ 1 := by
  rw [taskRisk_eq]
  have h := m.normalized
  unfold Normalized at h
  constructor
  · exact sq_nonneg _
  · nlinarith [sq_nonneg m.signal]

theorem objective_signal_axis (u v : ℝ) :
    objective u v (signalPredictor u) = v ^ 2 := by
  norm_num [objective, signalPredictor, loss_exact]

theorem objective_nuisance_axis (u v : ℝ) :
    objective u v (nuisancePredictor v) = u ^ 2 := by
  norm_num [objective, nuisancePredictor, loss_exact]

theorem objective_signal_form (u v : ℝ) (m : Predictor) :
    objective u v m = v ^ 2 + (u ^ 2 - v ^ 2) * taskRisk m +
      (m.firstHead - u * m.signal) ^ 2 +
      (m.secondHead - v * m.nuisance) ^ 2 := by
  rw [objective, loss_complete_square _ _ _ _ _ _ m.normalized, taskRisk_eq]
  have h := congrArg (fun x : ℝ => v ^ 2 * x) m.normalized
  unfold reducedLoss
  nlinarith

theorem objective_nuisance_form (u v : ℝ) (m : Predictor) :
    objective u v m = u ^ 2 + (v ^ 2 - u ^ 2) * (1 - taskRisk m) +
      (m.firstHead - u * m.signal) ^ 2 +
      (m.secondHead - v * m.nuisance) ^ 2 := by
  rw [objective_signal_form]
  ring

theorem raw_objective (s n : ℝ) (m : Predictor) :
    rawLoss s n m.signal m.nuisance m.firstHead m.secondHead = objective s n m := rfl

theorem hidden_objective (s n c d : ℝ) (m : Predictor) :
    hiddenLoss s n c d m.signal m.nuisance m.firstHead m.secondHead =
      objective (c * s) (d * n) m :=
  hiddenLoss_eq_loss _ _ _ _ _ _ _ _

end HJepa.MaskedPrediction
