import HJepa.TemporalPrediction.Comparison.Methods

/-!
# Queue entry V10: a masked term and a temporal term on one shared encoder

Unreviewed Frontier candidate; see `FUZZ_QUEUE.md`.

The reviewed library compares a masked learner with a temporal learner that are
trained separately. H-WavJEPA instead adds the two losses and trains one
encoder. This module defines that sum for the scalar two-sign model: the masked
loss of `MaskedPrediction`, weighted by `w`, plus the temporal loss of
`TemporalPrediction`, both evaluated on the same normalized encoder.

The masked part is reconstruction of a fixed target with scales
`signalScale` and `nuisanceScale` under a fixed mask. It is not WavJEPA's
moving EMA target, and no optimizer is modelled.
-/

namespace HJepa.Frontier.JointObjective

noncomputable section

/-- One normalized scalar encoder with two masked heads and one temporal head. -/
structure JointLearner where
  signal : ℝ
  nuisance : ℝ
  firstHead : ℝ
  secondHead : ℝ
  coefficient : ℝ
  normalized : MaskedPrediction.Normalized signal nuisance

/-- The shared encoder with its masked heads only. -/
def JointLearner.masked (m : JointLearner) : MaskedPrediction.Predictor :=
  ⟨m.signal, m.nuisance, m.firstHead, m.secondHead, m.normalized⟩

/-- The shared encoder with its temporal head only. -/
def JointLearner.temporal (m : JointLearner) : TemporalPrediction.Learner :=
  ⟨m.signal, m.nuisance, m.coefficient, m.normalized⟩

/-- Masked loss weighted by `w`, plus the temporal loss. -/
def jointObjective (w : ℝ) (p : TemporalPrediction.Problem) (m : JointLearner) : ℝ :=
  w * MaskedPrediction.objective p.signalScale p.nuisanceScale m.masked +
    TemporalPrediction.objective p.dynamics m.temporal

/-- Linear-probe risk for the signal; the two parts share this evaluation. -/
def taskRisk (m : JointLearner) : ℝ := TemporalPrediction.taskRisk m.temporal

def GlobalMinimum (w : ℝ) (p : TemporalPrediction.Problem) (m : JointLearner) : Prop :=
  ∀ other : JointLearner, jointObjective w p m ≤ jointObjective w p other

/-- The signal's energy advantage in the masked target, weighted by `w`, plus
its advantage in squared persistence. Its sign decides what every optimum keeps. -/
def selectionGap (w : ℝ) (p : TemporalPrediction.Problem) : ℝ :=
  w * (p.signalScale ^ 2 - p.nuisanceScale ^ 2) +
    (p.dynamics.signalPersistence ^ 2 - p.dynamics.nuisancePersistence ^ 2)

/-- Objective value of the signal-only encoder with fitted heads. -/
def signalValue (w : ℝ) (p : TemporalPrediction.Problem) : ℝ :=
  w * p.nuisanceScale ^ 2 + (1 - p.dynamics.signalPersistence ^ 2)

/-- Objective value of the nuisance-only encoder with fitted heads. -/
def nuisanceValue (w : ℝ) (p : TemporalPrediction.Problem) : ℝ :=
  w * p.signalScale ^ 2 + (1 - p.dynamics.nuisancePersistence ^ 2)

/-- Head misfit of both parts and the temporal mixing penalty. -/
def slack (w : ℝ) (p : TemporalPrediction.Problem) (m : JointLearner) : ℝ :=
  w * ((m.masked.firstHead - p.signalScale * m.masked.signal) ^ 2 +
      (m.masked.secondHead - p.nuisanceScale * m.masked.nuisance) ^ 2) +
    TemporalPrediction.mixingPenalty p.dynamics m.temporal +
    TemporalPrediction.headPenalty p.dynamics m.temporal

/-- Keeps only the signal, with the best heads for both parts. -/
def signalLearner (p : TemporalPrediction.Problem) : JointLearner :=
  ⟨1, 0, p.signalScale, 0, p.dynamics.signalPersistence,
    by norm_num [MaskedPrediction.Normalized]⟩

/-- Keeps only the nuisance, with the best heads for both parts. -/
def nuisanceLearner (p : TemporalPrediction.Problem) : JointLearner :=
  ⟨0, 1, 0, p.nuisanceScale, p.dynamics.nuisancePersistence,
    by norm_num [MaskedPrediction.Normalized]⟩

end

end HJepa.Frontier.JointObjective
