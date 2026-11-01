import HJepa.TemporalPrediction.Controls.FutureRaw
import HJepa.MaskedPrediction.Characterization

namespace HJepa.TemporalPrediction.Controls

/-- The full future-raw population objective on the common normalized class. -/
noncomputable def futureRawObjective (d : Dynamics) (action : Bool)
    (s n : ℝ) (m : MaskedPrediction.Predictor) : ℝ :=
  futureRawLoss d action s n m.signal m.nuisance m.firstHead m.secondHead

/-- Encoder and both raw heads are optimized jointly. -/
def FutureRawGlobal (d : Dynamics) (action : Bool)
    (s n : ℝ) (m : MaskedPrediction.Predictor) : Prop :=
  ∀ other : MaskedPrediction.Predictor,
    futureRawObjective d action s n m ≤ futureRawObjective d action s n other

theorem futureRawObjective_eq (d : Dynamics) (action : Bool)
    (s n : ℝ) (m : MaskedPrediction.Predictor) :
    futureRawObjective d action s n m = futureRawNoise d s n +
      MaskedPrediction.objective (s * d.signalPersistence)
        (n * d.nuisancePersistence) m :=
  futureRawLoss_eq_maskedLoss d action s n _ _ _ _

/-- Adding the encoder-independent noise floor preserves every global optimum. -/
theorem futureRawGlobal_iff (d : Dynamics) (action : Bool)
    (s n : ℝ) (m : MaskedPrediction.Predictor) :
    FutureRawGlobal d action s n m ↔
      MaskedPrediction.GlobalMinimum (s * d.signalPersistence)
        (n * d.nuisancePersistence) m := by
  unfold FutureRawGlobal MaskedPrediction.GlobalMinimum
  simp only [futureRawObjective_eq, add_le_add_iff_left]

theorem futureRaw_signal_task_zero (d : Dynamics) (action : Bool)
    (s n : ℝ) (m : MaskedPrediction.Predictor)
    (gap : (n * d.nuisancePersistence) ^ 2 < (s * d.signalPersistence) ^ 2)
    (hm : FutureRawGlobal d action s n m) : MaskedPrediction.taskRisk m = 0 :=
  MaskedPrediction.signal_global_task_zero _ _ gap m
    ((futureRawGlobal_iff d action s n m).mp hm)

theorem futureRaw_nuisance_task_one (d : Dynamics) (action : Bool)
    (s n : ℝ) (m : MaskedPrediction.Predictor)
    (gap : (s * d.signalPersistence) ^ 2 < (n * d.nuisancePersistence) ^ 2)
    (hm : FutureRawGlobal d action s n m) : MaskedPrediction.taskRisk m = 1 :=
  MaskedPrediction.nuisance_global_task_one _ _ gap m
    ((futureRawGlobal_iff d action s n m).mp hm)

/-- Equal predictable energies leave every normalized encoder feasible at optimum. -/
theorem futureRaw_equal_energy_iff (d : Dynamics) (action : Bool)
    (s n : ℝ) (m : MaskedPrediction.Predictor)
    (equal : (s * d.signalPersistence) ^ 2 = (n * d.nuisancePersistence) ^ 2) :
    FutureRawGlobal d action s n m ↔
      MaskedPrediction.HeadsFit (s * d.signalPersistence)
        (n * d.nuisancePersistence) m := by
  rw [futureRawGlobal_iff]
  exact MaskedPrediction.equal_energy_global_iff _ _ equal m

end HJepa.TemporalPrediction.Controls
