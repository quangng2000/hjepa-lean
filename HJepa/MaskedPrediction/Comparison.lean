import HJepa.MaskedPrediction.Optima

namespace HJepa.MaskedPrediction

/-- Raw masked coordinates and a fixed diagonal layer acting on their targets. -/
structure Targets where
  signal : ℝ
  nuisance : ℝ
  signalScale : ℝ
  nuisanceScale : ℝ

noncomputable def Targets.hiddenSignal (t : Targets) : ℝ := t.signalScale * t.signal
noncomputable def Targets.hiddenNuisance (t : Targets) : ℝ := t.nuisanceScale * t.nuisance

def RawOptimal (t : Targets) (m : Predictor) : Prop :=
  GlobalMinimum t.signal t.nuisance m

def HiddenOptimal (t : Targets) (m : Predictor) : Prop :=
  GlobalMinimum t.hiddenSignal t.hiddenNuisance m

/-- Opposite energy orderings give strict separation for every pair of optima. -/
theorem hidden_better (t : Targets) (raw hidden : Predictor)
    (hr : RawOptimal t raw) (hh : HiddenOptimal t hidden)
    (gr : t.signal ^ 2 < t.nuisance ^ 2)
    (gh : t.hiddenNuisance ^ 2 < t.hiddenSignal ^ 2) :
    taskRisk hidden < taskRisk raw := by
  rw [signal_global_task_zero _ _ gh hidden hh,
    nuisance_global_task_one _ _ gr raw hr]
  norm_num

theorem raw_better (t : Targets) (raw hidden : Predictor)
    (hr : RawOptimal t raw) (hh : HiddenOptimal t hidden)
    (gr : t.nuisance ^ 2 < t.signal ^ 2)
    (gh : t.hiddenSignal ^ 2 < t.hiddenNuisance ^ 2) :
    taskRisk raw < taskRisk hidden := by
  rw [signal_global_task_zero _ _ gr raw hr,
    nuisance_global_task_one _ _ gh hidden hh]
  norm_num

/-- Away from equal energies, this is a necessary and sufficient condition. -/
theorem hidden_better_iff (t : Targets) (raw hidden : Predictor)
    (hr : RawOptimal t raw) (hh : HiddenOptimal t hidden)
    (nr : t.signal ^ 2 ≠ t.nuisance ^ 2)
    (nh : t.hiddenSignal ^ 2 ≠ t.hiddenNuisance ^ 2) :
    taskRisk hidden < taskRisk raw ↔
      t.signal ^ 2 < t.nuisance ^ 2 ∧ t.hiddenNuisance ^ 2 < t.hiddenSignal ^ 2 := by
  rcases lt_or_gt_of_ne nr with gr | gr <;> rcases lt_or_gt_of_ne nh with gh | gh
  · rw [nuisance_global_task_one _ _ gr raw hr,
      nuisance_global_task_one _ _ gh hidden hh]; constructor <;> intro h <;> linarith
  · exact ⟨fun _ => ⟨gr, gh⟩, fun _ => hidden_better t raw hidden hr hh gr gh⟩
  · rw [signal_global_task_zero _ _ gr raw hr,
      nuisance_global_task_one _ _ gh hidden hh]; constructor <;> intro h <;> linarith
  · rw [signal_global_task_zero _ _ gr raw hr,
      signal_global_task_zero _ _ gh hidden hh]; constructor <;> intro h <;> linarith

theorem raw_better_iff (t : Targets) (raw hidden : Predictor)
    (hr : RawOptimal t raw) (hh : HiddenOptimal t hidden)
    (nr : t.signal ^ 2 ≠ t.nuisance ^ 2)
    (nh : t.hiddenSignal ^ 2 ≠ t.hiddenNuisance ^ 2) :
    taskRisk raw < taskRisk hidden ↔
      t.nuisance ^ 2 < t.signal ^ 2 ∧ t.hiddenSignal ^ 2 < t.hiddenNuisance ^ 2 := by
  rcases lt_or_gt_of_ne nr with gr | gr <;> rcases lt_or_gt_of_ne nh with gh | gh
  · rw [nuisance_global_task_one _ _ gr raw hr,
      nuisance_global_task_one _ _ gh hidden hh]; constructor <;> intro h <;> linarith
  · rw [nuisance_global_task_one _ _ gr raw hr,
      signal_global_task_zero _ _ gh hidden hh]; constructor <;> intro h <;> linarith
  · exact ⟨fun _ => ⟨gr, gh⟩, fun _ => raw_better t raw hidden hr hh gr gh⟩
  · rw [signal_global_task_zero _ _ gr raw hr,
      signal_global_task_zero _ _ gh hidden hh]; constructor <;> intro h <;> linarith

theorem identity_layer_same_objective (s n a b p q : ℝ) :
    hiddenLoss s n 1 1 a b p q = rawLoss s n a b p q := by
  simp [hiddenLoss_eq_loss, rawLoss]

theorem matching_strict_order_ties (t : Targets) (raw hidden : Predictor)
    (hr : RawOptimal t raw) (hh : HiddenOptimal t hidden)
    (same : (t.nuisance ^ 2 < t.signal ^ 2 ∧ t.hiddenNuisance ^ 2 < t.hiddenSignal ^ 2) ∨
      (t.signal ^ 2 < t.nuisance ^ 2 ∧ t.hiddenSignal ^ 2 < t.hiddenNuisance ^ 2)) :
    taskRisk raw = taskRisk hidden := by
  rcases same with ⟨gr, gh⟩ | ⟨gr, gh⟩
  · rw [signal_global_task_zero _ _ gr raw hr, signal_global_task_zero _ _ gh hidden hh]
  · rw [nuisance_global_task_one _ _ gr raw hr, nuisance_global_task_one _ _ gh hidden hh]

end HJepa.MaskedPrediction
