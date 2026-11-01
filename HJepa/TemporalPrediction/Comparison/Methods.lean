import HJepa.TemporalPrediction.Comparison.Characterization

namespace HJepa.TemporalPrediction

structure Problem where
  signalScale : ℝ
  nuisanceScale : ℝ
  dynamics : Dynamics

def RawOptimal (p : Problem) (raw : HJepa.MaskedPrediction.Predictor) : Prop :=
  HJepa.MaskedPrediction.GlobalMinimum p.signalScale p.nuisanceScale raw

def AROptimal (p : Problem) (ar : Learner) : Prop := GlobalMinimum p.dynamics ar

theorem autoregressive_better (p : Problem) (raw : HJepa.MaskedPrediction.Predictor)
    (ar : Learner) (hr : RawOptimal p raw) (ha : AROptimal p ar)
    (gr : p.signalScale ^ 2 < p.nuisanceScale ^ 2)
    (ga : p.dynamics.nuisancePersistence ^ 2 < p.dynamics.signalPersistence ^ 2) :
    taskRisk ar < HJepa.MaskedPrediction.taskRisk raw := by
  rw [signal_global_task_zero _ ga ar ha,
    HJepa.MaskedPrediction.nuisance_global_task_one _ _ gr raw hr]
  norm_num

theorem reconstruction_better (p : Problem) (raw : HJepa.MaskedPrediction.Predictor)
    (ar : Learner) (hr : RawOptimal p raw) (ha : AROptimal p ar)
    (gr : p.nuisanceScale ^ 2 < p.signalScale ^ 2)
    (ga : p.dynamics.signalPersistence ^ 2 < p.dynamics.nuisancePersistence ^ 2) :
    HJepa.MaskedPrediction.taskRisk raw < taskRisk ar := by
  rw [nuisance_global_task_one _ ga ar ha,
    HJepa.MaskedPrediction.signal_global_task_zero _ _ gr raw hr]
  norm_num

theorem autoregressive_better_iff (p : Problem) (raw : HJepa.MaskedPrediction.Predictor)
    (ar : Learner) (hr : RawOptimal p raw) (ha : AROptimal p ar)
    (nr : p.signalScale ^ 2 ≠ p.nuisanceScale ^ 2)
    (na : p.dynamics.signalPersistence ^ 2 ≠ p.dynamics.nuisancePersistence ^ 2) :
    taskRisk ar < HJepa.MaskedPrediction.taskRisk raw ↔
      p.signalScale ^ 2 < p.nuisanceScale ^ 2 ∧
      p.dynamics.nuisancePersistence ^ 2 < p.dynamics.signalPersistence ^ 2 := by
  rcases lt_or_gt_of_ne nr with gr | gr <;> rcases lt_or_gt_of_ne na with ga | ga
  · rw [nuisance_global_task_one _ ga ar ha,
      HJepa.MaskedPrediction.nuisance_global_task_one _ _ gr raw hr]
    constructor <;> intro h <;> linarith
  · exact ⟨fun _ => ⟨gr, ga⟩, fun _ => autoregressive_better p raw ar hr ha gr ga⟩
  · rw [nuisance_global_task_one _ ga ar ha,
      HJepa.MaskedPrediction.signal_global_task_zero _ _ gr raw hr]
    constructor <;> intro h <;> linarith
  · rw [signal_global_task_zero _ ga ar ha,
      HJepa.MaskedPrediction.signal_global_task_zero _ _ gr raw hr]
    constructor <;> intro h <;> linarith

theorem reconstruction_better_iff (p : Problem) (raw : HJepa.MaskedPrediction.Predictor)
    (ar : Learner) (hr : RawOptimal p raw) (ha : AROptimal p ar)
    (nr : p.signalScale ^ 2 ≠ p.nuisanceScale ^ 2)
    (na : p.dynamics.signalPersistence ^ 2 ≠ p.dynamics.nuisancePersistence ^ 2) :
    HJepa.MaskedPrediction.taskRisk raw < taskRisk ar ↔
      p.nuisanceScale ^ 2 < p.signalScale ^ 2 ∧
      p.dynamics.signalPersistence ^ 2 < p.dynamics.nuisancePersistence ^ 2 := by
  rcases lt_or_gt_of_ne nr with gr | gr <;> rcases lt_or_gt_of_ne na with ga | ga
  · rw [nuisance_global_task_one _ ga ar ha,
      HJepa.MaskedPrediction.nuisance_global_task_one _ _ gr raw hr]
    constructor <;> intro h <;> linarith
  · rw [signal_global_task_zero _ ga ar ha,
      HJepa.MaskedPrediction.nuisance_global_task_one _ _ gr raw hr]
    constructor <;> intro h <;> linarith
  · exact ⟨fun _ => ⟨gr, ga⟩, fun _ => reconstruction_better p raw ar hr ha gr ga⟩
  · rw [signal_global_task_zero _ ga ar ha,
      HJepa.MaskedPrediction.signal_global_task_zero _ _ gr raw hr]
    constructor <;> intro h <;> linarith

theorem matching_strict_order_ties (p : Problem) (raw : HJepa.MaskedPrediction.Predictor)
    (ar : Learner) (hr : RawOptimal p raw) (ha : AROptimal p ar)
    (same : (p.nuisanceScale ^ 2 < p.signalScale ^ 2 ∧
        p.dynamics.nuisancePersistence ^ 2 < p.dynamics.signalPersistence ^ 2) ∨
      (p.signalScale ^ 2 < p.nuisanceScale ^ 2 ∧
        p.dynamics.signalPersistence ^ 2 < p.dynamics.nuisancePersistence ^ 2)) :
    HJepa.MaskedPrediction.taskRisk raw = taskRisk ar := by
  rcases same with ⟨gr, ga⟩ | ⟨gr, ga⟩
  · rw [signal_global_task_zero _ ga ar ha,
      HJepa.MaskedPrediction.signal_global_task_zero _ _ gr raw hr]
  · rw [nuisance_global_task_one _ ga ar ha,
      HJepa.MaskedPrediction.nuisance_global_task_one _ _ gr raw hr]

end HJepa.TemporalPrediction
