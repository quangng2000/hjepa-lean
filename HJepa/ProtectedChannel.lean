import HJepa.Objective

/-!
# A task-protected channel beside a predictive channel

The scalar predictive objective in `Objective.lean` discards an unpredictable
current-task bit at every global optimum. Here a two-channel representation
separates the obligations: predict the future of one channel, and recover the
specified current task from the other. Both channels satisfy the same moment
normalization used in that counterexample. An explicit feasible model attains
zero for both losses, so every global optimum preserves the task, and an
approximate optimum gives quantitative task and prediction bounds.

This is a supervised population objective for a specified task, with an extra
scalar channel. It is not an equal-width comparison, a convergence theorem,
a claim about unknown tasks, or a proof about the full H-JEPA objective. The
protected channel is deliberately not required to predict its own future.
The three-level construction carries this channel unchanged; compression is
allowed only in the separate predictive channel.
-/

namespace HJepa.ProtectedChannel

open HJepa.Abstraction HJepa.Objective

/-- Predictive and task-protected coordinates have independent readouts. -/
structure Model where
  predictiveChannel : Input → ℝ
  taskChannel : Input → ℝ
  predictor : ℝ → ℝ
  taskHead : ℝ → ℝ

def Admissible (m : Model) : Prop :=
  MomentNormalized m.predictiveChannel ∧ MomentNormalized m.taskChannel

noncomputable def taskRisk (m : Model) : ℝ :=
  downstreamRisk m.taskChannel m.taskHead

noncomputable def objective (weight : ℝ) (m : Model) : ℝ :=
  predictionLoss m.predictiveChannel m.predictor + weight * taskRisk m

/-- These are population comparison predicates, not optimizer algorithms. -/
def GlobalMinimum (weight : ℝ) (m : Model) : Prop :=
  Admissible m ∧ ∀ alternative : Model, Admissible alternative →
    objective weight m ≤ objective weight alternative

def ApproximateGlobalMinimum (weight η : ℝ) (m : Model) : Prop :=
  Admissible m ∧ ∀ alternative : Model, Admissible alternative →
    objective weight m ≤ objective weight alternative + η

/-- Two independent fair bits fit in two normalized scalar channels. -/
def witness : Model where
  predictiveChannel := backgroundEncoder
  taskChannel := taskEncoder
  predictor := id
  taskHead := id

theorem witness_admissible : Admissible witness :=
  ⟨background_normalized, task_normalized⟩

theorem taskRisk_nonneg (m : Model) : 0 ≤ taskRisk m := by
  exact risk_nonneg fairTrajectory currentTask (m.taskHead ∘ m.taskChannel ∘ current)

theorem objective_nonneg (weight : ℝ) (hweight : 0 ≤ weight) (m : Model) :
    0 ≤ objective weight m := by
  exact add_nonneg (prediction_loss_nonneg _ _)
    (mul_nonneg hweight (taskRisk_nonneg m))

theorem witness_objective_zero (weight : ℝ) : objective weight witness = 0 := by
  simp [objective, taskRisk, witness, background_prediction_zero,
    task_downstream_zero]

theorem witness_global_minimum (weight : ℝ) (hweight : 0 ≤ weight) :
    GlobalMinimum weight witness := by
  refine ⟨witness_admissible, ?_⟩
  intro alternative _
  rw [witness_objective_zero]
  exact objective_nonneg weight hweight alternative

/-- Positivity of the protection coefficient is essential. -/
theorem objective_zero_iff (weight : ℝ) (hweight : 0 < weight) (m : Model) :
    objective weight m = 0 ↔
      predictionLoss m.predictiveChannel m.predictor = 0 ∧ taskRisk m = 0 := by
  have hp := prediction_loss_nonneg m.predictiveChannel m.predictor
  have ht := taskRisk_nonneg m
  unfold objective
  constructor
  · intro hz
    have hw : 0 ≤ weight * taskRisk m := mul_nonneg (le_of_lt hweight) ht
    constructor <;> nlinarith
  · rintro ⟨hpz, htz⟩
    rw [hpz, htz]
    ring

/-- Unlike the original scalar predictive objective, every global optimum of
this feasible objective has zero risk for the designated current task. -/
theorem every_global_minimum_preserves_task (weight : ℝ) (hweight : 0 < weight)
    (m : Model) (hmin : GlobalMinimum weight m) :
    predictionLoss m.predictiveChannel m.predictor = 0 ∧ taskRisk m = 0 := by
  have hle := hmin.2 witness witness_admissible
  rw [witness_objective_zero] at hle
  exact (objective_zero_iff weight hweight m).mp
    (le_antisymm hle (objective_nonneg weight (le_of_lt hweight) m))

/-- Approximate global optimality controls both branches separately. No
convergence or empirical-to-population transfer is assumed implicitly. -/
theorem approximate_minimum_bounds (weight η : ℝ) (hweight : 0 < weight)
    (m : Model) (hmin : ApproximateGlobalMinimum weight η m) :
    predictionLoss m.predictiveChannel m.predictor ≤ η ∧ taskRisk m ≤ η / weight := by
  have hle := hmin.2 witness witness_admissible
  rw [witness_objective_zero, zero_add] at hle
  have hp := prediction_loss_nonneg m.predictiveChannel m.predictor
  have ht := taskRisk_nonneg m
  have hw : 0 ≤ weight * taskRisk m := mul_nonneg (le_of_lt hweight) ht
  change predictionLoss m.predictiveChannel m.predictor + weight * taskRisk m ≤ η at hle
  constructor
  · linarith
  · apply (le_div_iff₀ hweight).2
    nlinarith

theorem approximate_minimum_task_risk_lt_one (weight η : ℝ) (hweight : 0 < weight)
    (hη : η < weight) (m : Model) (hmin : ApproximateGlobalMinimum weight η m) :
    taskRisk m < 1 := by
  have hb := (approximate_minimum_bounds weight η hweight m hmin).2
  have hdiv : η / weight < 1 := (div_lt_one hweight).2 hη
  exact lt_of_le_of_lt hb hdiv

/-- A direct comparison with every optimum of the old scalar objective, for
the same law and task. The protected model has an additional scalar channel. -/
theorem approximate_protection_beats_old_global_optimum
    (weight η : ℝ) (hweight : 0 < weight) (hη : η < weight)
    (m : Model) (hmin : ApproximateGlobalMinimum weight η m)
    (encoder : Input → ℝ) (predictor : ℝ → ℝ)
    (hold : ∀ (alternative : Input → ℝ) (otherPredictor : ℝ → ℝ),
      MomentNormalized alternative →
      predictionLoss encoder predictor ≤ predictionLoss alternative otherPredictor)
    (readout : ℝ → ℝ) :
    taskRisk m < downstreamRisk encoder readout := by
  exact lt_of_lt_of_le (approximate_minimum_task_risk_lt_one weight η hweight hη m hmin)
    (every_global_optimum_task_risk_at_least_one encoder predictor hold readout)

theorem exact_protection_beats_old_global_optimum
    (weight : ℝ) (hweight : 0 < weight) (m : Model) (hmin : GlobalMinimum weight m)
    (encoder : Input → ℝ) (predictor : ℝ → ℝ)
    (hold : ∀ (alternative : Input → ℝ) (otherPredictor : ℝ → ℝ),
      MomentNormalized alternative →
      predictionLoss encoder predictor ≤ predictionLoss alternative otherPredictor)
    (readout : ℝ → ℝ) :
    taskRisk m < downstreamRisk encoder readout := by
  rw [(every_global_minimum_preserves_task weight hweight m hmin).2]
  linarith [every_global_optimum_task_risk_at_least_one encoder predictor hold readout]

/-- Without a positive task weight, even the extra channel can be useless. -/
def unprotectedWitness : Model where
  predictiveChannel := backgroundEncoder
  taskChannel := backgroundEncoder
  predictor := id
  taskHead := fun _ => 0

theorem zero_weight_has_task_discarding_global_minimum :
    GlobalMinimum 0 unprotectedWitness ∧ taskRisk unprotectedWitness = 1 := by
  constructor
  · refine ⟨⟨background_normalized, background_normalized⟩, ?_⟩
    intro alternative _
    have hz : objective 0 unprotectedWitness = 0 := by
      simp [objective, unprotectedWitness, background_prediction_zero]
    rw [hz]
    exact objective_nonneg 0 (by norm_num) alternative
  · norm_num [taskRisk, unprotectedWitness, background_downstream_exact]

/-- Zero task risk is exact recovery of the specified task on every input,
because every trajectory in this finite example has positive probability. -/
theorem zero_task_risk_recovers_task (m : Model) (hzero : taskRisk m = 0)
    (b u : Bool) : m.taskHead (m.taskChannel (b, u)) = signedBit u := by
  have hpoint := (risk_eq_zero_iff fairTrajectory currentTask
    (m.taskHead ∘ m.taskChannel ∘ current)).mp hzero
  have hp := hpoint (b, u, false) (by norm_num [fairTrajectory])
  exact hp.symm

/-- Perfect recovery of the independent current-task bit is incompatible with
zero next-latent prediction loss on that same protected scalar channel. Thus
excluding this channel from the predictive target is substantive, not notation. -/
theorem task_preservation_forces_positive_task_channel_prediction
    (m : Model) (hzero : taskRisk m = 0) (predictor : ℝ → ℝ) :
    0 < predictionLoss m.taskChannel predictor := by
  by_contra hnot
  have hz : predictionLoss m.taskChannel predictor = 0 :=
    le_antisymm (le_of_not_gt hnot) (prediction_loss_nonneg _ _)
  have he := zero_prediction_downstream_exact m.taskChannel predictor m.taskHead hz
  change downstreamRisk m.taskChannel m.taskHead = 0 at hzero
  rw [hzero] at he
  nlinarith [sq_nonneg (m.taskHead (m.taskChannel (true, false))),
    sq_nonneg (m.taskHead (m.taskChannel (false, false)))]

/-- Predictive abstraction leaves the protected coordinate accessible. -/
def lowRepresentation (m : Model) (x : Input) : ℝ × ℝ :=
  (m.predictiveChannel x, m.taskChannel x)

def carry (g : ℝ → ℝ) (z : ℝ × ℝ) : ℝ × ℝ := (g z.1, z.2)

def middleRepresentation (m : Model) (g : ℝ → ℝ) : Input → ℝ × ℝ :=
  carry g ∘ lowRepresentation m

def highRepresentation (m : Model) (g h : ℝ → ℝ) : Input → ℝ × ℝ :=
  carry h ∘ middleRepresentation m g

def protectedReadout (m : Model) (z : ℝ × ℝ) : ℝ := m.taskHead z.2

noncomputable def representationTaskRisk (z : Input → ℝ × ℝ)
    (readout : ℝ × ℝ → ℝ) : ℝ :=
  risk fairTrajectory currentTask (readout ∘ z ∘ current)

/-- No task information is lost by either abstraction map, since the proof
exhibits the same task predictor at all three levels. -/
theorem three_level_task_risks_equal (m : Model) (g h : ℝ → ℝ) :
    representationTaskRisk (lowRepresentation m) (protectedReadout m) = taskRisk m ∧
    representationTaskRisk (middleRepresentation m g) (protectedReadout m) = taskRisk m ∧
    representationTaskRisk (highRepresentation m g h) (protectedReadout m) = taskRisk m := by
  exact ⟨rfl, rfl, rfl⟩

theorem three_levels_preserve_task_at_global_minimum
    (weight : ℝ) (hweight : 0 < weight) (m : Model) (hmin : GlobalMinimum weight m)
    (g h : ℝ → ℝ) :
    representationTaskRisk (lowRepresentation m) (protectedReadout m) = 0 ∧
    representationTaskRisk (middleRepresentation m g) (protectedReadout m) = 0 ∧
    representationTaskRisk (highRepresentation m g h) (protectedReadout m) = 0 := by
  have hz := (every_global_minimum_preserves_task weight hweight m hmin).2
  exact ⟨hz, hz, hz⟩

theorem three_levels_approximate_task_bounds
    (weight η : ℝ) (hweight : 0 < weight) (m : Model)
    (hmin : ApproximateGlobalMinimum weight η m) (g h : ℝ → ℝ) :
    representationTaskRisk (lowRepresentation m) (protectedReadout m) ≤ η / weight ∧
    representationTaskRisk (middleRepresentation m g) (protectedReadout m) ≤ η / weight ∧
    representationTaskRisk (highRepresentation m g h) (protectedReadout m) ≤ η / weight := by
  have hb := (approximate_minimum_bounds weight η hweight m hmin).2
  exact ⟨hb, hb, hb⟩

end HJepa.ProtectedChannel
