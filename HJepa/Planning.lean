import Mathlib.Data.Real.CompleteField
import Mathlib.Order.ConditionallyCompleteLattice.Basic
import Mathlib.Tactic.Linarith

/-!
# Task-calibrated guarantees for composed planners

`A` is the type of feasible primitive plans, `B` the type of macro plans, and
`refine : B → A` is the *actual* refinement algorithm. The calibration hypothesis
compares the surrogate with the true cost of the refined plan. It is stronger
than accurate latent prediction. All optimization assumptions concern the whole
composed planner, not just individual local solvers.

The infimum theorem does not assume that either true optimum is attained.
-/

namespace HJepa.Planning

/-- An optimizer whose objective value is within `η` of every competitor. -/
def ApproxMinimizer {A : Type*} (f : A → ℝ) (candidate : A) (η : ℝ) : Prop :=
  ∀ alternative, f candidate ≤ f alternative + η

/-- Uniform error measured in the units of the actual task cost. -/
def Calibrated {A B : Type*} (J : A → ℝ) (Q : B → ℝ)
    (refine : B → A) (δ : ℝ) : Prop :=
  ∀ b, |Q b - J (refine b)| ≤ δ

/-- A candidate witnesses nonemptiness and forces the tolerance nonnegative. -/
theorem optimization_tolerance_nonnegative {A : Type*}
    (f : A → ℝ) (candidate : A) (η : ℝ)
    (hopt : ApproxMinimizer f candidate η) : 0 ≤ η := by
  have h := hopt candidate
  linarith

/-- Calibration on a nonempty candidate set forces a nonnegative error bound. -/
theorem calibration_tolerance_nonnegative {A B : Type*}
    (J : A → ℝ) (Q : B → ℝ) (refine : B → A)
    (candidate : B) (δ : ℝ) (hcal : Calibrated J Q refine δ) : 0 ≤ δ :=
  (abs_nonneg _).trans (hcal candidate)

/-- Comparing the returned plan with any realizable comparator needs no minima. -/
theorem regret_against_comparator {A B : Type*}
    (J : A → ℝ) (Q : B → ℝ) (refine : B → A)
    (candidate comparator : B) (δ η : ℝ)
    (hcal : Calibrated J Q refine δ)
    (hopt : ApproxMinimizer Q candidate η) :
    J (refine candidate) - J (refine comparator) ≤ 2 * δ + η := by
  have hc := (abs_le.mp (hcal candidate)).1
  have hb := (abs_le.mp (hcal comparator)).2
  have ho := hopt comparator
  linarith

/-- The infimum formulation of approximate optimization implies `ApproxMinimizer`.
The boundedness hypothesis makes the real infimum meaningful. -/
theorem approxMinimizer_of_infimum {B : Type*} (Q : B → ℝ)
    (candidate : B) (η : ℝ) (hbounded : BddBelow (Set.range Q))
    (hopt : Q candidate ≤ sInf (Set.range Q) + η) :
    ApproxMinimizer Q candidate η := by
  intro b
  have hb : sInf (Set.range Q) ≤ Q b := csInf_le hbounded ⟨b, rfl⟩
  linarith

/-- Uniform calibration transfers a true-cost lower bound to the proxy. -/
theorem calibrated_objective_bounded_below {A B : Type*}
    (J : A → ℝ) (Q : B → ℝ) (refine : B → A) (δ : ℝ)
    (hbounded : BddBelow (Set.range J))
    (hcal : Calibrated J Q refine δ) : BddBelow (Set.range Q) := by
  rcases hbounded with ⟨lower, hlower⟩
  refine ⟨lower - δ, ?_⟩
  rintro _ ⟨b, rfl⟩
  have hj : lower ≤ J (refine b) := hlower ⟨refine b, rfl⟩
  have hc := (abs_le.mp (hcal b)).1
  linarith

/-- Restricting primitive plans to the image of refinement cannot improve the
optimal true cost. Nonemptiness is witnessed by `candidate`. -/
theorem restriction_cost_nonnegative {A B : Type*}
    (J : A → ℝ) (refine : B → A) (candidate : B)
    (hbounded : BddBelow (Set.range J)) :
    0 ≤ sInf (Set.range (J ∘ refine)) - sInf (Set.range J) := by
  have hle : sInf (Set.range J) ≤ sInf (Set.range (J ∘ refine)) := by
    have hnonempty : (Set.range (J ∘ refine)).Nonempty :=
      ⟨J (refine candidate), ⟨candidate, rfl⟩⟩
    apply le_csInf hnonempty
    rintro _ ⟨b, rfl⟩
    exact csInf_le hbounded ⟨refine b, rfl⟩
  linarith

/-- Full composed-planner regret, with the restriction cost expressed as the
difference of infima. The lower bound on `J` and the candidate ensure finite,
nonempty true-cost ranges; no minimizing plans need to exist. -/
theorem composite_planner_regret {A B : Type*}
    (J : A → ℝ) (Q : B → ℝ) (refine : B → A)
    (candidate : B) (δ η : ℝ)
    (hbounded : BddBelow (Set.range J))
    (hcal : Calibrated J Q refine δ)
    (hopt : ApproxMinimizer Q candidate η) :
    J (refine candidate) - sInf (Set.range J) ≤
      2 * δ + η +
        (sInf (Set.range (J ∘ refine)) - sInf (Set.range J)) := by
  have hrestricted : BddBelow (Set.range (J ∘ refine)) := by
    rcases hbounded with ⟨lower, hlower⟩
    exact ⟨lower, by rintro _ ⟨b, rfl⟩; exact hlower ⟨refine b, rfl⟩⟩
  have hle : J (refine candidate) - (2 * δ + η) ≤
      sInf (Set.range (J ∘ refine)) := by
    apply (le_csInf_iff hrestricted ⟨J (refine candidate), ⟨candidate, rfl⟩⟩).2
    rintro _ ⟨b, rfl⟩
    have hb := regret_against_comparator J Q refine candidate b δ η hcal hopt
    change J (refine candidate) - (2 * δ + η) ≤ J (refine b)
    linarith
  linarith

/-- The paper-note formulation, assuming optimization within `η` of the proxy
infimum. Calibration itself supplies the required finite proxy lower bound. -/
theorem composite_planner_regret_of_infimum {A B : Type*}
    (J : A → ℝ) (Q : B → ℝ) (refine : B → A)
    (candidate : B) (δ η : ℝ)
    (hbounded : BddBelow (Set.range J))
    (hcal : Calibrated J Q refine δ)
    (hopt : Q candidate ≤ sInf (Set.range Q) + η) :
    J (refine candidate) - sInf (Set.range J) ≤
      2 * δ + η +
        (sInf (Set.range (J ∘ refine)) - sInf (Set.range J)) := by
  exact composite_planner_regret J Q refine candidate δ η hbounded hcal
    (approxMinimizer_of_infimum Q candidate η
      (calibrated_objective_bounded_below J Q refine δ hbounded hcal) hopt)

/-- The flat planner is the identity-refinement special case. -/
theorem flat_planner_regret {A : Type*}
    (J Q : A → ℝ) (candidate : A) (δ η : ℝ)
    (hbounded : BddBelow (Set.range J))
    (hcal : Calibrated J Q id δ)
    (hopt : ApproxMinimizer Q candidate η) :
    J candidate - sInf (Set.range J) ≤ 2 * δ + η := by
  simpa using composite_planner_regret J Q id candidate δ η hbounded hcal hopt

/-- Exact task calibration, exact optimization, and no restriction gap imply
attainment of the true optimal value. Perfect latent prediction alone is not
one of these hypotheses. -/
theorem exact_calibration_optimality {A B : Type*}
    (J : A → ℝ) (Q : B → ℝ) (refine : B → A) (candidate : B)
    (hbounded : BddBelow (Set.range J))
    (hcal : Calibrated J Q refine 0)
    (hopt : ApproxMinimizer Q candidate 0)
    (hgap : sInf (Set.range (J ∘ refine)) = sInf (Set.range J)) :
    J (refine candidate) = sInf (Set.range J) := by
  have hupper := composite_planner_regret J Q refine candidate 0 0 hbounded hcal hopt
  have hlower : sInf (Set.range J) ≤ J (refine candidate) :=
    csInf_le hbounded ⟨refine candidate, rfl⟩
  rw [hgap] at hupper
  linarith

/-- Without a relation between proxy and true task cost, even an exact global
proxy optimizer can have any prescribed nonnegative regret. This is a limitation
of an uncalibrated objective, not a claim of inevitable failure of H-JEPA. -/
theorem uncalibrated_optimizer_arbitrary_regret (M : ℝ) (hM : 0 ≤ M) :
    ∃ J Q : Bool → ℝ,
      ApproxMinimizer Q false 0 ∧
      (∀ b, J true ≤ J b) ∧
      J false - J true = M := by
  refine ⟨(fun b => if b then 0 else M), (fun _ => 0), ?_, ?_, ?_⟩
  · intro b
    simp
  · intro b
    cases b <;> simp [hM]
  · simp

end HJepa.Planning
