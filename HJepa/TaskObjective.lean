import HJepa.Learning

/-!
# Task-aware objectives with explicit information-preservation guarantees

We add a supervised squared-loss term for a specified task to a predictive
objective. The theorem is about a deterministic learned abstraction of a fixed
lower context under one fixed finite population law. It does not assert that
unlabeled future prediction discovers the task, that arbitrary tasks are
preserved, or that a practical optimizer reaches a global minimum.

The central feasibility condition is explicit: a comparator must attain both
the predictive-cost floor and the Bayes task risk available from the lower
context. A protected task branch can make this condition possible even when
the task signal itself is not temporally predictable. If simultaneous
attainment is unavailable, the bound includes the comparator's predictive and
task costs rather than claiming exact preservation.

All objectives below are population objectives. The final theorem separates
an empirical objective from its population counterpart and requires a uniform
deviation bound; that hypothesis is not a proved sample-complexity result.
-/

namespace HJepa.TaskObjective

open HJepa.Abstraction HJepa.Learning

variable {C Ω I J : Type*} [Fintype Ω]

/-- Prediction and a known task share a candidate, but need not share a head
or force every protected coordinate to predict its own future value. -/
def weightedObjective (predictiveCost taskCost : C → ℝ) (α : ℝ) (c : C) : ℝ :=
  predictiveCost c + α * taskCost c

/-- A weighted objective controls task cost only after accounting for a
predictive-cost floor and an explicit comparator. Neither comparator budget
is silently set to zero. -/
theorem weighted_comparison_bound
    (predictiveChosen predictiveComparator taskChosen taskComparator
      floor reference α predictiveSlack taskSlack η : ℝ)
    (hα : 0 < α)
    (hfloor : floor ≤ predictiveChosen)
    (hpredictive : predictiveComparator ≤ floor + predictiveSlack)
    (htask : taskComparator ≤ reference + taskSlack)
    (hcomparison : predictiveChosen + α * taskChosen ≤
      predictiveComparator + α * taskComparator + η) :
    taskChosen - reference ≤ taskSlack + (predictiveSlack + η) / α := by
  have hweighted : (taskChosen - reference - taskSlack) * α ≤
      predictiveSlack + η := by
    have htaskWeighted := mul_le_mul_of_nonneg_left htask (le_of_lt hα)
    nlinarith
  have hquotient := (le_div_iff₀ hα).2 hweighted
  linarith

/-- A finite population visits only finitely many representation values,
even if the code type is a real vector space. Restricting to that image
extends the risk decomposition to arbitrary ambient representation spaces. -/
theorem risk_pythagorean_finite_support
    (p : FiniteLaw Ω) (y : Ω → ℝ) (representation : Ω → J) (head : J → ℝ) :
    risk p y (head ∘ representation) =
      risk p y (bayesPredictor p y representation) +
        risk p (bayesPredictor p y representation) (head ∘ representation) := by
  classical
  let labels : Finset J := Finset.univ.image representation
  let finiteRepresentation : Ω → labels := fun ω =>
    ⟨representation ω, Finset.mem_image.mpr ⟨ω, Finset.mem_univ _, rfl⟩⟩
  have hmean : bayesPredictor p y finiteRepresentation =
      bayesPredictor p y representation := by
    funext ω
    simp [bayesPredictor, conditionalMean, fiberMass, finiteRepresentation]
  have h := risk_pythagorean p y finiteRepresentation (fun j => head j.val)
  simpa only [hmean, Function.comp_def] using h

/-- Bayes optimality for an arbitrary code type over a finite population. -/
theorem bayes_optimal_finite_support
    (p : FiniteLaw Ω) (y : Ω → ℝ) (representation : Ω → J) (head : J → ℝ) :
    risk p y (bayesPredictor p y representation) ≤
      risk p y (head ∘ representation) := by
  rw [risk_pythagorean_finite_support]
  exact le_add_of_nonneg_right (risk_nonneg _ _ _)

/-- The information-gap identity also only needs the population to be
finite, not the ambient lower or upper code space. -/
theorem nested_risk_identity_finite_support
    (p : FiniteLaw Ω) (y : Ω → ℝ) (z : Ω → I) (encoder : I → J) :
    risk p y (bayesPredictor p y (encoder ∘ z)) -
      risk p y (bayesPredictor p y z) = informationGap p y z encoder := by
  have h := risk_pythagorean_finite_support p y z
    (conditionalMean p y (encoder ∘ z) ∘ encoder)
  change risk p y (bayesPredictor p y (encoder ∘ z)) =
    risk p y (bayesPredictor p y z) + informationGap p y z encoder at h
  linarith

/-- The upper representation's information loss is no greater than the
excess task risk of any actual head using that representation. -/
theorem information_gap_le_task_excess
    (p : FiniteLaw Ω) (y : Ω → ℝ) (z : Ω → I)
    (encoder : I → J) (head : J → ℝ) :
    informationGap p y z encoder ≤
      risk p y (head ∘ (encoder ∘ z)) - risk p y (bayesPredictor p y z) := by
  have hupper := bayes_optimal_finite_support p y (encoder ∘ z) head
  have hidentity := nested_risk_identity_finite_support p y z encoder
  linarith

/-- One population-objective comparison already gives a quantitative
task-risk bound. The lower reference is the Bayes risk for the original
context, not an assumed zero target error. -/
theorem task_excess_bound_of_comparison
    (p : FiniteLaw Ω) (y : Ω → ℝ) (z : Ω → I)
    (encoder : C → I → J) (head : C → J → ℝ)
    (predictiveCost : C → ℝ) (chosen comparator : C)
    (floor α predictiveSlack taskSlack η : ℝ)
    (hα : 0 < α)
    (hfloor : floor ≤ predictiveCost chosen)
    (hpredictive : predictiveCost comparator ≤ floor + predictiveSlack)
    (htask : risk p y (head comparator ∘ (encoder comparator ∘ z)) ≤
      risk p y (bayesPredictor p y z) + taskSlack)
    (hcomparison : weightedObjective predictiveCost
      (fun c => risk p y (head c ∘ (encoder c ∘ z))) α chosen ≤
      weightedObjective predictiveCost
        (fun c => risk p y (head c ∘ (encoder c ∘ z))) α comparator + η) :
    risk p y (head chosen ∘ (encoder chosen ∘ z)) -
        risk p y (bayesPredictor p y z) ≤
      taskSlack + (predictiveSlack + η) / α := by
  exact weighted_comparison_bound _ _ _ _ _ _ _ _ _ _
    hα hfloor hpredictive htask hcomparison

/-- Approximate global population minimization controls both the chosen
head's excess task risk and information lost by its learned encoder.

The candidate class, law, target, lower context, and objective are fixed.
Existence of a suitable comparator and attainment of approximate global
optimality are hypotheses, not consequences of gradient descent. -/
theorem information_bound_of_approximate_minimum
    (p : FiniteLaw Ω) (y : Ω → ℝ) (z : Ω → I)
    (encoder : C → I → J) (head : C → J → ℝ)
    (predictiveCost : C → ℝ) (admissible : C → Prop)
    (chosen comparator : C) (floor α predictiveSlack taskSlack η : ℝ)
    (hα : 0 < α)
    (hfloor : ∀ c, admissible c → floor ≤ predictiveCost c)
    (hcomparator : admissible comparator)
    (hpredictive : predictiveCost comparator ≤ floor + predictiveSlack)
    (htask : risk p y (head comparator ∘ (encoder comparator ∘ z)) ≤
      risk p y (bayesPredictor p y z) + taskSlack)
    (hfit : ApproximateERM admissible
      (weightedObjective predictiveCost
        (fun c => risk p y (head c ∘ (encoder c ∘ z))) α) chosen η) :
    (risk p y (head chosen ∘ (encoder chosen ∘ z)) -
        risk p y (bayesPredictor p y z) ≤
      taskSlack + (predictiveSlack + η) / α) ∧
    informationGap p y z (encoder chosen) ≤
      taskSlack + (predictiveSlack + η) / α := by
  have hrisk := task_excess_bound_of_comparison p y z encoder head
    predictiveCost chosen comparator floor α predictiveSlack taskSlack η hα
    (hfloor chosen hfit.1) hpredictive htask (hfit.2 comparator hcomparator)
  exact ⟨hrisk, le_trans (information_gap_le_task_excess p y z
    (encoder chosen) (head chosen)) hrisk⟩

/-- When predictive optimality and Bayes task performance are jointly
feasible, the entire task-excess and information-loss budget is `η / α`.
This is a stability guarantee for approximate population minimization, not
only an assertion about exact zero-loss solutions. -/
theorem jointly_feasible_approximate_minimum_bound
    (p : FiniteLaw Ω) (y : Ω → ℝ) (z : Ω → I)
    (encoder : C → I → J) (head : C → J → ℝ)
    (predictiveCost : C → ℝ) (admissible : C → Prop)
    (chosen comparator : C) (floor α η : ℝ)
    (hα : 0 < α)
    (hfloor : ∀ c, admissible c → floor ≤ predictiveCost c)
    (hcomparator : admissible comparator)
    (hpredictive : predictiveCost comparator = floor)
    (htask : risk p y (head comparator ∘ (encoder comparator ∘ z)) =
      risk p y (bayesPredictor p y z))
    (hfit : ApproximateERM admissible
      (weightedObjective predictiveCost
        (fun c => risk p y (head c ∘ (encoder c ∘ z))) α) chosen η) :
    (risk p y (head chosen ∘ (encoder chosen ∘ z)) -
      risk p y (bayesPredictor p y z) ≤ η / α) ∧
    informationGap p y z (encoder chosen) ≤ η / α := by
  simpa only [zero_add] using information_bound_of_approximate_minimum p y z
    encoder head predictiveCost admissible chosen comparator floor α 0 0 η hα
    hfloor hcomparator (by simp [hpredictive]) (by simp [htask]) hfit

/-- A feasible comparator simultaneously attaining the predictive floor and
lower-context Bayes risk forces every exact global minimum to preserve this
task's conditional mean on support. Irreducible label noise is permitted:
the lower-context Bayes risk need not be zero. -/
theorem exact_minimum_preserves_task_information
    (p : FiniteLaw Ω) (y : Ω → ℝ) (z : Ω → I)
    (encoder : C → I → J) (head : C → J → ℝ)
    (predictiveCost : C → ℝ) (admissible : C → Prop)
    (chosen comparator : C) (floor α : ℝ)
    (hα : 0 < α)
    (hfloor : ∀ c, admissible c → floor ≤ predictiveCost c)
    (hcomparator : admissible comparator)
    (hpredictive : predictiveCost comparator = floor)
    (htask : risk p y (head comparator ∘ (encoder comparator ∘ z)) =
      risk p y (bayesPredictor p y z))
    (hfit : ApproximateERM admissible
      (weightedObjective predictiveCost
        (fun c => risk p y (head c ∘ (encoder c ∘ z))) α) chosen 0) :
    informationGap p y z (encoder chosen) = 0 := by
  have hbound := (information_bound_of_approximate_minimum p y z encoder head
    predictiveCost admissible chosen comparator floor α 0 0 0 hα hfloor
    hcomparator (by simp [hpredictive]) (by simp [htask]) hfit).2
  simp only [add_zero, zero_div] at hbound
  exact le_antisymm hbound (informationGap_nonneg p y z (encoder chosen))

/-- An approximate empirical optimizer also gets a population guarantee if
the entire weighted objective has a certified uniform deviation bound.
This includes the statistical price `2 * ε`; empirical loss alone is not a
certificate of preserved population information. -/
theorem information_bound_of_empirical_minimum
    (p : FiniteLaw Ω) (y : Ω → ℝ) (z : Ω → I)
    (encoder : C → I → J) (head : C → J → ℝ)
    (predictiveCost empirical : C → ℝ) (admissible : C → Prop)
    (chosen comparator : C) (floor α predictiveSlack taskSlack ε η : ℝ)
    (hα : 0 < α)
    (hfloor : ∀ c, admissible c → floor ≤ predictiveCost c)
    (hcomparator : admissible comparator)
    (hpredictive : predictiveCost comparator ≤ floor + predictiveSlack)
    (htask : risk p y (head comparator ∘ (encoder comparator ∘ z)) ≤
      risk p y (bayesPredictor p y z) + taskSlack)
    (hdev : UniformDeviation admissible
      (weightedObjective predictiveCost
        (fun c => risk p y (head c ∘ (encoder c ∘ z))) α) empirical ε)
    (hfit : ApproximateERM admissible empirical chosen η) :
    informationGap p y z (encoder chosen) ≤
      taskSlack + (predictiveSlack + (2 * ε + η)) / α := by
  have horacle := population_oracle_inequality admissible
    (weightedObjective predictiveCost
      (fun c => risk p y (head c ∘ (encoder c ∘ z))) α)
    empirical chosen ε η hdev hfit comparator hcomparator
  have hcomparison : weightedObjective predictiveCost
      (fun c => risk p y (head c ∘ (encoder c ∘ z))) α chosen ≤
      weightedObjective predictiveCost
        (fun c => risk p y (head c ∘ (encoder c ∘ z))) α comparator +
          (2 * ε + η) := by linarith
  have hrisk := task_excess_bound_of_comparison p y z encoder head
    predictiveCost chosen comparator floor α predictiveSlack taskSlack (2 * ε + η)
    hα (hfloor chosen hfit.1) hpredictive htask hcomparison
  exact le_trans (information_gap_le_task_excess p y z
    (encoder chosen) (head chosen)) hrisk

end HJepa.TaskObjective
