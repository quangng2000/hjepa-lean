import Mathlib.Data.Fintype.BigOperators
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring

/-!
# Explicit finite counterexamples to unconditional hierarchy advantages

All probabilities below are actual finite averages. No probabilistic axiom or
assumed risk gap is used. The first model has an independent uniform Boolean
background and hidden mode; abstraction retains the background. Its abstract
dynamics are perfectly predictable at every finite horizon, yet every
deterministic policy using only that abstraction has success probability 1/2.
A policy observing the mode succeeds with probability 1.

The second model discards the sign of a fair ±1 target by squaring it. Its best
constant prediction has squared error 1, while retaining the sign allows zero
error. These examples disprove unconditional implications from predictable
abstract latents to good prediction/control. They are not models of the full
H-JEPA architecture, whose lower levels may retain the missing information.
-/

namespace HJepa.Counterexamples

open scoped BigOperators

structure State where
  background : Bool
  mode : Bool
  goalReached : Bool
  deriving DecidableEq

def initial (c m : Bool) : State := ⟨c, m, false⟩

/-- An action succeeds exactly when it matches the hidden mode. -/
def step (s : State) (a : Bool) : State :=
  ⟨s.background, s.mode, decide (a = s.mode)⟩

def abstraction (s : State) : Bool := s.background

def abstractStep (c _a : Bool) : Bool := c

def rollout (s : State) : List Bool → State
  | [] => s
  | a :: rest => rollout (step s a) rest

/-- The retained background varies across states, so this is not collapse. -/
theorem abstraction_nonconstant :
    ∃ s t : State, abstraction s ≠ abstraction t := by
  exact ⟨initial false false, initial true false, by decide⟩

/-- Prediction of the retained background is exact for every action and state. -/
theorem abstract_dynamics_exact (s : State) (a : Bool) :
    abstraction (step s a) = abstractStep (abstraction s) a := rfl

/-- Exact latent prediction persists through any finite action sequence. -/
theorem abstract_rollout_exact (s : State) (actions : List Bool) :
    abstraction (rollout s actions) = abstraction s := by
  induction actions generalizing s with
  | nil => rfl
  | cons a rest ih =>
    exact ih (step s a)

def goalIndicator (s : State) : ℚ := if s.goalReached then 1 else 0

/-- Uniform expectation over the four equally likely `(background, mode)` states. -/
def successRate (policy : Bool → Bool → Bool) : ℚ :=
  (∑ c : Bool, ∑ m : Bool, goalIndicator (step (initial c m) (policy c m))) / 4

/-- Every deterministic background-only policy succeeds on exactly half the states. -/
theorem abstract_policy_success_half (policy : Bool → Bool) :
    successRate (fun c _m => policy c) = 1 / 2 := by
  unfold successRate
  simp only [Fintype.sum_bool]
  cases htrue : policy true <;> cases hfalse : policy false <;>
    norm_num [htrue, hfalse, goalIndicator, step, initial]

/-- The full-state policy chooses the hidden mode and succeeds on all four states. -/
theorem full_policy_success_one :
    successRate (fun _c m => m) = 1 := by
  norm_num [successRate, goalIndicator, step, initial]

/-- A strict control disadvantage coexists with nonconstant, exact latent prediction. -/
theorem abstract_policy_strictly_worse (policy : Bool → Bool) :
    successRate (fun c _m => policy c) < successRate (fun _c m => m) := by
  rw [abstract_policy_success_half, full_policy_success_one]
  norm_num

def sign (b : Bool) : ℝ := if b then 1 else -1

/-- The abstraction loses only sign: both fair outcomes map to 1. -/
def squaredSign (b : Bool) : ℝ := sign b ^ 2

theorem squared_sign_eq_one (b : Bool) : squaredSign b = 1 := by
  cases b <;> norm_num [squaredSign, sign]

/-- Squared-loss risk averaged over the two equally likely target signs. -/
noncomputable def signRisk (predictor : Bool → ℝ) : ℝ :=
  (∑ b : Bool, (sign b - predictor b) ^ 2) / 2

theorem constant_sign_risk (a : ℝ) : signRisk (fun _ => a) = 1 + a ^ 2 := by
  simp only [signRisk, Fintype.sum_bool, sign, Bool.false_eq_true, ↓reduceIte]
  ring

/-- Every predictor that sees only the squared sign incurs at least unit loss. -/
theorem squared_sign_predictor_risk (f : ℝ → ℝ) :
    signRisk (fun b => f (squaredSign b)) = 1 + f 1 ^ 2 := by
  simp only [squared_sign_eq_one]
  exact constant_sign_risk (f 1)

theorem squared_sign_risk_at_least_one (f : ℝ → ℝ) :
    1 ≤ signRisk (fun b => f (squaredSign b)) := by
  rw [squared_sign_predictor_risk]
  exact le_add_of_nonneg_right (sq_nonneg (f 1))

theorem zero_predictor_attains_one : signRisk (fun _ => 0) = 1 := by
  norm_num [constant_sign_risk]

/-- Retaining the sign allows exact prediction, hence zero squared loss. -/
theorem retained_sign_risk_zero : signRisk sign = 0 := by
  simp [signRisk]

end HJepa.Counterexamples
