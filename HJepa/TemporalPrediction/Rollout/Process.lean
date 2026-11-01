import HJepa.TemporalPrediction.Model

/-! Genuine autoregression and the finite Markov expectation along an action list. -/

namespace HJepa.TemporalPrediction

open HJepa.Abstraction
open scoped BigOperators

noncomputable def actionProduct : List Bool → ℝ
  | [] => 1
  | action :: actions => signedBit action * actionProduct actions

/-- Every next prediction consumes the previous prediction, not a true state. -/
noncomputable def rollout (k : ℝ) : List Bool → ℝ → ℝ
  | [], z => z
  | action :: actions, z => rollout k actions (predictor action k z)

/-- This recursively sums over fresh independent innovations at each step. -/
noncomputable def pathExpectation (d : Dynamics) :
    List Bool → ((ℝ × ℝ) → ℝ) → (ℝ × ℝ) → ℝ
  | [], f, x => f x
  | action :: actions, f, x => ∑ e, (innovationLaw d).weight e *
      pathExpectation d actions f (step action e x)

theorem actionProduct_sq (actions : List Bool) : actionProduct actions ^ 2 = 1 := by
  induction actions with
  | nil => norm_num [actionProduct]
  | cons action actions ih =>
    cases action <;> simpa [actionProduct, signedBit] using ih

/-- The predictor is repeatedly composed for the full horizon. -/
theorem rollout_closed_form (k z : ℝ) (actions : List Bool) :
    rollout k actions z = k ^ actions.length * actionProduct actions * z := by
  induction actions generalizing z with
  | nil => simp [rollout, actionProduct]
  | cons action actions ih =>
    rw [rollout, ih]
    simp only [List.length_cons, actionProduct, predictor, pow_succ]
    ring

theorem pathExpectation_const (d : Dynamics) (actions : List Bool)
    (c : ℝ) (x : ℝ × ℝ) : pathExpectation d actions (fun _ => c) x = c := by
  induction actions generalizing x with
  | nil => rfl
  | cons action actions ih =>
    simp only [pathExpectation, ih, ← Finset.sum_mul,
      (innovationLaw d).total_weight, one_mul]

theorem pathExpectation_nonneg (d : Dynamics) (actions : List Bool)
    (f : (ℝ × ℝ) → ℝ) (hf : ∀ x, 0 ≤ f x) (x : ℝ × ℝ) :
    0 ≤ pathExpectation d actions f x := by
  induction actions generalizing x with
  | nil => exact hf x
  | cons action actions ih =>
    exact Finset.sum_nonneg fun e _ => mul_nonneg ((innovationLaw d).nonneg e) (ih _)

end HJepa.TemporalPrediction
