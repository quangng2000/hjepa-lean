import HJepa.TemporalPrediction.Rollout.Process

namespace HJepa.TemporalPrediction

open HJepa.Abstraction
open scoped BigOperators

/-- Coefficients of a quadratic observable of the two-dimensional state. -/
structure RolloutQuadratic where
  constant : ℝ
  signal : ℝ
  nuisance : ℝ
  signalSquare : ℝ
  nuisanceSquare : ℝ
  cross : ℝ

noncomputable def RolloutQuadratic.eval (q : RolloutQuadratic) (x : ℝ × ℝ) : ℝ :=
  q.constant + q.signal * x.1 + q.nuisance * x.2 +
    q.signalSquare * x.1 ^ 2 + q.nuisanceSquare * x.2 ^ 2 + q.cross * x.1 * x.2

/-- Backward propagation of an observable through one actual Markov step. -/
noncomputable def RolloutQuadratic.advance (q : RolloutQuadratic)
    (d : Dynamics) (action : Bool) : RolloutQuadratic :=
  ⟨q.constant, q.signal * signedBit action * d.signalPersistence,
    q.nuisance * signedBit action * d.nuisancePersistence,
    q.signalSquare, q.nuisanceSquare,
    q.cross * (d.signalPersistence * d.nuisancePersistence)⟩

/-- Candidate closed form; the expectation identity below proves it from the kernel. -/
noncomputable def RolloutQuadratic.transport (q : RolloutQuadratic)
    (d : Dynamics) (actions : List Bool) : RolloutQuadratic :=
  ⟨q.constant, q.signal * actionProduct actions * d.signalPersistence ^ actions.length,
    q.nuisance * actionProduct actions * d.nuisancePersistence ^ actions.length,
    q.signalSquare, q.nuisanceSquare,
    q.cross * (d.signalPersistence * d.nuisancePersistence) ^ actions.length⟩

/-- Direct expectation over four independent innovation atoms. -/
theorem quadratic_step (q : RolloutQuadratic) (d : Dynamics)
    (action : Bool) (x : ℝ × ℝ) :
    (∑ e, (innovationLaw d).weight e * q.eval (step action e x)) =
      (q.advance d action).eval x := by
  cases action <;>
    norm_num [innovationLaw, signWeight, signedBit, step, RolloutQuadratic.eval,
      RolloutQuadratic.advance, Fintype.sum_prod_type, Fintype.sum_bool] <;> ring

theorem quadratic_transport_step (q : RolloutQuadratic) (d : Dynamics)
    (action : Bool) (actions : List Bool) :
    (q.transport d actions).advance d action = q.transport d (action :: actions) := by
  cases q
  simp only [RolloutQuadratic.transport, RolloutQuadratic.advance,
    actionProduct, List.length_cons, pow_succ]
  congr 1 <;> ring

/-- The quadratic moments used below follow from iterating the finite transition kernel. -/
theorem pathExpectation_quadratic (q : RolloutQuadratic) (d : Dynamics)
    (actions : List Bool) (x : ℝ × ℝ) :
    pathExpectation d actions q.eval x = (q.transport d actions).eval x := by
  induction actions generalizing x with
  | nil => simp [pathExpectation, RolloutQuadratic.transport, actionProduct]
  | cons action actions ih =>
    simp only [pathExpectation, ih]
    rw [quadratic_step, quadratic_transport_step]

end HJepa.TemporalPrediction
