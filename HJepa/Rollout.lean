import Mathlib.Algebra.Ring.GeomSum
import Mathlib.Data.Real.Basic
import Mathlib.Topology.MetricSpace.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-!
# Rollout error, task-cost calibration, and temporal grouping

These are deterministic fixed-horizon statements. Uniform one-step error and
Lipschitz assumptions must be supplied by an application; the results do not
infer them from a training loss.
-/

namespace HJepa.Rollout

/-- The finite geometric sum, including the cases `L = 1` and `n = 0`. -/
def geomSum (L : ℝ) (n : ℕ) : ℝ := ∑ i ∈ Finset.range n, L ^ i

@[simp] theorem geomSum_zero (L : ℝ) : geomSum L 0 = 0 := by
  simp [geomSum]

theorem geomSum_succ (L : ℝ) (n : ℕ) :
    geomSum L (n + 1) = L * geomSum L n + 1 := by
  exact geom_sum_succ

/-- Solving the scalar recurrence requires only a nonnegative multiplier.
Neither zero initial error nor a contractive predictor is assumed. -/
theorem rollout_error_bound (e : ℕ → ℝ) (L ε : ℝ) (hL : 0 ≤ L)
    (hstep : ∀ n, e (n + 1) ≤ ε + L * e n) (n : ℕ) :
    e n ≤ L ^ n * e 0 + ε * geomSum L n := by
  induction n with
  | zero => simp
  | succ n ih =>
    calc
      e (n + 1) ≤ ε + L * e n := hstep n
      _ ≤ ε + L * (L ^ n * e 0 + ε * geomSum L n) :=
        add_le_add_left (mul_le_mul_of_nonneg_left ih hL) ε
      _ = L ^ (n + 1) * e 0 + ε * geomSum L (n + 1) := by
        rw [pow_succ, geomSum_succ]
        ring

/-- The usual triangle-inequality derivation of the scalar recurrence for an
actual predicted trajectory. The local error includes abstraction/refinement. -/
theorem dynamics_error_recurrence {X A : Type*} [PseudoMetricSpace X]
    (G : X → A → X) (actions : ℕ → A) (actual predicted : ℕ → X)
    (L ε : ℝ)
    (hpred : ∀ n, predicted (n + 1) = G (predicted n) (actions n))
    (hlocal : ∀ n, dist (actual (n + 1)) (G (actual n) (actions n)) ≤ ε)
    (hlip : ∀ n,
      dist (G (actual n) (actions n)) (G (predicted n) (actions n)) ≤
        L * dist (actual n) (predicted n)) (n : ℕ) :
    dist (actual (n + 1)) (predicted (n + 1)) ≤
      ε + L * dist (actual n) (predicted n) := by
  rw [hpred n]
  exact (dist_triangle _ (G (actual n) (actions n)) _).trans
    (add_le_add (hlocal n) (hlip n))

/-- Blocking `k` elementary steps into each of `N` macro steps leaves the
inherited geometric error factor exactly unchanged. This holds for every real
`L`, and includes `k = 0`, `N = 0`, and `L = 1`. -/
theorem geometric_blocking (L : ℝ) (k N : ℕ) :
    geomSum L k * geomSum (L ^ k) N = geomSum L (k * N) := by
  by_cases hL : L = 1
  · subst L
    simp [geomSum, Nat.cast_mul]
  · apply mul_right_cancel₀ (sub_ne_zero.mpr hL)
    calc
      (geomSum L k * geomSum (L ^ k) N) * (L - 1) =
          geomSum (L ^ k) N * (geomSum L k * (L - 1)) := by ring
      _ = geomSum (L ^ k) N * (L ^ k - 1) := by
        have hgeom : geomSum L k * (L - 1) = L ^ k - 1 := geom_sum_mul L k
        rw [hgeom]
      _ = (L ^ k) ^ N - 1 := geom_sum_mul (L ^ k) N
      _ = L ^ (k * N) - 1 := by rw [pow_mul]
      _ = geomSum L (k * N) * (L - 1) := (geom_sum_mul L (k * N)).symm

/-- Exact equality of inherited accumulated-error bounds, not just an
inequality. A shorter macro horizon alone therefore provides no improvement. -/
theorem inherited_error_equality (L ε : ℝ) (k N : ℕ) :
    (ε * geomSum L k) * geomSum (L ^ k) N = ε * geomSum L (k * N) := by
  rw [mul_assoc, geometric_blocking]

/-- The same equality includes propagated initial-state error. -/
theorem inherited_full_bound_equality (L ε e₀ : ℝ) (k N : ℕ) :
    (L ^ k) ^ N * e₀ + (ε * geomSum L k) * geomSum (L ^ k) N =
      L ^ (k * N) * e₀ + ε * geomSum L (k * N) := by
  rw [← pow_mul, inherited_error_equality]

/-- A separate task-calibration hypothesis and Lipschitz cost convert a state
error bound into a task-cost error bound. For squared latent distance, establishing
the Lipschitz hypothesis generally needs a bounded relevant domain. -/
theorem task_cost_calibration {X : Type*} [PseudoMetricSpace X]
    (C : X → ℝ) (actual predicted : X) (J K κ E : ℝ)
    (hK : 0 ≤ K)
    (hcal : |J - C actual| ≤ κ)
    (hlip : |C predicted - C actual| ≤ K * dist actual predicted)
    (herror : dist actual predicted ≤ E) :
    |C predicted - J| ≤ κ + K * E := by
  have hcal' : |C actual - J| ≤ κ := by
    simpa only [abs_sub_comm] using hcal
  calc
    |C predicted - J| ≤ |C predicted - C actual| + |C actual - J| :=
      abs_sub_le _ _ _
    _ ≤ K * dist actual predicted + κ := add_le_add hlip hcal'
    _ ≤ κ + K * E := by
      have hmul := mul_le_mul_of_nonneg_left herror hK
      linarith

end HJepa.Rollout
