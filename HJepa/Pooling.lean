import HJepa.Abstraction
import Mathlib.Tactic.FieldSimp

/-!
# Temporal pooling: explicit noise reduction, bias cost, and correlation floor

This is a finite-law calculation for an aligned scalar observation model. Each
index observes the same target plus an error. Averaging can reduce error when
the cross moments are small; signal variation or misalignment may introduce
bias. Neither alignment, zero means, nor decorrelation is asserted to follow
from hierarchical JEPA training. An unrestricted predictor of all raw
observations can perform the same averaging, so this is a possible mechanism,
not an information-theoretic advantage exclusive to hierarchy.
-/

namespace HJepa.Pooling

open scoped BigOperators
open HJepa.Abstraction

variable {Ω I : Type*} [Fintype Ω] [Fintype I]

/-- Expectation under the finite probability law. -/
def expectation (p : FiniteLaw Ω) (f : Ω → ℝ) : ℝ :=
  ∑ ω, p.weight ω * f ω

/-- Average residual over an observation window. Positivity of the window
size is required by the nontrivial statistical conclusions below. -/
noncomputable def meanError (e : I → Ω → ℝ) (ω : Ω) : ℝ :=
  (∑ i, e i ω) / (Fintype.card I : ℝ)

/-- A pooled prediction in the aligned observation model `y + error`.
This parameterization describes the prediction error; it is not an algorithm
that is allowed to read the unknown target. -/
noncomputable def pooledPrediction (y : Ω → ℝ) (e : I → Ω → ℝ) : Ω → ℝ :=
  fun ω => y ω + meanError e ω

/-- Pairwise raw error second moment. For centered errors it is covariance. -/
def crossMoment (p : FiniteLaw Ω) (e : I → Ω → ℝ) (i j : I) : ℝ :=
  expectation p (fun ω => e i ω * e j ω)

omit [Fintype Ω] in
/-- In the aligned model, the pooled prediction is the arithmetic mean of
the observed values. It does not require access to the target at inference. -/
theorem pooledPrediction_eq_observation_average (y : Ω → ℝ)
    (e : I → Ω → ℝ) (hn : (Fintype.card I : ℝ) ≠ 0) (ω : Ω) :
    pooledPrediction y e ω =
      (∑ i, (y ω + e i ω)) / (Fintype.card I : ℝ) := by
  simp only [pooledPrediction, meanError, Finset.sum_add_distrib,
    Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
  field_simp

theorem expectation_add (p : FiniteLaw Ω) (f g : Ω → ℝ) :
    expectation p (fun ω => f ω + g ω) = expectation p f + expectation p g := by
  simp only [expectation, mul_add, Finset.sum_add_distrib]

theorem expectation_const (p : FiniteLaw Ω) (c : ℝ) :
    expectation p (fun _ => c) = c := by
  simp only [expectation, ← Finset.sum_mul, p.total_weight, one_mul]

theorem expectation_mul_const (p : FiniteLaw Ω) (f : Ω → ℝ) (c : ℝ) :
    expectation p (fun ω => f ω * c) = expectation p f * c := by
  simp only [expectation, ← mul_assoc, Finset.sum_mul]

/-- Exact pooled MSE is the sum of *all* pairwise error second moments,
divided by the square of the window size. No independence assumption is used. -/
theorem pooled_risk_pairwise (p : FiniteLaw Ω) (y : Ω → ℝ)
    (e : I → Ω → ℝ) :
    risk p y (pooledPrediction y e) =
      (∑ i, ∑ j, crossMoment p e i j) / (Fintype.card I : ℝ) ^ 2 := by
  have hsquare (ω : Ω) :
      (y ω - (y ω + meanError e ω)) ^ 2 =
        (∑ i, ∑ j, e i ω * e j ω) / (Fintype.card I : ℝ) ^ 2 := by
    simp only [meanError]
    rw [show y ω - (y ω + (∑ i, e i ω) / (Fintype.card I : ℝ)) =
      -((∑ i, e i ω) / (Fintype.card I : ℝ)) by ring]
    rw [neg_sq, div_pow, pow_two, Finset.sum_mul]
    simp only [Finset.mul_sum]
  simp only [risk, pooledPrediction, hsquare, ← mul_div_assoc, ← Finset.sum_div]
  congr 1
  simp only [Finset.mul_sum, crossMoment, expectation]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro i _
  rw [Finset.sum_comm]

/-- The mean residual is centered if each constituent residual is centered. -/
theorem meanError_centered (p : FiniteLaw Ω) (e : I → Ω → ℝ)
    (hmean : ∀ i, expectation p (e i) = 0) :
    expectation p (meanError e) = 0 := by
  simp only [expectation, meanError, ← mul_div_assoc, ← Finset.sum_div,
    Finset.mul_sum]
  rw [Finset.sum_comm]
  change (∑ i, expectation p (e i)) / (Fintype.card I : ℝ) = 0
  simp only [hmean, Finset.sum_const_zero, zero_div]

/-- Equal marginal second moments and zero off-diagonal cross moments give
the familiar `1 / n` error reduction. Centered independence is sufficient but
stronger than necessary; centering is not separately needed for this
raw-second-moment formulation. -/
theorem pooled_risk_uncorrelated (p : FiniteLaw Ω) (y : Ω → ℝ)
    (e : I → Ω → ℝ) (v : ℝ) (hn : (Fintype.card I : ℝ) ≠ 0)
    (hdiag : ∀ i, crossMoment p e i i = v)
    (hoff : ∀ i j, i ≠ j → crossMoment p e i j = 0) :
    risk p y (pooledPrediction y e) = v / (Fintype.card I : ℝ) := by
  classical
  rw [pooled_risk_pairwise]
  have hrow (i : I) : (∑ j, crossMoment p e i j) = v := by
    rw [Finset.sum_eq_single i]
    · exact hdiag i
    · intro j _ hji
      exact hoff i j (Ne.symm hji)
    · simp
  simp only [hrow, Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
  field_simp

/-- Constant cross moment `c` creates a correlation floor:
`v / n + (n - 1) * c / n`. With centered errors `c` is the common covariance. -/
theorem pooled_risk_equicorrelated (p : FiniteLaw Ω) (y : Ω → ℝ)
    (e : I → Ω → ℝ) (v c : ℝ) (hn : (Fintype.card I : ℝ) ≠ 0)
    (hdiag : ∀ i, crossMoment p e i i = v)
    (hoff : ∀ i j, i ≠ j → crossMoment p e i j = c) :
    risk p y (pooledPrediction y e) =
      v / (Fintype.card I : ℝ) +
        ((Fintype.card I : ℝ) - 1) * c / (Fintype.card I : ℝ) := by
  classical
  rw [pooled_risk_pairwise]
  have hrow (i : I) : (∑ j, crossMoment p e i j) =
      v + ((Fintype.card I : ℝ) - 1) * c := by
    have hentry (j : I) : crossMoment p e i j =
        (if j = i then v - c else 0) + c := by
      by_cases h : j = i
      · subst j
        simp [hdiag]
      · simp [h, hoff i j (Ne.symm h)]
    simp only [hentry, Finset.sum_add_distrib]
    simp
    ring
  simp only [hrow, Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
  field_simp

/-- Perfectly shared error gives no averaging gain. -/
theorem pooled_risk_shared_error (p : FiniteLaw Ω) (y : Ω → ℝ)
    (e : I → Ω → ℝ) (v : ℝ) (hn : (Fintype.card I : ℝ) ≠ 0)
    (hall : ∀ i j, crossMoment p e i j = v) :
    risk p y (pooledPrediction y e) = v := by
  rw [pooled_risk_equicorrelated p y e v v hn (fun i => hall i i)
    (fun i j _ => hall i j)]
  field_simp
  ring

/-- Averaging a centered error and adding a deterministic scalar bias adds
exactly the squared bias to risk. Nonstationarity is one possible source of
such a bias, but this theorem does not assert that real audio has constant bias. -/
theorem pooled_risk_add_bias (p : FiniteLaw Ω) (y : Ω → ℝ)
    (e : I → Ω → ℝ) (b : ℝ)
    (hmean : ∀ i, expectation p (e i) = 0) :
    risk p y (fun ω => pooledPrediction y e ω + b) =
      risk p y (pooledPrediction y e) + b ^ 2 := by
  have hm := meanError_centered p e hmean
  have hexpand : risk p y (fun ω => pooledPrediction y e ω + b) =
      risk p y (pooledPrediction y e) +
        2 * b * expectation p (meanError e) + b ^ 2 := by
    calc
      _ = (∑ ω, p.weight ω * (y ω - pooledPrediction y e ω) ^ 2) +
          (2 * b) * (∑ ω, p.weight ω * meanError e ω) +
          (∑ ω, p.weight ω) * b ^ 2 := by
        simp only [risk, ← Finset.sum_add_distrib, Finset.mul_sum, Finset.sum_mul]
        apply Finset.sum_congr rfl
        intro ω _
        simp only [pooledPrediction]
        ring
      _ = _ := by rw [p.total_weight, one_mul]; rfl
  rw [hm, mul_zero, add_zero] at hexpand
  exact hexpand

/-- Bias-variance tradeoff for a centered, pairwise-uncorrelated window. -/
theorem pooled_risk_biased_uncorrelated (p : FiniteLaw Ω) (y : Ω → ℝ)
    (e : I → Ω → ℝ) (v b : ℝ) (hn : (Fintype.card I : ℝ) ≠ 0)
    (hmean : ∀ i, expectation p (e i) = 0)
    (hdiag : ∀ i, crossMoment p e i i = v)
    (hoff : ∀ i j, i ≠ j → crossMoment p e i j = 0) :
    risk p y (fun ω => pooledPrediction y e ω + b) =
      b ^ 2 + v / (Fintype.card I : ℝ) := by
  rw [pooled_risk_add_bias p y e b hmean,
    pooled_risk_uncorrelated p y e v hn hdiag hoff]
  ring

/-- Both failure modes remain visible: a constant bias and correlated errors. -/
theorem pooled_risk_biased_equicorrelated (p : FiniteLaw Ω) (y : Ω → ℝ)
    (e : I → Ω → ℝ) (v c b : ℝ) (hn : (Fintype.card I : ℝ) ≠ 0)
    (hmean : ∀ i, expectation p (e i) = 0)
    (hdiag : ∀ i, crossMoment p e i i = v)
    (hoff : ∀ i j, i ≠ j → crossMoment p e i j = c) :
    risk p y (fun ω => pooledPrediction y e ω + b) =
      b ^ 2 + v / (Fintype.card I : ℝ) +
        ((Fintype.card I : ℝ) - 1) * c / (Fintype.card I : ℝ) := by
  rw [pooled_risk_add_bias p y e b hmean,
    pooled_risk_equicorrelated p y e v c hn hdiag hoff]
  ring

omit [Fintype I] in
/-- A single aligned observation has its marginal error second moment as risk. -/
theorem single_risk (p : FiniteLaw Ω) (y : Ω → ℝ)
    (e : I → Ω → ℝ) (i : I) :
    risk p y (fun ω => y ω + e i ω) = crossMoment p e i i := by
  unfold risk crossMoment expectation
  apply Finset.sum_congr rfl
  intro ω _
  ring

/-- An exact, testable threshold: the squared pooling bias must be smaller
than the variance saved relative to any single constituent observation. -/
theorem biased_pooling_improves_iff (p : FiniteLaw Ω) (y : Ω → ℝ)
    (e : I → Ω → ℝ) (v b : ℝ) (hn : (Fintype.card I : ℝ) ≠ 0)
    (hmean : ∀ i, expectation p (e i) = 0)
    (hdiag : ∀ i, crossMoment p e i i = v)
    (hoff : ∀ i j, i ≠ j → crossMoment p e i j = 0) (i : I) :
    risk p y (fun ω => pooledPrediction y e ω + b) <
        risk p y (fun ω => y ω + e i ω) ↔
      b ^ 2 < v * (1 - 1 / (Fintype.card I : ℝ)) := by
  rw [pooled_risk_biased_uncorrelated p y e v b hn hmean hdiag hoff,
    single_risk, hdiag]
  have heq : v * (1 - 1 / (Fintype.card I : ℝ)) =
      v - v / (Fintype.card I : ℝ) := by ring
  rw [heq]
  constructor <;> intro h <;> linarith

/-- With correlated centered errors, only the unshared component `v - c`
is reduced. The bias must cost less than this available reduction. -/
theorem biased_correlated_pooling_improves_iff (p : FiniteLaw Ω) (y : Ω → ℝ)
    (e : I → Ω → ℝ) (v c b : ℝ) (hn : (Fintype.card I : ℝ) ≠ 0)
    (hmean : ∀ i, expectation p (e i) = 0)
    (hdiag : ∀ i, crossMoment p e i i = v)
    (hoff : ∀ i j, i ≠ j → crossMoment p e i j = c) (i : I) :
    risk p y (fun ω => pooledPrediction y e ω + b) <
        risk p y (fun ω => y ω + e i ω) ↔
      b ^ 2 < (v - c) * (1 - 1 / (Fintype.card I : ℝ)) := by
  rw [pooled_risk_biased_equicorrelated p y e v c b hn hmean hdiag hoff,
    single_risk, hdiag]
  have heq : (v - c) * (1 - 1 / (Fintype.card I : ℝ)) =
      v - (v / (Fintype.card I : ℝ) +
        ((Fintype.card I : ℝ) - 1) * c / (Fintype.card I : ℝ)) := by
    field_simp
    ring
  rw [heq]
  constructor <;> intro h <;> linarith

end HJepa.Pooling
