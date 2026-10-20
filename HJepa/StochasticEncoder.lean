import HJepa.EncoderLearning

/-!
# Actual stochastic learning of a protected encoder from raw inputs

The trainable encoder sees the two raw fair bits, not a precomputed task
feature. Its two coefficients are updated by the sample squared-error
gradient defined in `EncoderLearning`. The decoder is fixed to the identity,
the predictive background branch is fixed, and the labels are the noiseless
current task bit. This is a supervised linear encoder-learning result, not
convergence of an unsupervised nonlinear JEPA implementation.

The finite sample average explicitly enumerates independent raw-input draws.
With effective step size `1/4`, expected population task risk is multiplied by
`1/2` at every update, from arbitrary initial coefficients. This expectation
does not imply success for every finite sample sequence: repeated inputs in a
single feature direction can leave a task-irrelevant encoder unchanged.
-/

namespace HJepa.StochasticEncoder

open HJepa.Abstraction HJepa.Objective HJepa.EncoderLearning
open scoped BigOperators

/-- Independent fair background and task bits, before encoding. -/
noncomputable def fairInput : FiniteLaw Input where
  weight := fun _ => 1 / 4
  nonneg := by intro _; norm_num
  total_weight := by norm_num [Fintype.sum_prod_type, Fintype.sum_bool]

/-- One actual sample-gradient update of the encoder coefficients.
For the protection objective multiplied by `lambda > 0`, this is the update
with optimizer step size `alpha / lambda` on that weighted objective. -/
noncomputable def sampleStep (alpha : ℝ) (c : Coefficients) (x : Input) :
    Coefficients :=
  (c.1 - alpha * (sampleGradient c x).1,
    c.2 - alpha * (sampleGradient c x).2)

/-- An exact first-order expansion certifies the sample gradient of the
actual squared labeled residual; the remainder is quadratic in the update. -/
theorem exact_sample_loss_increment (c delta : Coefficients) (x : Input) :
    residual (c.1 + delta.1, c.2 + delta.2) x ^ 2 =
      residual c x ^ 2 + (sampleGradient c x).1 * delta.1 +
        (sampleGradient c x).2 * delta.2 +
        (delta.1 * signedBit x.1 + delta.2 * signedBit x.2) ^ 2 := by
  simp only [residual, encoder, sampleGradient]
  ring

/-- The effective update is exactly SGD on the sample protection loss
multiplied by `weight`, using learning rate `alpha / weight`. The frozen
background predictive branch contributes zero loss and zero gradient. -/
theorem sampleStep_is_weighted_objective_step (weight alpha : ℝ)
    (hweight : weight ≠ 0) (c : Coefficients) (x : Input) :
    sampleStep alpha c x =
      (c.1 - (alpha / weight) * (weight * (sampleGradient c x).1),
       c.2 - (alpha / weight) * (weight * (sampleGradient c x).2)) := by
  have hcancel : (alpha / weight) * weight = alpha := div_mul_cancel₀ _ hweight
  simp only [sampleStep, ← mul_assoc, hcancel]

/-- The halving rate is ordinary weighted-loss SGD with learning rate
`1 / (4 * weight)`, for any positive protection weight. -/
theorem quarter_step_weighted_learning_rate (weight : ℝ) (hweight : 0 < weight)
    (c : Coefficients) (x : Input) :
    sampleStep (1 / 4) c x =
      (c.1 - (1 / (4 * weight)) * (weight * (sampleGradient c x).1),
       c.2 - (1 / (4 * weight)) * (weight * (sampleGradient c x).2)) := by
  simpa only [div_div] using
    sampleStep_is_weighted_objective_step weight (1 / 4) (ne_of_gt hweight) c x

/-- Exact expected population risk after one independent raw-input update.
This identity derives the contraction from the actual gradient, not from an
assumed expected improvement or uniform-deviation hypothesis. -/
theorem one_sample_expected_risk (alpha : ℝ) (c : Coefficients) :
    (∑ x : Input, fairInput.weight x * populationRisk (sampleStep alpha c x)) =
      (1 - 4 * alpha + 8 * alpha ^ 2) * populationRisk c := by
  simp only [population_risk_formula]
  norm_num [fairInput, sampleStep, sampleGradient, residual, encoder, signedBit,
    Fintype.sum_prod_type, Fintype.sum_bool]
  ring

/-- The stable effective step `1/4` halves expected task risk in one update. -/
theorem quarter_step_expected_risk (c : Coefficients) :
    (∑ x : Input, fairInput.weight x * populationRisk (sampleStep (1 / 4) c x)) =
      (1 / 2) * populationRisk c := by
  rw [one_sample_expected_risk]
  norm_num

/-- The expected-risk factor is strictly less than one for every positive
effective step below `1/2`. The special step `1/4` minimizes this quadratic. -/
theorem expected_factor_lt_one (alpha : ℝ) (hpos : 0 < alpha)
    (hlt : alpha < 1 / 2) : 1 - 4 * alpha + 8 * alpha ^ 2 < 1 := by
  nlinarith [mul_pos hpos (show 0 < 1 / 2 - alpha by linarith)]

theorem quarter_step_minimizes_expected_factor (alpha : ℝ) :
    1 / 2 ≤ 1 - 4 * alpha + 8 * alpha ^ 2 := by
  nlinarith [sq_nonneg (alpha - 1 / 4)]

/-- Actual sequential SGD: every input changes the encoder before the next
input is processed. No fitted coefficient or optimum is supplied to it. -/
noncomputable def iterate (alpha : ℝ) : Coefficients → List Input → Coefficients
  | c, [] => c
  | c, x :: rest => iterate alpha (sampleStep alpha c x) rest

/-- Exact expectation over all ordered IID raw samples of length `n`.
The recursion gives each of the `4^n` raw samples weight `1 / 4^n`. -/
noncomputable def rawSampleAverage : ℕ → (List Input → ℝ) → ℝ
  | 0, f => f []
  | n + 1, f => ∑ x : Input,
      fairInput.weight x * rawSampleAverage n (fun rest => f (x :: rest))

/-- The full finite-sample law of expected population risk for this SGD
algorithm, including arbitrary initialization and arbitrary fixed step size.
The right-hand side is derived by integrating actual raw-input iterates. -/
theorem iid_expected_population_risk (n : ℕ) (alpha : ℝ) (c : Coefficients) :
    rawSampleAverage n (fun sample => populationRisk (iterate alpha c sample)) =
      (1 - 4 * alpha + 8 * alpha ^ 2) ^ n * populationRisk c := by
  induction n generalizing c with
  | zero => simp [rawSampleAverage, iterate]
  | succ n ih =>
      calc
        _ = ∑ x : Input, fairInput.weight x *
            ((1 - 4 * alpha + 8 * alpha ^ 2) ^ n *
              populationRisk (sampleStep alpha c x)) := by
          simp only [rawSampleAverage, iterate, ih]
        _ = (1 - 4 * alpha + 8 * alpha ^ 2) ^ n *
            (∑ x : Input, fairInput.weight x * populationRisk (sampleStep alpha c x)) := by
          rw [Finset.mul_sum]
          apply Finset.sum_congr rfl
          intro x _
          ring
        _ = (1 - 4 * alpha + 8 * alpha ^ 2) ^ (n + 1) * populationRisk c := by
          rw [one_sample_expected_risk, pow_succ]
          ring

/-- From any initialization, expected task risk is exactly halved per SGD
update at effective step `1/4`. This is not a samplewise guarantee. -/
theorem quarter_step_iid_expected_risk (n : ℕ) (c : Coefficients) :
    rawSampleAverage n (fun sample => populationRisk (iterate (1 / 4) c sample)) =
      (1 / 2 : ℝ) ^ n * populationRisk c := by
  rw [iid_expected_population_risk]
  norm_num

/-- At every sample size, all three levels have the same geometrically
decreasing expected task risk. Their predictive abstraction maps may be
arbitrary because the learned task channel is carried without alteration. -/
theorem quarter_step_three_level_expected_risks (n : ℕ) (c : Coefficients)
    (g h : ℝ → ℝ) :
    let learned := fun sample => model (iterate (1 / 4) c sample)
    rawSampleAverage n (fun sample =>
      ProtectedChannel.representationTaskRisk
        (ProtectedChannel.lowRepresentation (learned sample))
        (ProtectedChannel.protectedReadout (learned sample))) =
        (1 / 2 : ℝ) ^ n * populationRisk c ∧
    rawSampleAverage n (fun sample =>
      ProtectedChannel.representationTaskRisk
        (ProtectedChannel.middleRepresentation (learned sample) g)
        (ProtectedChannel.protectedReadout (learned sample))) =
        (1 / 2 : ℝ) ^ n * populationRisk c ∧
    rawSampleAverage n (fun sample =>
      ProtectedChannel.representationTaskRisk
        (ProtectedChannel.highRepresentation (learned sample) g h)
        (ProtectedChannel.protectedReadout (learned sample))) =
        (1 / 2 : ℝ) ^ n * populationRisk c := by
  have hr := quarter_step_iid_expected_risk n c
  exact ⟨hr, hr, hr⟩

/-- A task-irrelevant background encoder already fits the single raw input
`(true,true)`, so repeating only that input yields a zero sample gradient. -/
theorem repeated_direction_step_fixed (alpha : ℝ) :
    sampleStep alpha (1, 0) (true, true) = (1, 0) := by
  norm_num [sampleStep, sampleGradient, residual, encoder, signedBit]

theorem repeated_direction_iterate_fixed (n : ℕ) (alpha : ℝ) :
    iterate alpha (1, 0) (List.replicate n (true, true)) = (1, 0) := by
  induction n with
  | zero => rfl
  | succ n ih => simpa [List.replicate_succ, iterate, repeated_direction_step_fixed] using ih

/-- Arbitrarily long uninformative finite samples can retain population risk
two, despite the geometric IID expectation theorem. -/
theorem repeated_direction_risk_two (n : ℕ) (alpha : ℝ) :
    populationRisk (iterate alpha (1, 0) (List.replicate n (true, true))) = 2 := by
  rw [repeated_direction_iterate_fixed, population_risk_formula]
  norm_num

end HJepa.StochasticEncoder
