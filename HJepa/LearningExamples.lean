import HJepa.Expressivity
import HJepa.Learning

/-!
# An exactly balanced two-example readout-fitting construction

This is a deterministic existence witness, not a statistical sample-complexity
claim. Two specifically selected raw four-sign inputs have opposite parity
targets and opposite high features. Equal weighting of those two examples
therefore reproduces the full sixteen-outcome population squared risk of every
affine high-feature readout. The zero uniform deviation bound is proved by
calculation, not assumed.

Consequently any eta-approximate empirical minimizer in this high affine class
has population risk at most eta, and beats every low/middle affine readout when
eta is less than one. The identity coefficients are an actual exact empirical
minimizer, so the fitting hypothesis is nonvacuous. An optimizer's convergence
is still not proved. Neither IID/random sampling nor learning the handcrafted
feature maps is involved; this is not a theorem about JEPA encoder training or
real audio. The low/middle comparison remains restricted to affine readouts.
-/

namespace HJepa.LearningExamples

open HJepa.Abstraction HJepa.Expressivity HJepa.Learning

/-- A fixed raw example with positive parity, before any feature encoding. -/
def positiveExample : FourSigns := ((true, true), (true, true))

/-- A fixed raw example with negative parity, before any feature encoding. -/
def negativeExample : FourSigns := ((false, true), (true, true))

/-- Both the actual target and encoded feature are balanced on these examples. -/
theorem examples_are_balanced :
    parityTarget positiveExample = 1 ∧ high positiveExample = true ∧
      parityTarget negativeExample = -1 ∧ high negativeExample = false := by
  norm_num [positiveExample, negativeExample, parityTarget, high, toHigh,
    middle, toMiddle, low, productBit, signedBit]

/-- Affine readout parameters: first coordinate is bias, second is slope. -/
abbrev Coefficients := ℝ × ℝ

/-- Empirical squared risk computed from two raw inputs and their actual
parity labels, with equal weight. It is not defined through population risk. -/
noncomputable def balancedEmpiricalRisk (coefficients : Coefficients) : ℝ :=
  ((parityTarget positiveExample -
      highAffine coefficients.1 coefficients.2 (high positiveExample)) ^ 2 +
    (parityTarget negativeExample -
      highAffine coefficients.1 coefficients.2 (high negativeExample)) ^ 2) / 2

/-- Population risk under the common full sixteen-outcome law. -/
noncomputable def highPopulationRisk (coefficients : Coefficients) : ℝ :=
  risk fairFour parityTarget (highAffine coefficients.1 coefficients.2 ∘ high)

theorem balanced_empirical_formula (coefficients : Coefficients) :
    balancedEmpiricalRisk coefficients =
      (1 - coefficients.2) ^ 2 + coefficients.1 ^ 2 := by
  norm_num [balancedEmpiricalRisk, positiveExample, negativeExample, parityTarget,
    highAffine, high, toHigh, middle, toMiddle, low, productBit, signedBit]
  ring

/-- Exact empirical/population equality for every high affine readout follows
from this chosen balanced design, not from a generalization assumption. -/
theorem balanced_empirical_eq_population (coefficients : Coefficients) :
    balancedEmpiricalRisk coefficients = highPopulationRisk coefficients := by
  rw [balanced_empirical_formula, highPopulationRisk, high_affine_risk]

/-- The general learning module's uniform deviation hypothesis is discharged
with epsilon zero for the entire high affine class in this concrete example. -/
theorem balanced_uniform_deviation_zero :
    UniformDeviation (fun _ : Coefficients => True)
      highPopulationRisk balancedEmpiricalRisk 0 := by
  intro coefficients _
  rw [balanced_empirical_eq_population, sub_self, abs_zero]

/-- The identity high readout is a genuine exact empirical minimizer. -/
theorem identity_exact_empirical_minimizer :
    ApproximateERM (fun _ : Coefficients => True)
      balancedEmpiricalRisk (0, 1) 0 := by
  refine ⟨trivial, ?_⟩
  intro coefficients _
  rw [balanced_empirical_formula, balanced_empirical_formula]
  dsimp
  nlinarith [sq_nonneg (1 - coefficients.2), sq_nonneg coefficients.1]

/-- Every approximate empirical minimizer has population risk at most eta.
There is no assumed positive generalization penalty in this balanced design. -/
theorem approximate_fit_population_bound (chosen : Coefficients) (η : ℝ)
    (hfit : ApproximateERM (fun _ : Coefficients => True)
      balancedEmpiricalRisk chosen η) :
    highPopulationRisk chosen ≤ η := by
  have horacle := population_oracle_inequality (fun _ : Coefficients => True)
    highPopulationRisk balancedEmpiricalRisk chosen 0 η
    balanced_uniform_deviation_zero hfit (0, 1) trivial
  have hzero : highPopulationRisk (0, 1) = 0 := high_identity_risk_zero
  rw [hzero] at horacle
  linarith

/-- The same guarantee directly controls error in the learned affine coefficients. -/
theorem approximate_fit_coefficient_error (chosen : Coefficients) (η : ℝ)
    (hfit : ApproximateERM (fun _ : Coefficients => True)
      balancedEmpiricalRisk chosen η) :
    (1 - chosen.2) ^ 2 + chosen.1 ^ 2 ≤ η := by
  have h := approximate_fit_population_bound chosen η hfit
  rwa [highPopulationRisk, high_affine_risk] at h

/-- A fitted, rather than merely exhibited, high affine readout strictly beats
every raw/middle affine competitor if its empirical fitting tolerance is less
than one. All risks use the same full population law, target, and squared loss. -/
theorem approximate_fit_beats_all_affine_readouts (chosen : Coefficients) (η : ℝ)
    (hfit : ApproximateERM (fun _ : Coefficients => True)
      balancedEmpiricalRisk chosen η) (hη : η < 1)
    (bias a b c d middleBias middleA middleB : ℝ) :
    highPopulationRisk chosen < risk fairFour parityTarget (lowAffine bias a b c d ∘ low) ∧
      highPopulationRisk chosen <
        risk fairFour parityTarget (middleAffine middleBias middleA middleB ∘ middle) := by
  have hfitted := approximate_fit_population_bound chosen η hfit
  constructor
  · linarith [low_affine_risk_lower_bound bias a b c d]
  · linarith [middle_affine_risk_lower_bound middleBias middleA middleB]

end HJepa.LearningExamples
