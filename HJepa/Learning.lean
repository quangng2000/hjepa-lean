import HJepa.ThreeLevel
import Mathlib.Tactic.NormNum

/-!
# Conditional learning guarantees for fixed readout classes

The exact information-loss identities do not by themselves establish what a
training algorithm learns. This module supplies a deterministic bridge: on a
specified admissible candidate class, a uniform empirical/population deviation
bound and approximate empirical minimization imply an oracle inequality.

The bound is an explicit hypothesis, not a concentration theorem. No sample
size, IID sampling, high-probability event, or convergence of a JEPA optimizer
is proved here. The same warning applies if a class or encoder is selected
using the evaluation data: the deviation hypothesis must still be justified
for that selection. For squared loss over unbounded affine coefficients, a
finite uniform deviation fails if the empirical and population quadratic risks
differ in any coefficient-dependent term. Applications may instead use bounded
or finite classes with separately justified bounds. These results can describe
supervised readout fitting on fixed features and a common population law,
target, and loss. Comparing two
self-supervised losses with different representation targets does not supply
such a common objective.

Strict superiority is proved only against a valid population-risk lower bound
for the competing class (or from a sufficiently separated empirical risk and
valid deviation bounds). Merely ordering two risk upper bounds is insufficient.
-/

namespace HJepa.Learning

variable {C D : Type*}

/-- A uniform absolute deviation bound restricted to the admissible class.
The class need not be finite, but this definition does not prove the bound. -/
def UniformDeviation (admissible : C → Prop) (population empirical : C → ℝ)
    (ε : ℝ) : Prop :=
  ∀ c, admissible c → |population c - empirical c| ≤ ε

/-- An admissible selection whose empirical risk is within `η` of every
admissible competitor. Existence and optimizer convergence are not asserted. -/
def ApproximateERM (admissible : C → Prop) (empirical : C → ℝ)
    (chosen : C) (η : ℝ) : Prop :=
  admissible chosen ∧ ∀ c, admissible c → empirical chosen ≤ empirical c + η

/-- A witnessed nonempty class forces a valid deviation budget to be nonnegative. -/
theorem deviation_budget_nonneg (admissible : C → Prop)
    (population empirical : C → ℝ) (ε : ℝ)
    (hdev : UniformDeviation admissible population empirical ε)
    (c : C) (hc : admissible c) : 0 ≤ ε :=
  le_trans (abs_nonneg _) (hdev c hc)

/-- The approximate minimization budget is nonnegative by comparison with the
chosen candidate itself; a separate nonemptiness assumption is unnecessary. -/
theorem optimization_budget_nonneg (admissible : C → Prop)
    (empirical : C → ℝ) (chosen : C) (η : ℝ)
    (hfit : ApproximateERM admissible empirical chosen η) : 0 ≤ η := by
  have hself := hfit.2 chosen hfit.1
  linarith

/-- The standard deterministic oracle inequality: approximate empirical
minimization incurs at most `2 * ε + η` beyond each admissible comparator's
population risk. Both chosen and comparator are covered by the uniform bound. -/
theorem population_oracle_inequality (admissible : C → Prop)
    (population empirical : C → ℝ) (chosen : C) (ε η : ℝ)
    (hdev : UniformDeviation admissible population empirical ε)
    (hfit : ApproximateERM admissible empirical chosen η)
    (competitor : C) (hc : admissible competitor) :
    population chosen ≤ population competitor + 2 * ε + η := by
  have hchosen := (abs_le.mp (hdev chosen hfit.1)).2
  have hcompetitor := (abs_le.mp (hdev competitor hc)).1
  have hmin := hfit.2 competitor hc
  linarith

/-- A witnessed approximation guarantee plus the oracle inequality gives a
total risk budget relative to any reference risk. No infimum is assumed attained. -/
theorem population_bound_of_approximation_witness (admissible : C → Prop)
    (population empirical : C → ℝ) (chosen : C) (ε η reference approximation : ℝ)
    (hdev : UniformDeviation admissible population empirical ε)
    (hfit : ApproximateERM admissible empirical chosen η)
    (competitor : C) (hc : admissible competitor)
    (happrox : population competitor ≤ reference + approximation) :
    population chosen ≤ reference + approximation + 2 * ε + η := by
  have h := population_oracle_inequality admissible population empirical chosen
    ε η hdev hfit competitor hc
  linarith

/-- A genuine sufficient condition for the selected hierarchical predictor to
beat every admissible flat predictor: a good hierarchical witness, a valid
population lower bound for the entire flat class, and a strict separation
larger than the empirical-to-population and optimization penalty.

`hierarchicalRisk` and `flatRisk` must describe the same prediction objective in
an application. A flat risk upper bound cannot replace `hlower`. -/
theorem strict_superiority_of_separated_lower_bound
    (hierarchicalClass : C → Prop) (flatClass : D → Prop)
    (hierarchicalRisk empirical : C → ℝ) (flatRisk : D → ℝ)
    (chosen : C) (ε η lowerBound : ℝ)
    (hdev : UniformDeviation hierarchicalClass hierarchicalRisk empirical ε)
    (hfit : ApproximateERM hierarchicalClass empirical chosen η)
    (competitor : C) (hc : hierarchicalClass competitor)
    (hlower : ∀ d, flatClass d → lowerBound ≤ flatRisk d)
    (hseparated : hierarchicalRisk competitor + 2 * ε + η < lowerBound) :
    ∀ d, flatClass d → hierarchicalRisk chosen < flatRisk d := by
  have horacle := population_oracle_inequality hierarchicalClass hierarchicalRisk
    empirical chosen ε η hdev hfit competitor hc
  intro d hd
  exact lt_of_lt_of_le (lt_of_le_of_lt horacle hseparated) (hlower d hd)

/-- The same strict comparison in budget form. `flatGap` is a certified lower
gap above the common reference, not the flat model's generalization upper bound. -/
theorem strict_superiority_of_budget_separation
    (hierarchicalClass : C → Prop) (flatClass : D → Prop)
    (hierarchicalRisk empirical : C → ℝ) (flatRisk : D → ℝ)
    (chosen : C) (ε η reference approximation flatGap : ℝ)
    (hdev : UniformDeviation hierarchicalClass hierarchicalRisk empirical ε)
    (hfit : ApproximateERM hierarchicalClass empirical chosen η)
    (competitor : C) (hc : hierarchicalClass competitor)
    (happrox : hierarchicalRisk competitor ≤ reference + approximation)
    (hlower : ∀ d, flatClass d → reference + flatGap ≤ flatRisk d)
    (hmargin : approximation + 2 * ε + η < flatGap) :
    ∀ d, flatClass d → hierarchicalRisk chosen < flatRisk d := by
  apply strict_superiority_of_separated_lower_bound hierarchicalClass flatClass
    hierarchicalRisk empirical flatRisk chosen ε η (reference + flatGap)
    hdev hfit competitor hc hlower
  linarith

/-- A certified empirical margin also establishes an actual risk ordering.
This does not require ERM, but both population deviation bounds must hold.
An ordinary validation-point estimate alone does not certify these hypotheses. -/
theorem population_lt_of_empirical_margin
    (population₁ empirical₁ population₂ empirical₂ ε₁ ε₂ : ℝ)
    (hdev₁ : |population₁ - empirical₁| ≤ ε₁)
    (hdev₂ : |population₂ - empirical₂| ≤ ε₂)
    (hmargin : empirical₁ + ε₁ + ε₂ < empirical₂) :
    population₁ < population₂ := by
  have h₁ := (abs_le.mp hdev₁).2
  have h₂ := (abs_le.mp hdev₂).1
  linarith

open HJepa.Abstraction

/-- Specialization to actual finite-law squared-loss readouts. The reference
is the unrestricted Bayes risk for the fixed encoder, and `approximation` is
a witnessed restricted-class excess risk. All empirical risks must estimate
this same law, target, and squared-loss objective. -/
theorem finite_readout_excess_bound {Ω I : Type*} [Fintype Ω]
    (p : FiniteLaw Ω) (y : Ω → ℝ) (z : Ω → I)
    (readout : C → I → ℝ) (admissible : C → Prop) (empirical : C → ℝ)
    (chosen : C) (ε η approximation : ℝ)
    (hdev : UniformDeviation admissible (fun c => risk p y (readout c ∘ z))
      empirical ε)
    (hfit : ApproximateERM admissible empirical chosen η)
    (competitor : C) (hc : admissible competitor)
    (happrox : excessRisk p y z (readout competitor) ≤ approximation) :
    excessRisk p y z (readout chosen) ≤ approximation + 2 * ε + η := by
  have horacle := population_oracle_inequality admissible
    (fun c => risk p y (readout c ∘ z)) empirical chosen ε η hdev hfit competitor hc
  unfold excessRisk at happrox ⊢
  linarith

/-- Three-level total error budget for a highest-level readout: lowest-level
Bayes risk plus the two information gaps, the comparator approximation error,
and the conditional statistical/optimization penalty. This is an upper bound,
not a claim that it is smaller than a trained flat model's actual risk. -/
theorem three_level_population_bound {Ω I J K : Type*}
    [Fintype Ω] [Fintype I] [Fintype J]
    (p : FiniteLaw Ω) (y : Ω → ℝ) (z : Ω → I) (g : I → J) (h : J → K)
    (readout : C → K → ℝ) (admissible : C → Prop) (empirical : C → ℝ)
    (chosen : C) (ε η approximation : ℝ)
    (hdev : UniformDeviation admissible
      (fun c => risk p y (readout c ∘ (h ∘ (g ∘ z)))) empirical ε)
    (hfit : ApproximateERM admissible empirical chosen η)
    (competitor : C) (hc : admissible competitor)
    (happrox : excessRisk p y (h ∘ (g ∘ z)) (readout competitor) ≤ approximation) :
    risk p y (readout chosen ∘ (h ∘ (g ∘ z))) ≤
      risk p y (bayesPredictor p y z) + informationGap p y z g +
        informationGap p y (g ∘ z) h + approximation + 2 * ε + η := by
  have hexcess := finite_readout_excess_bound p y (h ∘ (g ∘ z)) readout
    admissible empirical chosen ε η approximation hdev hfit competitor hc happrox
  have hgap := HJepa.ThreeLevel.bayes_risk_difference p y z g h
  unfold excessRisk at hexcess
  linarith

/-- A sufficient three-level superiority condition on one common finite-law
squared-loss task. Every admissible flat readout must have a certified excess
risk of at least `flatGap`; the entire hierarchical information, approximation,
deviation, and optimization budget must be smaller. If the flat class contains
the unrestricted Bayes predictor, a positive lower gap cannot be certified.
Thus this does not claim hierarchy beats unrestricted optimal prediction. -/
theorem three_level_strict_superiority {Ω I J K : Type*}
    [Fintype Ω] [Fintype I] [Fintype J]
    (p : FiniteLaw Ω) (y : Ω → ℝ) (z : Ω → I) (g : I → J) (h : J → K)
    (readout : C → K → ℝ) (admissible : C → Prop) (empirical : C → ℝ)
    (flatReadout : D → I → ℝ) (flatClass : D → Prop)
    (chosen : C) (ε η approximation flatGap : ℝ)
    (hdev : UniformDeviation admissible
      (fun c => risk p y (readout c ∘ (h ∘ (g ∘ z)))) empirical ε)
    (hfit : ApproximateERM admissible empirical chosen η)
    (competitor : C) (hc : admissible competitor)
    (happrox : excessRisk p y (h ∘ (g ∘ z)) (readout competitor) ≤ approximation)
    (hlower : ∀ d, flatClass d → flatGap ≤ excessRisk p y z (flatReadout d))
    (hmargin : informationGap p y z g + informationGap p y (g ∘ z) h +
      approximation + 2 * ε + η < flatGap) :
    ∀ d, flatClass d →
      risk p y (readout chosen ∘ (h ∘ (g ∘ z))) < risk p y (flatReadout d ∘ z) := by
  have hbound := three_level_population_bound p y z g h readout admissible
    empirical chosen ε η approximation hdev hfit competitor hc happrox
  intro d hd
  have hflat := hlower d hd
  unfold excessRisk at hflat
  linarith

/-- Counterexample to comparing certificates as if they were actual risks:
nonnegative risks `2` and `1` satisfy strictly ordered upper bounds `3 < 4`,
yet the risk with the smaller upper bound is worse. -/
theorem ordered_upper_bounds_do_not_order_risks :
    ∃ hierarchicalRisk flatRisk hierarchicalBound flatBound : ℝ,
      0 ≤ hierarchicalRisk ∧ 0 ≤ flatRisk ∧
      hierarchicalRisk ≤ hierarchicalBound ∧ flatRisk ≤ flatBound ∧
      hierarchicalBound < flatBound ∧ flatRisk < hierarchicalRisk := by
  exact ⟨2, 1, 3, 4, by norm_num⟩

end HJepa.Learning
