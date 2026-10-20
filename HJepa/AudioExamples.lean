import HJepa.Abstraction
import HJepa.Readout
import Mathlib.Data.Fintype.BigOperators

/-!
# Finite three-level witnesses for the H-WavJEPA research question

Two independent fair signs encode two pieces of task signal. The low feature
retains both, the middle feature retains the first, and the high feature is
constant. The target is the first sign plus twice the second. These are exact
finite mathematical witnesses, not waveforms, neural training runs, or measured
audio results. In particular, the constant high feature deliberately models
complete compression; it is not an assumption that real high-level features
must collapse.
-/

namespace HJepa.AudioExamples

open HJepa.Abstraction
open scoped BigOperators

/-- Four equally likely pairs of independent Boolean signs. -/
noncomputable def fairPair : FiniteLaw (Bool × Bool) where
  weight := fun _ => 1 / 4
  nonneg := by intro _; norm_num
  total_weight := by norm_num [Fintype.sum_prod_type, Fintype.sum_bool]

/-- A common target with one unit of first-sign variance and four of second-sign variance. -/
def target (s : Bool × Bool) : ℝ := signedBit s.1 + 2 * signedBit s.2

def low : Bool × Bool → Bool × Bool := id

def toMiddle : Bool × Bool → Bool := Prod.fst

def toHigh : Bool → Unit := fun _ => ()

def middle : Bool × Bool → Bool := toMiddle ∘ low

def high : Bool × Bool → Unit := toHigh ∘ middle

/-- The actual conditional means, calculated from the finite law, at all three levels. -/
theorem conditional_means (s : Bool × Bool) :
    bayesPredictor fairPair target low s = target s ∧
      bayesPredictor fairPair target middle s = signedBit s.1 ∧
      bayesPredictor fairPair target high s = 0 := by
  rcases s with ⟨a, b⟩
  cases a <;> cases b <;>
    simp only [bayesPredictor, conditionalMean, fiberMass, Finset.sum_filter] <;>
    norm_num [fairPair, target, signedBit, low, middle, high, toMiddle, toHigh,
      Fintype.sum_prod_type, Fintype.sum_bool]

/-- Each compression step strictly increases optimal risk: zero, four, then five. -/
theorem three_level_bayes_risks :
    risk fairPair target (bayesPredictor fairPair target low) = 0 ∧
      risk fairPair target (bayesPredictor fairPair target middle) = 4 ∧
      risk fairPair target (bayesPredictor fairPair target high) = 5 := by
  simp only [risk, (conditional_means _).1, (conditional_means _).2.1,
    (conditional_means _).2.2]
  norm_num [fairPair, target, signedBit, Fintype.sum_prod_type, Fintype.sum_bool]

/-- Information penalties are computed, not postulated: four plus one equals five. -/
theorem adjacent_information_gaps :
    informationGap fairPair target low toMiddle = 4 ∧
      informationGap fairPair target middle toHigh = 1 ∧
      informationGap fairPair target low (toHigh ∘ toMiddle) = 5 := by
  have h12 := nested_risk_identity fairPair target low toMiddle
  have h23 := nested_risk_identity fairPair target middle toHigh
  have h13 := nested_risk_identity fairPair target low (toHigh ∘ toMiddle)
  have hl := three_level_bayes_risks.1
  have hm := three_level_bayes_risks.2.1
  have hh := three_level_bayes_risks.2.2
  change risk fairPair target (bayesPredictor fairPair target middle) -
    risk fairPair target (bayesPredictor fairPair target low) = _ at h12
  change risk fairPair target (bayesPredictor fairPair target high) -
    risk fairPair target (bayesPredictor fairPair target middle) = _ at h23
  change risk fairPair target (bayesPredictor fairPair target high) -
    risk fairPair target (bayesPredictor fairPair target low) = _ at h13
  constructor
  · linarith
  constructor <;> linarith

/-- A selected high-level predictor can still beat a poorly chosen low-level predictor. -/
theorem imperfect_predictor_gain :
    risk fairPair target ((fun _ : Unit => (0 : ℝ)) ∘ high) = 5 ∧
      risk fairPair target ((fun _ : Bool × Bool => (3 : ℝ)) ∘ low) = 14 := by
  norm_num [risk, fairPair, target, signedBit, low, high, middle, toHigh,
    toMiddle, Fintype.sum_prod_type, Fintype.sum_bool]

/-- Discarded information costs five, but the chosen predictor's excess-risk reduction is fourteen. -/
theorem exact_gain_loss_budget :
    informationGap fairPair target low (toHigh ∘ toMiddle) = 5 ∧
      excessRisk fairPair target low (fun _ => 3) = 14 ∧
      excessRisk fairPair target high (fun _ => 0) = 0 := by
  refine ⟨adjacent_information_gaps.2.2, ?_, ?_⟩
  · unfold excessRisk
    rw [imperfect_predictor_gain.2, three_level_bayes_risks.1]
    norm_num
  · unfold excessRisk
    rw [imperfect_predictor_gain.1, three_level_bayes_risks.2.2]
    norm_num

/-- Unlike the highest level alone, the retained bundle still permits zero Bayes risk. -/
theorem combined_bayes_risk_zero :
    risk fairPair target
      (bayesPredictor fairPair target (Readout.threeLevelBundle toMiddle toHigh ∘ low)) =
        0 := by
  rw [Readout.three_level_bundle_bayes_risk_eq]
  exact three_level_bayes_risks.1

/-- Duplicate coordinates make the sum-loss convention explicit. -/
def repeatedTarget (s : Bool × Bool) (_ : Bool) : ℝ := target s

/-- Two identical target coordinates double each risk; this is not a dimension average. -/
theorem vector_three_level_bayes_risks :
    Vector.risk fairPair repeatedTarget (Vector.bayesPredictor fairPair repeatedTarget low) = 0 ∧
      Vector.risk fairPair repeatedTarget
        (Vector.bayesPredictor fairPair repeatedTarget middle) = 8 ∧
      Vector.risk fairPair repeatedTarget
        (Vector.bayesPredictor fairPair repeatedTarget high) = 10 := by
  simp only [Vector.risk, Vector.bayesPredictor, repeatedTarget, Fintype.sum_bool,
    three_level_bayes_risks.1, three_level_bayes_risks.2.1, three_level_bayes_risks.2.2]
  norm_num

/-- The empty-coordinate boundary is permitted and has zero summed loss. -/
theorem empty_vector_risk (y f : (Bool × Bool) → Fin 0 → ℝ) :
    Vector.risk fairPair y f = 0 := by
  simp [Vector.risk]

/-- A law with one null outcome exercises the support, rather than pointwise, boundary. -/
noncomputable def pointLaw : FiniteLaw Bool where
  weight := fun b => if b then 1 else 0
  nonneg := by intro b; cases b <;> norm_num
  total_weight := by norm_num [Fintype.sum_bool]

/-- Losing a distinction only on a null outcome has zero risk despite pointwise disagreement. -/
theorem null_outcome_example :
    risk pointLaw signedBit (fun _ => 1) = 0 ∧
      signedBit false ≠ (1 : ℝ) ∧
      informationGap pointLaw signedBit (id : Bool → Bool) (fun _ => ()) = 0 := by
  simp only [informationGap, risk, bayesPredictor, conditionalMean, fiberMass,
    Finset.sum_filter]
  norm_num [pointLaw, signedBit, Fintype.sum_bool]

end HJepa.AudioExamples
