import HJepa.Expressivity

/-!
# An exact random-sample learning guarantee for fixed compositional features

The input to this learner is the fixed high parity feature from `Expressivity`,
paired with its target sign. Its law under `fairFour` is proved to be uniform.
The sample expectation below explicitly averages the full binary sample tree:
each of the `2^n` ordered samples has weight `1 / 2^n`.

The learner reads arbitrary real labels and fits the affine function through
the observed negative/positive features. On this target it fits the identity
once both signs have appeared; with one sign it uses the observed label as a
constant. The simpler sign-state machine is proved equivalent on this data.
It always interpolates its training examples. Nevertheless, its population
risk is two on a one-sign sample and zero when both signs occur. Averaging
over IID samples gives exactly `4 / 2^n` for `n >= 1`; three examples give
expected risk `1/2`, below every raw/middle affine readout's risk of at least one.

This is supervised fitting of a fixed, handcrafted feature, not JEPA encoder
learning. A nonlinear raw-input predictor can also solve parity. The theorem
is an expectation guarantee, not samplewise superiority or a real-audio claim.
-/

namespace HJepa.SampleLearning

open HJepa.Abstraction HJepa.Expressivity
open scoped BigOperators

/-- Exact pushforward of the raw sixteen-outcome population to the high bit. -/
theorem high_feature_fair (f : Bool → ℝ) :
    (∑ s : FourSigns, fairFour.weight s * f (high s)) =
      (f false + f true) / 2 := by
  norm_num [fairFour, high, toHigh, middle, toMiddle, low, productBit,
    Fintype.sum_prod_type, Fintype.sum_bool]
  ring

/-- The target equals the signed high feature, for every raw input. -/
theorem target_is_high_sign (s : FourSigns) :
    parityTarget s = signedBit (high s) := by
  simp [parityTarget, high, toHigh, middle, toMiddle, low, signed_productBit,
    Function.comp_def]

/-- A sufficient statistic of the labels/features observed so far. -/
inductive FitState where
  | unseen
  | one (bit : Bool)
  | both
  deriving DecidableEq

def observe : FitState → Bool → FitState
  | .unseen, b => .one b
  | .one a, b => if a = b then .one a else .both
  | .both, _ => .both

/-- Concrete affine coefficients, bias first and slope second. -/
noncomputable def coefficients : FitState → ℝ × ℝ
  | .unseen => (0, 0)
  | .one b => (signedBit b, 0)
  | .both => (0, 1)

def fitFrom (state : FitState) : List Bool → FitState
  | [] => state
  | b :: rest => fitFrom (observe state b) rest

def fit (sample : List Bool) : FitState := fitFrom .unseen sample

noncomputable def prediction (state : FitState) (b : Bool) : ℝ :=
  highAffine (coefficients state).1 (coefficients state).2 b

noncomputable def populationRisk (state : FitState) : ℝ :=
  risk fairFour parityTarget
    (highAffine (coefficients state).1 (coefficients state).2 ∘ high)

theorem population_risk_unseen : populationRisk .unseen = 1 := by
  norm_num [populationRisk, coefficients, high_affine_risk]

theorem population_risk_one (b : Bool) : populationRisk (.one b) = 2 := by
  cases b <;> norm_num [populationRisk, coefficients, high_affine_risk, signedBit]

theorem population_risk_both : populationRisk .both = 0 := by
  norm_num [populationRisk, coefficients, high_affine_risk]

theorem observe_fits (state : FitState) (b : Bool) :
    prediction (observe state b) b = signedBit b := by
  cases state with
  | unseen => simp [observe, prediction, coefficients, highAffine]
  | one a => cases a <;> cases b <;>
      norm_num [observe, prediction, coefficients, highAffine, signedBit]
  | both => simp [observe, prediction, coefficients, highAffine]

theorem observe_preserves_fit (state : FitState) (a b : Bool)
    (h : prediction state a = signedBit a) :
    prediction (observe state b) a = signedBit a := by
  cases state with
  | unseen => cases a <;> norm_num [prediction, coefficients, highAffine, signedBit] at h
  | one c => cases c <;> cases a <;> cases b <;>
      norm_num [observe, prediction, coefficients, highAffine, signedBit] at *
  | both => simp [observe, prediction, coefficients, highAffine]

theorem fitFrom_preserves_fit (sample : List Bool) (state : FitState) (a : Bool)
    (h : prediction state a = signedBit a) :
    prediction (fitFrom state sample) a = signedBit a := by
  induction sample generalizing state with
  | nil => exact h
  | cons b rest ih =>
      exact ih (observe state b) (observe_preserves_fit state a b h)

/-- Every observed example is fitted exactly, even for one-sign samples. -/
theorem fitFrom_interpolates (sample : List Bool) (state : FitState) (a : Bool)
    (ha : a ∈ sample) : prediction (fitFrom state sample) a = signedBit a := by
  induction sample generalizing state with
  | nil => simp at ha
  | cons b rest ih =>
      rcases List.mem_cons.mp ha with hab | ha
      · subst a
        exact fitFrom_preserves_fit rest (observe state b) b (observe_fits state b)
      · exact ih (observe state b) ha

noncomputable def empiricalLoss (sample : List Bool) (state : FitState) : ℝ :=
  (sample.map fun b => (signedBit b - prediction state b) ^ 2).sum

/-- Unnormalized empirical squared loss is zero, hence its average is too. -/
theorem training_loss_zero (sample : List Bool) : empiricalLoss sample (fit sample) = 0 := by
  apply List.sum_eq_zero
  intro x hx
  obtain ⟨b, hb, rfl⟩ := List.mem_map.mp hx
  rw [show prediction (fit sample) b = signedBit b from
    fitFrom_interpolates sample .unseen b hb]
  simp

/-- Exact expectation over an IID sequence of fair Boolean observations.
Each recursion independently gives both next signs probability one half. -/
noncomputable def fairSampleAverage : ℕ → (List Bool → ℝ) → ℝ
  | 0, f => f []
  | n + 1, f =>
      (fairSampleAverage n (fun rest => f (false :: rest)) +
        fairSampleAverage n (fun rest => f (true :: rest))) / 2

/-- Exact expectation over IID raw four-sign inputs from the original law. -/
noncomputable def rawSampleAverage : ℕ → (List FourSigns → ℝ) → ℝ
  | 0, f => f []
  | n + 1, f => ∑ s : FourSigns,
      fairFour.weight s * rawSampleAverage n (fun rest => f (s :: rest))

/-- The feature-sample calculation is exactly the pushforward of IID raw
samples, not a separately assumed data distribution. -/
theorem raw_sample_pushforward (n : ℕ) (f : List Bool → ℝ) :
    rawSampleAverage n (fun sample => f (sample.map high)) =
      fairSampleAverage n f := by
  induction n generalizing f with
  | zero => rfl
  | succ n ih =>
      simp only [rawSampleAverage, List.map_cons]
      calc
        _ = ∑ s : FourSigns, fairFour.weight s *
            fairSampleAverage n (fun rest => f (high s :: rest)) := by
          apply Finset.sum_congr rfl
          intro s _
          congr 1
          exact ih (fun rest => f (high s :: rest))
        _ = _ := high_feature_fair
          (fun b => fairSampleAverage n (fun rest => f (b :: rest)))

noncomputable def expectedRiskFrom (n : ℕ) (state : FitState) : ℝ :=
  fairSampleAverage n (fun sample => populationRisk (fitFrom state sample))

theorem expected_risk_zero_samples (state : FitState) :
    expectedRiskFrom 0 state = populationRisk state := rfl

theorem expected_risk_step (n : ℕ) (state : FitState) :
    expectedRiskFrom (n + 1) state =
      (expectedRiskFrom n (observe state false) +
        expectedRiskFrom n (observe state true)) / 2 := rfl

theorem expected_risk_both (n : ℕ) : expectedRiskFrom n .both = 0 := by
  induction n with
  | zero => exact population_risk_both
  | succ n ih => simp [expected_risk_step, observe, ih]

theorem expected_risk_one (n : ℕ) (b : Bool) :
    expectedRiskFrom n (.one b) = 2 / (2 : ℝ) ^ n := by
  induction n with
  | zero => simpa [expected_risk_zero_samples] using population_risk_one b
  | succ n ih =>
      cases b <;>
        simp [expected_risk_step, observe, expected_risk_both, ih, pow_succ] <;> ring

/-- Exact expected population risk, derived by averaging actual fitted models. -/
theorem iid_expected_population_risk (n : ℕ) :
    expectedRiskFrom (n + 1) .unseen = 4 / (2 : ℝ) ^ (n + 1) := by
  rw [expected_risk_step]
  simp only [observe, expected_risk_one, pow_succ]
  ring

/-- End-to-end version: sample raw contexts IID, encode their high features,
fit the concrete readout, and evaluate on the full original population. -/
theorem iid_raw_expected_population_risk (n : ℕ) :
    rawSampleAverage (n + 1)
      (fun sample => populationRisk (fit (sample.map high))) =
      4 / (2 : ℝ) ^ (n + 1) := by
  exact (raw_sample_pushforward (n + 1)
    (fun sample => populationRisk (fit sample))).trans (iid_expected_population_risk n)

/-- Three IID examples suffice for strict superiority in expectation over
every fixed affine raw or middle readout, under the same population law. -/
theorem three_samples_expected_risk : expectedRiskFrom 3 .unseen = 1 / 2 := by
  norm_num [show 3 = 2 + 1 from rfl, iid_expected_population_risk]

theorem three_samples_beat_affine_readouts
    (bias a b c d middleBias middleA middleB : ℝ) :
    expectedRiskFrom 3 .unseen <
        risk fairFour parityTarget (lowAffine bias a b c d ∘ low) ∧
      expectedRiskFrom 3 .unseen <
        risk fairFour parityTarget (middleAffine middleBias middleA middleB ∘ middle) := by
  rw [three_samples_expected_risk]
  constructor
  · linarith [low_affine_risk_lower_bound bias a b c d]
  · linarith [middle_affine_risk_lower_bound middleBias middleA middleB]

/-! ## Label-based learning

The state above simplifies the exact probability calculation. The actual
interpolator below accepts arbitrary real labels, remembers the first observed
label at each of the two feature values, and solves the two affine equations.
It never assigns the identity coefficients without reading the labels.
Repeated contradictory labels are not claimed to be interpolated.
-/

/-- First observed negative-feature label and positive-feature label. -/
abbrev LabelState := Option ℝ × Option ℝ

def remember (old : Option ℝ) (label : ℝ) : Option ℝ :=
  match old with
  | none => some label
  | some previous => some previous

def observeLabel (state : LabelState) (datum : Bool × ℝ) : LabelState :=
  if datum.1 then (state.1, remember state.2 datum.2)
  else (remember state.1 datum.2, state.2)

def fitLabelsFrom (state : LabelState) : List (Bool × ℝ) → LabelState
  | [] => state
  | datum :: rest => fitLabelsFrom (observeLabel state datum) rest

/-- Bias and slope fitted to the labels, not to a hardcoded target rule. -/
noncomputable def labelCoefficients : LabelState → ℝ × ℝ
  | (none, none) => (0, 0)
  | (some negative, none) => (negative, 0)
  | (none, some positive) => (positive, 0)
  | (some negative, some positive) =>
      ((positive + negative) / 2, (positive - negative) / 2)

noncomputable def learn (sample : List (Bool × ℝ)) : ℝ × ℝ :=
  labelCoefficients (fitLabelsFrom (none, none) sample)

theorem interpolates_negative_label (negative : ℝ) (positive : Option ℝ) :
    highAffine (labelCoefficients (some negative, positive)).1
      (labelCoefficients (some negative, positive)).2 false = negative := by
  cases positive <;> simp [labelCoefficients, highAffine, signedBit]
  ring

theorem interpolates_positive_label (negative : Option ℝ) (positive : ℝ) :
    highAffine (labelCoefficients (negative, some positive)).1
      (labelCoefficients (negative, some positive)).2 true = positive := by
  cases negative <;> simp [labelCoefficients, highAffine, signedBit]
  ring

noncomputable def canonicalLabels : FitState → LabelState
  | .unseen => (none, none)
  | .one false => (some (-1), none)
  | .one true => (none, some 1)
  | .both => (some (-1), some 1)

theorem canonical_observe (state : FitState) (b : Bool) :
    observeLabel (canonicalLabels state) (b, signedBit b) =
      canonicalLabels (observe state b) := by
  cases state with
  | unseen => cases b <;> simp [observeLabel, canonicalLabels, observe, remember, signedBit]
  | one a => cases a <;> cases b <;>
      simp [observeLabel, canonicalLabels, observe, remember, signedBit]
  | both => cases b <;> simp [observeLabel, canonicalLabels, observe, remember]

theorem canonical_coefficients (state : FitState) :
    labelCoefficients (canonicalLabels state) = coefficients state := by
  cases state with
  | unseen => rfl
  | one b => cases b <;> rfl
  | both => norm_num [labelCoefficients, canonicalLabels, coefficients]

theorem labeled_fit_state_eq (sample : List Bool) (state : FitState) :
    fitLabelsFrom (canonicalLabels state) (sample.map fun b => (b, signedBit b)) =
      canonicalLabels (fitFrom state sample) := by
  induction sample generalizing state with
  | nil => rfl
  | cons b rest ih =>
      simp only [List.map_cons, fitLabelsFrom, canonical_observe, fitFrom]
      exact ih (observe state b)

/-- The probability-analysis state is the actual label-based learner on this
target distribution; it is not an additional learning assumption. -/
theorem labeled_coefficients_eq (sample : List Bool) :
    learn (sample.map fun b => (b, signedBit b)) = coefficients (fit sample) := by
  unfold learn
  change labelCoefficients
    (fitLabelsFrom (canonicalLabels .unseen) (sample.map fun b => (b, signedBit b))) = _
  rw [labeled_fit_state_eq, canonical_coefficients]
  rfl

/-- Actual supervised examples: the encoded input paired with its raw target. -/
def labelRawSample (sample : List FourSigns) : List (Bool × ℝ) :=
  sample.map fun s => (high s, parityTarget s)

theorem raw_labels_consistent (sample : List FourSigns) :
    labelRawSample sample = (sample.map high).map (fun b => (b, signedBit b)) := by
  simp only [labelRawSample, List.map_map]
  congr 1
  funext s
  simp [target_is_high_sign]

noncomputable def learnedPopulationRisk (sample : List (Bool × ℝ)) : ℝ :=
  risk fairFour parityTarget (highAffine (learn sample).1 (learn sample).2 ∘ high)

theorem learned_raw_risk_eq (sample : List FourSigns) :
    learnedPopulationRisk (labelRawSample sample) =
      populationRisk (fit (sample.map high)) := by
  rw [learnedPopulationRisk, raw_labels_consistent, labeled_coefficients_eq]
  rfl

theorem learned_raw_interpolates (sample : List FourSigns) (s : FourSigns)
    (hs : s ∈ sample) :
    highAffine (learn (labelRawSample sample)).1
      (learn (labelRawSample sample)).2 (high s) = parityTarget s := by
  rw [raw_labels_consistent, labeled_coefficients_eq, target_is_high_sign]
  exact fitFrom_interpolates (sample.map high) .unseen (high s)
    (List.mem_map.mpr ⟨s, hs, rfl⟩)

/-- Training error of the real-label learner is zero on every raw sample. -/
theorem learned_raw_training_loss_zero (sample : List FourSigns) :
    (sample.map fun s => (parityTarget s -
      highAffine (learn (labelRawSample sample)).1
        (learn (labelRawSample sample)).2 (high s)) ^ 2).sum = 0 := by
  apply List.sum_eq_zero
  intro x hx
  obtain ⟨s, hs, rfl⟩ := List.mem_map.mp hx
  rw [learned_raw_interpolates sample s hs]
  simp

/-- Genuine label-based finite-sample theorem: all real coefficients are fit
from observed labels, the IID raw sample distribution is explicitly averaged,
and test risk is evaluated against the full original raw-input population. -/
theorem iid_labeled_expected_population_risk (n : ℕ) :
    rawSampleAverage (n + 1)
      (fun sample => learnedPopulationRisk (labelRawSample sample)) =
      4 / (2 : ℝ) ^ (n + 1) := by
  simp_rw [learned_raw_risk_eq]
  exact iid_raw_expected_population_risk n

theorem raw_average_const (n : ℕ) (c : ℝ) :
    rawSampleAverage n (fun _ => c) = c := by
  induction n with
  | zero => rfl
  | succ n ih =>
      simp only [rawSampleAverage, ih]
      rw [← Finset.sum_mul, fairFour.total_weight, one_mul]

theorem raw_average_mono (n : ℕ) (f g : List FourSigns → ℝ)
    (h : ∀ sample, f sample ≤ g sample) :
    rawSampleAverage n f ≤ rawSampleAverage n g := by
  induction n generalizing f g with
  | zero => exact h []
  | succ n ih =>
      apply Finset.sum_le_sum
      intro s _
      exact mul_le_mul_of_nonneg_left
        (ih (fun rest => f (s :: rest)) (fun rest => g (s :: rest))
          (fun rest => h (s :: rest))) (fairFour.nonneg s)

/-- Every raw affine training algorithm is bounded below, even when each
coefficient is an arbitrary function of the entire random training sample. -/
theorem trained_low_expected_risk_lower_bound (n : ℕ)
    (bias a b c d : List FourSigns → ℝ) :
    1 ≤ rawSampleAverage n (fun sample =>
      risk fairFour parityTarget
        (lowAffine (bias sample) (a sample) (b sample) (c sample) (d sample) ∘ low)) := by
  calc
    1 = rawSampleAverage n (fun _ => 1) := (raw_average_const n 1).symm
    _ ≤ _ := raw_average_mono n _ _ (fun sample =>
      low_affine_risk_lower_bound (bias sample) (a sample) (b sample) (c sample) (d sample))

theorem trained_middle_expected_risk_lower_bound (n : ℕ)
    (bias a b : List FourSigns → ℝ) :
    1 ≤ rawSampleAverage n (fun sample =>
      risk fairFour parityTarget
        (middleAffine (bias sample) (a sample) (b sample) ∘ middle)) := by
  calc
    1 = rawSampleAverage n (fun _ => 1) := (raw_average_const n 1).symm
    _ ≤ _ := raw_average_mono n _ _ (fun sample =>
      middle_affine_risk_lower_bound (bias sample) (a sample) (b sample))

theorem three_raw_labeled_samples_expected_risk :
    rawSampleAverage 3 (fun sample => learnedPopulationRisk (labelRawSample sample)) =
      1 / 2 := by
  have h := iid_labeled_expected_population_risk 2
  norm_num at h
  exact h

/-- Three IID raw examples give strictly smaller expected test risk than any
sample-dependent raw/middle affine fitting rule. All methods use the same
population, target, sample law and sample count; feature construction is fixed. -/
theorem three_samples_beat_trained_affine_readouts
    (bias a b c d middleBias middleA middleB : List FourSigns → ℝ) :
    rawSampleAverage 3 (fun sample => learnedPopulationRisk (labelRawSample sample)) <
        rawSampleAverage 3 (fun sample => risk fairFour parityTarget
          (lowAffine (bias sample) (a sample) (b sample) (c sample) (d sample) ∘ low)) ∧
      rawSampleAverage 3 (fun sample => learnedPopulationRisk (labelRawSample sample)) <
        rawSampleAverage 3 (fun sample => risk fairFour parityTarget
          (middleAffine (middleBias sample) (middleA sample) (middleB sample) ∘ middle)) := by
  rw [three_raw_labeled_samples_expected_risk]
  constructor
  · linarith [trained_low_expected_risk_lower_bound 3 bias a b c d]
  · linarith [trained_middle_expected_risk_lower_bound 3 middleBias middleA middleB]

end HJepa.SampleLearning
