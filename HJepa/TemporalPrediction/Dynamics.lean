import HJepa.MaskedPrediction.Model

/-! A controlled stationary two-sign Markov model, with independent innovations. -/

namespace HJepa.TemporalPrediction

open HJepa.Abstraction
open scoped BigOperators

abbrev Input := HJepa.MaskedPrediction.Input
abbrev Innovation := Input
abbrev Trajectory := Input × Innovation

/-- Nonnegative persistence parameters allow deterministic endpoints. -/
structure Dynamics where
  signalPersistence : ℝ
  nuisancePersistence : ℝ
  signal_nonneg : 0 ≤ signalPersistence
  signal_le_one : signalPersistence ≤ 1
  nuisance_nonneg : 0 ≤ nuisancePersistence
  nuisance_le_one : nuisancePersistence ≤ 1

/-- An innovation is positive with probability `(1 + persistence) / 2`. -/
noncomputable def signWeight (persistence : ℝ) (bit : Bool) : ℝ :=
  (1 + persistence * signedBit bit) / 2

theorem signWeight_nonneg (persistence : ℝ) (hlo : 0 ≤ persistence)
    (hhi : persistence ≤ 1) (bit : Bool) : 0 ≤ signWeight persistence bit := by
  cases bit <;> simp [signWeight, signedBit] <;> linarith

/-- The signal and nuisance innovations are independent biased signs. -/
noncomputable def innovationLaw (d : Dynamics) : FiniteLaw Innovation where
  weight := fun e => signWeight d.signalPersistence e.1 *
    signWeight d.nuisancePersistence e.2
  nonneg := fun e => mul_nonneg
    (signWeight_nonneg _ d.signal_nonneg d.signal_le_one e.1)
    (signWeight_nonneg _ d.nuisance_nonneg d.nuisance_le_one e.2)
  total_weight := by
    norm_num [signWeight, signedBit, Fintype.sum_prod_type, Fintype.sum_bool]
    ring

/-- Sixteen atoms: two initial independent fair signs and two innovations. -/
noncomputable def trajectoryLaw (d : Dynamics) : FiniteLaw Trajectory where
  weight := fun t => (1 / 4) * (innovationLaw d).weight t.2
  nonneg := fun t => mul_nonneg (by norm_num) ((innovationLaw d).nonneg t.2)
  total_weight := by
    norm_num [innovationLaw, signWeight, signedBit,
      Fintype.sum_prod_type, Fintype.sum_bool]
    ring

def current (x : Input) : ℝ × ℝ := (signedBit x.1, signedBit x.2)

/-- A known action multiplies both next-state coordinates by its sign. -/
noncomputable def step (action : Bool) (e : Innovation) (x : ℝ × ℝ) : ℝ × ℝ :=
  (signedBit action * signedBit e.1 * x.1,
    signedBit action * signedBit e.2 * x.2)

noncomputable def future (action : Bool) (x : Input) (e : Innovation) : ℝ × ℝ :=
  step action e (current x)

theorem innovation_signal_mean (d : Dynamics) :
    (∑ e, (innovationLaw d).weight e * signedBit e.1) = d.signalPersistence := by
  norm_num [innovationLaw, signWeight, signedBit,
    Fintype.sum_prod_type, Fintype.sum_bool]
  ring

theorem innovation_nuisance_mean (d : Dynamics) :
    (∑ e, (innovationLaw d).weight e * signedBit e.2) = d.nuisancePersistence := by
  norm_num [innovationLaw, signWeight, signedBit,
    Fintype.sum_prod_type, Fintype.sum_bool]
  ring

end HJepa.TemporalPrediction
