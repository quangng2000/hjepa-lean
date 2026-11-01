import HJepa.TemporalPrediction.Model

namespace HJepa.TemporalPrediction

open HJepa.Abstraction

/-- Direct evaluation of all sixteen trajectory atoms, for either action. -/
theorem loss_exact (d : Dynamics) (action : Bool) (a b k : ℝ) :
    loss d action a b k = (a ^ 2 + b ^ 2) * (1 + k ^ 2) -
      2 * k * correlation d a b := by
  cases action <;>
    norm_num [loss, trajectoryLaw, innovationLaw, signWeight, future, step,
      current, predictor, encoder, HJepa.MaskedPrediction.encoder,
      signedBit, correlation, Fintype.sum_prod_type, Fintype.sum_bool] <;> ring

theorem loss_action_invariant (d : Dynamics) (a b k : ℝ) :
    loss d false a b k = loss d true a b k := by
  rw [loss_exact, loss_exact]

/-- Jointly learned target and predictor have this exact normalized objective. -/
theorem loss_complete_square (d : Dynamics) (action : Bool) (a b k : ℝ)
    (h : Normalized a b) :
    loss d action a b k = (k - correlation d a b) ^ 2 + reducedLoss d a b := by
  rw [loss_exact, h]
  unfold reducedLoss
  ring

theorem loss_optimal_predictor (d : Dynamics) (action : Bool) (a b : ℝ)
    (h : Normalized a b) :
    loss d action a b (correlation d a b) = reducedLoss d a b := by
  rw [loss_complete_square d action a b _ h]
  simp

theorem reducedLoss_le_loss (d : Dynamics) (action : Bool) (a b k : ℝ)
    (h : Normalized a b) : reducedLoss d a b ≤ loss d action a b k := by
  rw [loss_complete_square d action a b k h]
  exact le_add_of_nonneg_left (sq_nonneg _)

theorem loss_eq_reducedLoss_iff (d : Dynamics) (action : Bool) (a b k : ℝ)
    (h : Normalized a b) :
    loss d action a b k = reducedLoss d a b ↔ k = correlation d a b := by
  rw [loss_complete_square d action a b k h]
  constructor
  · intro heq
    exact sub_eq_zero.mp (sq_eq_zero_iff.mp (by linarith :
      (k - correlation d a b) ^ 2 = 0))
  · intro heq
    simp [heq]

theorem correlation_nonneg (d : Dynamics) (a b : ℝ) :
    0 ≤ correlation d a b :=
  add_nonneg (mul_nonneg d.signal_nonneg (sq_nonneg a))
    (mul_nonneg d.nuisance_nonneg (sq_nonneg b))

theorem correlation_le_one (d : Dynamics) (a b : ℝ) (h : Normalized a b) :
    correlation d a b ≤ 1 := by
  calc
    _ ≤ 1 * a ^ 2 + 1 * b ^ 2 := add_le_add
      (mul_le_mul_of_nonneg_right d.signal_le_one (sq_nonneg a))
      (mul_le_mul_of_nonneg_right d.nuisance_le_one (sq_nonneg b))
    _ = 1 := by simpa using h

end HJepa.TemporalPrediction
