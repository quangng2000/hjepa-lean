import HJepa.MaskedPrediction.Model

namespace HJepa.MaskedPrediction

open HJepa.Abstraction

/-- Exact expectation over all four equally likely visible contexts. -/
theorem loss_exact (u v a b p q : ℝ) :
    loss u v a b p q =
      (u - p * a) ^ 2 + (p * b) ^ 2 + (q * a) ^ 2 + (v - q * b) ^ 2 := by
  norm_num [loss, fairInput, pairError, rawTarget, rawExample, decoder, encoder,
    mask, taskSign, nuisanceSign, signedBit, Fintype.sum_prod_type, Fintype.sum_bool]
  ring

/-- Under unit second moment, training the heads is an exact least-squares problem. -/
theorem loss_complete_square (u v a b p q : ℝ) (h : Normalized a b) :
    loss u v a b p q =
      (p - u * a) ^ 2 + (q - v * b) ^ 2 + reducedLoss u v a b := by
  rw [loss_exact]
  unfold Normalized at h
  unfold reducedLoss
  nlinarith [congrArg (fun t : ℝ => (p ^ 2 + q ^ 2 - u ^ 2 - v ^ 2) * t) h]

theorem loss_optimal_heads (u v a b : ℝ) (h : Normalized a b) :
    loss u v a b (u * a) (v * b) = reducedLoss u v a b := by
  rw [loss_complete_square u v a b (u * a) (v * b) h]
  simp

theorem reducedLoss_le_loss (u v a b p q : ℝ) (h : Normalized a b) :
    reducedLoss u v a b ≤ loss u v a b p q := by
  rw [loss_complete_square u v a b p q h]
  nlinarith [sq_nonneg (p - u * a), sq_nonneg (q - v * b)]

theorem loss_eq_reducedLoss_iff (u v a b p q : ℝ) (h : Normalized a b) :
    loss u v a b p q = reducedLoss u v a b ↔ p = u * a ∧ q = v * b := by
  rw [loss_complete_square u v a b p q h]
  constructor
  · intro heq
    have hp : (p - u * a) ^ 2 = 0 := by nlinarith [sq_nonneg (q - v * b)]
    have hq : (q - v * b) ^ 2 = 0 := by nlinarith [sq_nonneg (p - u * a)]
    exact ⟨sub_eq_zero.mp (sq_eq_zero_iff.mp hp), sub_eq_zero.mp (sq_eq_zero_iff.mp hq)⟩
  · rintro ⟨rfl, rfl⟩
    simp

/-- Parameter normalization gives zero mean and unit variance of the actual encoder. -/
theorem encoded_moment_normalized (a b : ℝ) (h : Normalized a b) :
    HJepa.Objective.MomentNormalized (encoded a b) := by
  unfold Normalized at h
  constructor <;>
    norm_num [HJepa.Objective.encoderMean, HJepa.Objective.encoderSecondMoment,
      encoded, encoder, mask, rawExample, taskSign, nuisanceSign,
      signedBit, Fintype.sum_bool] <;> nlinarith

theorem downstreamRisk_exact (a b k : ℝ) :
    downstreamRisk a b k = (1 - k * a) ^ 2 + (k * b) ^ 2 := by
  norm_num [downstreamRisk, risk, fairInput, encoded, encoder, mask, rawExample,
    taskSign, nuisanceSign, signedBit, Fintype.sum_prod_type, Fintype.sum_bool]
  ring

theorem downstreamRisk_complete_square (a b k : ℝ) (h : Normalized a b) :
    downstreamRisk a b k = (k - a) ^ 2 + b ^ 2 := by
  rw [downstreamRisk_exact]
  unfold Normalized at h
  nlinarith [congrArg (fun t : ℝ => k ^ 2 * t) h]

/-- The optimal linear task readout has coefficient `a`. -/
theorem downstreamRisk_optimal (a b : ℝ) (h : Normalized a b) :
    downstreamRisk a b a = b ^ 2 := by
  rw [downstreamRisk_complete_square a b a h]
  simp

theorem downstreamRisk_lower_bound (a b k : ℝ) (h : Normalized a b) :
    b ^ 2 ≤ downstreamRisk a b k := by
  rw [downstreamRisk_complete_square a b k h]
  exact le_add_of_nonneg_left (sq_nonneg (k - a))

end HJepa.MaskedPrediction
