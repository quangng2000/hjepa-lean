import HJepa.TemporalPrediction.Model

namespace HJepa.TemporalPrediction

open HJepa.Abstraction
open scoped BigOperators

theorem current_latent_mean (d : Dynamics) (a b : ℝ) :
    (∑ t, (trajectoryLaw d).weight t * encoder a b (current t.1)) = 0 := by
  norm_num [trajectoryLaw, innovationLaw, signWeight, current, encoder,
    HJepa.MaskedPrediction.encoder, signedBit, Fintype.sum_prod_type, Fintype.sum_bool]
  ring

theorem current_latent_secondMoment (d : Dynamics) (a b : ℝ) :
    (∑ t, (trajectoryLaw d).weight t * encoder a b (current t.1) ^ 2) =
      a ^ 2 + b ^ 2 := by
  norm_num [trajectoryLaw, innovationLaw, signWeight, current, encoder,
    HJepa.MaskedPrediction.encoder, signedBit, Fintype.sum_prod_type, Fintype.sum_bool]
  ring

theorem future_latent_mean (d : Dynamics) (action : Bool) (a b : ℝ) :
    (∑ t, (trajectoryLaw d).weight t * encoder a b (future action t.1 t.2)) = 0 := by
  cases action <;>
    norm_num [trajectoryLaw, innovationLaw, signWeight, future, step, current, encoder,
      HJepa.MaskedPrediction.encoder, signedBit,
      Fintype.sum_prod_type, Fintype.sum_bool] <;> ring

theorem future_latent_secondMoment (d : Dynamics) (action : Bool) (a b : ℝ) :
    (∑ t, (trajectoryLaw d).weight t * encoder a b (future action t.1 t.2) ^ 2) =
      a ^ 2 + b ^ 2 := by
  cases action <;>
    norm_num [trajectoryLaw, innovationLaw, signWeight, future, step, current, encoder,
      HJepa.MaskedPrediction.encoder, signedBit,
      Fintype.sum_prod_type, Fintype.sum_bool] <;> ring

/-- Unit-variance normalization constrains both the context and learned target. -/
theorem learned_latents_normalized (d : Dynamics) (action : Bool) (a b : ℝ)
    (h : Normalized a b) :
    (∑ t, (trajectoryLaw d).weight t * encoder a b (current t.1)) = 0 ∧
    (∑ t, (trajectoryLaw d).weight t * encoder a b (current t.1) ^ 2) = 1 ∧
    (∑ t, (trajectoryLaw d).weight t * encoder a b (future action t.1 t.2)) = 0 ∧
    (∑ t, (trajectoryLaw d).weight t * encoder a b (future action t.1 t.2) ^ 2) = 1 := by
  exact ⟨current_latent_mean d a b, (current_latent_secondMoment d a b).trans h,
    future_latent_mean d action a b, (future_latent_secondMoment d action a b).trans h⟩

end HJepa.TemporalPrediction
