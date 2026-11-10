import HJepa.TemporalPrediction.Rollout.Moments

namespace HJepa.TemporalPrediction

open HJepa.Abstraction

/-- The learned linear encoder inherits both persistence rates. -/
theorem pathExpectation_encoder_product (d : Dynamics) (actions : List Bool)
    (a b c : ℝ) (x : ℝ × ℝ) :
    pathExpectation d actions (fun y => c * encoder a b y) x =
      c * actionProduct actions *
        (a * d.signalPersistence ^ actions.length * x.1 +
          b * d.nuisancePersistence ^ actions.length * x.2) := by
  have heq : (fun y => c * encoder a b y) =
      (RolloutQuadratic.mk 0 (c * a) (c * b) 0 0 0).eval := by
    funext y
    simp only [encoder, HJepa.MaskedPrediction.encoder, RolloutQuadratic.eval]
    ring
  rw [heq, pathExpectation_quadratic]
  simp only [RolloutQuadratic.transport, RolloutQuadratic.eval]
  ring

/-- Exact latent lag second moment, obtained from the finite Markov kernel. -/
theorem encoded_lag_moment (d : Dynamics) (actions : List Bool) (a b : ℝ) :
    populationExpectation d actions (fun x y => encoder a b x * encoder a b y) =
      actionProduct actions * (a ^ 2 * d.signalPersistence ^ actions.length +
        b ^ 2 * d.nuisancePersistence ^ actions.length) := by
  unfold populationExpectation
  simp only [pathExpectation_encoder_product]
  norm_num [HJepa.MaskedPrediction.fairInput, current, signedBit, encoder,
    HJepa.MaskedPrediction.encoder, Fintype.sum_prod_type, Fintype.sum_bool]
  ring

/-- The encoder has zero mean at every step under the fair initial law. -/
theorem encoded_future_mean (d : Dynamics) (actions : List Bool) (a b : ℝ) :
    populationExpectation d actions (fun _ y => encoder a b y) = 0 := by
  have h (x : ℝ × ℝ) :
      pathExpectation d actions (fun y => encoder a b y) x =
        actionProduct actions * (a * d.signalPersistence ^ actions.length * x.1 +
          b * d.nuisancePersistence ^ actions.length * x.2) := by
    simpa only [one_mul] using pathExpectation_encoder_product d actions a b 1 x
  unfold populationExpectation
  simp only [h]
  norm_num [HJepa.MaskedPrediction.fairInput, current, signedBit,
    Fintype.sum_prod_type, Fintype.sum_bool]
  ring

end HJepa.TemporalPrediction
