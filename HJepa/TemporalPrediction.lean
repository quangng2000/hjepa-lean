import HJepa.TemporalPrediction.Moments
import HJepa.TemporalPrediction.Comparison.Examples
import HJepa.TemporalPrediction.Comparison.Boundary
import HJepa.TemporalPrediction.Comparison.FutureTask
import HJepa.TemporalPrediction.Controls.Examples
import HJepa.TemporalPrediction.Rollout.Examples
import HJepa.TemporalPrediction.Rollout.Correlation

/-!
# Masked reconstruction versus autoregressive learned-latent prediction

Exact population comparisons in a controlled finite linear model. This models
shared-encoder temporal prediction and genuine recurrent rollouts, but not the
full H-JEPA hierarchy, SIGReg regularizer, neural optimizer, or planner.
-/
