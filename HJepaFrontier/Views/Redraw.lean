import HJepa.TemporalPrediction.Dynamics

/-!
# Queue entry V1: redrawing a factor between views sets its persistence

Unreviewed Frontier candidate; see `FUZZ_QUEUE.md`.

Suppose the second view keeps a sign factor with probability `1 - r` and
otherwise redraws it as a fair sign, independently of the first view. This
module checks that the resulting weights for "unchanged" and "flipped" are the
innovation weights of the temporal model at persistence `1 - r`. A shared scene
is `r = 0` (persistence one); an independently drawn scene is `r = 1`
(persistence zero).

Only the weights are compared. No `Dynamics` is built from a rendering pipeline.
-/

namespace HJepa.Frontier.Views

noncomputable section

/-- One when the factor is unchanged (`true`), zero when it is flipped (`false`). -/
def unchangedIndicator (bit : Bool) : ℝ := (1 + Abstraction.signedBit bit) / 2

/-- Keep the factor with probability `1 - r`; otherwise redraw it as a fair sign. -/
def redrawWeight (r : ℝ) (bit : Bool) : ℝ :=
  (1 - r) * unchangedIndicator bit + r * (1 / 2)

theorem unchangedIndicator_true : unchangedIndicator true = 1 := by
  norm_num [unchangedIndicator, Abstraction.signedBit]

theorem unchangedIndicator_false : unchangedIndicator false = 0 := by
  norm_num [unchangedIndicator, Abstraction.signedBit]

/-- Redrawing with probability `r` is the innovation law at persistence `1 - r`. -/
theorem redrawWeight_eq_signWeight (r : ℝ) (bit : Bool) :
    redrawWeight r bit = TemporalPrediction.signWeight (1 - r) bit := by
  unfold redrawWeight unchangedIndicator TemporalPrediction.signWeight
  ring

/-- An independently drawn factor is a fair sign whatever the first view held. -/
theorem redrawWeight_independent (bit : Bool) : redrawWeight 1 bit = 1 / 2 := by
  unfold redrawWeight
  ring

/-- A shared factor keeps its sign with certainty. -/
theorem redrawWeight_shared (bit : Bool) :
    redrawWeight 0 bit = unchangedIndicator bit := by
  unfold redrawWeight
  ring

end

end HJepa.Frontier.Views
