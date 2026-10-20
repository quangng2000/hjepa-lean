import HJepa.Abstraction

/-!
# Three-level prediction trade-offs for H-WavJEPA

Fix a finite probability law, a scalar target, and deterministic nested
representations `z`, `g ∘ z`, and `h ∘ (g ∘ z)`. The information loss from the
first representation to the third is exactly the sum of the two adjacent
losses. This yields a three-level gain/loss identity for arbitrary chosen
predictors and a necessary and sufficient condition for strict improvement.

The levels may be interpreted as low, medium, and high temporal abstraction,
but temporal scales and audio semantics are not assumptions or conclusions of
these theorems. The encoders and law are fixed: the results do not compare
independently trained networks or guarantee that training reduces excess risk.
-/

namespace HJepa.ThreeLevel

open HJepa.Abstraction

variable {Ω I J K : Type*} [Fintype Ω]

/-- The cumulative information gap is the sum of the adjacent gaps for three
fixed deterministic nested representations. No finiteness of `K` is needed. -/
theorem informationGap_telescope [Fintype I] [Fintype J]
    (p : FiniteLaw Ω) (y : Ω → ℝ) (z : Ω → I) (g : I → J) (h : J → K) :
    informationGap p y z (h ∘ g) =
      informationGap p y z g + informationGap p y (g ∘ z) h := by
  have h₁₂ := nested_risk_identity p y z g
  have h₂₃ := nested_risk_identity p y (g ∘ z) h
  have h₁₃ := nested_risk_identity p y z (h ∘ g)
  change risk p y (bayesPredictor p y (h ∘ (g ∘ z))) -
    risk p y (bayesPredictor p y z) = informationGap p y z (h ∘ g) at h₁₃
  linarith

/-- Optimal squared-loss prediction risk can only increase along the two
deterministic abstraction steps. -/
theorem bayes_risk_monotone_chain [Fintype I] [Fintype J]
    (p : FiniteLaw Ω) (y : Ω → ℝ) (z : Ω → I) (g : I → J) (h : J → K) :
    risk p y (bayesPredictor p y z) ≤ risk p y (bayesPredictor p y (g ∘ z)) ∧
      risk p y (bayesPredictor p y (g ∘ z)) ≤
        risk p y (bayesPredictor p y (h ∘ (g ∘ z))) := by
  exact ⟨abstraction_bayes_risk_monotone p y z g,
    abstraction_bayes_risk_monotone p y (g ∘ z) h⟩

/-- The difference between highest- and lowest-level optimal risks is the
sum of the adjacent conditional-mean losses. -/
theorem bayes_risk_difference [Fintype I] [Fintype J]
    (p : FiniteLaw Ω) (y : Ω → ℝ) (z : Ω → I) (g : I → J) (h : J → K) :
    risk p y (bayesPredictor p y (h ∘ (g ∘ z))) -
      risk p y (bayesPredictor p y z) =
        informationGap p y z g + informationGap p y (g ∘ z) h := by
  calc
    _ = informationGap p y z (h ∘ g) := nested_risk_identity p y z (h ∘ g)
    _ = _ := informationGap_telescope p y z g h

/-- Exact three-level gain/loss identity: a highest-level chosen predictor
pays both information gaps, offset by any reduction in excess risk. -/
theorem three_level_gain_loss_identity [Fintype I] [Fintype J]
    (p : FiniteLaw Ω) (y : Ω → ℝ) (z : Ω → I) (g : I → J) (h : J → K)
    (lower : I → ℝ) (upper : K → ℝ) :
    risk p y (upper ∘ (h ∘ (g ∘ z))) - risk p y (lower ∘ z) =
      informationGap p y z g + informationGap p y (g ∘ z) h -
        (excessRisk p y z lower - excessRisk p y (h ∘ (g ∘ z)) upper) := by
  have hgap := bayes_risk_difference p y z g h
  unfold excessRisk
  linarith

/-- The highest-level chosen predictor strictly improves on the lowest-level
one exactly when its excess-risk reduction exceeds both adjacent losses. -/
theorem three_level_improves_iff [Fintype I] [Fintype J]
    (p : FiniteLaw Ω) (y : Ω → ℝ) (z : Ω → I) (g : I → J) (h : J → K)
    (lower : I → ℝ) (upper : K → ℝ) :
    risk p y (upper ∘ (h ∘ (g ∘ z))) < risk p y (lower ∘ z) ↔
      informationGap p y z g + informationGap p y (g ∘ z) h <
        excessRisk p y z lower - excessRisk p y (h ∘ (g ∘ z)) upper := by
  have hid := three_level_gain_loss_identity p y z g h lower upper
  constructor <;> intro hh <;> linarith

/-- Conversely, the highest-level chosen predictor is strictly worse exactly
when its excess-risk reduction is smaller than the cumulative information loss. -/
theorem three_level_worsens_iff [Fintype I] [Fintype J]
    (p : FiniteLaw Ω) (y : Ω → ℝ) (z : Ω → I) (g : I → J) (h : J → K)
    (lower : I → ℝ) (upper : K → ℝ) :
    risk p y (lower ∘ z) < risk p y (upper ∘ (h ∘ (g ∘ z))) ↔
      excessRisk p y z lower - excessRisk p y (h ∘ (g ∘ z)) upper <
        informationGap p y z g + informationGap p y (g ∘ z) h := by
  have hid := three_level_gain_loss_identity p y z g h lower upper
  constructor <;> intro hh <;> linarith

/-- The lowest- and highest-level chosen predictors have equal risk exactly
when the excess-risk reduction balances both adjacent information losses. -/
theorem three_level_equal_risk_iff [Fintype I] [Fintype J]
    (p : FiniteLaw Ω) (y : Ω → ℝ) (z : Ω → I) (g : I → J) (h : J → K)
    (lower : I → ℝ) (upper : K → ℝ) :
    risk p y (lower ∘ z) = risk p y (upper ∘ (h ∘ (g ∘ z))) ↔
      excessRisk p y z lower - excessRisk p y (h ∘ (g ∘ z)) upper =
        informationGap p y z g + informationGap p y (g ∘ z) h := by
  have hid := three_level_gain_loss_identity p y z g h lower upper
  constructor <;> intro hh <;> linarith

/-- Zero cumulative information loss holds exactly when neither adjacent
step loses conditional-mean information for this particular target. -/
theorem cumulative_gap_eq_zero_iff [Fintype I] [Fintype J]
    (p : FiniteLaw Ω) (y : Ω → ℝ) (z : Ω → I) (g : I → J) (h : J → K) :
    informationGap p y z (h ∘ g) = 0 ↔
      informationGap p y z g = 0 ∧ informationGap p y (g ∘ z) h = 0 := by
  rw [informationGap_telescope]
  have h₁₂ := informationGap_nonneg p y z g
  have h₂₃ := informationGap_nonneg p y (g ∘ z) h
  constructor
  · intro hzero
    constructor <;> linarith
  · rintro ⟨hzero₁₂, hzero₂₃⟩
    rw [hzero₁₂, hzero₂₃, add_zero]

/-- There is strict cumulative information loss exactly when at least one
of the two abstraction steps has strict information loss. -/
theorem cumulative_gap_pos_iff [Fintype I] [Fintype J]
    (p : FiniteLaw Ω) (y : Ω → ℝ) (z : Ω → I) (g : I → J) (h : J → K) :
    0 < informationGap p y z (h ∘ g) ↔
      0 < informationGap p y z g ∨ 0 < informationGap p y (g ∘ z) h := by
  rw [informationGap_telescope]
  have h₁₂ := informationGap_nonneg p y z g
  have h₂₃ := informationGap_nonneg p y (g ∘ z) h
  constructor
  · intro hpos
    by_cases hfirst : 0 < informationGap p y z g
    · exact Or.inl hfirst
    · exact Or.inr (by linarith)
  · rintro (hfirst | hsecond) <;> linarith

/-- Each adjacent information gap is bounded by the cumulative gap, so a
later deterministic abstraction cannot undo an earlier information loss. -/
theorem adjacent_gaps_le_cumulative [Fintype I] [Fintype J]
    (p : FiniteLaw Ω) (y : Ω → ℝ) (z : Ω → I) (g : I → J) (h : J → K) :
    informationGap p y z g ≤ informationGap p y z (h ∘ g) ∧
      informationGap p y (g ∘ z) h ≤ informationGap p y z (h ∘ g) := by
  rw [informationGap_telescope]
  have h₁₂ := informationGap_nonneg p y z g
  have h₂₃ := informationGap_nonneg p y (g ∘ z) h
  constructor <;> linarith

end HJepa.ThreeLevel
