import HJepa.Abstraction

/-!
# Finite-dimensional prediction under hierarchical abstraction

This module lifts the scalar finite-law results coordinatewise to a target
`y : Ω → D → ℝ`, where `D` is a finite coordinate type. The loss is the **sum**
of squared coordinate errors, equivalently squared Euclidean error. It is not
divided by the number of coordinates, and permits even an empty coordinate type.

All representations below are fixed deterministic maps on the same finite
probability space. Three levels mean `z`, `g ∘ z`, and `h ∘ (g ∘ z)`; they do not
assert that separately trained audio encoders are nested. These identities do
not establish a training guarantee or a continuous-audio population theorem.
-/

namespace HJepa.Vector

open scoped BigOperators

open HJepa.Abstraction (FiniteLaw)

variable {Ω D I J K : Type*} [Fintype Ω] [Fintype D]

/-- Expected squared Euclidean prediction error, without dimension normalization. -/
def risk (p : FiniteLaw Ω) (y f : Ω → D → ℝ) : ℝ :=
  ∑ d, Abstraction.risk p (fun ω => y ω d) (fun ω => f ω d)

/-- The conditional mean of each target coordinate, evaluated at the encoder. -/
noncomputable def bayesPredictor (p : FiniteLaw Ω) (y : Ω → D → ℝ)
    (z : Ω → I) : Ω → D → ℝ :=
  fun ω d => Abstraction.bayesPredictor p (fun ω' => y ω' d) z ω

/-- Sum of the coordinatewise prediction information gaps. -/
noncomputable def informationGap (p : FiniteLaw Ω) (y : Ω → D → ℝ)
    (z : Ω → I) (g : I → J) : ℝ :=
  ∑ d, Abstraction.informationGap p (fun ω => y ω d) z g

/-- Sum of coordinatewise excess risks of a chosen feature predictor. -/
noncomputable def excessRisk (p : FiniteLaw Ω) (y : Ω → D → ℝ)
    (z : Ω → I) (a : I → D → ℝ) : ℝ :=
  ∑ d, Abstraction.excessRisk p (fun ω => y ω d) z (fun i => a i d)

/-- The coordinate-sum definition is expected squared Euclidean error. -/
theorem risk_eq_sum_squared_error (p : FiniteLaw Ω) (y f : Ω → D → ℝ) :
    risk p y f = ∑ ω, p.weight ω * ∑ d, (y ω d - f ω d) ^ 2 := by
  simp only [risk, Abstraction.risk, Finset.mul_sum]
  exact Finset.sum_comm

theorem risk_nonneg (p : FiniteLaw Ω) (y f : Ω → D → ℝ) :
    0 ≤ risk p y f := by
  exact Finset.sum_nonneg fun d _ => Abstraction.risk_nonneg _ _ _

/-- Zero error means coordinatewise equality on support; null outcomes are unrestricted. -/
theorem risk_eq_zero_iff (p : FiniteLaw Ω) (y f : Ω → D → ℝ) :
    risk p y f = 0 ↔ ∀ ω, 0 < p.weight ω → ∀ d, y ω d = f ω d := by
  constructor
  · intro h ω hω d
    have hd : Abstraction.risk p (fun ω => y ω d) (fun ω => f ω d) = 0 :=
      (Finset.sum_eq_zero_iff_of_nonneg
        (fun d _ => Abstraction.risk_nonneg p (fun ω => y ω d)
          (fun ω => f ω d))).mp h d (Finset.mem_univ _)
    exact (Abstraction.risk_eq_zero_iff _ _ _).mp hd ω hω
  · intro h
    apply Finset.sum_eq_zero
    intro d _
    exact (Abstraction.risk_eq_zero_iff _ _ _).mpr fun ω hω => h ω hω d

/-- Coordinatewise orthogonality yields the vector Pythagorean identity. -/
theorem risk_pythagorean [Fintype I] (p : FiniteLaw Ω) (y : Ω → D → ℝ)
    (z : Ω → I) (a : I → D → ℝ) :
    risk p y (a ∘ z) = risk p y (bayesPredictor p y z) +
      risk p (bayesPredictor p y z) (a ∘ z) := by
  unfold risk
  rw [← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl fun d _ =>
    Abstraction.risk_pythagorean p (fun ω => y ω d) z (fun i => a i d)

/-- The vector conditional mean is optimal among all vector functions of `z`. -/
theorem bayes_optimal [Fintype I] (p : FiniteLaw Ω) (y : Ω → D → ℝ)
    (z : Ω → I) (a : I → D → ℝ) :
    risk p y (bayesPredictor p y z) ≤ risk p y (a ∘ z) := by
  rw [risk_pythagorean]
  exact le_add_of_nonneg_right (risk_nonneg _ _ _)

theorem informationGap_eq_risk (p : FiniteLaw Ω) (y : Ω → D → ℝ)
    (z : Ω → I) (g : I → J) :
    informationGap p y z g =
      risk p (bayesPredictor p y z) (bayesPredictor p y (g ∘ z)) := rfl

theorem informationGap_nonneg (p : FiniteLaw Ω) (y : Ω → D → ℝ)
    (z : Ω → I) (g : I → J) : 0 ≤ informationGap p y z g := by
  exact risk_nonneg _ _ _

/-- Conditional-mean agreement is target-specific, not distributional sufficiency. -/
theorem informationGap_eq_zero_iff (p : FiniteLaw Ω) (y : Ω → D → ℝ)
    (z : Ω → I) (g : I → J) :
    informationGap p y z g = 0 ↔ ∀ ω, 0 < p.weight ω → ∀ d,
      bayesPredictor p y z ω d = bayesPredictor p y (g ∘ z) ω d :=
  risk_eq_zero_iff _ _ _

/-- The Bayes-risk increment is exactly the sum of coordinate information gaps. -/
theorem nested_risk_identity [Fintype I] (p : FiniteLaw Ω) (y : Ω → D → ℝ)
    (z : Ω → I) (g : I → J) :
    risk p y (bayesPredictor p y (g ∘ z)) - risk p y (bayesPredictor p y z) =
      informationGap p y z g := by
  unfold risk informationGap
  rw [← Finset.sum_sub_distrib]
  exact Finset.sum_congr rfl fun d _ =>
    Abstraction.nested_risk_identity p (fun ω => y ω d) z g

/-- Deterministic compression cannot improve unrestricted vector Bayes risk. -/
theorem abstraction_bayes_risk_monotone [Fintype I] (p : FiniteLaw Ω)
    (y : Ω → D → ℝ) (z : Ω → I) (g : I → J) :
    risk p y (bayesPredictor p y z) ≤ risk p y (bayesPredictor p y (g ∘ z)) := by
  have hi := nested_risk_identity p y z g
  have hn := informationGap_nonneg p y z g
  linarith

theorem excessRisk_eq_risk_sub (p : FiniteLaw Ω) (y : Ω → D → ℝ)
    (z : Ω → I) (a : I → D → ℝ) :
    excessRisk p y z a = risk p y (a ∘ z) - risk p y (bayesPredictor p y z) := by
  simp only [excessRisk, Abstraction.excessRisk, risk, bayesPredictor,
    Function.comp_apply, Finset.sum_sub_distrib]
  rfl

theorem excessRisk_nonneg [Fintype I] (p : FiniteLaw Ω) (y : Ω → D → ℝ)
    (z : Ω → I) (a : I → D → ℝ) : 0 ≤ excessRisk p y z a := by
  rw [excessRisk_eq_risk_sub]
  exact sub_nonneg.mpr (bayes_optimal p y z a)

/-- Vector excess risk equals squared distance from the vector Bayes predictor. -/
theorem excessRisk_eq_distance [Fintype I] (p : FiniteLaw Ω)
    (y : Ω → D → ℝ) (z : Ω → I) (a : I → D → ℝ) :
    excessRisk p y z a = risk p (bayesPredictor p y z) (a ∘ z) := by
  rw [excessRisk_eq_risk_sub, risk_pythagorean]
  ring

/-- Exact vector gain/loss accounting for two arbitrary chosen feature predictors. -/
theorem learning_gain_loss_identity [Fintype I] (p : FiniteLaw Ω)
    (y : Ω → D → ℝ) (z : Ω → I) (g : I → J)
    (lower : I → D → ℝ) (upper : J → D → ℝ) :
    risk p y (upper ∘ (g ∘ z)) - risk p y (lower ∘ z) =
      informationGap p y z g -
        (excessRisk p y z lower - excessRisk p y (g ∘ z) upper) := by
  have h := nested_risk_identity p y z g
  rw [excessRisk_eq_risk_sub, excessRisk_eq_risk_sub]
  linarith

/-- Improvement requires the excess-risk reduction to outweigh information loss. -/
theorem abstraction_improves_iff [Fintype I] (p : FiniteLaw Ω)
    (y : Ω → D → ℝ) (z : Ω → I) (g : I → J)
    (lower : I → D → ℝ) (upper : J → D → ℝ) :
    risk p y (upper ∘ (g ∘ z)) < risk p y (lower ∘ z) ↔
      informationGap p y z g <
        excessRisk p y z lower - excessRisk p y (g ∘ z) upper := by
  have h := learning_gain_loss_identity p y z g lower upper
  constructor <;> intro hh <;> linarith

/-! ## Three fixed, nested representation levels -/

/-- Low-to-high Bayes-risk growth telescopes into the two adjacent information gaps. -/
theorem three_level_risk_identity [Fintype I] [Fintype J]
    (p : FiniteLaw Ω) (y : Ω → D → ℝ) (z : Ω → I) (g : I → J) (h : J → K) :
    risk p y (bayesPredictor p y (h ∘ (g ∘ z))) -
        risk p y (bayesPredictor p y z) =
      informationGap p y z g + informationGap p y (g ∘ z) h := by
  have h12 := nested_risk_identity p y z g
  have h23 := nested_risk_identity p y (g ∘ z) h
  linarith

/-- Direct low-to-high compression loses exactly the sum lost at each adjacent level. -/
theorem three_level_informationGap_identity [Fintype I] [Fintype J]
    (p : FiniteLaw Ω) (y : Ω → D → ℝ) (z : Ω → I) (g : I → J) (h : J → K) :
    informationGap p y z (h ∘ g) =
      informationGap p y z g + informationGap p y (g ∘ z) h := by
  have h13 := nested_risk_identity p y z (h ∘ g)
  have ht := three_level_risk_identity p y z g h
  simpa only [Function.comp_assoc] using h13.symm.trans ht

/-- No cumulative vector information loss occurs exactly when neither step loses it. -/
theorem three_level_cumulative_gap_eq_zero_iff [Fintype I] [Fintype J]
    (p : FiniteLaw Ω) (y : Ω → D → ℝ) (z : Ω → I) (g : I → J) (h : J → K) :
    informationGap p y z (h ∘ g) = 0 ↔
      informationGap p y z g = 0 ∧ informationGap p y (g ∘ z) h = 0 := by
  rw [three_level_informationGap_identity]
  have h12 := informationGap_nonneg p y z g
  have h23 := informationGap_nonneg p y (g ∘ z) h
  constructor
  · intro hzero
    constructor <;> linarith
  · rintro ⟨hzero12, hzero23⟩
    rw [hzero12, hzero23, add_zero]

/-- Strict cumulative loss occurs exactly when at least one adjacent step loses information. -/
theorem three_level_cumulative_gap_pos_iff [Fintype I] [Fintype J]
    (p : FiniteLaw Ω) (y : Ω → D → ℝ) (z : Ω → I) (g : I → J) (h : J → K) :
    0 < informationGap p y z (h ∘ g) ↔
      0 < informationGap p y z g ∨ 0 < informationGap p y (g ∘ z) h := by
  rw [three_level_informationGap_identity]
  have h12 := informationGap_nonneg p y z g
  have h23 := informationGap_nonneg p y (g ∘ z) h
  constructor
  · intro hpos
    by_cases hfirst : 0 < informationGap p y z g
    · exact Or.inl hfirst
    · exact Or.inr (by linarith)
  · rintro (hfirst | hsecond) <;> linarith

/-- The highest level has at least the low and medium levels' unrestricted Bayes risk. -/
theorem three_level_bayes_risk_monotone [Fintype I] [Fintype J]
    (p : FiniteLaw Ω) (y : Ω → D → ℝ) (z : Ω → I) (g : I → J) (h : J → K) :
    risk p y (bayesPredictor p y z) ≤ risk p y (bayesPredictor p y (g ∘ z)) ∧
      risk p y (bayesPredictor p y (g ∘ z)) ≤
        risk p y (bayesPredictor p y (h ∘ (g ∘ z))) :=
  ⟨abstraction_bayes_risk_monotone p y z g,
    abstraction_bayes_risk_monotone p y (g ∘ z) h⟩

/-- Three-level vector gain/loss identity; the intermediate predictor cancels. -/
theorem three_level_gain_loss_identity [Fintype I] [Fintype J]
    (p : FiniteLaw Ω) (y : Ω → D → ℝ) (z : Ω → I) (g : I → J) (h : J → K)
    (lower : I → D → ℝ) (upper : K → D → ℝ) :
    risk p y (upper ∘ (h ∘ (g ∘ z))) - risk p y (lower ∘ z) =
      informationGap p y z g + informationGap p y (g ∘ z) h -
        (excessRisk p y z lower - excessRisk p y (h ∘ (g ∘ z)) upper) := by
  have ht := three_level_risk_identity p y z g h
  rw [excessRisk_eq_risk_sub, excessRisk_eq_risk_sub]
  linarith

/-- A high-level learned predictor improves iff its excess-risk saving covers both gaps. -/
theorem three_level_improves_iff [Fintype I] [Fintype J]
    (p : FiniteLaw Ω) (y : Ω → D → ℝ) (z : Ω → I) (g : I → J) (h : J → K)
    (lower : I → D → ℝ) (upper : K → D → ℝ) :
    risk p y (upper ∘ (h ∘ (g ∘ z))) < risk p y (lower ∘ z) ↔
      informationGap p y z g + informationGap p y (g ∘ z) h <
        excessRisk p y z lower - excessRisk p y (h ∘ (g ∘ z)) upper := by
  have ht := three_level_gain_loss_identity p y z g h lower upper
  constructor <;> intro hh <;> linarith

/-- Equal chosen-predictor risks occur exactly when the excess-risk saving equals both gaps. -/
theorem three_level_equal_risk_iff [Fintype I] [Fintype J]
    (p : FiniteLaw Ω) (y : Ω → D → ℝ) (z : Ω → I) (g : I → J) (h : J → K)
    (lower : I → D → ℝ) (upper : K → D → ℝ) :
    risk p y (upper ∘ (h ∘ (g ∘ z))) = risk p y (lower ∘ z) ↔
      excessRisk p y z lower - excessRisk p y (h ∘ (g ∘ z)) upper =
        informationGap p y z g + informationGap p y (g ∘ z) h := by
  have ht := three_level_gain_loss_identity p y z g h lower upper
  constructor <;> intro hh <;> linarith

/-- The same accounting explicitly characterizes a harmful high-level predictor. -/
theorem three_level_worsens_iff [Fintype I] [Fintype J]
    (p : FiniteLaw Ω) (y : Ω → D → ℝ) (z : Ω → I) (g : I → J) (h : J → K)
    (lower : I → D → ℝ) (upper : K → D → ℝ) :
    risk p y (lower ∘ z) < risk p y (upper ∘ (h ∘ (g ∘ z))) ↔
      excessRisk p y z lower - excessRisk p y (h ∘ (g ∘ z)) upper <
        informationGap p y z g + informationGap p y (g ∘ z) h := by
  have ht := three_level_gain_loss_identity p y z g h lower upper
  constructor <;> intro hh <;> linarith

end HJepa.Vector
