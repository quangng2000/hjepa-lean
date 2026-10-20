import HJepa.Abstraction
import HJepa.Vector
import Mathlib.Data.Fintype.Prod

/-!
# Readouts that retain the lowest-level representation

Appending deterministic higher-level features to a fixed encoder cannot change
its unrestricted Bayes risk when the original features can be recovered. The
proof uses Bayes-risk monotonicity in both directions, not a claim about a
particular neural head or a training procedure.

For a three-level hierarchy, `threeLevelBundle g h` stores the low, medium, and
high features together. Every lower-level predictor has an equal-risk lifted
predictor on this bundle; every predictor on the bundle pulls back to a
lower-level predictor. These are statements about all functions of the fixed
features. A restricted readout class need not contain these compositions, and
training need not find them.
-/

namespace HJepa.Readout

open HJepa.Abstraction

variable {Ω I J K : Type*} [Fintype Ω]

/-- A deterministic feature augmentation with a left inverse on the realized
features preserves unrestricted scalar Bayes risk. Recovery is only required
on the image of `z`, not on unused labels. -/
theorem recoverable_bayes_risk_eq [Fintype I] [Fintype J]
    (p : FiniteLaw Ω) (y : Ω → ℝ) (z : Ω → I)
    (augment : I → J) (recover : J → I)
    (hrecover : ∀ ω, recover (augment (z ω)) = z ω) :
    risk p y (bayesPredictor p y (augment ∘ z)) =
      risk p y (bayesPredictor p y z) := by
  apply le_antisymm
  · have h := abstraction_bayes_risk_monotone p y (augment ∘ z) recover
    have heq : recover ∘ (augment ∘ z) = z := funext hrecover
    rw [heq] at h
    exact h
  · exact abstraction_bayes_risk_monotone p y z augment

/-- Recoverable deterministic augmentation introduces no squared-loss
information gap for the fixed target. -/
theorem recoverable_informationGap_eq_zero [Fintype I] [Fintype J]
    (p : FiniteLaw Ω) (y : Ω → ℝ) (z : Ω → I)
    (augment : I → J) (recover : J → I)
    (hrecover : ∀ ω, recover (augment (z ω)) = z ω) :
    informationGap p y z augment = 0 := by
  have hgap := nested_risk_identity p y z augment
  rw [recoverable_bayes_risk_eq p y z augment recover hrecover, sub_self] at hgap
  exact hgap.symm

/-- A lower-level predictor can be lifted through a recovery map without
changing its predictions or risk. -/
theorem recoverable_lift_risk_eq
    (p : FiniteLaw Ω) (y : Ω → ℝ) (z : Ω → I)
    (augment : I → J) (recover : J → I)
    (hrecover : ∀ ω, recover (augment (z ω)) = z ω)
    (lower : I → ℝ) :
    risk p y ((lower ∘ recover) ∘ (augment ∘ z)) =
      risk p y (lower ∘ z) := by
  have heq : (lower ∘ recover) ∘ (augment ∘ z) = lower ∘ z := by
    funext ω
    simp only [Function.comp_apply, hrecover]
  rw [heq]

/-- Any predictor of deterministically augmented features is already a
function of the original features. No recovery assumption is needed. -/
theorem augmentation_pullback_risk_eq
    (p : FiniteLaw Ω) (y : Ω → ℝ) (z : Ω → I)
    (augment : I → J) (combined : J → ℝ) :
    risk p y (combined ∘ (augment ∘ z)) =
      risk p y ((combined ∘ augment) ∘ z) := rfl

/-- The low, medium, and high representations retained in a single readout.
The first coordinate is exactly the original low-level feature. -/
def threeLevelBundle (g : I → J) (h : J → K) : I → I × J × K :=
  fun i => (i, g i, h (g i))

/-- Projection of the retained three-level bundle exactly recovers the low
representation. This is a feature identity, independent of a probability law. -/
theorem threeLevelBundle_recover (g : I → J) (h : J → K) (i : I) :
    (threeLevelBundle g h i).1 = i := rfl

/-- Retaining all three fixed representations preserves the unrestricted
scalar Bayes risk of the lowest-level representation. -/
theorem three_level_bundle_bayes_risk_eq [Fintype I] [Fintype J] [Fintype K]
    (p : FiniteLaw Ω) (y : Ω → ℝ) (z : Ω → I) (g : I → J) (h : J → K) :
    risk p y (bayesPredictor p y (threeLevelBundle g h ∘ z)) =
      risk p y (bayesPredictor p y z) :=
  recoverable_bayes_risk_eq p y z (threeLevelBundle g h) Prod.fst (fun _ => rfl)

/-- There is no information-loss term from retaining the three levels. -/
theorem three_level_bundle_informationGap_eq_zero
    [Fintype I] [Fintype J] [Fintype K]
    (p : FiniteLaw Ω) (y : Ω → ℝ) (z : Ω → I) (g : I → J) (h : J → K) :
    informationGap p y z (threeLevelBundle g h) = 0 :=
  recoverable_informationGap_eq_zero p y z (threeLevelBundle g h)
    Prod.fst (fun _ => rfl)

/-- A low-level predictor has an equal-risk combined readout: ignore the
medium and high coordinates, and apply the low-level predictor. -/
theorem three_level_lift_risk_eq
    (p : FiniteLaw Ω) (y : Ω → ℝ) (z : Ω → I) (g : I → J) (h : J → K)
    (lower : I → ℝ) :
    risk p y ((lower ∘ Prod.fst) ∘ (threeLevelBundle g h ∘ z)) =
      risk p y (lower ∘ z) := rfl

/-- Conversely, a predictor on all three levels pulls back to a predictor
on the low level, because the other two levels are deterministic functions. -/
theorem three_level_pullback_risk_eq
    (p : FiniteLaw Ω) (y : Ω → ℝ) (z : Ω → I) (g : I → J) (h : J → K)
    (combined : I × J × K → ℝ) :
    risk p y (combined ∘ (threeLevelBundle g h ∘ z)) =
      risk p y ((combined ∘ threeLevelBundle g h) ∘ z) := rfl

/-- Two selected readouts can still have different risks. With the low
features retained, their difference is exactly the excess-risk difference;
there is no guarantee that either training procedure minimizes excess risk. -/
theorem three_level_combined_gain_loss_identity
    [Fintype I] [Fintype J] [Fintype K]
    (p : FiniteLaw Ω) (y : Ω → ℝ) (z : Ω → I) (g : I → J) (h : J → K)
    (lower : I → ℝ) (combined : I × J × K → ℝ) :
    risk p y (combined ∘ (threeLevelBundle g h ∘ z)) - risk p y (lower ∘ z) =
      excessRisk p y (threeLevelBundle g h ∘ z) combined -
        excessRisk p y z lower := by
  unfold excessRisk
  rw [three_level_bundle_bayes_risk_eq]
  ring

/-- The exact set of achievable risks is unchanged when unrestricted
predictors may use the retained three-level bundle. This does not assert
equality for two arbitrary restricted neural readout families. -/
theorem three_level_achievable_risk_iff
    (p : FiniteLaw Ω) (y : Ω → ℝ) (z : Ω → I) (g : I → J) (h : J → K)
    (r : ℝ) :
    (∃ combined : I × J × K → ℝ,
      risk p y (combined ∘ (threeLevelBundle g h ∘ z)) = r) ↔
    (∃ lower : I → ℝ, risk p y (lower ∘ z) = r) := by
  constructor
  · rintro ⟨combined, hr⟩
    exact ⟨combined ∘ threeLevelBundle g h, hr⟩
  · rintro ⟨lower, hr⟩
    exact ⟨lower ∘ Prod.fst, hr⟩

/-! ## Finite-dimensional targets and squared Euclidean loss -/

variable {D : Type*} [Fintype D]

/-- Coordinatewise scalar preservation gives vector Bayes-risk preservation. -/
theorem vector_recoverable_bayes_risk_eq [Fintype I] [Fintype J]
    (p : FiniteLaw Ω) (y : Ω → D → ℝ) (z : Ω → I)
    (augment : I → J) (recover : J → I)
    (hrecover : ∀ ω, recover (augment (z ω)) = z ω) :
    Vector.risk p y (Vector.bayesPredictor p y (augment ∘ z)) =
      Vector.risk p y (Vector.bayesPredictor p y z) := by
  unfold Vector.risk
  exact Finset.sum_congr rfl fun d _ =>
    recoverable_bayes_risk_eq p (fun ω => y ω d) z augment recover hrecover

/-- Retaining all three fixed levels preserves the unrestricted vector Bayes
risk. The dimension may be any finite type; the loss sums squared errors. -/
theorem vector_three_level_bundle_bayes_risk_eq
    [Fintype I] [Fintype J] [Fintype K]
    (p : FiniteLaw Ω) (y : Ω → D → ℝ) (z : Ω → I) (g : I → J) (h : J → K) :
    Vector.risk p y (Vector.bayesPredictor p y (threeLevelBundle g h ∘ z)) =
      Vector.risk p y (Vector.bayesPredictor p y z) :=
  vector_recoverable_bayes_risk_eq p y z (threeLevelBundle g h)
    Prod.fst (fun _ => rfl)

/-- The vector readout may ignore upper-level features and reproduce every
low-level prediction exactly. -/
theorem vector_three_level_lift_risk_eq
    (p : FiniteLaw Ω) (y : Ω → D → ℝ) (z : Ω → I) (g : I → J) (h : J → K)
    (lower : I → D → ℝ) :
    Vector.risk p y ((lower ∘ Prod.fst) ∘ (threeLevelBundle g h ∘ z)) =
      Vector.risk p y (lower ∘ z) := rfl

/-- Every vector combined predictor pulls back to the low-level features. -/
theorem vector_three_level_pullback_risk_eq
    (p : FiniteLaw Ω) (y : Ω → D → ℝ) (z : Ω → I) (g : I → J) (h : J → K)
    (combined : I × J × K → D → ℝ) :
    Vector.risk p y (combined ∘ (threeLevelBundle g h ∘ z)) =
      Vector.risk p y ((combined ∘ threeLevelBundle g h) ∘ z) := rfl

/-- Under retained features, the vector prediction risk difference between
two chosen readouts is exactly their excess-risk difference. -/
theorem vector_three_level_combined_gain_loss_identity
    [Fintype I] [Fintype J] [Fintype K]
    (p : FiniteLaw Ω) (y : Ω → D → ℝ) (z : Ω → I) (g : I → J) (h : J → K)
    (lower : I → D → ℝ) (combined : I × J × K → D → ℝ) :
    Vector.risk p y (combined ∘ (threeLevelBundle g h ∘ z)) -
        Vector.risk p y (lower ∘ z) =
      Vector.excessRisk p y (threeLevelBundle g h ∘ z) combined -
        Vector.excessRisk p y z lower := by
  rw [Vector.excessRisk_eq_risk_sub, Vector.excessRisk_eq_risk_sub,
    vector_three_level_bundle_bayes_risk_eq]
  ring

/-- The entire achievable-risk set is also preserved for vector targets and
unrestricted readouts, with no assumption that a learned head finds an optimum. -/
theorem vector_three_level_achievable_risk_iff
    (p : FiniteLaw Ω) (y : Ω → D → ℝ) (z : Ω → I) (g : I → J) (h : J → K)
    (r : ℝ) :
    (∃ combined : I × J × K → D → ℝ,
      Vector.risk p y (combined ∘ (threeLevelBundle g h ∘ z)) = r) ↔
    (∃ lower : I → D → ℝ, Vector.risk p y (lower ∘ z) = r) := by
  constructor
  · rintro ⟨combined, hr⟩
    exact ⟨combined ∘ threeLevelBundle g h, hr⟩
  · rintro ⟨lower, hr⟩
    exact ⟨lower ∘ Prod.fst, hr⟩

end HJepa.Readout
