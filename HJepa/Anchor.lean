import HJepa.Vector

/-!
# Frozen-reference protection without downstream labels

Reconstruct a fixed finite-dimensional reference from the protected channel.
Zero population reconstruction error preserves every readout of that reference,
even if the reference itself is an imperfect predictor. Approximate recovery
protects readouts only under an explicit sensitivity bound. The reference is
fixed in these statements: jointly collapsing a trainable reference and its
decoder would make this loss vacuous. Nothing here asserts that a reference
already contains all information in raw audio or that an optimizer converges.
-/

namespace HJepa.Anchor

open HJepa.Abstraction
open scoped BigOperators

variable {Ω D J : Type*} [Fintype Ω] [Fintype D]

/-- Squared Euclidean distance, expressed without additional norm imports. -/
def squaredDistance (a b : D → ℝ) : ℝ := ∑ d, (a d - b d) ^ 2

/-- The target `reference` is held fixed while `code` and `decode` are trained. -/
def reconstructionLoss (p : FiniteLaw Ω) (reference : Ω → D → ℝ)
    (code : Ω → J) (decode : J → D → ℝ) : ℝ :=
  Vector.risk p reference (decode ∘ code)

/-- Any nonnegative predictive cost can be combined with reference protection.
It may, for example, be a positive weighted sum over three predictive levels. -/
def objective (p : FiniteLaw Ω) (reference : Ω → D → ℝ)
    (code : Ω → J) (decode : J → D → ℝ) (predictiveCost weight : ℝ) : ℝ :=
  predictiveCost + weight * reconstructionLoss p reference code decode

/-- Zero reconstruction preserves each reference readout's actual population
risk, not just its ability to fit the reconstruction target. -/
theorem exact_reconstruction_preserves_readout
    (p : FiniteLaw Ω) (y : Ω → ℝ) (reference : Ω → D → ℝ)
    (code : Ω → J) (decode : J → D → ℝ)
    (hzero : reconstructionLoss p reference code decode = 0)
    (readout : (D → ℝ) → ℝ) :
    risk p y (readout ∘ decode ∘ code) = risk p y (readout ∘ reference) := by
  have hpoint := (Vector.risk_eq_zero_iff p reference (decode ∘ code)).mp hzero
  apply Finset.sum_congr rfl
  intro ω _
  by_cases hpos : 0 < p.weight ω
  · have heq : reference ω = decode (code ω) := funext (hpoint ω hpos)
    simp only [Function.comp_apply, heq]
  · have hw : p.weight ω = 0 := le_antisymm (le_of_not_gt hpos) (p.nonneg ω)
    simp [hw]

/-- A squared sensitivity condition. For nonnegative `L`, this is the usual
Euclidean L-Lipschitz condition squared. It is an assumption on the readout,
not a statement that every future task is robust to approximate recovery. -/
def SquaredLipschitz (readout : (D → ℝ) → ℝ) (L : ℝ) : Prop :=
  ∀ a b, (readout a - readout b) ^ 2 ≤ L ^ 2 * squaredDistance a b

/-- Reconstruction error controls the error of a sensitive reference readout. -/
theorem readout_distortion_le
    (p : FiniteLaw Ω) (reference : Ω → D → ℝ)
    (code : Ω → J) (decode : J → D → ℝ)
    (readout : (D → ℝ) → ℝ) (L : ℝ) (hL : SquaredLipschitz readout L) :
    risk p (readout ∘ reference) (readout ∘ decode ∘ code) ≤
      L ^ 2 * reconstructionLoss p reference code decode := by
  rw [reconstructionLoss, Vector.risk_eq_sum_squared_error, Finset.mul_sum]
  apply Finset.sum_le_sum
  intro ω _
  have h := mul_le_mul_of_nonneg_left
    (hL (reference ω) (decode (code ω))) (p.nonneg ω)
  simpa only [Function.comp_apply, squaredDistance, mul_left_comm] using h

/-- Squared loss obeys a two-term bound even when the reference readout has
nonzero task error. The factor two must not be dropped without more structure. -/
theorem risk_via_reference_le (p : FiniteLaw Ω) (y f g : Ω → ℝ) :
    risk p y g ≤ 2 * risk p y f + 2 * risk p f g := by
  unfold risk
  rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
  apply Finset.sum_le_sum
  intro ω _
  have hsq : (y ω - g ω) ^ 2 ≤
      2 * (y ω - f ω) ^ 2 + 2 * (f ω - g ω) ^ 2 := by
    nlinarith [sq_nonneg ((y ω - f ω) - (f ω - g ω))]
  have h := mul_le_mul_of_nonneg_left hsq (p.nonneg ω)
  nlinarith

/-- Protection transfers the fixed reference's quality, not unknown ground
truth semantics: its existing task error remains in the bound. -/
theorem task_risk_le_reference_and_reconstruction
    (p : FiniteLaw Ω) (y : Ω → ℝ) (reference : Ω → D → ℝ)
    (code : Ω → J) (decode : J → D → ℝ)
    (readout : (D → ℝ) → ℝ) (L : ℝ) (hL : SquaredLipschitz readout L) :
    risk p y (readout ∘ decode ∘ code) ≤
      2 * risk p y (readout ∘ reference) +
      2 * L ^ 2 * reconstructionLoss p reference code decode := by
  have h := risk_via_reference_le p y (readout ∘ reference)
    (readout ∘ decode ∘ code)
  have hd := readout_distortion_le p reference code decode readout L hL
  linarith

/-- A population objective certificate gives an explicit reconstruction budget.
Obtaining this certificate by finite-sample training is a separate obligation. -/
theorem reconstruction_le_of_objective_le
    (p : FiniteLaw Ω) (reference : Ω → D → ℝ)
    (code : Ω → J) (decode : J → D → ℝ)
    (predictiveCost weight η : ℝ) (hpred : 0 ≤ predictiveCost) (hw : 0 < weight)
    (hobjective : objective p reference code decode predictiveCost weight ≤ η) :
    reconstructionLoss p reference code decode ≤ η / weight := by
  apply (le_div_iff₀ hw).mpr
  unfold objective at hobjective
  nlinarith

/-- With positive protection weight, an exact zero population objective
preserves every reference readout. Attainability of zero is not assumed to
follow merely from having a reconstruction decoder. -/
theorem zero_objective_preserves_readout
    (p : FiniteLaw Ω) (y : Ω → ℝ) (reference : Ω → D → ℝ)
    (code : Ω → J) (decode : J → D → ℝ)
    (predictiveCost weight : ℝ) (hpred : 0 ≤ predictiveCost) (hw : 0 < weight)
    (hzero : objective p reference code decode predictiveCost weight = 0)
    (readout : (D → ℝ) → ℝ) :
    risk p y (readout ∘ decode ∘ code) = risk p y (readout ∘ reference) := by
  have hrec := reconstruction_le_of_objective_le
    p reference code decode predictiveCost weight 0 hpred hw (le_of_eq hzero)
  rw [zero_div] at hrec
  have hz : reconstructionLoss p reference code decode = 0 :=
    le_antisymm hrec (Vector.risk_nonneg _ _ _)
  exact exact_reconstruction_preserves_readout p y reference code decode hz readout

/-- Copying the entire reference is a feasible exact reconstruction, with no
claim of compression or simultaneous minimization of predictive cost. -/
theorem copy_reference_zero_loss (p : FiniteLaw Ω) (reference : Ω → D → ℝ) :
    reconstructionLoss p reference reference id = 0 := by
  simp [reconstructionLoss, Vector.risk, risk]

theorem copy_reference_objective_eq_predictive_cost
    (p : FiniteLaw Ω) (reference : Ω → D → ℝ) (predictiveCost weight : ℝ) :
    objective p reference reference id predictiveCost weight = predictiveCost := by
  simp [objective, copy_reference_zero_loss]

/-- Approximate label-free reference protection gives a downstream guarantee
for any readout obeying the sensitivity bound. No task labels enter the loss. -/
theorem objective_bounds_task_risk
    (p : FiniteLaw Ω) (y : Ω → ℝ) (reference : Ω → D → ℝ)
    (code : Ω → J) (decode : J → D → ℝ)
    (predictiveCost weight η : ℝ) (hpred : 0 ≤ predictiveCost) (hw : 0 < weight)
    (hobjective : objective p reference code decode predictiveCost weight ≤ η)
    (readout : (D → ℝ) → ℝ) (L : ℝ) (hL : SquaredLipschitz readout L) :
    risk p y (readout ∘ decode ∘ code) ≤
      2 * risk p y (readout ∘ reference) + 2 * L ^ 2 * (η / weight) := by
  have htask := task_risk_le_reference_and_reconstruction
    p y reference code decode readout L hL
  have hrec := reconstruction_le_of_objective_le
    p reference code decode predictiveCost weight η hpred hw hobjective
  have hmul := mul_le_mul_of_nonneg_left hrec (by positivity : 0 ≤ 2 * L ^ 2)
  linarith

/-- An unconstrained, jointly trainable reference can collapse to a constant
along with its decoder. Freezing or otherwise protecting the reference matters. -/
theorem constant_reference_zero_loss (p : FiniteLaw Ω)
    (code : Ω → J) (value : D → ℝ) :
    reconstructionLoss p (fun _ => value) code (fun _ => value) = 0 := by
  simp [reconstructionLoss, Vector.risk, risk]

end HJepa.Anchor
