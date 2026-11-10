import HJepa.Abstraction
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Tactic.NormNum

namespace HJepa.Spoofing

open HJepa.Abstraction
open scoped BigOperators

/-- Actual finite-population misclassification probability, not a squared surrogate. -/
noncomputable def classificationRisk {Ω Z : Type*} [Fintype Ω] (p : FiniteLaw Ω)
    (labels : Ω → Bool) (observation : Ω → Z) (classifier : Z → Bool) : ℝ :=
  ∑ ω, p.weight ω * if classifier (observation ω) = labels ω then 0 else 1

noncomputable def toyLaw : FiniteLaw Bool where
  weight := fun _ => 1 / 2
  nonneg := by intro _; norm_num
  total_weight := by norm_num [Fintype.sum_bool]

/-- The label itself is the equiprobable source variable in these examples. -/
noncomputable def toyError {Z : Type*} (observation : Bool → Z)
    (classifier : Z → Bool) : ℝ := classificationRisk toyLaw id observation classifier

theorem classificationRisk_zero_of_correct {Ω Z : Type*} [Fintype Ω]
    (p : FiniteLaw Ω) (labels : Ω → Bool) (observation : Ω → Z)
    (classifier : Z → Bool) (h : ∀ ω, classifier (observation ω) = labels ω) :
    classificationRisk p labels observation classifier = 0 := by
  simp [classificationRisk, h]

theorem toyError_constant {Z : Type*} (z : Z) (classifier : Z → Bool) :
    toyError (fun _ => z) classifier = 1 / 2 := by
  cases hc : classifier z <;>
    norm_num [toyError, classificationRisk, toyLaw, Fintype.sum_bool, hc]

/-- A collision of opposite equally probable labels leaves chance error. -/
theorem toyError_collapsed {Z : Type*} (observation : Bool → Z)
    (classifier : Z → Bool) (h : observation false = observation true) :
    toyError observation classifier = 1 / 2 := by
  have he : observation = fun _ => observation false := by
    funext label
    cases label
    · rfl
    · exact h.symm
  rw [he]
  exact toyError_constant _ _

/-- No deterministic next encoder can distinguish labels already merged. -/
theorem toyError_postencoder_collapsed {Z W : Type*} (observation : Bool → Z)
    (h : observation false = observation true) (encoder : Z → W)
    (classifier : W → Bool) :
    toyError (fun label => encoder (observation label)) classifier = 1 / 2 := by
  apply toyError_collapsed
  rw [h]

end HJepa.Spoofing
