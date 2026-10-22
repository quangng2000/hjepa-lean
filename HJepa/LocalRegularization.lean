import Mathlib.Algebra.BigOperators.Field
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring

/-!
# Centered local regularization: a conditional, modality-independent property

LocaLeJEPA (arXiv:2610.28458), Eq. 7, subtracts each cluster's mean before
distribution matching. For a fixed nonempty finite cluster, translating every
point by the same vector preserves all residuals and hence any loss of them.

This is exact real arithmetic. It does not assert invariance of recomputed
cosine k-means assignments, preservation of task information, non-collapse,
optimizer convergence, floating-point equality, or superiority on audio.
-/

namespace HJepa.LocalRegularization

open scoped BigOperators

variable {ι κ : Type*}

/-- Coordinatewise mean of a finite cluster. -/
noncomputable def clusterMean (s : Finset ι) (z : ι → κ → ℝ) (j : κ) : ℝ :=
  (∑ i ∈ s, z i j) / (s.card : ℝ)

/-- Residuals only for points belonging to the fixed cluster. -/
noncomputable def centered (s : Finset ι) (z : ι → κ → ℝ)
    (i : {i // i ∈ s}) (j : κ) : ℝ :=
  z i j - clusterMean s z j

theorem clusterMean_translate (s : Finset ι) (hs : s.Nonempty)
    (z : ι → κ → ℝ) (a : κ → ℝ) (j : κ) :
    clusterMean s (fun i j => z i j + a j) j = clusterMean s z j + a j := by
  have hn : (s.card : ℝ) ≠ 0 := by exact_mod_cast hs.card_ne_zero
  simp only [clusterMean, Finset.sum_add_distrib, Finset.sum_const, nsmul_eq_mul]
  field_simp [hn]

theorem centered_translate (s : Finset ι) (hs : s.Nonempty)
    (z : ι → κ → ℝ) (a : κ → ℝ) :
    centered s (fun i j => z i j + a j) = centered s z := by
  funext i j
  simp only [centered, clusterMean_translate s hs]
  ring

/-- Any fixed-target statistic of centered residuals inherits the invariance. -/
theorem centered_loss_translate (s : Finset ι) (hs : s.Nonempty)
    (z : ι → κ → ℝ) (a : κ → ℝ)
    (loss : ({i // i ∈ s} → κ → ℝ) → ℝ) :
    loss (centered s (fun i j => z i j + a j)) = loss (centered s z) := by
  rw [centered_translate s hs]

/-- Each fixed cluster may be translated independently before its loss is summed.
The supplied cluster data and weights are fixed; no reclustering is performed. -/
theorem sum_centered_losses_translate {q : Type*} (clusters : Finset q)
    (s : q → Finset ι) (hs : ∀ c ∈ clusters, (s c).Nonempty)
    (z : q → ι → κ → ℝ) (a : q → κ → ℝ) (weight : q → ℝ)
    (loss : (c : q) → ({i // i ∈ s c} → κ → ℝ) → ℝ) :
    (∑ c ∈ clusters, weight c * loss c (centered (s c) (fun i j => z c i j + a c j))) =
      ∑ c ∈ clusters, weight c * loss c (centered (s c) (z c)) := by
  apply Finset.sum_congr rfl
  intro c hc
  rw [centered_translate (s c) (hs c hc)]

end HJepa.LocalRegularization
