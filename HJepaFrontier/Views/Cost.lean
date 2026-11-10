import HJepaFrontier.Views.Persistence
import Mathlib.Tactic.LinearCombination

/-!
# Queue entry V6: a redrawn factor cannot be read out afterwards

Unreviewed Frontier candidate; see `FUZZ_QUEUE.md`.

Good views remove a factor from the representation by redrawing it. If a
downstream task needs that factor, the representation is useless for it: every
linear probe has risk at least one, the risk of predicting zero. In audio terms,
redrawing the room and source position between views removes spatial cues.

The claim is about linear probes of the scalar encoder at a global optimum.
-/

namespace HJepa.Frontier.Views

/-- Risk of the linear probe `k * z` for the nuisance sign, mirroring
`MaskedPrediction.downstreamRisk`, which targets the signal. -/
noncomputable def nuisanceProbeRisk (a b k : ℝ) : ℝ :=
  Abstraction.risk MaskedPrediction.fairInput MaskedPrediction.nuisanceSign
    (fun ω => k * MaskedPrediction.encoded a b ω)

/-- Exact expectation over the four equally likely contexts. -/
theorem nuisanceProbeRisk_exact (a b k : ℝ) :
    nuisanceProbeRisk a b k = (1 - k * b) ^ 2 + (k * a) ^ 2 := by
  norm_num [nuisanceProbeRisk, Abstraction.risk, MaskedPrediction.fairInput,
    MaskedPrediction.encoded, MaskedPrediction.encoder, MaskedPrediction.mask,
    MaskedPrediction.rawExample, MaskedPrediction.taskSign,
    MaskedPrediction.nuisanceSign, Abstraction.signedBit,
    Fintype.sum_prod_type, Fintype.sum_bool]
  ring

theorem nuisanceProbeRisk_complete_square (a b k : ℝ)
    (h : MaskedPrediction.Normalized a b) :
    nuisanceProbeRisk a b k = (k - b) ^ 2 + a ^ 2 := by
  have hnorm : a ^ 2 + b ^ 2 = 1 := h
  rw [nuisanceProbeRisk_exact]
  linear_combination (k ^ 2 - 1) * hnorm

/-- Predicting zero has risk one, so a lower bound of one means no probe helps. -/
theorem nuisanceProbeRisk_zero_probe (a b : ℝ) : nuisanceProbeRisk a b 0 = 1 := by
  rw [nuisanceProbeRisk_exact]
  ring

/-- A signal-only encoder carries nothing a linear probe can use for the nuisance. -/
theorem nuisanceProbeRisk_of_task_zero (m : TemporalPrediction.Learner)
    (hzero : TemporalPrediction.taskRisk m = 0) (k : ℝ) :
    1 ≤ nuisanceProbeRisk m.signal m.nuisance k := by
  have hnuisance : m.nuisance ^ 2 = 0 :=
    (TemporalPrediction.taskRisk_eq m).symm.trans hzero
  have hnorm : m.signal ^ 2 + m.nuisance ^ 2 = 1 := m.normalized
  rw [nuisanceProbeRisk_complete_square m.signal m.nuisance k m.normalized]
  linarith [sq_nonneg (k - m.nuisance)]

/-- Under good views every global optimum is blind, for linear probes, to the
factor that was redrawn between the views. -/
theorem redrawn_factor_unreadable (d : TemporalPrediction.Dynamics)
    (hnuisance : d.nuisancePersistence = 0) (hsignal : 0 < d.signalPersistence)
    (m : TemporalPrediction.Learner) (hm : TemporalPrediction.GlobalMinimum d m)
    (k : ℝ) : 1 ≤ nuisanceProbeRisk m.signal m.nuisance k :=
  nuisanceProbeRisk_of_task_zero m
    (zero_nuisance_persistence_task_zero d hnuisance hsignal m hm) k

end HJepa.Frontier.Views
