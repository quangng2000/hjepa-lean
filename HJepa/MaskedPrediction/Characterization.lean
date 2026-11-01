import HJepa.MaskedPrediction.Optima

namespace HJepa.MaskedPrediction

/-- The two decoder heads solve least squares for this normalized encoder. -/
def HeadsFit (u v : ℝ) (m : Predictor) : Prop :=
  m.firstHead = u * m.signal ∧ m.secondHead = v * m.nuisance

/-- A joint global optimum must also optimize its heads with its encoder fixed. -/
theorem global_heads_fit (u v : ℝ) (m : Predictor) (hm : GlobalMinimum u v m) :
    HeadsFit u v m := by
  have hc := hm ⟨m.signal, m.nuisance, u * m.signal, v * m.nuisance, m.normalized⟩
  change loss u v m.signal m.nuisance m.firstHead m.secondHead ≤
    loss u v m.signal m.nuisance (u * m.signal) (v * m.nuisance) at hc
  rw [loss_optimal_heads _ _ _ _ m.normalized] at hc
  have he := le_antisymm hc
    (reducedLoss_le_loss u v m.signal m.nuisance m.firstHead m.secondHead m.normalized)
  exact (loss_eq_reducedLoss_iff _ _ _ _ _ _ m.normalized).mp he

/-- In the strict signal regime, this characterizes every joint global optimum. -/
theorem signal_global_iff (u v : ℝ) (h : v ^ 2 < u ^ 2) (m : Predictor) :
    GlobalMinimum u v m ↔ taskRisk m = 0 ∧ HeadsFit u v m := by
  constructor
  · intro hm
    exact ⟨signal_global_task_zero u v h m hm, global_heads_fit u v m hm⟩
  · rintro ⟨hr, hp, hq⟩
    apply (signal_global_value u v (le_of_lt h) m).mpr
    rw [objective_signal_form, hr, hp, hq]
    ring

/-- In the strict nuisance regime, every joint optimum has linear-probe risk one. -/
theorem nuisance_global_iff (u v : ℝ) (h : u ^ 2 < v ^ 2) (m : Predictor) :
    GlobalMinimum u v m ↔ taskRisk m = 1 ∧ HeadsFit u v m := by
  constructor
  · intro hm
    exact ⟨nuisance_global_task_one u v h m hm, global_heads_fit u v m hm⟩
  · rintro ⟨hr, hp, hq⟩
    apply (nuisance_global_value u v (le_of_lt h) m).mpr
    rw [objective_nuisance_form, hr, hp, hq]
    ring

/-- Equal target energies leave the encoder unrestricted by global optimality. -/
theorem equal_energy_global_iff (u v : ℝ) (h : u ^ 2 = v ^ 2) (m : Predictor) :
    GlobalMinimum u v m ↔ HeadsFit u v m := by
  constructor
  · exact global_heads_fit u v m
  · rintro ⟨hp, hq⟩
    apply (signal_global_value u v (le_of_eq h.symm) m).mpr
    rw [objective_signal_form, h, hp, hq]
    ring

end HJepa.MaskedPrediction
