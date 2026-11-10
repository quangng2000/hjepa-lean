import HJepa.MaskedPrediction.Comparison

namespace HJepa.MaskedPrediction

/-- Excess population loss controls linear task risk even with imperfect heads. -/
theorem signal_tolerance (u v η : ℝ) (m : Predictor)
    (h : objective u v m ≤ v ^ 2 + η) :
    (u ^ 2 - v ^ 2) * taskRisk m ≤ η := by
  rw [objective_signal_form] at h
  nlinarith [sq_nonneg (m.firstHead - u * m.signal),
    sq_nonneg (m.secondHead - v * m.nuisance)]

theorem nuisance_tolerance (u v η : ℝ) (m : Predictor)
    (h : objective u v m ≤ u ^ 2 + η) :
    (v ^ 2 - u ^ 2) * (1 - taskRisk m) ≤ η := by
  rw [objective_nuisance_form] at h
  nlinarith [sq_nonneg (m.firstHead - u * m.signal),
    sq_nonneg (m.secondHead - v * m.nuisance)]

theorem signal_risk_le (u v η : ℝ) (m : Predictor)
    (gap : v ^ 2 < u ^ 2) (h : objective u v m ≤ v ^ 2 + η) :
    taskRisk m ≤ η / (u ^ 2 - v ^ 2) := by
  apply (le_div_iff₀ (sub_pos.mpr gap)).mpr
  simpa [mul_comm] using signal_tolerance u v η m h

theorem nuisance_risk_ge (u v η : ℝ) (m : Predictor)
    (gap : u ^ 2 < v ^ 2) (h : objective u v m ≤ u ^ 2 + η) :
    1 - η / (v ^ 2 - u ^ 2) ≤ taskRisk m := by
  have hb : 1 - taskRisk m ≤ η / (v ^ 2 - u ^ 2) := by
    apply (le_div_iff₀ (sub_pos.mpr gap)).mpr
    simpa [mul_comm] using nuisance_tolerance u v η m h
  linarith

/-- Both losses are near their own minima; the comparison is on the common task. -/
theorem approximate_hidden_better (t : Targets) (raw hidden : Predictor) (εr εh : ℝ)
    (gr : t.signal ^ 2 < t.nuisance ^ 2)
    (gh : t.hiddenNuisance ^ 2 < t.hiddenSignal ^ 2)
    (hr : objective t.signal t.nuisance raw ≤ t.signal ^ 2 + εr)
    (hh : objective t.hiddenSignal t.hiddenNuisance hidden ≤ t.hiddenNuisance ^ 2 + εh)
    (small : εh / (t.hiddenSignal ^ 2 - t.hiddenNuisance ^ 2) +
      εr / (t.nuisance ^ 2 - t.signal ^ 2) < 1) : taskRisk hidden < taskRisk raw := by
  have rh := signal_risk_le _ _ _ hidden gh hh
  have rr := nuisance_risk_ge _ _ _ raw gr hr
  linarith

theorem approximate_raw_better (t : Targets) (raw hidden : Predictor) (εr εh : ℝ)
    (gr : t.nuisance ^ 2 < t.signal ^ 2)
    (gh : t.hiddenSignal ^ 2 < t.hiddenNuisance ^ 2)
    (hr : objective t.signal t.nuisance raw ≤ t.nuisance ^ 2 + εr)
    (hh : objective t.hiddenSignal t.hiddenNuisance hidden ≤ t.hiddenSignal ^ 2 + εh)
    (small : εr / (t.signal ^ 2 - t.nuisance ^ 2) +
      εh / (t.hiddenNuisance ^ 2 - t.hiddenSignal ^ 2) < 1) : taskRisk raw < taskRisk hidden := by
  have rr := signal_risk_le _ _ _ raw gr hr
  have rh := nuisance_risk_ge _ _ _ hidden gh hh
  linarith

end HJepa.MaskedPrediction
