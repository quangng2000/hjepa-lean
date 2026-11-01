import HJepa.Spoofing.Filters.Causal

namespace HJepa.Spoofing

def pairProjection (a b : ℝ) (tokens : ℝ × ℝ) : ℝ :=
  a * tokens.1 + b * tokens.2

/-- Every scalar linear reduction of an unrestricted pair has a collision. -/
theorem pair_projection_collision (a b : ℝ) :
    ∃ x y : ℝ × ℝ, x ≠ y ∧ pairProjection a b x = pairProjection a b y := by
  by_cases ha : a = 0
  · refine ⟨(1, 0), (-1, 0), ?_, ?_⟩
    · intro h
      have hf := congrArg Prod.fst h
      norm_num at hf
    · simp [pairProjection, ha]
  · refine ⟨(b, -a), (-b, a), ?_, ?_⟩
    · intro h
      have hs := congrArg Prod.snd h
      apply ha
      dsimp at hs
      linarith
    · simp only [pairProjection]
      ring

/-- A subsequent encoder, even nonlinear, cannot separate an existing collision. -/
theorem scalar_projection_then_encoder_not_injective {V : Type*}
    (encode : ℝ → V) (a b : ℝ) :
    ¬ Function.Injective (fun tokens => encode (pairProjection a b tokens)) := by
  intro h
  obtain ⟨x, y, hxy, he⟩ := pair_projection_collision a b
  exact hxy (h (congrArg encode he))

end HJepa.Spoofing
