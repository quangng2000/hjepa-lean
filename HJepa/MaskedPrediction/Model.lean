import HJepa.Objective

/-! A fixed mask exposes two independent signs and hides two scaled copies.
The encoder has one real coordinate; both objectives use two linear heads.
The hidden target map is fixed, so no target collapse or training dynamics is assumed. -/

namespace HJepa.MaskedPrediction

open HJepa.Abstraction
open scoped BigOperators

abbrev Input := HJepa.Objective.Input
abbrev Example := (ℝ × ℝ) × (ℝ × ℝ)

/-- The two context bits are independent and uniform. -/
noncomputable def fairInput : FiniteLaw Input where
  weight := fun _ => 1 / 4
  nonneg := by intro _; norm_num
  total_weight := by norm_num [Fintype.sum_prod_type, Fintype.sum_bool]

def taskSign (ω : Input) : ℝ := signedBit ω.1
def nuisanceSign (ω : Input) : ℝ := signedBit ω.2

/-- Full example `(S,N,s*S,n*N)`, grouped into visible and hidden pairs. -/
noncomputable def rawExample (s n : ℝ) (ω : Input) : Example :=
  ((taskSign ω, nuisanceSign ω), (s * taskSign ω, n * nuisanceSign ω))

def mask (x : Example) : ℝ × ℝ := x.1
def rawTarget (x : Example) : ℝ × ℝ := x.2

/-- A frozen diagonal hidden-layer map applied only to the masked target. -/
noncomputable def frozenTarget (c d : ℝ) (x : Example) : ℝ × ℝ :=
  (c * (rawTarget x).1, d * (rawTarget x).2)

noncomputable def encoder (a b : ℝ) (visible : ℝ × ℝ) : ℝ :=
  a * visible.1 + b * visible.2

noncomputable def encoded (a b : ℝ) (ω : Input) : ℝ :=
  encoder a b (mask (rawExample 1 1 ω))

noncomputable def decoder (p q : ℝ) (z : ℝ) : ℝ × ℝ := (p * z, q * z)

noncomputable def pairError (target prediction : ℝ × ℝ) : ℝ :=
  (target.1 - prediction.1) ^ 2 + (target.2 - prediction.2) ^ 2

/-- Population reconstruction loss; the encoder receives only the masked view. -/
noncomputable def loss (u v a b p q : ℝ) : ℝ :=
  ∑ ω, (fairInput.weight ω) * pairError (rawTarget (rawExample u v ω))
    (decoder p q (encoder a b (mask (rawExample u v ω))))

noncomputable def rawLoss (s n a b p q : ℝ) : ℝ := loss s n a b p q

/-- Same observation and predictor class, with a frozen hidden-layer target. -/
noncomputable def hiddenLoss (s n c d a b p q : ℝ) : ℝ :=
  ∑ ω, (fairInput.weight ω) * pairError (frozenTarget c d (rawExample s n ω))
    (decoder p q (encoder a b (mask (rawExample s n ω))))

def Normalized (a b : ℝ) : Prop := a ^ 2 + b ^ 2 = 1

noncomputable def reducedLoss (u v a b : ℝ) : ℝ := u ^ 2 * b ^ 2 + v ^ 2 * a ^ 2

/-- Linear-probe risk only: a generic scalar mixture can be injective on these
four atoms, so this is not a claim about unrestricted Bayes risk. -/
noncomputable def downstreamRisk (a b k : ℝ) : ℝ :=
  risk fairInput taskSign (fun ω => k * encoded a b ω)

theorem mask_independent_of_target_scales (s n u v : ℝ) (ω : Input) :
    mask (rawExample s n ω) = mask (rawExample u v ω) := rfl

theorem rawLoss_eq_loss (s n a b p q : ℝ) :
    rawLoss s n a b p q = loss s n a b p q := rfl

theorem hiddenLoss_eq_loss (s n c d a b p q : ℝ) :
    hiddenLoss s n c d a b p q = loss (c * s) (d * n) a b p q := by
  unfold hiddenLoss loss
  apply Finset.sum_congr rfl
  intro ω _
  simp only [pairError, frozenTarget, rawTarget, rawExample, decoder, encoder, mask]
  ring

end HJepa.MaskedPrediction
