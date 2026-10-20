import HJepa.Abstraction

/-!
# Preserving every task requires recovering the retained features

The task-specific protection objective can preserve a declared target without
preserving every possible future target. This module characterizes that stronger
requirement on a finite population: every deterministic scalar task of the old
features is exactly readable from the compressed features if and only if no two
distinct old features on positive-probability outcomes are merged. Equivalently,
there is a decoder recovering the old features on the probability support.

This is a population and unrestricted-readout statement. It neither says that
finite training data certify recovery nor that a neural decoder learns it. The
old features themselves may already have discarded information from raw audio.
-/

namespace HJepa.UniversalPreservation

open HJepa.Abstraction
open scoped BigOperators

variable {Ω I J : Type*} [Fintype Ω]

/-- Compression merges no distinct realized features of positive probability. -/
def SupportInjective (p : FiniteLaw Ω) (z : Ω → I) (compress : I → J) : Prop :=
  ∀ a b, 0 < p.weight a → 0 < p.weight b →
    compress (z a) = compress (z b) → z a = z b

/-- The original features can be decoded on the probability support. Null
outcomes and feature labels not realized by the population need not recover. -/
def SupportRecovery (p : FiniteLaw Ω) (z : Ω → I)
    (compress : I → J) (recover : J → I) : Prop :=
  ∀ ω, 0 < p.weight ω → recover (compress (z ω)) = z ω

/-- Every deterministic real-valued task of the original features has a
zero-risk readout on the compressed representation. -/
def PreservesEveryTask (p : FiniteLaw Ω) (z : Ω → I) (compress : I → J) : Prop :=
  ∀ task : I → ℝ, ∃ readout : J → ℝ,
    risk p (task ∘ z) (readout ∘ (compress ∘ z)) = 0

/-- A finite probability law has at least one positive-probability outcome. -/
theorem exists_positive_weight (p : FiniteLaw Ω) :
    ∃ ω, 0 < p.weight ω := by
  classical
  have hsum : 0 < ∑ ω, p.weight ω := by rw [p.total_weight]; norm_num
  obtain ⟨ω, _, hω⟩ :=
    (Finset.sum_pos_iff_of_nonneg (fun ω _ => p.nonneg ω)).mp hsum
  exact ⟨ω, hω⟩

/-- Recovery preserves the risk of every original-feature predictor, including
its error on an arbitrary noisy target; it does not assert that this error is zero. -/
theorem recovery_preserves_readout_risk (p : FiniteLaw Ω) (z : Ω → I)
    (compress : I → J) (recover : J → I)
    (hrecover : SupportRecovery p z compress recover)
    (target : Ω → ℝ) (readout : I → ℝ) :
    risk p target ((readout ∘ recover) ∘ (compress ∘ z)) =
      risk p target (readout ∘ z) := by
  apply Finset.sum_congr rfl
  intro ω _
  by_cases hw : p.weight ω = 0
  · simp [hw]
  · have hp : 0 < p.weight ω := lt_of_le_of_ne (p.nonneg ω) (Ne.symm hw)
    simp only [Function.comp_apply, hrecover ω hp]

/-- Recovering the old feature is sufficient to preserve all its deterministic tasks. -/
theorem recovery_preserves_every_task (p : FiniteLaw Ω) (z : Ω → I)
    (compress : I → J) (recover : J → I)
    (hrecover : SupportRecovery p z compress recover) :
    PreservesEveryTask p z compress := by
  intro task
  refine ⟨task ∘ recover, (risk_eq_zero_iff _ _ _).mpr ?_⟩
  intro ω hω
  simp only [Function.comp_apply, hrecover ω hω]

/-- All-task preservation rules out a collision: the binary indicator of either
colliding feature would otherwise require two different predictions from one code. -/
theorem every_task_implies_support_injective (p : FiniteLaw Ω) (z : Ω → I)
    (compress : I → J) (hpreserve : PreservesEveryTask p z compress) :
    SupportInjective p z compress := by
  classical
  intro a b ha hb hcode
  by_contra hne
  let task : I → ℝ := fun i => if i = z a then 0 else 1
  obtain ⟨readout, hrisk⟩ := hpreserve task
  have hzero := (risk_eq_zero_iff _ _ _).mp hrisk
  have hfirst := hzero a ha
  have hsecond := hzero b hb
  simp [task] at hfirst
  have hba : z b ≠ z a := Ne.symm hne
  simp only [Function.comp_apply, task, if_neg hba] at hsecond
  rw [hcode] at hfirst
  linarith

/-- Support-injectivity constructs a decoder. Values off the realized support
are filled with one arbitrary realized feature and have no effect on risk. -/
theorem support_injective_has_recovery (p : FiniteLaw Ω) (z : Ω → I)
    (compress : I → J) (hinj : SupportInjective p z compress) :
    ∃ recover : J → I, SupportRecovery p z compress recover := by
  classical
  obtain ⟨defaultOutcome, _⟩ := exists_positive_weight p
  let recover : J → I := fun j =>
    if h : ∃ ω, 0 < p.weight ω ∧ compress (z ω) = j then
      z (Classical.choose h)
    else z defaultOutcome
  refine ⟨recover, ?_⟩
  intro ω hω
  have hex : ∃ a, 0 < p.weight a ∧ compress (z a) = compress (z ω) :=
    ⟨ω, hω, rfl⟩
  simp only [recover, dif_pos hex]
  exact hinj _ ω (Classical.choose_spec hex).1 hω (Classical.choose_spec hex).2

/-- A left inverse on support prohibits merging distinct positive-mass features. -/
theorem recovery_implies_support_injective (p : FiniteLaw Ω) (z : Ω → I)
    (compress : I → J) (recover : J → I)
    (hrecover : SupportRecovery p z compress recover) :
    SupportInjective p z compress := by
  intro a b ha hb hcode
  calc
    z a = recover (compress (z a)) := (hrecover a ha).symm
    _ = recover (compress (z b)) := congrArg recover hcode
    _ = z b := hrecover b hb

/-- Exact preservation of every possible deterministic task is equivalent to
injectivity on realized positive-probability features, not merely a variance constraint. -/
theorem preserves_every_task_iff_support_injective
    (p : FiniteLaw Ω) (z : Ω → I) (compress : I → J) :
    PreservesEveryTask p z compress ↔ SupportInjective p z compress := by
  constructor
  · exact every_task_implies_support_injective p z compress
  · intro hinj
    obtain ⟨recover, hrecover⟩ := support_injective_has_recovery p z compress hinj
    exact recovery_preserves_every_task p z compress recover hrecover

/-- All-task preservation is exactly support recoverability of the old features. -/
theorem preserves_every_task_iff_recoverable
    (p : FiniteLaw Ω) (z : Ω → I) (compress : I → J) :
    PreservesEveryTask p z compress ↔
      ∃ recover : J → I, SupportRecovery p z compress recover := by
  constructor
  · intro hp
    exact support_injective_has_recovery p z compress
      (every_task_implies_support_injective p z compress hp)
  · rintro ⟨recover, hrecover⟩
    exact recovery_preserves_every_task p z compress recover hrecover

/-- If two positive-mass outcomes with different binary task labels collide,
every readout pays at least the optimal two-point weighted squared-error cost.
Other outcomes can only increase the error. -/
theorem collision_binary_risk_lower_bound
    (p : FiniteLaw Ω) (z : Ω → I) (compress : I → J)
    (task : I → ℝ) (a b : Ω)
    (ha : 0 < p.weight a) (hb : 0 < p.weight b)
    (hfirst : task (z a) = 0) (hsecond : task (z b) = 1)
    (hcode : compress (z a) = compress (z b))
    (readout : J → ℝ) :
    (p.weight a * p.weight b) / (p.weight a + p.weight b) ≤
      risk p (task ∘ z) (readout ∘ (compress ∘ z)) := by
  classical
  have hab : a ≠ b := by
    intro hab
    subst b
    linarith
  have hpair :
      p.weight a * (0 - readout (compress (z a))) ^ 2 +
      p.weight b * (1 - readout (compress (z a))) ^ 2 ≤
        risk p (task ∘ z) (readout ∘ (compress ∘ z)) := by
    have hsum := Finset.sum_le_sum_of_subset_of_nonneg
      (s := ({a, b} : Finset Ω)) (t := Finset.univ)
      (f := fun ω => p.weight ω *
        ((task ∘ z) ω - (readout ∘ (compress ∘ z)) ω) ^ 2)
      (Finset.subset_univ _) (fun ω _ _ =>
        mul_nonneg (p.nonneg ω) (sq_nonneg _))
    simpa [risk, hab, Function.comp_apply, hfirst, hsecond, ← hcode] using hsum
  have htotal : 0 < p.weight a + p.weight b := add_pos ha hb
  apply (div_le_iff₀ htotal).mpr
  have hmul := mul_le_mul_of_nonneg_left hpair (le_of_lt htotal)
  have hsquare := sq_nonneg
    ((p.weight a + p.weight b) * readout (compress (z a)) - p.weight b)
  nlinarith

/-- A genuinely lossy compression has an explicit 0/1 task for which every
readout has strictly positive population error. This task is relative to the
old features; it need not be one of the semantic tasks chosen in advance. -/
theorem collision_has_unrecoverable_binary_task
    (p : FiniteLaw Ω) (z : Ω → I) (compress : I → J)
    (a b : Ω) (ha : 0 < p.weight a) (hb : 0 < p.weight b)
    (hne : z a ≠ z b) (hcode : compress (z a) = compress (z b)) :
    ∃ task : I → ℝ, (∀ i, task i = 0 ∨ task i = 1) ∧
      ∀ readout : J → ℝ,
        0 < risk p (task ∘ z) (readout ∘ (compress ∘ z)) := by
  classical
  let task : I → ℝ := fun i => if i = z a then 0 else 1
  refine ⟨task, ?_, ?_⟩
  · intro i
    by_cases hi : i = z a
    · exact Or.inl (by simp [task, hi])
    · exact Or.inr (by simp [task, hi])
  · intro readout
    have hbound := collision_binary_risk_lower_bound p z compress task a b ha hb
      (by simp [task]) (by simp [task, Ne.symm hne]) hcode readout
    exact lt_of_lt_of_le (div_pos (mul_pos ha hb) (add_pos ha hb)) hbound

end HJepa.UniversalPreservation
