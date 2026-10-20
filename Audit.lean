import HJepa
import Lean.Util.CollectAxioms
import Lean.Elab.Command

/-!
# Trust audit

Check every declaration in the `HJepa` namespace, including definitions used by
the theorems. Only Lean's standard classical axioms are permitted transitively.
This rejects unfinished proofs and additional axioms even when hidden behind
helper declarations. The audit does not certify the mathematical relevance of
a theorem's hypotheses; that requires reading its statement.
-/

open Lean Elab Command in
elab "#audit_hjepa" : command => do
  let env ← getEnv
  let allowed : Array Name := #[``propext, ``Classical.choice, ``Quot.sound]
  let names := env.constants.toList.filterMap fun (name, _) =>
    if (`HJepa).isPrefixOf (Lean.privateToUserName name) then some name else none
  let names := names.toArray.qsort fun a b => a.toString < b.toString
  if names.isEmpty then
    throwError "No HJepa declarations found: the audit must not pass vacuously."
  let mut theoremCount := 0
  for name in names do
    let axioms ← Lean.collectAxioms name
    for ax in axioms do
      unless allowed.contains ax do
        throwError "Unapproved axiom in {name}: {ax}"
    match env.find? name with
    | some (.thmInfo _) =>
      theoremCount := theoremCount + 1
      logInfo m!"{name}: {axioms}"
    | _ => pure ()
  logInfo m!"PASS: audited {names.size} declarations, including {theoremCount} theorems."
  logInfo "Allowed axioms only: propext, Classical.choice, Quot.sound."

#audit_hjepa
