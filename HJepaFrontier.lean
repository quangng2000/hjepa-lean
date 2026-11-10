import HJepaFrontier.Views.Persistence
import HJepaFrontier.Views.Redraw
import HJepaFrontier.Views.Cost
import HJepaFrontier.Horizon.Margin
import HJepaFrontier.JointObjective.Design

/-!
# Frontier: machine-checked candidates awaiting statement review

Modules under `HJepaFrontier/` hold candidate theorems from the proof-fuzzing
queue in `FUZZ_QUEUE.md`. Lean checks every proof here, and `Audit.lean` applies
the same axiom audit as it does to the reviewed library.

What has not happened yet is a person confirming that each statement and its
hypotheses say what the queue entry claims. Do not cite a Frontier result in
`README.md` or `THEORY.md` until it has been reviewed and moved into `HJepa/`.

Conventions, enforced by `scripts/check_frontier.sh`:

* every module declares its results in the namespace `HJepa.Frontier`, because
  the audit selects declarations by the `HJepa` prefix;
* every module is imported from this file, because Lean checks only what is built;
* Frontier may import the reviewed library, never the reverse.
-/
