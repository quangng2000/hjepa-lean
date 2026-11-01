import HJepa.TokenAggregation.Examples.Stable
import HJepa.TokenAggregation.Examples.Alternating
import HJepa.TokenAggregation.Examples.Skipped

namespace HJepa.TokenAggregation.Examples

/-- Each operator has an attained winner against every readout of the other. -/
theorem neither_aggregation_dominates :
    (∃ tokens averageReadout, ∀ selectedReadout,
      averagedRisk tokens averageReadout < selectedRisk tokens selectedReadout) ∧
    (∃ tokens selectedReadout, ∀ averageReadout,
      selectedRisk tokens selectedReadout < averagedRisk tokens averageReadout) := by
  exact ⟨⟨stablePair, id, stable_average_strictly_better⟩,
    ⟨alternatingPair, id, alternating_selected_strictly_better⟩⟩

end HJepa.TokenAggregation.Examples
