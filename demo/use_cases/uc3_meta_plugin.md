# Use Case 3: Plugin Self-Improvement (Meta-Test)

**Task:** AttractorFlow monitors its OWN improvement.

The classifier currently has 6 regimes. Add a 7th: **PLATEAU**

A PLATEAU is when:
- λ ≈ 0 (neutral, not strictly negative)
- Distance series is decreasing (converging direction)
- BUT velocity is very low (almost no movement)
- Agent is making tiny refinements but not truly converging

This is different from STUCK (v≈0 with no trend) and CONVERGING (clear negative λ).
PLATEAU = slow drift toward solution, needs a "nudge" not a full perturbation.

## Files to modify
- `attractorflow/mcp-server/classifier.py` — add Regime.PLATEAU + _PlateauStrategy
- `attractorflow/mcp-server/server.py` — no changes needed (regime auto-included)

## Why this is the meta-test
The agent making this change IS being monitored by AttractorFlow.
Each code edit is recorded as a state. The monitoring should show:
- EXPLORING phase while reading/understanding the classifier
- CONVERGING phase as the implementation crystallizes
- CHECKPOINT saved when tests pass

This proves the harness works on the very tool building it.
