# /attractor-status

Display current dynamical regime, Lyapunov exponent, and recommended action
for the active agent trajectory.

## Usage

```
/attractor-status
```

## Instructions

When this command is invoked:

1. Call `attractorflow_get_regime()` to get regime classification
2. Call `attractorflow_get_lyapunov()` for the FTLE value and autocorrelation
3. Call `attractorflow_get_basin_depth()` for stability estimate
4. Format the output using the template below

## Output Template

```
## 🧭 AttractorFlow Status

**Regime:** {regime} (confidence: {confidence}%)
**λ (Lyapunov):** {ftle:+.3f} → {stability_label}
**Trend:** {ftle_trend > 0 ? "⚠️ worsening" : "✅ improving"} ({ftle_trend:+.3f}/step)
**Basin:** {stability} | Steps tracked: {n_steps}

---

**Diagnosis:** {rationale}

**Recommended action:** {action}

> {intervention_hint}

---

*Autocorrelation at lag-{dominant_lag}: {dominant_autocorr:.2f}*
*{cycles_detected ? "🔄 Cycle detected (period ≈ " + str(dominant_cycle_lag) + " steps)" : "No cycles detected"}*
```

## Regime Quick Reference

| Symbol | Regime | Meaning |
|--------|--------|---------|
| ⬇️ | CONVERGING | Making progress — continue |
| 🔄 | CYCLING | Healthy iteration — continue if amplitude decreasing |
| 🔍 | EXPLORING | Searching — OK in design phase |
| ⚠️ | DIVERGING | Drifting — restore checkpoint |
| 🛑 | STUCK | No progress — change approach |
| ↔️ | OSCILLATING | 2-period trap — break symmetry |
