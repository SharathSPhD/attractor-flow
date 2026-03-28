# /phase-portrait

Generate a 2D ASCII phase portrait of the agent's trajectory through
embedding space, plus distance series and basin depth estimate.

## Usage

```
/phase-portrait [n_steps=20]
```

## Instructions

When this command is invoked:

1. Call `attractorflow_get_trajectory(n_steps=20)` (or n_steps from arg)
2. Call `attractorflow_detect_bifurcation()` for basin clustering info
3. Render an ASCII phase portrait using the pca_2d data
4. Render a distance-over-time sparkline

## ASCII Phase Portrait Renderer

Scale the pca_2d points to fit in a 40×20 character grid.
Mark trajectory steps with numbers (1..N) or dots.
Mark goal position with ★ if goal_distances are available.
Mark cluster centroids with ◈ if bifurcation detected.

Example output format:
```
## 📐 Phase Portrait (last {n} steps)

     ┌────────────────────────────────────────┐
  +1 │                    3                   │
     │              2         4               │
     │          1                 5           │
   0 │        ★                     6         │
     │                              7·8       │
     │                                9       │
  -1 │                                 10     │
     └────────────────────────────────────────┘
       -1                               +1
                  PCA Component 1

Distance series (step → step):
  ▁▂▄▆▄▃▂▁▁▂  mean={mean_d:.3f}  trend={trend:+.3f}

{bifurcation_detected ?
  "⚡ PITCHFORK detected — two basins emerging (◈ markers)\n" +
  "   Basin A centroid: ({cx1:.2f}, {cy1:.2f})\n" +
  "   Basin B centroid: ({cx2:.2f}, {cy2:.2f})"
  : "No bifurcation detected."
}
```

## Sparkline Helper

For the distance series, map values to block characters:
- 0.00–0.05 → ▁
- 0.05–0.10 → ▂
- 0.10–0.15 → ▃
- 0.15–0.20 → ▄
- 0.20–0.30 → ▅
- 0.30–0.40 → ▆
- 0.40–0.60 → ▇
- > 0.60    → █
