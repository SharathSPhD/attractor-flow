---
name: attractorflow
description: >
  Apply dynamical systems theory to Claude Code agentic orchestration using the
  AttractorFlow MCP server. Use this skill whenever you are: (1) orchestrating
  multi-step or multi-agent Claude Code tasks, (2) noticing an agent looping,
  drifting, oscillating, or making no progress, (3) deciding whether to decompose
  a complex task into parallel subagents, (4) wanting real-time convergence
  diagnostics (Lyapunov exponent, regime classification), or (5) designing
  attractor-engineered agent pipelines. Trigger on: "agent is stuck", "is the
  agent making progress?", "why is it looping", "should I split this task",
  "attractor status", "Lyapunov", "phase portrait", "/attractor-status",
  "agent drift", "convergence check".
---

# AttractorFlow Skill

You have access to the **AttractorFlow MCP server** (`attractorflow_mcp`), which
exposes 8 tools for treating agent task execution as **dynamical systems** rather
than reward optimization. The core insight: stop asking "did the agent converge to
the right answer?" and start asking "is the agent's *trajectory through solution
space* exhibiting healthy dynamics?"

---

## Setup

**First time in a project:**
```bash
python3.12 -m venv .venv
.venv/bin/pip install -r attractorflow/mcp-server/requirements.txt
```

The `.mcp.json` at project root registers the server automatically. Restart Claude
Code after first-time setup to activate MCP tool access.

**MCP tool calling — IMPORTANT:**
AttractorFlow tools (`attractorflow_record_state`, `attractorflow_get_regime`, etc.)
are called as **direct Claude tool calls**, not via Bash. They appear in your tool
list when the MCP server is connected. Do NOT use Bash to invoke Python scripts for
AttractorFlow monitoring — call the tools directly.

**Subagent Bash permission:**
Subagents launched via the `Agent` tool need Bash access if they also write code.
Ensure your session settings allow Bash, or use `/attractor-status` from the main
Claude Code session where MCP is always connected.

---

## Core Concepts

**Lyapunov Exponent (λ)** — a single number characterising trajectory health:
- λ < -0.2 → Strongly converging (approaching fixed-point attractor)
- λ ∈ [-0.2, -0.05] → Gently converging (healthy iteration)
- λ ∈ [-0.05, 0.05] → Neutral (limit cycle, plateau, or transition)
- λ ∈ [0.05, 0.25] → Expanding (exploring; OK in design phase only)
- λ > 0.25 **or** drift_diverging signal → Diverging (intervene now)

**Eight Regimes** — the classifier maps trajectories to actionable states:

| Regime | What it means | λ signal | Action |
|--------|---------------|----------|--------|
| CONVERGING | Consistent progress toward goal | < -0.05 | `REDUCE_TEMPERATURE` |
| CYCLING | Healthy dev loop (code→test→fix) | ≈ 0, autocorr | `CONTINUE` (if amp ↓) |
| EXPLORING | Searching solution space | 0.05–0.25 | OK in design; pressure in impl |
| DIVERGING | Drifting from goal | trend > 0 + mean_d > 1 | `RESTORE_CHECKPOINT` |
| STUCK | Zero progress, fixed-point trap | v ≈ 0, no trend | `SPAWN_EXPLORER` |
| OSCILLATING | Alternating between 2 states | lag-1 autocorr < -0.4 | `BREAK_SYMMETRY` |
| PLATEAU | Slow micro-refinement drift | v ≈ 0, trend < -0.01 | `NUDGE` |
| UNKNOWN | Insufficient data | < 3 steps | Wait for more data |

> **PLATEAU vs STUCK:** both have low velocity. Distinguisher: PLATEAU has a
> negative distance trend (drifting toward goal in tiny steps). STUCK has no trend.
> PLATEAU needs a small nudge; STUCK needs a full perturbation or explorer.

**Nine Orchestration Actions:**

| Action | Trigger regime | What to do |
|--------|---------------|------------|
| `CONTINUE` | CYCLING (healthy) | Let agent proceed unchanged |
| `REDUCE_TEMPERATURE` | CONVERGING | Add to prompt: "Commit to current approach. No alternatives." |
| `INJECT_PERTURBATION` | STUCK | Call `attractorflow_inject_perturbation`, prepend hint to context |
| `SPAWN_EXPLORER` | STUCK (deep) | Delegate to `explorer-agent` subagent |
| `BREAK_SYMMETRY` | OSCILLATING | Add asymmetric constraint: write a test only one option passes |
| `RESTORE_CHECKPOINT` | DIVERGING | Re-anchor to last checkpoint; re-read original goal |
| `DECOMPOSE_TASK` | BIFURCATION | Spawn two subagents, one per discovered cluster |
| `NUDGE` | PLATEAU | Add one small specific constraint to unstick the micro-drift |
| `HALT` | Any (confidence < 20%) | Stop and ask human for guidance |

**Bifurcation types:**
- PITCHFORK: task splits into 2 symmetric subtasks → spawn 2 parallel agents
- HOPF: direct process becomes iterative → add explicit exit criterion
- SADDLE_NODE: approach collapses → restart from checkpoint with new strategy

---

## Workflow Pattern

```
task start
  → attractorflow_record_state(state_text="<first step>", goal_text="<task goal>")
  → [agent does work step by step]
  → after every step: attractorflow_record_state("<what agent just did>")
  → every 3–5 steps: attractorflow_get_regime() → act on action field
  → every 10 steps: attractorflow_detect_bifurcation() → decompose if needed
  → on clean deliverable: attractorflow_checkpoint()
task end
```

---

## Intervention Recipes

**STUCK → escape sequence:**
```
# Level 1 — add a constraint
attractorflow_inject_perturbation(magnitude=0.3)
# Level 2 — switch tools or approach
attractorflow_inject_perturbation(magnitude=0.6)
# Level 3 — full restart
attractorflow_inject_perturbation(magnitude=1.0)
# OR spawn explorer
use explorer-agent subagent
```

**OSCILLATING → break symmetry:**
```
attractorflow_inject_perturbation(magnitude=0.5)
# Write a specific test that only one option can pass before choosing
```

**DIVERGING → re-anchor:**
```
attractorflow_checkpoint()   # save current position first
# Re-read original goal
# Narrow scope to single most important deliverable
```

**PLATEAU → nudge:**
```
# Add one small specific constraint: a test, a type annotation, a scope limit
# Do NOT inject full perturbation — that overshoots
```

**CYCLING (stable amplitude) → unstick:**
```
attractorflow_inject_perturbation(magnitude=0.4)
# Add a specific new test as deadline
```

**PITCHFORK bifurcation → decompose:**
```
# Spawn subagent A for cluster 1
# Spawn subagent B for cluster 2
# Orchestrator evaluates both, continues with convergent path
```

---

## Response Format for `/attractor-status`

```
## 🧭 AttractorFlow Status

**Regime:** [REGIME] (confidence: [X]%)
**λ (Lyapunov):** [value] ([converging/neutral/diverging])
**Steps tracked:** [n]

**Diagnosis:** [rationale from classifier]

**Recommended action:** [action]
> [intervention_hint — full text from MCP response]
```

---

## When NOT to use AttractorFlow

- Simple one-shot tasks (< 3 steps, no iteration)
- Tasks where the agent is visibly making progress and you have clear success criteria
- Purely informational queries (no code or multi-step work involved)

---

## Reference Files

- `references/dynamical_systems_primer.md` — Formal definitions of attractors, Lyapunov exponents, bifurcations
