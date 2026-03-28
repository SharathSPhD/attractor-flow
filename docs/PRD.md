# AttractorFlow — Product Requirements Document

**Version:** 0.1
**Date:** 2026-03-26
**Status:** Draft
**Author:** Attractor Project

---

## 1. Problem Statement

Agentic AI systems — Claude Code agents in particular — routinely exhibit pathological behaviors whose root cause is never diagnosed: they loop indefinitely on repetition, drift from their original goal, oscillate between two approaches without converging, or get stuck at local attractors and spin in place. Engineers respond to each of these with ad hoc fixes: "try again," "increase temperature," "reduce context." None of these are principled. There is no real-time diagnostic that tells an operator *what kind of stuck* an agent is, let alone *why*, or *what intervention the physics of the system demands*.

Recent research (Tacheny et al. 2025, Wang et al. 2025, Concept Attractors 2025, DMET 2025) has established that LLM inference and agentic loops are formally describable as dynamical systems with measurable attractors, Lyapunov exponents, and bifurcation points. Yet no engineering product has closed the loop — applying these observations to *design* orchestration systems with formal stability guarantees.

**AttractorFlow** fills that gap. It is a Claude Code plugin that treats task execution as attractor dynamics rather than reward optimization, providing real-time diagnosis and principled intervention for agent behavior.

---

## 2. Goals

### Primary
- Provide a **single interpretable real-time metric** (finite-time Lyapunov exponent λ) that characterizes agent health at each step
- **Classify agent state** into six dynamical regimes (CONVERGING, CYCLING, EXPLORING, DIVERGING, STUCK, OSCILLATING) with actionable recommendations for each
- **Detect bifurcation points** — complexity thresholds where single-basin approaches fail and task decomposition is required — using measurable signals rather than orchestrator intuition

### Secondary
- Visualize agent trajectories as phase portraits and energy landscapes for debugging
- Enable **attractor-engineered orchestration** where desired solution behaviors are the deepest basins and failure modes are structurally unstable
- Provide an **escape-injection mechanism** that perturbs agent context when trapped in pathological attractors

### Non-Goals (v0.1)
- No new model training — purely harness-level innovation on top of existing Claude models
- Not a general-purpose monitoring dashboard for LLM APIs
- Not a replacement for existing evaluation frameworks (e.g., evals, test suites)
- No support for non-Claude-Code agent frameworks in v0.1

---

## 3. Users and Use Cases

### Primary User: Claude Code Power User / Agentic Developer
Someone building multi-agent pipelines in Claude Code who currently has no diagnostic visibility into agent convergence. They need to know when to intervene and what kind of intervention is appropriate.

**Key pain points:**
- "My agent has been running for 20 minutes. Is it making progress or spinning in a loop?"
- "The orchestrator spawned 5 subagents. Two seem to be doing redundant work. Why?"
- "The agent keeps switching between two approaches. How do I break this cycle?"

### Secondary User: AI Systems Researcher
Building and evaluating novel agent architectures. Needs reproducible metrics for agent trajectory quality that go beyond task-completion rates.

---

## 4. Success Metrics

| Metric | Target |
|--------|--------|
| Regime classification accuracy vs. manual annotation | ≥ 80% on held-out agent traces |
| Time-to-detect DIVERGING regime before human notices | < 3 agent steps |
| False positive rate for STUCK classification | < 10% |
| Lyapunov computation latency (per step) | < 200ms |
| MCP server cold-start time | < 5s |
| Successful bifurcation detection (complex task) | Triggers decomposition 70% of the time when human would do same |

---

## 5. Feature Specification

### 5.1 Phase Space Monitor (Core)

**What it does:** At each agent step, receives a text state, embeds it into a dense vector, and appends to a rolling trajectory buffer.

**Inputs:**
- Agent output text (or a summarized version) at each step
- Optional: goal embedding to anchor the trajectory

**Outputs:**
- Trajectory buffer (last N=50 embeddings by default)
- Successive distance series d_i = ||e_i - e_{i-1}||

**Embedding model:** `all-MiniLM-L6-v2` (sentence-transformers) — 384 dims, ~22MB, runs on CPU in < 50ms per call.

### 5.2 Lyapunov Exponent Estimator

Computes finite-time Lyapunov exponent (FTLE) as:

```
λ(t) = (1 / window) × Σ ln(d_{i+1} / d_i)   for i in sliding window
```

**Regime thresholds:**
- λ < -0.2 → CONVERGING
- -0.2 ≤ λ < -0.05 → gentle convergence (healthy iteration)
- -0.05 ≤ λ ≤ 0.05 → neutral / CYCLING candidate
- 0.05 < λ ≤ 0.25 → EXPLORING
- λ > 0.25 → DIVERGING

### 5.3 Attractor Classifier

Six regimes, each with a prescribed orchestration response:

| Regime | Detection Criteria | Recommended Action |
|--------|-------------------|--------------------|
| CONVERGING | λ < -0.1, monotone distance decrease | Continue, reduce exploration temperature |
| CYCLING | Autocorrelation peak at lag k > 1, bounded variance | If amplitude decreasing: healthy, let run. If stable: inject perturbation |
| EXPLORING | λ ∈ (0, 0.25), bounded trajectory | OK in design phase; apply convergence pressure in impl phase |
| DIVERGING | λ > 0.25 for 3+ consecutive steps | Halt, re-anchor to goal embedding, restore checkpoint |
| STUCK | Distance < 0.02 for 3+ consecutive steps | Increase temperature, try alternative approach, spawn explorer |
| OSCILLATING | 2-period cycle in autocorrelation, peak at lag=2 | Break symmetry: add asymmetric constraint or inject external info |

### 5.4 Bifurcation Detector

Monitors task complexity indicators:
- **Sensitivity increase:** variance of output embeddings under small prompt perturbations → approaching bifurcation
- **Bimodal response distribution:** clustering of trajectory into 2 distinct regions → pitchfork bifurcation
- **Oscillation onset:** previously CONVERGING → now CYCLING → Hopf bifurcation
- **Output:** `BifurcationType` enum (NONE, PITCHFORK, HOPF, SADDLE_NODE) + proximity score 0–1

### 5.5 MCP Server Interface (8 tools)

Exposed via Claude Code's MCP protocol (stdio transport):

1. `attractorflow_record_state(state_text)` — Record a new agent step
2. `attractorflow_get_regime()` → regime + confidence + recommended action
3. `attractorflow_get_lyapunov()` → current FTLE + trend + window stats
4. `attractorflow_get_trajectory(n_steps)` → PCA-reduced 2D trajectory points
5. `attractorflow_get_basin_depth()` → variance-based depth estimate
6. `attractorflow_detect_bifurcation()` → type + proximity + trigger recommendation
7. `attractorflow_inject_perturbation(magnitude)` → returns prompt fragment to inject
8. `attractorflow_checkpoint()` → saves current state as stable reference

### 5.6 Attractor-Engineered Orchestrator (Claude Code Agents)

Three subagent personas triggered by regime:
- **attractor-orchestrator** — meta-agent that reads regime at each decision point
- **explorer-agent** — strange-attractor mode: higher temperature, broader tool access, no convergence pressure
- **convergence-agent** — fixed-point mode: lower temperature, tight scope, test-gating

### 5.7 Slash Commands

- `/attractor-status` — Human-readable regime summary + λ + recommended action
- `/phase-portrait` — ASCII or SVG phase portrait of recent trajectory

---

## 6. Technical Architecture (Summary)

```
Claude Code Agent Loop
        │
        ▼  (at each step)
attractorflow_record_state(text)
        │
        ▼
[Phase Space Monitor]
  sentence-transformers
  all-MiniLM-L6-v2
  trajectory buffer (FIFO)
        │
        ├──→ [Lyapunov Estimator] ──→ λ(t)
        │
        └──→ [Attractor Classifier]
                    │
                    ├──→ regime + confidence
                    └──→ prescribed action
                              │
                    [Bifurcation Detector]
                              │
                    orchestrator decision:
                    continue / perturb / decompose / halt
```

Transport: **stdio** (local, no auth, zero network overhead)
Language: **Python 3.11+**, FastMCP
Dependencies: sentence-transformers, numpy, scipy, scikit-learn, mcp

---

## 7. Phased Rollout

### Phase 1 — Core Diagnostics (v0.1, now)
- MCP server with 8 tools
- Regime classifier (all 6 regimes)
- Lyapunov estimator
- `/attractor-status` command
- Basic bifurcation detection

### Phase 2 — Orchestration Integration (v0.2)
- Attractor-orchestrator agent
- Explorer + convergence subagents
- Automatic perturbation injection
- Checkpoint/restore
- `/phase-portrait` command with SVG output

### Phase 3 — Energy Landscape (v0.3)
- Energy landscape visualization
- Separatrix detection (decision boundaries between approaches)
- Multi-agent trajectory coordination
- Simulation harness for offline evaluation of orchestration strategies

---

## 8. Open Questions

1. **Ground truth for regime classification:** How do we build a labeled dataset of agent trajectories with human-annotated regime labels?
2. **Embedding model selection:** MiniLM-L6-v2 is fast but domain-general. Would a code-specific embedding (e.g., CodeBERT) improve precision for coding agent trajectories?
3. **Window size tuning:** The FTLE window W (default 8) is a free parameter. What's the right tradeoff between responsiveness and noise?
4. **Perturbation quality:** `inject_perturbation` currently returns a generic prompt fragment. Domain-specific perturbations (e.g., "add a unit test", "refactor to functional style") would be more actionable.
5. **Multi-agent coordination:** How should bifurcation-spawned subagents share state with the Phase Space Monitor?
