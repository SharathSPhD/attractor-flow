# AttractorFlow — Project Context

This project uses the **AttractorFlow MCP server** to monitor and steer
agent trajectories using dynamical systems theory.

## Setup

The AttractorFlow MCP server is registered in `.mcp.json` and starts
automatically when Claude Code is opened in this project.

**First-time setup:**
```bash
python3.12 -m venv .venv
.venv/bin/pip install -r attractorflow/mcp-server/requirements.txt
```

The server embeds agent states using `all-MiniLM-L6-v2` (~22MB, auto-downloads on first run).
Restart Claude Code after first-time setup so the MCP server connects.

## Core Commands

- `/attractor-status` — Current regime, λ, recommended action
- `/phase-portrait` — ASCII trajectory visualization

## Available Agents

- `attractor-orchestrator` — Meta-orchestrator with regime-aware routing
- `explorer-agent` — Strange-attractor exploration (for STUCK / PITCHFORK)
- `convergence-agent` — Fixed-point convergence (for implementation phases)

## MCP Tools (attractorflow_mcp)

Call these as **direct Claude tool calls** — NOT via Bash or Python scripts.
The tools are listed in each agent's frontmatter and available when MCP is connected.

| Tool | When to call |
|------|-------------|
| `attractorflow_record_state` | After every agent step |
| `attractorflow_get_regime` | Every 3–5 steps |
| `attractorflow_get_lyapunov` | For detailed stability analysis |
| `attractorflow_get_trajectory` | For visualization |
| `attractorflow_get_basin_depth` | When deciding whether to commit to approach |
| `attractorflow_detect_bifurcation` | Every 10 steps or on complex tasks |
| `attractorflow_inject_perturbation` | When regime is STUCK or OSCILLATING |
| `attractorflow_checkpoint` | After all tests pass or clean deliverable |

## Workflow Pattern

```
task start
  → set goal: attractorflow_record_state(..., goal_text="<task goal>")
  → [agent does work]
  → every step: attractorflow_record_state("<what agent just did>")
  → every 3-5 steps: attractorflow_get_regime() → act on action field
  → every 10 steps: attractorflow_detect_bifurcation() → decompose if needed
  → on good state: attractorflow_checkpoint()
task end
```

## Regime Decision Reference

| Regime | λ signal | Action |
|--------|---------|--------|
| CONVERGING | < -0.05 | Continue, add convergence pressure |
| CYCLING | ≈ 0, autocorr peak | Continue if amplitude ↓, perturb if stable |
| EXPLORING | 0.05–0.25 | OK design phase, not impl phase |
| DIVERGING | trend > 0 + mean_d > 1 | Restore checkpoint immediately |
| STUCK | v≈0, no trend | Inject perturbation, spawn explorer |
| OSCILLATING | ≈ 0, lag-1 negative autocorr | Break symmetry with asymmetric constraint |
| PLATEAU | v≈0, trend < -0.01 | Nudge (small specific constraint) |
