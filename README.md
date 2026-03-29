# AttractorFlow

A Claude Code plugin that monitors and steers multi-agent trajectories using
dynamical systems theory. Agent states are embedded in semantic space;
finite-time Lyapunov exponents (FTLE) classify the trajectory regime and
prescribe interventions before a task gets stuck, oscillates, or diverges.

## How it works

Every agent step is embedded into a 384-dimensional vector using
`all-MiniLM-L6-v2`. The MCP server tracks a FIFO buffer of embeddings and
computes:

- **FTLE** — `λ = (1/W) × Σ ln(d_{i+1} / d_i)` over a sliding window
- **Autocorrelation** at lags 1–10 for cycle detection
- **Basin depth** from distance variance (stability estimate)
- **Bifurcation proximity** via k-means cluster separation

These signals drive regime classification and route the orchestrator to the
right intervention.

## Regimes

| Regime | λ signal | Action |
|--------|---------|--------|
| CONVERGING | < −0.05 | Continue, add convergence pressure |
| CYCLING | ≈ 0, autocorr peak | Continue if amplitude ↓, perturb if stable |
| EXPLORING | 0.05–0.25 | OK in design phase, pressure in impl phase |
| DIVERGING | trend > 0, mean_d > 1 | Restore checkpoint immediately |
| STUCK | v ≈ 0, no trend | Inject perturbation, spawn explorer |
| OSCILLATING | ≈ 0, lag-1 autocorr < −0.4 | Break symmetry with asymmetric constraint |
| PLATEAU | v ≈ 0, trend < −0.01 | Nudge (small specific constraint) |

## Setup

**Step 1** — Install [uv](https://docs.astral.sh/uv/) (one-time, ~5 MB binary):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Step 2** — Register the MCP server globally (one-time):

```bash
claude mcp add --scope user attractorflow_mcp -- sh -c \
  'PLUGIN_ROOT="${CLAUDE_PLUGIN_ROOT:-$HOME/.claude/plugins/cache/attractor-flow}" && cd "$PLUGIN_ROOT/attractorflow/mcp-server" && PATH="$PATH:$HOME/.local/bin" uv run server.py'
```

Restart Claude Code. Python dependencies (`sentence-transformers`, `scikit-learn`, etc.)
load automatically on the first `attractorflow_record_state` call — no venv or pip needed.

### Session scoping (optional)

For multi-project or benchmark use, set env vars in your shell or in a
project `.mcp.json` `env` block:

| Env var | Default | Effect |
|---------|---------|--------|
| `ATTRACTORFLOW_SESSION_ID` | _(none)_ | Named session → `~/.attractorflow/sessions/<id>.json` |
| `ATTRACTORFLOW_DISABLE_PERSISTENCE` | `0` | `1` = no disk I/O (benchmark / CI mode) |
| `ATTRACTORFLOW_BUFFER_CAPACITY` | `100` | Override buffer size |
| `ATTRACTORFLOW_WINDOW` | `8` | Override FTLE window |

## MCP tools

All tools are direct Claude tool calls — not bash commands.

| Tool | When to call |
|------|-------------|
| `attractorflow_record_state` | After every agent step |
| `attractorflow_get_regime` | Every 3–5 steps |
| `attractorflow_get_lyapunov` | For detailed stability analysis |
| `attractorflow_get_trajectory` | For visualization |
| `attractorflow_get_basin_depth` | Before committing to an approach |
| `attractorflow_detect_bifurcation` | Every 10 steps or on complex tasks |
| `attractorflow_inject_perturbation` | When STUCK or OSCILLATING |
| `attractorflow_checkpoint` | After tests pass or clean deliverable |

## Agents

Three specialist agents are defined in `.claude/agents/`:

- **attractor-orchestrator** — meta-orchestrator; reads regime every 3–5 steps
  and routes to the right subagent
- **explorer-agent** — covers solution space when STUCK or PITCHFORK detected
- **convergence-agent** — drives chosen approach to completion; halts if λ rises

## Commands

- `/attractor-status` — current regime, λ, recommended action
- `/phase-portrait` — ASCII trajectory visualization

## Project structure

```
attractorflow/
  mcp-server/          # FastMCP server (8 tools)
    server.py          # tool definitions and lifespan
    phase_space.py     # embedding buffer, PCA, distance series
    lyapunov.py        # FTLE computation, autocorrelation
    classifier.py      # regime classification, action prescription
    bifurcation.py     # k-means bifurcation detection
    requirements.txt
  claude-plugin/       # Claude Code plugin integration
  skill/               # SKILL.md and reference materials
.claude/
  CLAUDE.md            # project context loaded into every session
  agents/              # attractor-orchestrator, explorer, convergence
  commands/            # /attractor-status, /phase-portrait
.mcp.json              # MCP server registration
docs/
  PRD.md
  ADR-001.md
research.md            # theoretical foundations
simulation/            # synthetic trajectory scenarios
demo/                  # proof-of-work demo runner and dashboard
```

## References

- Drela, M. (1989). XFOIL: An analysis and design system for low Reynolds
  number airfoils. *Lecture Notes in Engineering*, 54.
- Wang, X. et al. (2025). Stable attractors in multi-agent LLM trajectories.
- Lyapunov, A. M. (1892). The general problem of the stability of motion.

## License

MIT — see [LICENSE](LICENSE).
