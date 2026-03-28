#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "mcp>=1.0.0",
#   "sentence-transformers>=3.0.0",
#   "numpy>=1.26.0",
#   "scipy>=1.12.0",
#   "scikit-learn>=1.4.0",
#   "pydantic>=2.6.0",
# ]
# ///
"""
AttractorFlow MCP Server

Exposes dynamical systems diagnostics for Claude Code agentic loops.
Treats agent task execution as attractor dynamics, providing real-time
Lyapunov exponent monitoring, regime classification, and bifurcation detection.

Run via: python server.py
Or register in .mcp.json:
  {
    "mcpServers": {
      "attractorflow": {
        "command": "python",
        "args": ["/path/to/attractorflow/mcp-server/server.py"]
      }
    }
  }

Research basis: Tacheny et al. 2025, Wang et al. 2025,
  DMET (arXiv:2505.20340), Concept Attractors (arXiv:2601.11575)
"""

from __future__ import annotations

import json
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ConfigDict
from mcp.server.fastmcp import FastMCP, Context

# Local modules
from phase_space import PhaseSpaceMonitor
from lyapunov import LyapunovEstimator
from classifier import AttractorClassifier, Regime
from bifurcation import BifurcationDetector


# ------------------------------------------------------------------
# Server singleton state (shared across all tool calls in session)
# ------------------------------------------------------------------

_monitor: Optional[PhaseSpaceMonitor] = None
_lyapunov: Optional[LyapunovEstimator] = None
_classifier: Optional[AttractorClassifier] = None
_bifurcation: Optional[BifurcationDetector] = None
_regime_history: List[Regime] = []
_ftle_history: List[float] = []


@asynccontextmanager
async def _lifespan(app):
    """Initialize the Phase Space Monitor and embedding model at startup."""
    global _monitor, _lyapunov, _classifier, _bifurcation
    _monitor = PhaseSpaceMonitor(capacity=100)
    _lyapunov = LyapunovEstimator(window=8)
    _classifier = AttractorClassifier()
    _bifurcation = BifurcationDetector()
    _monitor.load_model()
    restored = _monitor.load()   # restore previous session if within 24h
    if restored:
        import sys
        print(f"[attractorflow] Restored session: {_monitor.buffer_size} steps from disk.", file=sys.stderr)
    yield
    # Cleanup on shutdown — final save already handled per-record
    pass


mcp = FastMCP("attractorflow_mcp", lifespan=_lifespan)


# ------------------------------------------------------------------
# Pydantic Input Models
# ------------------------------------------------------------------

class RecordStateInput(BaseModel):
    """Input for recording a new agent state."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    state_text: str = Field(
        ...,
        description=(
            "The agent's current output, plan, code, or a summary of its most recent step. "
            "This text is embedded and added to the trajectory. "
            "Example: 'I wrote a Python function that parses JSON and returns a list of dicts. "
            "Tests are passing for the happy path but failing on empty input.'"
        ),
        min_length=1,
        max_length=4000,
    )
    goal_text: Optional[str] = Field(
        default=None,
        description=(
            "Optional: the task goal as a reference anchor. "
            "If provided and goal is not already set, this becomes the trajectory reference point. "
            "Only needs to be provided once (first call or when goal changes)."
        ),
        max_length=2000,
    )


class GetTrajectoryInput(BaseModel):
    """Input for retrieving trajectory data."""
    model_config = ConfigDict(extra="forbid")

    n_steps: int = Field(
        default=20,
        description="Number of recent trajectory steps to return (1–100).",
        ge=1,
        le=100,
    )


class InjectPerturbationInput(BaseModel):
    """Input for generating a context perturbation."""
    model_config = ConfigDict(extra="forbid")

    magnitude: float = Field(
        default=0.5,
        description=(
            "Perturbation magnitude from 0.0 (subtle reframe) to 1.0 (radical approach change). "
            "0.3 = add a new constraint; 0.6 = change primary tool or algorithm; "
            "1.0 = start from scratch with a different paradigm."
        ),
        ge=0.0,
        le=1.0,
    )


class SetGoalInput(BaseModel):
    """Input for setting the goal anchor."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    goal_text: str = Field(
        ...,
        description="The task goal or intended outcome, used as the reference attractor.",
        min_length=1,
        max_length=2000,
    )


# ------------------------------------------------------------------
# Tool 1: Record State
# ------------------------------------------------------------------

@mcp.tool(
    name="attractorflow_record_state",
    annotations={
        "title": "Record Agent State",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    },
)
async def attractorflow_record_state(params: RecordStateInput, ctx: Context) -> str:
    """
    Record a new agent state and update the trajectory.

    Call this at each agent step with the agent's current output or a
    summary of what it just did. The text is embedded into a 384-dim
    vector and appended to the FIFO trajectory buffer.

    Call this BEFORE querying regime or Lyapunov — the diagnostics
    reflect the trajectory up to and including the most recent record.

    Args:
        params (RecordStateInput):
            - state_text (str): Agent's current output or summary (max 4000 chars)
            - goal_text (Optional[str]): Task goal for reference anchor (first call only)

    Returns:
        str: JSON with step index, buffer size, and confirmation.

    Schema:
        {
          "step_index": int,
          "buffer_size": int,
          "goal_set": bool,
          "message": str
        }
    """
    global _monitor, _regime_history, _ftle_history

    # Optionally set goal on first call
    if params.goal_text and not _monitor.has_goal:
        _monitor.set_goal(params.goal_text)
        await ctx.log("info", f"Goal anchor set: {params.goal_text[:100]}")

    record = _monitor.record(params.state_text)
    await ctx.log("info", f"State recorded step={record.step_index} buffer={_monitor.buffer_size}")

    # Incrementally update FTLE history for bifurcation detector
    distances = _monitor.get_distance_series()
    if len(distances) >= 2:
        embeddings = _monitor.get_embeddings_matrix()
        result = _lyapunov.compute(distances, embeddings_matrix=embeddings)
        _ftle_history.append(result.ftle)

    return json.dumps({
        "step_index": record.step_index,
        "buffer_size": _monitor.buffer_size,
        "goal_set": _monitor.has_goal,
        "message": f"Recorded step {record.step_index}. Buffer: {_monitor.buffer_size}/100.",
    }, indent=2)


# ------------------------------------------------------------------
# Tool 2: Get Regime
# ------------------------------------------------------------------

@mcp.tool(
    name="attractorflow_get_regime",
    annotations={
        "title": "Get Current Dynamical Regime",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def attractorflow_get_regime(ctx: Context) -> str:
    """
    Classify the current agent trajectory into one of six dynamical regimes.

    Returns the regime, confidence, recommended action, and a concrete
    prompt fragment (intervention_hint) the orchestrator can use.

    Six regimes:
      CONVERGING  — approaching solution; apply convergence pressure
      CYCLING     — healthy iteration (if amplitude decreasing) or cycle trap
      EXPLORING   — bounded divergence; OK in design phase
      DIVERGING   — drifting away; restore checkpoint
      STUCK       — near-zero velocity; spawn explorer
      OSCILLATING — 2-period Wang attractor; break symmetry

    Returns:
        str: JSON with full classification result.

    Schema:
        {
          "regime": str,
          "confidence": float,       # 0–1
          "action": str,             # OrchestratorAction enum value
          "intervention_hint": str,  # paste into agent context to intervene
          "rationale": str,
          "lambda": float,
          "n_steps": int,
          "cycles_detected": bool,
          "dominant_cycle_lag": int
        }
    """
    global _monitor, _lyapunov, _classifier, _regime_history

    distances = _monitor.get_distance_series()
    stats = _monitor.get_stats()
    embeddings = _monitor.get_embeddings_matrix()
    lya = _lyapunov.compute(distances, embeddings_matrix=embeddings)
    result = _classifier.classify(lya, stats)

    # Track regime history for bifurcation detector
    _regime_history.append(result.regime)
    if len(_regime_history) > 50:
        _regime_history = _regime_history[-50:]

    return json.dumps({
        "regime": result.regime.value,
        "confidence": round(result.confidence, 3),
        "action": result.action.value,
        "intervention_hint": result.intervention_hint,
        "rationale": result.rationale,
        "lambda": round(result.λ, 4),
        "n_steps": result.n_steps,
        "cycles_detected": result.cycles_detected,
        "dominant_cycle_lag": result.dominant_cycle_lag,
    }, indent=2)


# ------------------------------------------------------------------
# Tool 3: Get Lyapunov Exponent
# ------------------------------------------------------------------

@mcp.tool(
    name="attractorflow_get_lyapunov",
    annotations={
        "title": "Get Finite-Time Lyapunov Exponent",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def attractorflow_get_lyapunov(ctx: Context) -> str:
    """
    Compute the current finite-time Lyapunov exponent (FTLE) for the trajectory.

    FTLE interpretation:
      λ < -0.2  → CONVERGING strongly (trajectory contracting fast)
      -0.2–-0.05 → CONVERGING weakly (healthy iteration)
      -0.05–0.05 → NEUTRAL (limit cycle candidate)
      0.05–0.25  → EXPLORING (mild expansion)
      λ > 0.25   → DIVERGING (intervention needed)

    Also returns autocorrelation data for cycle detection and a trend
    indicating whether λ is improving or worsening over recent steps.

    Returns:
        str: JSON with full Lyapunov analysis.

    Schema:
        {
          "ftle": float,
          "ftle_trend": float,         # slope: positive = worsening
          "window_size": int,
          "n_valid_steps": int,
          "is_stuck": bool,
          "autocorrelation": [float],  # lags 1..10
          "dominant_lag": int,
          "dominant_autocorr": float,
          "stability_label": str,
          "message": str
        }
    """
    distances = _monitor.get_distance_series()
    embeddings = _monitor.get_embeddings_matrix()
    result = _lyapunov.compute(distances, embeddings_matrix=embeddings)

    return json.dumps({
        "ftle": round(result.ftle, 4),
        "step_growth_rate": round(result.step_growth_rate, 4),
        "isotropy_ratio": round(result.isotropy_ratio, 4),
        "singular_values": [round(s, 4) for s in result.singular_values],
        "ftle_trend": round(result.ftle_trend, 4),
        "window_size": result.window_size,
        "n_valid_steps": result.n_valid_steps,
        "is_stuck": result.is_stuck,
        "autocorrelation": [round(v, 3) for v in result.autocorrelation],
        "dominant_lag": result.dominant_lag,
        "dominant_autocorr": round(result.dominant_autocorr, 3),
        "stability_label": result.stability_label,
        "message": result.message,
    }, indent=2)


# ------------------------------------------------------------------
# Tool 4: Get Trajectory
# ------------------------------------------------------------------

@mcp.tool(
    name="attractorflow_get_trajectory",
    annotations={
        "title": "Get Agent Trajectory (PCA 2D)",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def attractorflow_get_trajectory(params: GetTrajectoryInput, ctx: Context) -> str:
    """
    Return the recent agent trajectory projected to 2D via PCA.

    Useful for visualizing attractor geometry. Each point represents
    one recorded agent state in a 2D embedding space.

    Args:
        params (GetTrajectoryInput):
            - n_steps (int): Number of recent steps to include (default 20, max 100)

    Returns:
        str: JSON with trajectory points and distance series.

    Schema:
        {
          "n_steps": int,
          "pca_2d": [[float, float]],  # [x, y] for each step
          "distances": [float],        # ||e_i - e_{i-1}||
          "goal_distances": [float],   # ||e_i - goal|| (empty if no goal set)
          "mean_distance": float,
          "distance_trend": float      # positive = diverging
        }
    """
    stats = _monitor.get_stats()
    pca_2d = stats.pca_2d[-params.n_steps:]
    distances = stats.distances[-(params.n_steps - 1):] if stats.distances else []
    goal_distances = stats.goal_distances[-params.n_steps:] if stats.goal_distances else []

    return json.dumps({
        "n_steps": len(pca_2d),
        "pca_2d": [[round(x, 4), round(y, 4)] for x, y in pca_2d],
        "distances": [round(d, 4) for d in distances],
        "goal_distances": [round(d, 4) for d in goal_distances],
        "mean_distance": round(stats.mean_distance, 4),
        "distance_trend": round(stats.distance_trend, 4),
    }, indent=2)


# ------------------------------------------------------------------
# Tool 5: Get Basin Depth
# ------------------------------------------------------------------

@mcp.tool(
    name="attractorflow_get_basin_depth",
    annotations={
        "title": "Estimate Attractor Basin Depth",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def attractorflow_get_basin_depth(ctx: Context) -> str:
    """
    Estimate the depth of the current attractor basin.

    Basin depth is estimated from the variance of recent distances:
    low variance + decreasing distances = deep, stable basin.
    High variance + large distances = shallow or unstable basin.

    A deeper basin means the current approach is more robust to perturbations.
    A shallow basin means small changes in context could flip the approach.

    Returns:
        str: JSON with basin depth estimate and interpretation.

    Schema:
        {
          "basin_depth": float,      # 0.0 (very shallow) to 1.0 (very deep)
          "stability": str,          # "deep" | "moderate" | "shallow" | "unstable"
          "variance_of_distances": float,
          "interpretation": str
        }
    """
    stats = _monitor.get_stats()

    if not stats.distances:
        return json.dumps({
            "basin_depth": 0.0,
            "stability": "unknown",
            "variance_of_distances": 0.0,
            "interpretation": "Insufficient data.",
        }, indent=2)

    import numpy as np
    d = np.array(stats.distances)
    variance = float(d.var())
    mean = stats.mean_distance
    trend = stats.distance_trend

    # Depth score: inversely related to variance and mean distance
    # Bonus for negative trend (converging)
    raw_depth = 1.0 / (1.0 + variance * 10 + mean * 2)
    if trend < 0:
        raw_depth = min(1.0, raw_depth * 1.3)  # converging bonus
    elif trend > 0.01:
        raw_depth = max(0.0, raw_depth * 0.7)  # diverging penalty

    basin_depth = round(raw_depth, 3)

    if basin_depth > 0.7:
        stability = "deep"
        interpretation = "Strong basin — robust to perturbations. Current approach is well-established."
    elif basin_depth > 0.4:
        stability = "moderate"
        interpretation = "Moderate basin — approach is viable but not deeply committed."
    elif basin_depth > 0.2:
        stability = "shallow"
        interpretation = "Shallow basin — approach is fragile. Small changes could shift the trajectory."
    else:
        stability = "unstable"
        interpretation = "Near-zero basin depth — the current approach has no structural stability."

    return json.dumps({
        "basin_depth": basin_depth,
        "stability": stability,
        "variance_of_distances": round(variance, 4),
        "interpretation": interpretation,
    }, indent=2)


# ------------------------------------------------------------------
# Tool 6: Detect Bifurcation
# ------------------------------------------------------------------

@mcp.tool(
    name="attractorflow_detect_bifurcation",
    annotations={
        "title": "Detect Task Bifurcation Point",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def attractorflow_detect_bifurcation(ctx: Context) -> str:
    """
    Detect if the task has crossed a bifurcation point requiring decomposition.

    Three bifurcation types:
      PITCHFORK   — task splits into two symmetric subtasks (spawn 2 subagents)
      HOPF        — convergent process has become cyclic (add exit criterion)
      SADDLE_NODE — current approach is critically unstable (restart with new approach)

    Call this periodically (every 5–10 steps) to check if the task needs decomposition.
    The orchestrator should act on the decomposition_hint when detected=True.

    Returns:
        str: JSON with bifurcation analysis.

    Schema:
        {
          "detected": bool,
          "bifurcation_type": str,
          "proximity": float,          # 0–1 imminence score
          "recommended_n_subtasks": int,
          "decomposition_hint": str,   # orchestration instruction
          "evidence": str,
          "cluster_centroids": [[float, float]]
        }
    """
    embeddings = _monitor.get_embeddings_matrix()
    result = _bifurcation.analyze(embeddings, _regime_history, _ftle_history)

    return json.dumps({
        "detected": result.detected,
        "bifurcation_type": result.bifurcation_type.value,
        "proximity": round(result.proximity, 3),
        "recommended_n_subtasks": result.recommended_n_subtasks,
        "decomposition_hint": result.decomposition_hint,
        "evidence": result.evidence,
        "cluster_centroids": [[round(x, 3), round(y, 3)] for x, y in result.cluster_centroids],
    }, indent=2)


# ------------------------------------------------------------------
# Tool 7: Inject Perturbation
# ------------------------------------------------------------------

@mcp.tool(
    name="attractorflow_inject_perturbation",
    annotations={
        "title": "Generate Context Perturbation",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    },
)
async def attractorflow_inject_perturbation(
    params: InjectPerturbationInput, ctx: Context
) -> str:
    """
    Generate a prompt fragment to inject into the agent context to escape a
    pathological attractor (STUCK, OSCILLATING, or CYCLING trap).

    Perturbation magnitude controls the intensity of the intervention:
      0.0–0.3: subtle — reframe the problem or add a constraint
      0.3–0.6: moderate — change primary tool or algorithm
      0.6–1.0: radical — restart with a fundamentally different approach

    The returned perturbation_text should be injected into the agent's
    next message as a system-level context addition.

    Args:
        params (InjectPerturbationInput):
            - magnitude (float): 0.0–1.0 perturbation intensity

    Returns:
        str: JSON with perturbation text and strategy.

    Schema:
        {
          "perturbation_text": str,  # inject this into agent context
          "strategy": str,
          "magnitude": float,
          "current_regime": str
        }
    """
    distances = _monitor.get_distance_series()
    stats = _monitor.get_stats()
    lya = _lyapunov.compute(distances)
    classification = _classifier.classify(lya, stats)
    regime = classification.regime

    perturbation = _generate_perturbation(regime, params.magnitude)

    return json.dumps({
        "perturbation_text": perturbation["text"],
        "strategy": perturbation["strategy"],
        "magnitude": params.magnitude,
        "current_regime": regime.value,
    }, indent=2)


# ------------------------------------------------------------------
# Tool 8: Checkpoint
# ------------------------------------------------------------------

@mcp.tool(
    name="attractorflow_checkpoint",
    annotations={
        "title": "Save Trajectory Checkpoint",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def attractorflow_checkpoint(ctx: Context) -> str:
    """
    Save the current trajectory as a stable reference checkpoint.

    Call this when the agent has just reached a verified-good state
    (all tests passing, clean design, stable output). If the trajectory
    later diverges, the orchestrator can restore this checkpoint to
    re-anchor from the last known good position.

    Returns:
        str: JSON confirming checkpoint save.

    Schema:
        {
          "saved": bool,
          "step_index": int,
          "buffer_size": int,
          "message": str
        }
    """
    _monitor.checkpoint()
    await ctx.log("info", f"Checkpoint saved step={_monitor.n_steps} size={_monitor.buffer_size}")

    return json.dumps({
        "saved": True,
        "step_index": _monitor.n_steps,
        "buffer_size": _monitor.buffer_size,
        "message": (
            f"Checkpoint saved at step {_monitor.n_steps}. "
            "Use attractorflow_restore_checkpoint to return here if trajectory diverges."
        ),
    }, indent=2)


# ------------------------------------------------------------------
# Perturbation generator (private)
# ------------------------------------------------------------------

def _generate_perturbation(regime: Regime, magnitude: float) -> Dict[str, str]:
    """Generate regime-appropriate perturbation text."""
    if regime == Regime.STUCK:
        if magnitude < 0.4:
            return {
                "strategy": "constraint_addition",
                "text": (
                    "[ATTRACTOR INTERVENTION — STUCK] "
                    "You have been in the same place for several steps. "
                    "Add one specific constraint to the problem: pick a simpler version, "
                    "a specific test case, or a smaller scope. Make progress on that "
                    "before returning to the full problem."
                ),
            }
        elif magnitude < 0.7:
            return {
                "strategy": "tool_switch",
                "text": (
                    "[ATTRACTOR INTERVENTION — STUCK] "
                    "Stop using your current approach entirely. "
                    "Switch to a different tool or algorithm: if you were writing code, "
                    "search for an existing library; if you were reading docs, write a prototype; "
                    "if you were debugging, write a minimal reproduction case."
                ),
            }
        else:
            return {
                "strategy": "fresh_start",
                "text": (
                    "[ATTRACTOR INTERVENTION — STUCK] "
                    "Abandon the current approach completely. Re-read the original task. "
                    "List three fundamentally different ways to solve it. "
                    "Pick the one you have not tried yet and start fresh."
                ),
            }

    elif regime == Regime.OSCILLATING:
        return {
            "strategy": "symmetry_breaking",
            "text": (
                "[ATTRACTOR INTERVENTION — OSCILLATING] "
                "You are alternating between two approaches. Break the symmetry: "
                "impose an asymmetric test that only one approach can pass. "
                "Write the test first, before choosing the implementation. "
                "Commit to whichever approach makes the test easiest to pass."
            ),
        }

    elif regime == Regime.DIVERGING:
        return {
            "strategy": "re_anchor",
            "text": (
                "[ATTRACTOR INTERVENTION — DIVERGING] "
                "STOP. You have drifted from the original goal. "
                "Before writing any more code or text, re-read the original task statement "
                "in full. Write one sentence: what is the single most important deliverable? "
                "Discard everything that does not directly contribute to that deliverable."
            ),
        }

    else:
        # Generic perturbation for CYCLING trap or other states
        return {
            "strategy": "perspective_shift",
            "text": (
                f"[ATTRACTOR INTERVENTION — {regime.value}] "
                "Shift your perspective: instead of asking 'how do I do X?', "
                "ask 'what is the simplest proof that X is done?' "
                "Write that proof first, then work backwards to the implementation."
            ),
        }


# ------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
