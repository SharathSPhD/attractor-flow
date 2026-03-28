"""
classifier.py — Attractor Regime Classifier

Takes output from LyapunovEstimator + PhaseSpaceMonitor stats and
produces a definitive regime classification with a prescribed
orchestration action.

Seven regimes (from research.md):
  CONVERGING  — trajectory approaching fixed-point attractor
  CYCLING     — healthy limit-cycle iteration (develop→test→refactor)
  EXPLORING   — structured bounded divergence (design phase)
  DIVERGING   — uncontrolled trajectory expansion (intervention needed)
  STUCK       — near-zero velocity, fixed-point trap
  OSCILLATING — 2-period Wang attractor (symmetry must be broken)
  PLATEAU     — slow drift toward goal with minimal velocity (needs gentle nudge)

Each regime maps to a Strategy that provides:
  - recommend_action() → OrchestratorAction
  - intervention_hint() → str (prompt fragment for orchestrator)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from lyapunov import LyapunovResult
from phase_space import TrajectoryStats


# ------------------------------------------------------------------
# Enums
# ------------------------------------------------------------------

class Regime(str, Enum):
    CONVERGING = "CONVERGING"
    CYCLING = "CYCLING"
    EXPLORING = "EXPLORING"
    DIVERGING = "DIVERGING"
    STUCK = "STUCK"
    OSCILLATING = "OSCILLATING"
    PLATEAU = "PLATEAU"
    UNKNOWN = "UNKNOWN"


class OrchestratorAction(str, Enum):
    CONTINUE = "CONTINUE"               # let the agent keep running
    REDUCE_TEMPERATURE = "REDUCE_TEMPERATURE"  # apply convergence pressure
    INJECT_PERTURBATION = "INJECT_PERTURBATION"  # escape local attractor
    DECOMPOSE_TASK = "DECOMPOSE_TASK"   # bifurcation — spawn subagents
    RESTORE_CHECKPOINT = "RESTORE_CHECKPOINT"  # re-anchor from stable state
    BREAK_SYMMETRY = "BREAK_SYMMETRY"   # asymmetric constraint for 2-period orbit
    SPAWN_EXPLORER = "SPAWN_EXPLORER"   # launch explorer-agent for stuck state
    NUDGE = "NUDGE"                     # gentle constraint for plateau — small push forward
    HALT = "HALT"                       # stop and escalate to human


@dataclass
class ClassificationResult:
    regime: Regime
    confidence: float                   # 0–1 confidence in this classification
    action: OrchestratorAction
    intervention_hint: str              # concrete prompt fragment
    rationale: str                      # human-readable explanation
    λ: float                            # FTLE at classification time
    n_steps: int                        # trajectory length
    cycles_detected: bool
    dominant_cycle_lag: int


# ------------------------------------------------------------------
# Strategy Protocol + implementations
# ------------------------------------------------------------------

class _ConvergingStrategy:
    def classify(self, lya: LyapunovResult, stats: TrajectoryStats) -> ClassificationResult:
        strong = lya.ftle < -0.2
        confidence = min(1.0, abs(lya.ftle) / 0.3)
        return ClassificationResult(
            regime=Regime.CONVERGING,
            confidence=confidence,
            action=OrchestratorAction.REDUCE_TEMPERATURE,
            intervention_hint=(
                "You are approaching a solution. Narrow your scope: commit to the "
                "current approach and add tests to lock in progress. Avoid exploring "
                "alternative approaches unless tests fail."
            ),
            rationale=(
                f"λ={lya.ftle:.3f} ({'strong' if strong else 'weak'} convergence). "
                f"Mean step distance={stats.mean_distance:.4f}, "
                f"trend={stats.distance_trend:+.4f}. "
                "Trajectory is contracting toward attractor basin."
            ),
            λ=lya.ftle,
            n_steps=stats.n_steps,
            cycles_detected=False,
            dominant_cycle_lag=0,
        )


class _CyclingStrategy:
    def classify(self, lya: LyapunovResult, stats: TrajectoryStats) -> ClassificationResult:
        # Determine if amplitude is decreasing (healthy) or stable/growing (trap)
        d = stats.distances
        if len(d) >= 6:
            first_half = sum(d[:len(d)//2]) / (len(d)//2)
            second_half = sum(d[len(d)//2:]) / (len(d) - len(d)//2)
            amplitude_decreasing = second_half < first_half
        else:
            amplitude_decreasing = True  # assume healthy with too few steps

        if amplitude_decreasing:
            action = OrchestratorAction.CONTINUE
            hint = (
                "You are in a healthy development cycle. Continue iterating — "
                "each loop is improving the solution. The amplitude of changes "
                "is decreasing, which indicates convergence toward a stable solution."
            )
            rationale = "Limit cycle with decreasing amplitude — healthy iteration."
        else:
            action = OrchestratorAction.INJECT_PERTURBATION
            hint = (
                "You appear to be caught in a stable loop without making progress. "
                "Try a qualitatively different approach: change the tool you are using, "
                "add an explicit constraint to the problem, or solve a sub-problem first."
            )
            rationale = "Limit cycle with stable/growing amplitude — approaching cycle trap."

        return ClassificationResult(
            regime=Regime.CYCLING,
            confidence=min(1.0, abs(lya.dominant_autocorr)),
            action=action,
            intervention_hint=hint,
            rationale=rationale,
            λ=lya.ftle,
            n_steps=stats.n_steps,
            cycles_detected=True,
            dominant_cycle_lag=lya.dominant_lag,
        )


class _ExploringStrategy:
    def classify(self, lya: LyapunovResult, stats: TrajectoryStats) -> ClassificationResult:
        return ClassificationResult(
            regime=Regime.EXPLORING,
            confidence=0.7,
            action=OrchestratorAction.CONTINUE,
            intervention_hint=(
                "You are in exploration mode — bounded divergence is appropriate. "
                "If this is a design or planning phase, continue. "
                "If you are in an implementation phase, narrow your scope: "
                "pick the most promising approach and commit to it."
            ),
            rationale=(
                f"λ={lya.ftle:.3f}: Mild trajectory expansion. "
                "Structured exploration — trajectory is bounded but not repeating."
            ),
            λ=lya.ftle,
            n_steps=stats.n_steps,
            cycles_detected=False,
            dominant_cycle_lag=0,
        )


class _DivergingStrategy:
    def classify(self, lya: LyapunovResult, stats: TrajectoryStats) -> ClassificationResult:
        severe = lya.ftle > 0.5
        return ClassificationResult(
            regime=Regime.DIVERGING,
            confidence=min(1.0, lya.ftle / 0.5),
            action=OrchestratorAction.RESTORE_CHECKPOINT if not severe else OrchestratorAction.HALT,
            intervention_hint=(
                "ALERT: Your trajectory is diverging — you are drifting from the original goal. "
                "Stop the current approach immediately. Re-read the original task statement. "
                "Restore context to the last stable checkpoint and restart with a narrower scope."
            ),
            rationale=(
                f"λ={lya.ftle:.3f} ({'severe' if severe else 'moderate'} divergence). "
                f"ftle_trend={lya.ftle_trend:+.3f}. "
                "Trajectory expanding away from goal basin."
            ),
            λ=lya.ftle,
            n_steps=stats.n_steps,
            cycles_detected=False,
            dominant_cycle_lag=0,
        )


class _StuckStrategy:
    def classify(self, lya: LyapunovResult, stats: TrajectoryStats) -> ClassificationResult:
        return ClassificationResult(
            regime=Regime.STUCK,
            confidence=0.9,
            action=OrchestratorAction.SPAWN_EXPLORER,
            intervention_hint=(
                "You are stuck at a fixed-point attractor. Your recent steps are not "
                "changing the solution meaningfully. Try a completely different approach: "
                "use a different tool, decompose the problem into smaller sub-problems, "
                "or ask for clarification about the requirements."
            ),
            rationale=(
                f"Near-zero velocity in embedding space "
                f"(max_distance={stats.max_distance:.4f} < 0.40 threshold). "
                "Agent is in a fixed-point trap."
            ),
            λ=lya.ftle,
            n_steps=stats.n_steps,
            cycles_detected=False,
            dominant_cycle_lag=0,
        )


class _OscillatingStrategy:
    def classify(self, lya: LyapunovResult, stats: TrajectoryStats) -> ClassificationResult:
        return ClassificationResult(
            regime=Regime.OSCILLATING,
            confidence=min(1.0, abs(lya.dominant_autocorr)),
            action=OrchestratorAction.BREAK_SYMMETRY,
            intervention_hint=(
                "You are oscillating between exactly two states — a Wang 2-period attractor. "
                "Break the symmetry by adding an asymmetric constraint: "
                "require one of the two approaches to pass a specific test, "
                "add a new requirement that differentiates the options, "
                "or bring in external information (documentation, examples) "
                "that makes one option strictly preferable."
            ),
            rationale=(
                f"λ={lya.ftle:.3f}. "
                f"lag-1 autocorr={lya.autocorrelation[0]:.2f} — "
                "Detected 2-period attractor — alternating between two states without "
                "convergence. (Empirical: lag-1 anticorrelation characteristic of "
                "2-period limit cycles in LLM generation.)"
            ),
            λ=lya.ftle,
            n_steps=stats.n_steps,
            cycles_detected=True,
            dominant_cycle_lag=lya.dominant_lag,
        )


class _PlateauStrategy:
    """
    PLATEAU regime: slow drift toward goal with minimal velocity.

    Characteristics:
    - lambda near zero (-0.05 < ftle < 0.05)
    - Negative distance trend (drifting toward goal)
    - Very low velocity (tiny refinements only)

    Different from STUCK (no trend) and CONVERGING (clear lambda < -0.05).
    Action: NUDGE — a gentle push with a small specific constraint.
    """
    def classify(self, lya: LyapunovResult, stats: TrajectoryStats) -> ClassificationResult:
        return ClassificationResult(
            regime=Regime.PLATEAU,
            confidence=0.75,
            action=OrchestratorAction.NUDGE,
            intervention_hint=(
                "You are making slow but measurable progress toward the goal — a plateau state. "
                "Your trajectory shows gradual drift in the right direction, but velocity is low. "
                "Add a small, specific constraint to accelerate progress: "
                "write one concrete test case, implement one specific function, "
                "or answer one precise question about the design. "
                "Do not change your overall approach — just take one deliberate step forward."
            ),
            rationale=(
                f"λ={lya.ftle:.3f} (near-neutral). "
                f"Distance trend={stats.distance_trend:+.4f} (drifting toward goal). "
                f"Mean velocity={stats.mean_distance:.4f} (low). "
                "Plateau regime — slow progress needs gentle acceleration."
            ),
            λ=lya.ftle,
            n_steps=stats.n_steps,
            cycles_detected=False,
            dominant_cycle_lag=0,
        )


# ------------------------------------------------------------------
# Main Classifier
# ------------------------------------------------------------------

_STRATEGIES = {
    Regime.CONVERGING: _ConvergingStrategy(),
    Regime.CYCLING: _CyclingStrategy(),
    Regime.EXPLORING: _ExploringStrategy(),
    Regime.DIVERGING: _DivergingStrategy(),
    Regime.STUCK: _StuckStrategy(),
    Regime.OSCILLATING: _OscillatingStrategy(),
    Regime.PLATEAU: _PlateauStrategy(),
}


class AttractorClassifier:
    """
    Classifies agent dynamics into one of seven attractor regimes.

    Usage:
        classifier = AttractorClassifier()
        result = classifier.classify(lyapunov_result, trajectory_stats)
    """

    def classify(
        self,
        lya: LyapunovResult,
        stats: TrajectoryStats,
    ) -> ClassificationResult:
        """
        Determine the current dynamical regime and prescribe an action.

        Decision tree (order matters — checked top to bottom):
        1. STUCK: near-zero velocity regardless of λ
        2. OSCILLATING: 2-period cycle (lag=2, high autocorr)
        3. DIVERGING: λ > 0.25
        4. CYCLING: periodic orbit (lag>2, high autocorr)
        5. PLATEAU: λ ≈ 0 with negative distance trend and low velocity
        6. CONVERGING: λ < -0.05
        7. EXPLORING: λ ≥ -0.05 (residual)
        """
        if stats.n_steps < 3:
            return ClassificationResult(
                regime=Regime.UNKNOWN,
                confidence=0.0,
                action=OrchestratorAction.CONTINUE,
                intervention_hint="Insufficient data — record at least 3 states.",
                rationale=f"Only {stats.n_steps} steps recorded.",
                λ=lya.ftle,
                n_steps=stats.n_steps,
                cycles_detected=False,
                dominant_cycle_lag=0,
            )

        regime = self._select_regime(lya, stats)
        return _STRATEGIES[regime].classify(lya, stats)

    def _select_regime(self, lya: LyapunovResult, stats: TrajectoryStats) -> Regime:
        """Core decision logic."""
        # Priority 1a: PLATEAU — checked before STUCK because PLATEAU is a special case of
        # low-velocity movement that IS directed toward the goal (negative distance trend).
        # Physics: PLATEAU = slow drift in a shallow gradient.  STUCK = zero gradient.
        # Distinguisher: distance_trend.  If trend is clearly negative, the agent is moving
        # toward the goal in tiny steps; that is PLATEAU, not STUCK.
        if lya.is_stuck and stats.distance_trend < -0.01:
            return Regime.PLATEAU

        # Priority 1b: Stuck (near-zero velocity, no directional trend)
        if lya.is_stuck:
            return Regime.STUCK

        # Priority 2: 2-period oscillation (Wang attractor)
        # A 2-period orbit (A→B→A→B) produces EITHER:
        #   lag-1 strong negative autocorr (adjacent states anti-correlated), OR
        #   lag-2 strong positive autocorr (same state every 2 steps).
        autocorr = lya.autocorrelation
        lag1_autocorr = autocorr[0] if len(autocorr) >= 1 else 0.0
        lag2_autocorr = autocorr[1] if len(autocorr) >= 2 else 0.0
        is_two_period = (
            (lag1_autocorr < -0.4 and abs(lya.ftle) < 0.25)
            or (lag2_autocorr > 0.5 and abs(lya.ftle) < 0.15)
        )
        if is_two_period:
            return Regime.OSCILLATING

        # Priority 3: Diverging
        # Two detection paths:
        #   (a) High FTLE alone → clear exponential divergence
        #   (b) Sustained positive distance trend + large mean distance →
        #       drift-style divergence where λ hasn't yet crossed the hard threshold.
        #       Observed in real agent traces: topic drift produces λ ≈ 0.01–0.08
        #       but mean_distance > 1.0 and trend > 0.008 are unambiguous.
        drift_diverging = (
            stats.distance_trend > 0.008
            and stats.mean_distance > 1.0
            and lya.ftle > -0.05  # not clearly converging
        )
        if lya.ftle > 0.25 or drift_diverging:
            return Regime.DIVERGING

        # Priority 4: General cycling (limit cycle)
        # Does NOT fire when distance_trend is strongly negative — a falling trajectory
        # with high autocorr is likely a healthy convergence spiral, not a trap cycle.
        if (
            abs(lya.dominant_autocorr) > 0.5
            and lya.dominant_lag >= 2
            and lya.ftle > -0.05          # don't mask strong convergence
            and lya.ftle < 0.25
            and stats.distance_trend > -0.02  # not a clear falling trajectory
        ):
            return Regime.CYCLING

        # Priority 5: (PLATEAU handled at Priority 1a above)

        # Priority 6: Converging
        if lya.ftle < -0.05:
            return Regime.CONVERGING

        # Priority 7: Default — exploring
        return Regime.EXPLORING
