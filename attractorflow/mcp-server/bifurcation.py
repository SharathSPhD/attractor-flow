"""
bifurcation.py — Bifurcation Detector

Monitors agent trajectory for signs that a task has crossed a complexity
threshold requiring decomposition into parallel subtasks.

Three bifurcation types from dynamical systems theory:
  PITCHFORK  — single attractor splits into two (symmetric subtasks)
               Detection: bimodal clustering of trajectory in PCA space
  HOPF       — fixed point destabilizes into limit cycle
               Detection: CONVERGING → CYCLING transition with Lyapunov sign change
  SADDLE_NODE — two fixed points collide and annihilate (approach becomes infeasible)
               Detection: previously stable approach suddenly becomes unstable

When a bifurcation is detected, the orchestrator should spawn subagents
— one per emergent basin — rather than continuing with a single agent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

from classifier import Regime, ClassificationResult


class BifurcationType(str, Enum):
    NONE = "NONE"
    PITCHFORK = "PITCHFORK"     # task splits into symmetric subtasks
    HOPF = "HOPF"               # convergent process becomes cyclic
    SADDLE_NODE = "SADDLE_NODE"  # previously stable approach collapses


@dataclass
class BifurcationResult:
    """Output of bifurcation analysis."""
    detected: bool
    bifurcation_type: BifurcationType
    proximity: float                    # 0.0 = no signs; 1.0 = bifurcation imminent
    recommended_n_subtasks: int         # how many subagents to spawn
    decomposition_hint: str             # suggested decomposition strategy
    evidence: str                       # what signal triggered detection
    cluster_centroids: List[Tuple[float, float]]  # PCA centroids of emergent basins


class BifurcationDetector:
    """
    Detects bifurcation signatures in the agent trajectory.

    Usage:
        detector = BifurcationDetector()
        result = detector.analyze(pca_2d_trajectory, regime_history, lyapunov_history)
    """

    def __init__(
        self,
        bimodal_threshold: float = 0.6,     # silhouette score threshold for PITCHFORK
        hopf_window: int = 5,               # steps to look back for Hopf transition
        sensitivity_threshold: float = 0.3,  # FTLE variance for SADDLE_NODE
    ) -> None:
        self.bimodal_threshold = bimodal_threshold
        self.hopf_window = hopf_window
        self.sensitivity_threshold = sensitivity_threshold

    def analyze(
        self,
        pca_2d: List[Tuple[float, float]],
        regime_history: List[Regime],
        ftle_history: List[float],
    ) -> BifurcationResult:
        """
        Run all bifurcation checks and return the highest-priority result.

        Checks are run in priority order; the first detection returned.
        """
        if len(pca_2d) < 6:
            return _no_bifurcation("Insufficient trajectory data (need ≥ 6 steps).")

        # Check 1: Pitchfork (bimodal clustering)
        result = self._check_pitchfork(pca_2d)
        if result.detected:
            return result

        # Check 2: Hopf (CONVERGING → CYCLING transition)
        result = self._check_hopf(regime_history, ftle_history)
        if result.detected:
            return result

        # Check 3: Saddle-node (high FTLE variance in recent window)
        result = self._check_saddle_node(ftle_history)
        if result.detected:
            return result

        # No bifurcation detected — but compute proximity for monitoring
        proximity = self._compute_proximity(pca_2d, ftle_history)
        return BifurcationResult(
            detected=False,
            bifurcation_type=BifurcationType.NONE,
            proximity=proximity,
            recommended_n_subtasks=1,
            decomposition_hint="",
            evidence=f"No bifurcation. Proximity score: {proximity:.2f}",
            cluster_centroids=[],
        )

    # ------------------------------------------------------------------
    # Pitchfork detection — bimodal clustering in PCA space
    # ------------------------------------------------------------------

    def _check_pitchfork(
        self, pca_2d: List[Tuple[float, float]]
    ) -> BifurcationResult:
        """
        Detect if trajectory has split into two distinct clusters.
        Uses k-means with k=2 and silhouette score.
        """
        if len(pca_2d) < 8:
            return _no_bifurcation("Too few points for cluster analysis.")

        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
        except ImportError:
            return _no_bifurcation("scikit-learn not available for cluster analysis.")

        X = np.array(pca_2d)
        kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
        labels = kmeans.fit_predict(X)

        # Need at least 2 points per cluster
        if len(set(labels)) < 2 or min(np.bincount(labels)) < 2:
            return _no_bifurcation("Degenerate cluster assignment.")

        sil_score = float(silhouette_score(X, labels))
        centroids = [(float(c[0]), float(c[1])) for c in kmeans.cluster_centers_]

        if sil_score >= self.bimodal_threshold:
            proximity = min(1.0, sil_score)
            return BifurcationResult(
                detected=True,
                bifurcation_type=BifurcationType.PITCHFORK,
                proximity=proximity,
                recommended_n_subtasks=2,
                decomposition_hint=(
                    "Task has naturally split into two distinct approaches. "
                    "Spawn two subagents, one per identified cluster. "
                    "Provide each with the context corresponding to its basin: "
                    "subagent A focuses on the first approach, subagent B on the second. "
                    "Have the orchestrator evaluate both and select the convergent path."
                ),
                evidence=(
                    f"Silhouette score={sil_score:.3f} ≥ {self.bimodal_threshold} "
                    f"with 2 clusters in PCA space. "
                    f"Centroid separation={_centroid_distance(centroids):.3f}."
                ),
                cluster_centroids=centroids,
            )

        # Not yet bifurcated but approaching
        return _no_bifurcation(f"Silhouette={sil_score:.3f} below threshold {self.bimodal_threshold}.")

    # ------------------------------------------------------------------
    # Hopf detection — CONVERGING → CYCLING transition
    # ------------------------------------------------------------------

    def _check_hopf(
        self,
        regime_history: List[Regime],
        ftle_history: List[float],
    ) -> BifurcationResult:
        """
        Detect Hopf bifurcation: previously CONVERGING agent has become CYCLING.
        Signature: λ crosses zero from below, regime transitions from CONVERGING.
        """
        W = self.hopf_window
        if len(regime_history) < W + 2:
            return _no_bifurcation("Insufficient regime history for Hopf check.")

        recent = regime_history[-W:]
        prev_period = regime_history[-(W * 2):-W]

        was_converging = any(r == Regime.CONVERGING for r in prev_period)
        is_cycling = any(r in (Regime.CYCLING, Regime.OSCILLATING) for r in recent)

        if not (was_converging and is_cycling):
            return _no_bifurcation("No CONVERGING → CYCLING transition detected.")

        # Check λ sign change
        if len(ftle_history) >= W:
            prev_lambda = np.mean(ftle_history[-(W * 2):-W])
            curr_lambda = np.mean(ftle_history[-W:])
            lambda_crossed = prev_lambda < 0 < curr_lambda
        else:
            lambda_crossed = False

        proximity = 0.7 + 0.3 * float(lambda_crossed)

        return BifurcationResult(
            detected=True,
            bifurcation_type=BifurcationType.HOPF,
            proximity=proximity,
            recommended_n_subtasks=1,  # Hopf doesn't split — it creates iteration
            decomposition_hint=(
                "The task has undergone a Hopf bifurcation: a previously direct approach "
                "has become iterative. This is often healthy (TDD cycles, refinement loops). "
                "If cycles are not converging, add an explicit exit criterion: "
                "'stop when all N tests pass' or 'stop after K iterations regardless'. "
                "This converts the limit cycle into a controlled spiral toward the attractor."
            ),
            evidence=(
                f"Regime transition: CONVERGING → CYCLING over last {W * 2} steps. "
                + (f"λ sign change: {prev_lambda:.3f} → {curr_lambda:.3f}." if lambda_crossed else "")
            ),
            cluster_centroids=[],
        )

    # ------------------------------------------------------------------
    # Saddle-node detection — sudden instability (high FTLE variance)
    # ------------------------------------------------------------------

    def _check_saddle_node(self, ftle_history: List[float]) -> BifurcationResult:
        """
        Detect saddle-node bifurcation: approach suddenly becomes unstable.
        Signature: high variance in recent FTLE values (sensitivity to initial conditions).
        """
        W = self.hopf_window
        if len(ftle_history) < W:
            return _no_bifurcation("Insufficient FTLE history for saddle-node check.")

        recent_ftle = np.array(ftle_history[-W:])
        variance = float(recent_ftle.var())

        if variance >= self.sensitivity_threshold:
            return BifurcationResult(
                detected=True,
                bifurcation_type=BifurcationType.SADDLE_NODE,
                proximity=min(1.0, variance / (self.sensitivity_threshold * 2)),
                recommended_n_subtasks=0,
                decomposition_hint=(
                    "The current approach is critically unstable — highly sensitive to context. "
                    "This indicates a saddle-node bifurcation: the viable solution basin has "
                    "narrowed to the point of near-disappearance. "
                    "Recommended action: restart from the last stable checkpoint with a "
                    "fundamentally different approach (different algorithm, different tool, "
                    "different problem decomposition)."
                ),
                evidence=(
                    f"FTLE variance={variance:.3f} ≥ {self.sensitivity_threshold} "
                    f"over last {W} steps. "
                    f"FTLE range: [{recent_ftle.min():.3f}, {recent_ftle.max():.3f}]."
                ),
                cluster_centroids=[],
            )

        return _no_bifurcation(f"FTLE variance={variance:.3f} below threshold.")

    # ------------------------------------------------------------------
    # Proximity score for non-detected cases
    # ------------------------------------------------------------------

    def _compute_proximity(
        self,
        pca_2d: List[Tuple[float, float]],
        ftle_history: List[float],
    ) -> float:
        """Continuous 0–1 score even when no bifurcation is detected."""
        scores = []

        # PCA spread as proxy for approaching pitchfork
        if len(pca_2d) >= 4:
            X = np.array(pca_2d)
            spread = float(X.std(axis=0).mean())
            scores.append(min(1.0, spread / 2.0))

        # FTLE trend as proxy for approaching any bifurcation
        if len(ftle_history) >= 3:
            trend = float(np.polyfit(np.arange(len(ftle_history)), ftle_history, 1)[0])
            scores.append(min(1.0, max(0.0, trend * 5 + 0.5)))

        return float(np.mean(scores)) if scores else 0.0


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _no_bifurcation(reason: str) -> BifurcationResult:
    return BifurcationResult(
        detected=False,
        bifurcation_type=BifurcationType.NONE,
        proximity=0.0,
        recommended_n_subtasks=1,
        decomposition_hint="",
        evidence=reason,
        cluster_centroids=[],
    )


def _centroid_distance(centroids: List[Tuple[float, float]]) -> float:
    if len(centroids) < 2:
        return 0.0
    a, b = np.array(centroids[0]), np.array(centroids[1])
    return float(np.linalg.norm(a - b))
