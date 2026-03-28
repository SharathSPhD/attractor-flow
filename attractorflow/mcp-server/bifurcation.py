"""
bifurcation.py — Bifurcation Detector

Monitors agent trajectory for signs that a task has crossed a complexity
threshold requiring decomposition into parallel subtasks.

Three bifurcation/instability signals detected:

  PITCHFORK  — Trajectory clusters into two distinct semantic basins.
               Detection: k-means with k=2 directly in 384-dim embedding
               space; silhouette score ≥ threshold.
               Interpretation: task has naturally split into two independent
               approaches. Spawn two subagents, one per cluster.

  HOPF       — Trajectory transitions from contracting (CONVERGING) to
               neutral/periodic (CYCLING).
               Detection: λ history transitions from λ < -0.05 to |λ| < 0.05.
               Self-monitoring: fires from FTLE history alone, without
               requiring frequent get_regime() calls.
               Interpretation: direct process has become iterative; add an
               explicit exit criterion.

  SADDLE_NODE — Displacement matrix condition number exceeds threshold,
                indicating trajectory has collapsed to near-1D.
                Detection: σ_max / σ_min of the W×384 displacement matrix.
                High condition number = displacement vectors are nearly
                collinear = trajectory is exploring only one semantic
                direction = geometric signature of instability.
                Note: this is an instability signal, not a true saddle-node
                bifurcation in the strict sense (which requires two fixed
                points to collide). It reliably detects trajectory collapse
                before the agent fully stalls.

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
    PITCHFORK = "PITCHFORK"     # task splits into two distinct semantic basins
    HOPF = "HOPF"               # convergent process becomes cyclic
    SADDLE_NODE = "SADDLE_NODE"  # trajectory collapses to near-1D (instability event)


@dataclass
class BifurcationResult:
    """Output of bifurcation analysis."""
    detected: bool
    bifurcation_type: BifurcationType
    proximity: float                    # 0.0 = no signs; 1.0 = bifurcation imminent
    recommended_n_subtasks: int         # how many subagents to spawn
    decomposition_hint: str             # suggested decomposition strategy
    evidence: str                       # what signal triggered detection
    cluster_centroids: List[Tuple[float, float]]  # PCA-2D centroids of emergent basins


class BifurcationDetector:
    """
    Detects bifurcation signatures in the agent trajectory.

    Usage:
        detector = BifurcationDetector()
        result = detector.analyze(embeddings_matrix, regime_history, lyapunov_history)
    """

    def __init__(
        self,
        bimodal_threshold: float = 0.6,   # silhouette score threshold for PITCHFORK
        hopf_window: int = 5,             # steps to look back for Hopf transition
        cond_threshold: float = 20.0,     # condition number threshold for SADDLE_NODE
    ) -> None:
        self.bimodal_threshold = bimodal_threshold
        self.hopf_window = hopf_window
        self.cond_threshold = cond_threshold

    def analyze(
        self,
        embeddings_matrix: np.ndarray,
        regime_history: List[Regime],
        ftle_history: List[float],
    ) -> BifurcationResult:
        """
        Run all bifurcation checks and return the highest-priority result.

        Args:
            embeddings_matrix: (N, 384) array of recorded embeddings.
            regime_history: List of Regime enum values from past get_regime() calls.
            ftle_history: List of FTLE values from past record_state() calls.

        Checks are run in priority order; the first detection is returned.
        """
        n = len(embeddings_matrix)
        if n < 6:
            return _no_bifurcation("Insufficient trajectory data (need ≥ 6 steps).")

        # Check 1: Pitchfork (bimodal clustering in 384-dim space)
        result = self._check_pitchfork(embeddings_matrix)
        if result.detected:
            return result

        # Check 2: Hopf (CONVERGING → CYCLING transition)
        result = self._check_hopf(regime_history, ftle_history)
        if result.detected:
            return result

        # Check 3: Saddle-node (trajectory collapse via condition number)
        result = self._check_saddle_node(embeddings_matrix)
        if result.detected:
            return result

        # No bifurcation — compute proximity for monitoring
        proximity = self._compute_proximity(embeddings_matrix, ftle_history)
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
    # Pitchfork detection — bimodal clustering in 384-dim embedding space
    # ------------------------------------------------------------------

    def _check_pitchfork(self, embeddings_matrix: np.ndarray) -> BifurcationResult:
        """
        Detect if trajectory has split into two distinct semantic clusters.

        K-means with k=2 runs directly in 384-dim embedding space.
        Silhouette score is computed in 384-dim (not on PCA projection).
        Cluster centroids are then projected to PCA-2D for human-readable output.
        """
        n = len(embeddings_matrix)
        if n < 8:
            return _no_bifurcation("Too few points for cluster analysis.")

        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            from sklearn.decomposition import PCA
        except ImportError:
            return _no_bifurcation("scikit-learn not available for cluster analysis.")

        X = embeddings_matrix  # N × 384 — full embedding space
        kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
        labels = kmeans.fit_predict(X)

        # Need at least 2 points per cluster
        if len(set(labels)) < 2 or min(np.bincount(labels)) < 2:
            return _no_bifurcation("Degenerate cluster assignment.")

        # Silhouette score in 384-dim (higher fidelity than 2D)
        sil_score = float(silhouette_score(X, labels))

        # Project centroids to PCA-2D for visualization only
        centroids_384 = kmeans.cluster_centers_
        if n >= 3:
            pca = PCA(n_components=2)
            pca.fit(X)
            centroids_2d = pca.transform(centroids_384)
            centroids = [(float(c[0]), float(c[1])) for c in centroids_2d]
        else:
            centroids = [(0.0, 0.0), (0.0, 0.0)]

        centroid_dist_384 = float(np.linalg.norm(centroids_384[0] - centroids_384[1]))

        if sil_score >= self.bimodal_threshold:
            return BifurcationResult(
                detected=True,
                bifurcation_type=BifurcationType.PITCHFORK,
                proximity=min(1.0, sil_score),
                recommended_n_subtasks=2,
                decomposition_hint=(
                    "Task has naturally split into two distinct semantic approaches. "
                    "Spawn two subagents, one per identified cluster. "
                    "Provide each with the context corresponding to its basin: "
                    "subagent A focuses on the first approach, subagent B on the second. "
                    "Have the orchestrator evaluate both and select the convergent path."
                ),
                evidence=(
                    f"Silhouette score={sil_score:.3f} ≥ {self.bimodal_threshold} "
                    f"(k-means in 384-dim embedding space). "
                    f"Centroid separation in embedding space={centroid_dist_384:.3f}."
                ),
                cluster_centroids=centroids,
            )

        return _no_bifurcation(
            f"Silhouette={sil_score:.3f} below threshold {self.bimodal_threshold} "
            f"(384-dim k-means)."
        )

    # ------------------------------------------------------------------
    # Hopf detection — CONVERGING → CYCLING transition (self-monitoring)
    # ------------------------------------------------------------------

    def _check_hopf(
        self,
        regime_history: List[Regime],
        ftle_history: List[float],
    ) -> BifurcationResult:
        """
        Detect Hopf-like transition: previously contracting trajectory has
        become neutral/periodic.

        Primary path (self-monitoring): uses FTLE history directly.
            prev_converging = any(λ < -0.05) in prior window
            now_cycling = |mean(λ)| < 0.05 in recent window
        This fires correctly regardless of whether get_regime() was called.

        Secondary path: regime label transition (CONVERGING → CYCLING/OSCILLATING)
        for extra confirmation when regime history is available.
        """
        W = self.hopf_window
        ftle_detected = False
        lambda_crossed = False
        prev_lam_mean = 0.0
        curr_lam_mean = 0.0

        if len(ftle_history) >= W * 2:
            prev_ftle = np.array(ftle_history[-(W * 2):-W])
            curr_ftle = np.array(ftle_history[-W:])
            prev_lam_mean = float(prev_ftle.mean())
            curr_lam_mean = float(curr_ftle.mean())
            prev_converging = bool(np.any(prev_ftle < -0.05))
            now_cycling = bool(abs(curr_lam_mean) < 0.05)
            lambda_crossed = prev_lam_mean < 0 < curr_lam_mean
            ftle_detected = prev_converging and now_cycling

        regime_detected = False
        if len(regime_history) >= W + 2:
            recent = regime_history[-W:]
            prev_period = regime_history[-(W * 2):-W]
            was_converging = any(r == Regime.CONVERGING for r in prev_period)
            is_cycling = any(r in (Regime.CYCLING, Regime.OSCILLATING) for r in recent)
            regime_detected = was_converging and is_cycling

        if not (ftle_detected or regime_detected):
            return _no_bifurcation("No CONVERGING → CYCLING transition detected.")

        proximity = 0.7 + 0.3 * float(lambda_crossed)

        return BifurcationResult(
            detected=True,
            bifurcation_type=BifurcationType.HOPF,
            proximity=proximity,
            recommended_n_subtasks=1,
            decomposition_hint=(
                "The task has undergone a Hopf-like transition: a previously direct approach "
                "has become iterative. This is often healthy (TDD cycles, refinement loops). "
                "If cycles are not converging, add an explicit exit criterion: "
                "'stop when all N tests pass' or 'stop after K iterations regardless'. "
                "This converts the limit cycle into a controlled spiral toward the attractor."
            ),
            evidence=(
                f"Trajectory transition: CONVERGING → CYCLING over last {W * 2} steps "
                f"(detected via {'FTLE history' if ftle_detected else 'regime labels'}). "
                + (
                    f"λ sign change: {prev_lam_mean:.3f} → {curr_lam_mean:.3f}."
                    if lambda_crossed
                    else ""
                )
            ),
            cluster_centroids=[],
        )

    # ------------------------------------------------------------------
    # Saddle-node detection — condition number of displacement matrix
    # ------------------------------------------------------------------

    def _check_saddle_node(self, embeddings_matrix: np.ndarray) -> BifurcationResult:
        """
        Detect trajectory instability via condition number of the displacement matrix.

        Forms the W×384 displacement matrix M from the last W+1 embeddings.
        Computes condition number = σ_max / σ_min.

        High condition number means displacement vectors are nearly collinear
        — the trajectory has collapsed to near-1D in semantic space — which
        is the geometric precursor to trajectory stall or reversal.

        Note: this is an instability signal, not a strict saddle-node
        bifurcation (which requires tracking two fixed points). It reliably
        detects trajectory degeneration from high-dimensional exploration to
        near-1D movement.
        """
        W = self.hopf_window
        n = len(embeddings_matrix)
        if n < W + 1:
            return _no_bifurcation("Insufficient embeddings for condition number check.")

        recent = embeddings_matrix[-(W + 1):]
        deltas = np.diff(recent, axis=0)  # W × 384

        _, sigma, _ = np.linalg.svd(deltas, full_matrices=False)
        sigma_pos = sigma[sigma > 1e-8]
        if len(sigma_pos) < 2:
            return _no_bifurcation("Degenerate displacement matrix (rank < 2).")

        cond = float(sigma_pos[0] / sigma_pos[-1])
        # Normalize: cond=1 → proximity=0.02; cond=cond_threshold → proximity=0.5;
        # cond=2*threshold → proximity≈1.0
        proximity = min(1.0, cond / (self.cond_threshold * 2))

        if cond >= self.cond_threshold:
            return BifurcationResult(
                detected=True,
                bifurcation_type=BifurcationType.SADDLE_NODE,
                proximity=proximity,
                recommended_n_subtasks=0,
                decomposition_hint=(
                    "The trajectory has collapsed to near-1D in semantic space — "
                    "all recent steps are exploring only one conceptual direction. "
                    "This instability signal indicates the current approach is losing "
                    "structural robustness. "
                    "Recommended: restore the last stable checkpoint and restart with "
                    "a fundamentally different approach (different algorithm, tool, "
                    "or problem decomposition)."
                ),
                evidence=(
                    f"Displacement matrix condition number={cond:.1f} ≥ {self.cond_threshold} "
                    f"(σ_max={sigma_pos[0]:.4f}, σ_min={sigma_pos[-1]:.4f}). "
                    f"Trajectory is near-1D in 384-dim embedding space."
                ),
                cluster_centroids=[],
            )

        return _no_bifurcation(
            f"Condition number={cond:.1f} below threshold {self.cond_threshold}."
        )

    # ------------------------------------------------------------------
    # Proximity score for non-detected cases
    # ------------------------------------------------------------------

    def _compute_proximity(
        self,
        embeddings_matrix: np.ndarray,
        ftle_history: List[float],
    ) -> float:
        """Continuous 0–1 proximity score even when no bifurcation is detected."""
        scores = []

        # Embedding spread as proxy for approaching pitchfork
        n = len(embeddings_matrix)
        if n >= 4:
            # Use variance in 384-dim space (mean per-dimension std)
            spread = float(embeddings_matrix.std(axis=0).mean())
            scores.append(min(1.0, spread / 0.5))

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
