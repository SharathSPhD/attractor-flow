"""
lyapunov.py — Finite-Time Lyapunov Exponent (FTLE) Estimator

Computes trajectory stability metrics from the 384-dim embedding sequence
produced by PhaseSpaceMonitor.

Primary computation — SVD-based single-trajectory FTLE:

    M = [δ_{t-W}, ..., δ_{t-1}]     (W × 384 displacement matrix)
    δ_i = e_{i+1} − e_i              (step vector in embedding space)
    σ_max = largest singular value of M
    λ(t) = log(σ_max) / W

This measures the maximum linear stretching of the trajectory in 384-dim
embedding space over the window.  It is a genuine single-trajectory FTLE
approximation — it uses all 384 directions, unlike a scalar distance ratio
which discards directional information.

Secondary metric — step-size growth rate (kept for reference):

    r(t) = (1/W) × Σ ln(d_{i+1} / d_i)

where d_i = ||e_i − e_{i-1}||.  This is the mean log-ratio of scalar step
sizes — a 1D proxy retained for backward compatibility and as a sanity check
against the SVD-based λ.

Interpretation of λ (SVD-based):
    λ < -0.2   → strongly contractive (CONVERGING)
    λ ∈ [-0.2, -0.05] → weakly contractive (healthy iteration)
    λ ∈ [-0.05, 0.05]  → neutral / limit-cycle candidate
    λ ∈ [0.05, 0.25]   → mildly expanding (EXPLORING)
    λ > 0.25           → strongly expanding (DIVERGING)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# Thresholds for regime pre-classification (refined by classifier.py)
LAMBDA_CONVERGING_STRONG = -0.2
LAMBDA_CONVERGING_WEAK = -0.05
LAMBDA_NEUTRAL_HIGH = 0.05
LAMBDA_EXPLORING_HIGH = 0.25

# Minimum value to avoid log(0)
EPSILON = 1e-8

# Default window width
DEFAULT_WINDOW = 8

# Stuck threshold — calibrated for normalized all-MiniLM-L6-v2 embeddings.
# Normal inter-step distances: 0.5–1.4. A stuck agent repeating semantically
# near-identical text produces distances < 0.4. Cycling/exploring/diverging all
# produce distances > 0.5, safely above this threshold.
STUCK_ABSOLUTE_THRESHOLD = 0.40


@dataclass
class LyapunovResult:
    """Full output of FTLE estimation."""
    ftle: float                        # SVD-based FTLE: log(σ_max) / W
    step_growth_rate: float            # Mean log-ratio of step sizes (1D proxy)
    isotropy_ratio: float              # σ_min / σ_max: 1=isotropic, 0=collapsed to 1D
    singular_values: List[float]       # Top-3 singular values of displacement matrix
    ftle_trend: float                  # Slope of λ over recent history (>0 = worsening)
    window_size: int                   # W used for computation
    n_valid_steps: int                 # Number of steps in current window
    is_stuck: bool                     # Distances below threshold
    autocorrelation: List[float]       # Autocorr at lags 1..min(10, n/2)
    dominant_lag: int                  # Lag with highest |autocorr| (cycle period estimate)
    dominant_autocorr: float           # Value at dominant_lag
    stability_label: str               # Human-readable pre-classification
    raw_increments: List[float]        # log(d_{i+1}/d_i) values in window
    message: str                       # Diagnostic message


class LyapunovEstimator:
    """
    Computes FTLE and supporting statistics from embedding trajectory data.

    Usage:
        estimator = LyapunovEstimator(window=8)
        result = estimator.compute(distances, embeddings_matrix=embeddings)
    """

    def __init__(self, window: int = DEFAULT_WINDOW) -> None:
        self.window = window

    def compute(
        self,
        distances: List[float],
        embeddings_matrix: Optional[np.ndarray] = None,
    ) -> LyapunovResult:
        """
        Compute FTLE from embedding trajectory data.

        Args:
            distances: List of ||e_i - e_{i-1}|| values from PhaseSpaceMonitor.
            embeddings_matrix: Optional (N, 384) array of raw embeddings.
                When provided with enough rows (>= window+1), the primary FTLE
                is computed via SVD of the displacement matrix in 384-dim space.
                Falls back to step-size log-ratio if not available.

        Returns:
            LyapunovResult with full diagnostic information.
        """
        d = np.array(distances, dtype=float)

        if len(d) < 2:
            return LyapunovResult(
                ftle=0.0,
                step_growth_rate=0.0,
                isotropy_ratio=1.0,
                singular_values=[],
                ftle_trend=0.0,
                window_size=self.window,
                n_valid_steps=0,
                is_stuck=False,
                autocorrelation=[],
                dominant_lag=0,
                dominant_autocorr=0.0,
                stability_label="INSUFFICIENT_DATA",
                raw_increments=[],
                message=f"Need ≥ 2 steps; have {len(d)}.",
            )

        # ------------------------------------------------------------------
        # Stuck detection — before Lyapunov (avoids log issues)
        # ------------------------------------------------------------------
        recent_window = d[-min(self.window, len(d)):]
        is_stuck = bool(np.all(recent_window < STUCK_ABSOLUTE_THRESHOLD))

        # ------------------------------------------------------------------
        # Step-size log-ratio (1D proxy, kept as secondary metric)
        # ------------------------------------------------------------------
        d_safe = np.maximum(d, EPSILON)
        log_ratios = np.log(d_safe[1:] / d_safe[:-1])  # length = len(d) - 1
        W = min(self.window, len(log_ratios))
        recent_increments = log_ratios[-W:]
        step_growth_rate = float(recent_increments.mean()) if len(recent_increments) > 0 else 0.0

        # ------------------------------------------------------------------
        # SVD-based FTLE (primary) — uses full 384-dim displacement matrix
        # ------------------------------------------------------------------
        if (
            embeddings_matrix is not None
            and len(embeddings_matrix) >= self.window + 1
        ):
            ftle, isotropy_ratio, singular_values = _compute_svd_ftle(
                embeddings_matrix, self.window
            )
        else:
            # Fallback: step-size growth rate (backward compat)
            ftle = step_growth_rate
            isotropy_ratio = 1.0
            singular_values = []

        # ------------------------------------------------------------------
        # FTLE trend: sliding log-ratio history then linear regression
        # (cheaper than recomputing SVD at every window position)
        # ------------------------------------------------------------------
        ftle_history = _compute_ftle_history(log_ratios, self.window)
        if len(ftle_history) >= 2:
            x = np.arange(len(ftle_history), dtype=float)
            ftle_trend = float(np.polyfit(x, ftle_history, 1)[0])
        else:
            ftle_trend = 0.0

        # ------------------------------------------------------------------
        # Autocorrelation (cycle detection)
        # ------------------------------------------------------------------
        autocorr, dominant_lag, dominant_autocorr = _compute_autocorrelation(d)

        # ------------------------------------------------------------------
        # Stability label
        # ------------------------------------------------------------------
        label, message = _classify_stability(ftle, is_stuck, dominant_autocorr, dominant_lag)
        if singular_values:
            message += (
                f" [SVD: σ_max={singular_values[0]:.4f}, "
                f"isotropy={isotropy_ratio:.3f}]"
            )

        return LyapunovResult(
            ftle=ftle,
            step_growth_rate=step_growth_rate,
            isotropy_ratio=isotropy_ratio,
            singular_values=singular_values,
            ftle_trend=ftle_trend,
            window_size=W,
            n_valid_steps=len(recent_increments),
            is_stuck=is_stuck,
            autocorrelation=autocorr,
            dominant_lag=dominant_lag,
            dominant_autocorr=dominant_autocorr,
            stability_label=label,
            raw_increments=recent_increments.tolist(),
            message=message,
        )


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------

def _compute_svd_ftle(
    embeddings: np.ndarray,
    window: int,
) -> tuple[float, float, list[float]]:
    """
    Single-trajectory FTLE via SVD of the W×384 displacement matrix.

    Forms the last W displacement vectors δ_i = e_{i+1} − e_i in 384-dim
    embedding space, stacks them as matrix M (W×384), and computes:

        λ = log(σ_max(M)) / W

    where σ_max is the largest singular value of M.  This measures the
    maximum linear stretching of the trajectory over the window in the
    full embedding space — a proper single-trajectory FTLE approximation.

    Also returns:
        isotropy_ratio = σ_min / σ_max  (1 = isotropic, 0 = collapsed to 1D)
        top3_sigma     = [σ_1, σ_2, σ_3] (dominant stretching magnitudes)
    """
    n = len(embeddings)
    W = min(window, n - 1)
    if W < 1:
        return 0.0, 1.0, []

    # W+1 consecutive embeddings → W displacement vectors
    recent = embeddings[-(W + 1):]      # shape: (W+1, 384)
    deltas = np.diff(recent, axis=0)    # shape: (W, 384)

    # Economy SVD: compute only min(W, 384) = W singular values
    # O(W² × 384) ≈ O(8² × 384) ≈ 24K ops — fast
    _, sigma, _ = np.linalg.svd(deltas, full_matrices=False)

    sigma_max = float(sigma[0]) + EPSILON
    sigma_min = float(sigma[-1]) + EPSILON

    ftle = float(np.log(sigma_max) / W)
    isotropy = float(sigma_min / sigma_max)
    top3 = [float(s) for s in sigma[:3]]

    return ftle, isotropy, top3


def _compute_ftle_history(log_ratios: np.ndarray, window: int) -> List[float]:
    """Sliding step-size FTLE over the full increment sequence."""
    history = []
    for i in range(window - 1, len(log_ratios)):
        window_vals = log_ratios[i - window + 1: i + 1]
        history.append(float(window_vals.mean()))
    return history


def _compute_autocorrelation(
    distances: np.ndarray,
) -> tuple[List[float], int, float]:
    """
    Normalized autocorrelation of the distance series at lags 1..max_lag.
    Returns (autocorr_list, dominant_lag, dominant_value).
    """
    n = len(distances)
    max_lag = min(10, n // 2)
    if max_lag < 1:
        return [], 0, 0.0

    d = distances - distances.mean()
    var = float((d ** 2).mean())
    if var < EPSILON:
        return [0.0] * max_lag, 0, 0.0

    autocorr = []
    for lag in range(1, max_lag + 1):
        cov = float((d[:-lag] * d[lag:]).mean())
        autocorr.append(cov / var)

    abs_autocorr = np.abs(autocorr)
    dominant_lag = int(np.argmax(abs_autocorr)) + 1  # +1 because lags start at 1
    dominant_value = float(autocorr[dominant_lag - 1])

    return autocorr, dominant_lag, dominant_value


def _classify_stability(
    ftle: float,
    is_stuck: bool,
    dominant_autocorr: float,
    dominant_lag: int,
) -> tuple[str, str]:
    """Pre-classify dynamics from FTLE and autocorrelation."""
    if is_stuck:
        return "STUCK", "Near-zero velocity — agent is not making progress in embedding space."

    if ftle < LAMBDA_CONVERGING_STRONG:
        return (
            "CONVERGING_STRONG",
            f"λ={ftle:.3f}: Trajectory contracting rapidly toward attractor basin.",
        )
    if ftle < LAMBDA_CONVERGING_WEAK:
        return (
            "CONVERGING_WEAK",
            f"λ={ftle:.3f}: Gentle convergence — healthy iterative progress.",
        )
    if abs(dominant_autocorr) > 0.5 and dominant_lag <= 4:
        if dominant_lag == 2 and dominant_autocorr > 0.5:
            return (
                "OSCILLATING",
                f"λ={ftle:.3f}: 2-period attractor detected (autocorr={dominant_autocorr:.2f} at lag={dominant_lag}).",
            )
        return (
            "CYCLING",
            f"λ={ftle:.3f}: Periodic orbit detected (autocorr={dominant_autocorr:.2f} at lag={dominant_lag}).",
        )
    if ftle < LAMBDA_NEUTRAL_HIGH:
        return (
            "NEUTRAL",
            f"λ={ftle:.3f}: Neutral dynamics — monitoring required.",
        )
    if ftle < LAMBDA_EXPLORING_HIGH:
        return (
            "EXPLORING",
            f"λ={ftle:.3f}: Mild trajectory expansion — structured exploration.",
        )
    return (
        "DIVERGING",
        f"λ={ftle:.3f}: Trajectory is expanding — agent may be drifting from goal.",
    )
