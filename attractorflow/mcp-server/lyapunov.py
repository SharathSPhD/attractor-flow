"""
lyapunov.py — Finite-Time Lyapunov Exponent (FTLE) Estimator

Computes trajectory stability metrics from the distance series produced
by PhaseSpaceMonitor.  The FTLE approximation used here:

    λ(t) = (1/W) × Σ_{i=t-W}^{t-1} ln(d_{i+1} / d_i)

where d_i = ||e_i - e_{i-1}|| is the embedding step distance and W is
the sliding window width.

Interpretation:
    λ < -0.2   → strongly contractive (CONVERGING)
    λ ∈ [-0.2, -0.05] → weakly contractive (healthy iteration)
    λ ∈ [-0.05, 0.05]  → neutral / limit-cycle candidate
    λ ∈ [0.05, 0.25]   → mildly expanding (EXPLORING)
    λ > 0.25           → strongly expanding (DIVERGING)

The estimator also computes autocorrelation to distinguish cyclic from
exploring dynamics (both can have near-zero λ).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


# Thresholds for regime pre-classification (refined by classifier.py)
LAMBDA_CONVERGING_STRONG = -0.2
LAMBDA_CONVERGING_WEAK = -0.05
LAMBDA_NEUTRAL_HIGH = 0.05
LAMBDA_EXPLORING_HIGH = 0.25

# Minimum distances to avoid log(0)
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
    ftle: float                        # current finite-time Lyapunov exponent
    ftle_trend: float                  # slope of λ over recent history (>0 = worsening)
    window_size: int                   # W used for computation
    n_valid_steps: int                 # number of steps in current window
    is_stuck: bool                     # distances below threshold
    autocorrelation: List[float]       # autocorr at lags 1..min(10, n/2)
    dominant_lag: int                  # lag with highest |autocorr| (cycle period estimate)
    dominant_autocorr: float           # value at dominant_lag
    stability_label: str               # human-readable pre-classification
    raw_increments: List[float]        # log(d_{i+1}/d_i) values in window
    message: str                       # diagnostic message


class LyapunovEstimator:
    """
    Computes FTLE and supporting statistics from a distance series.

    Usage:
        estimator = LyapunovEstimator(window=8)
        result = estimator.compute(distances)
    """

    def __init__(self, window: int = DEFAULT_WINDOW) -> None:
        self.window = window

    def compute(self, distances: List[float]) -> LyapunovResult:
        """
        Compute FTLE from the distance series.

        Args:
            distances: List of ||e_i - e_{i-1}|| values from PhaseSpaceMonitor.

        Returns:
            LyapunovResult with full diagnostic information.
        """
        d = np.array(distances, dtype=float)

        if len(d) < 2:
            return LyapunovResult(
                ftle=0.0,
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
        # Relative threshold: recent distances all below 50% of session mean
        # AND below an absolute cap. This calibrates correctly for normalized
        # MiniLM embeddings where typical inter-step distances are 0.5–1.4.
        # ------------------------------------------------------------------
        recent_window = d[-min(self.window, len(d)):]
        is_stuck = bool(np.all(recent_window < STUCK_ABSOLUTE_THRESHOLD))

        # ------------------------------------------------------------------
        # Lyapunov increments: log(d_{i+1} / d_i)
        # Clamp denominators to avoid log(0)
        # ------------------------------------------------------------------
        d_safe = np.maximum(d, EPSILON)
        log_ratios = np.log(d_safe[1:] / d_safe[:-1])  # length = len(d) - 1

        # Take the last W increments
        W = min(self.window, len(log_ratios))
        recent_increments = log_ratios[-W:]
        ftle = float(recent_increments.mean()) if len(recent_increments) > 0 else 0.0

        # ------------------------------------------------------------------
        # FTLE trend: compute λ at each step in a sliding fashion, then regress
        # ------------------------------------------------------------------
        ftle_history = _compute_ftle_history(log_ratios, self.window)
        if len(ftle_history) >= 2:
            x = np.arange(len(ftle_history), dtype=float)
            ftle_trend = float(np.polyfit(x, ftle_history, 1)[0])
        else:
            ftle_trend = 0.0

        # ------------------------------------------------------------------
        # Autocorrelation (to detect cyclic dynamics)
        # ------------------------------------------------------------------
        autocorr, dominant_lag, dominant_autocorr = _compute_autocorrelation(d)

        # ------------------------------------------------------------------
        # Stability label
        # ------------------------------------------------------------------
        label, message = _classify_stability(ftle, is_stuck, dominant_autocorr, dominant_lag)

        return LyapunovResult(
            ftle=ftle,
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

def _compute_ftle_history(log_ratios: np.ndarray, window: int) -> List[float]:
    """Sliding FTLE over the full increment sequence."""
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
