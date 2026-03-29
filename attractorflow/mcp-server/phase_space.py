"""
phase_space.py — Phase Space Monitor for AttractorFlow

Maintains the agent's trajectory in sentence-embedding space.
Each recorded state is embedded and appended to a fixed-size FIFO buffer.
Provides distance series and raw trajectory data to upstream estimators.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import os as _os

import numpy as np


def _resolve_persist_path() -> Path:
    """
    Compute session file path from environment variables.

    Priority:
      ATTRACTORFLOW_SESSION_ID=<name>  →  ~/.attractorflow/sessions/<name>.json
      (default)                         →  ~/.attractorflow/session.json

    Set ATTRACTORFLOW_DISABLE_PERSISTENCE=1 to skip all disk I/O without
    changing the resolved path (useful for benchmark / CI mode).
    """
    session_id = _os.environ.get("ATTRACTORFLOW_SESSION_ID", "").strip()
    if session_id:
        return Path.home() / ".attractorflow" / "sessions" / f"{session_id}.json"
    return Path.home() / ".attractorflow" / "session.json"


# Computed once at import — env vars must be set before the process starts.
PERSIST_PATH = _resolve_persist_path()
MAX_SESSION_AGE_SECONDS = int(_os.environ.get("ATTRACTORFLOW_SESSION_MAX_AGE_SECONDS", "86400"))

# Lazy import — loaded once at server start via lifespan
_model = None
EMBEDDING_DIM = 384
BUFFER_CAPACITY = 100  # keep last 100 states
GOAL_ANCHOR: Optional[np.ndarray] = None  # optional reference embedding


@dataclass
class StateRecord:
    """One recorded agent step."""
    text: str
    embedding: np.ndarray  # shape (384,)
    timestamp: float = field(default_factory=time.time)
    step_index: int = 0


@dataclass
class TrajectoryStats:
    """Derived statistics over the current trajectory buffer."""
    n_steps: int
    distances: List[float]          # d_i = ||e_i - e_{i-1}||
    mean_distance: float
    std_distance: float
    min_distance: float
    max_distance: float
    distance_trend: float           # slope of linear regression on distances (> 0 = diverging)
    goal_distances: List[float]     # ||e_i - goal|| if goal set, else []
    pca_2d: List[Tuple[float, float]]  # 2D projection for visualization


class PhaseSpaceMonitor:
    """
    Manages the agent state trajectory in embedding space.

    Usage:
        monitor = PhaseSpaceMonitor()
        monitor.load_model()           # call once at startup
        monitor.record("agent output text")
        stats = monitor.get_stats()
    """

    def __init__(self, capacity: int = BUFFER_CAPACITY) -> None:
        self._buffer: Deque[StateRecord] = deque(maxlen=capacity)
        self._step_counter: int = 0
        self._goal_embedding: Optional[np.ndarray] = None
        self._checkpoint: Optional[List[StateRecord]] = None

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Load the sentence-transformers model (call once at server startup)."""
        global _model
        if _model is None:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer("all-MiniLM-L6-v2")

    # ------------------------------------------------------------------
    # State recording
    # ------------------------------------------------------------------

    def record(self, text: str) -> StateRecord:
        """Embed text and append to trajectory buffer."""
        if _model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        embedding = _model.encode(text, normalize_embeddings=True)
        record = StateRecord(
            text=text,
            embedding=embedding,
            step_index=self._step_counter,
        )
        self._buffer.append(record)
        self._step_counter += 1
        self.save()  # persist after every new state (best-effort)
        return record

    def set_goal(self, goal_text: str) -> None:
        """Set an anchor embedding for the intended goal."""
        if _model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        self._goal_embedding = _model.encode(goal_text, normalize_embeddings=True)

    def checkpoint(self) -> None:
        """Save current buffer as a stable checkpoint."""
        self._checkpoint = list(self._buffer)

    def restore_checkpoint(self) -> bool:
        """Restore trajectory buffer from last checkpoint."""
        if self._checkpoint is None:
            return False
        self._buffer.clear()
        for rec in self._checkpoint:
            self._buffer.append(rec)
        return True

    def clear(self) -> None:
        """Clear trajectory for a new session."""
        self._buffer.clear()
        self._step_counter = 0
        self._goal_embedding = None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """
        Persist trajectory buffer to disk (best-effort — never raises).
        Saved to PERSIST_PATH (default: ~/.attractorflow/session.json).
        No-op when ATTRACTORFLOW_DISABLE_PERSISTENCE=1.
        """
        if _os.environ.get("ATTRACTORFLOW_DISABLE_PERSISTENCE", "").lower() in ("1", "true", "yes"):
            return
        try:
            data = {
                "version": 1,
                "saved_at": time.time(),
                "step_counter": self._step_counter,
                "goal_embedding": (
                    self._goal_embedding.tolist()
                    if self._goal_embedding is not None
                    else None
                ),
                "states": [
                    {
                        "text": r.text,
                        "embedding": r.embedding.tolist(),
                        "timestamp": r.timestamp,
                        "step_index": r.step_index,
                    }
                    for r in self._buffer
                ],
            }
            PERSIST_PATH.parent.mkdir(parents=True, exist_ok=True)
            PERSIST_PATH.write_text(json.dumps(data))
        except Exception:
            pass  # persistence is best-effort; never crash the server

    def load(self) -> bool:
        """
        Load trajectory buffer from disk on startup.

        Returns True if data was restored, False otherwise.
        Silently ignores corrupt or stale files.
        No-op when ATTRACTORFLOW_DISABLE_PERSISTENCE=1.
        """
        if _os.environ.get("ATTRACTORFLOW_DISABLE_PERSISTENCE", "").lower() in ("1", "true", "yes"):
            return False
        if not PERSIST_PATH.exists():
            return False
        try:
            data = json.loads(PERSIST_PATH.read_text())
            saved_at = float(data.get("saved_at", 0))
            if time.time() - saved_at > MAX_SESSION_AGE_SECONDS:
                return False
            self._step_counter = int(data.get("step_counter", 0))
            goal = data.get("goal_embedding")
            if goal is not None:
                self._goal_embedding = np.array(goal, dtype=np.float32)
            for s in data.get("states", []):
                rec = StateRecord(
                    text=s["text"],
                    embedding=np.array(s["embedding"], dtype=np.float32),
                    timestamp=float(s["timestamp"]),
                    step_index=int(s["step_index"]),
                )
                self._buffer.append(rec)
            return len(self._buffer) > 0
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_embeddings(self) -> np.ndarray:
        """Return trajectory as (N, 384) array."""
        if not self._buffer:
            return np.empty((0, EMBEDDING_DIM))
        return np.stack([r.embedding for r in self._buffer])

    def get_embeddings_matrix(self) -> np.ndarray:
        """Return trajectory as (N, 384) array (explicit alias for get_embeddings)."""
        return self.get_embeddings()

    def get_distance_series(self) -> List[float]:
        """Successive L2 distances between consecutive embeddings."""
        embeddings = self.get_embeddings()
        if len(embeddings) < 2:
            return []
        diffs = embeddings[1:] - embeddings[:-1]
        return np.linalg.norm(diffs, axis=1).tolist()

    def get_goal_distances(self) -> List[float]:
        """Distances from each state to the goal embedding."""
        if self._goal_embedding is None:
            return []
        embeddings = self.get_embeddings()
        if len(embeddings) == 0:
            return []
        diffs = embeddings - self._goal_embedding[np.newaxis, :]
        return np.linalg.norm(diffs, axis=1).tolist()

    def get_pca_2d(self) -> List[Tuple[float, float]]:
        """Project trajectory to 2D via PCA for visualization."""
        embeddings = self.get_embeddings()
        n = len(embeddings)
        if n < 2:
            return [(0.0, 0.0)] * n
        if n == 2:
            # Special case: PCA with 2 points
            mean = embeddings.mean(axis=0)
            centered = embeddings - mean
            direction = centered[0] / (np.linalg.norm(centered[0]) + 1e-8)
            projs = centered @ direction
            return [(float(p), 0.0) for p in projs]
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            coords = pca.fit_transform(embeddings)
            return [(float(x), float(y)) for x, y in coords]
        except ImportError:
            # Fallback: first 2 dims
            return [(float(e[0]), float(e[1])) for e in embeddings]

    def get_stats(self) -> TrajectoryStats:
        """Compute full trajectory statistics."""
        distances = self.get_distance_series()
        n = len(self._buffer)

        if not distances:
            return TrajectoryStats(
                n_steps=n,
                distances=[],
                mean_distance=0.0,
                std_distance=0.0,
                min_distance=0.0,
                max_distance=0.0,
                distance_trend=0.0,
                goal_distances=self.get_goal_distances(),
                pca_2d=self.get_pca_2d(),
            )

        d = np.array(distances)
        # Linear regression for trend
        x = np.arange(len(d), dtype=float)
        if len(d) >= 2:
            slope = float(np.polyfit(x, d, 1)[0])
        else:
            slope = 0.0

        return TrajectoryStats(
            n_steps=n,
            distances=distances,
            mean_distance=float(d.mean()),
            std_distance=float(d.std()),
            min_distance=float(d.min()),
            max_distance=float(d.max()),
            distance_trend=slope,
            goal_distances=self.get_goal_distances(),
            pca_2d=self.get_pca_2d(),
        )

    @property
    def n_steps(self) -> int:
        return self._step_counter

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    @property
    def has_goal(self) -> bool:
        return self._goal_embedding is not None

    @property
    def has_checkpoint(self) -> bool:
        return self._checkpoint is not None
