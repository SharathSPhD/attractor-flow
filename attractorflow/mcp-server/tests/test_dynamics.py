"""
test_dynamics.py — Unit tests for upgraded AttractorFlow dynamics fidelity.

Tests verify:
  1. SVD-based FTLE gives negative λ for contracting trajectories
  2. SVD-based FTLE gives positive λ for expanding trajectories
  3. SVD and log-ratio metrics agree on sign for clear cases
  4. PITCHFORK detected via 384-dim k-means for two-cluster trajectories
  5. PITCHFORK not detected for single-cluster random walk
  6. SADDLE_NODE detected when displacement matrix is near-degenerate
  7. Session persistence round-trip (save → load)
  8. HOPF self-monitoring fires from FTLE history alone (no regime_history)
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

# Add the mcp-server directory to sys.path for local imports
MCP_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(MCP_DIR))

from lyapunov import LyapunovEstimator, _compute_svd_ftle, EPSILON
from bifurcation import BifurcationDetector, BifurcationType
from classifier import Regime


# ------------------------------------------------------------------
# Fixtures: synthetic embedding trajectories
# ------------------------------------------------------------------

def make_contracting_embeddings(n=20, dim=384, seed=42):
    """
    Trajectory that spirals inward: each step is a scaled copy of the previous
    with the scale factor < 1. Displacement vectors shrink over time.
    → SVD σ_max decreasing → λ < 0
    """
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((1, dim))
    base /= np.linalg.norm(base)
    embeddings = [base.copy()]
    direction = rng.standard_normal(dim)
    direction /= np.linalg.norm(direction)
    scale = 0.9
    for i in range(1, n):
        step_size = 0.5 * (scale ** i)
        new = embeddings[-1] + step_size * direction
        new /= np.linalg.norm(new)
        embeddings.append(new)
    return np.stack([e.flatten() for e in embeddings])


def make_expanding_embeddings(n=20, dim=384, seed=42):
    """
    Random walk with growing step sizes.
    → SVD σ_max increasing → λ > 0
    """
    rng = np.random.default_rng(seed)
    embeddings = [rng.standard_normal(dim)]
    embeddings[0] /= np.linalg.norm(embeddings[0])
    for i in range(1, n):
        step_size = 0.1 * (1.1 ** i)
        step = rng.standard_normal(dim) * step_size
        new = embeddings[-1] + step
        new /= np.linalg.norm(new)
        embeddings.append(new)
    return np.stack(embeddings)


def make_two_cluster_embeddings(n=30, dim=384, seed=42):
    """
    Two tight clusters alternating in trajectory order.
    → k-means silhouette should be high → PITCHFORK detected
    """
    rng = np.random.default_rng(seed)
    center_a = rng.standard_normal(dim)
    center_a /= np.linalg.norm(center_a)
    center_b = -center_a + 0.05 * rng.standard_normal(dim)
    center_b /= np.linalg.norm(center_b)

    embeddings = []
    for i in range(n):
        center = center_a if i % 2 == 0 else center_b
        jitter = 0.01 * rng.standard_normal(dim)
        e = center + jitter
        e /= np.linalg.norm(e)
        embeddings.append(e)
    return np.stack(embeddings)


def make_single_cluster_embeddings(n=30, dim=384, seed=42):
    """
    Slow random walk — roughly one cluster.
    → silhouette score should be low → no PITCHFORK
    """
    rng = np.random.default_rng(seed)
    embeddings = [rng.standard_normal(dim)]
    embeddings[0] /= np.linalg.norm(embeddings[0])
    for _ in range(1, n):
        step = 0.05 * rng.standard_normal(dim)
        new = embeddings[-1] + step
        new /= np.linalg.norm(new)
        embeddings.append(new)
    return np.stack(embeddings)


def make_degenerate_embeddings(n=10, dim=384, seed=42):
    """
    Trajectory moving almost exclusively in one direction (near-1D).
    → displacement matrix is rank-1-like → high condition number → SADDLE_NODE
    """
    rng = np.random.default_rng(seed)
    direction = rng.standard_normal(dim)
    direction /= np.linalg.norm(direction)
    embeddings = []
    base = rng.standard_normal(dim)
    base /= np.linalg.norm(base)
    for i in range(n):
        # Move purely in 'direction' with tiny orthogonal noise
        e = base + i * 0.1 * direction + 1e-4 * rng.standard_normal(dim)
        e /= np.linalg.norm(e)
        embeddings.append(e)
    return np.stack(embeddings)


# ------------------------------------------------------------------
# Tests: SVD FTLE
# ------------------------------------------------------------------

class TestSVDFtle:

    def test_contracting_trajectory_gives_negative_ftle(self):
        """Spiraling inward → σ_max shrinks → log(σ_max)/W < 0."""
        embeddings = make_contracting_embeddings(n=20)
        ftle, isotropy, sigma = _compute_svd_ftle(embeddings, window=8)
        assert ftle < 0, f"Expected negative FTLE for contracting trajectory, got {ftle}"

    def test_expanding_trajectory_gives_positive_ftle(self):
        """Growing random walk → σ_max grows → log(σ_max)/W > 0."""
        embeddings = make_expanding_embeddings(n=20)
        ftle, isotropy, sigma = _compute_svd_ftle(embeddings, window=8)
        assert ftle > 0, f"Expected positive FTLE for expanding trajectory, got {ftle}"

    def test_isotropy_ratio_between_0_and_1(self):
        """σ_min ≤ σ_max so isotropy ∈ (0, 1]."""
        embeddings = make_single_cluster_embeddings(n=20)
        _, isotropy, _ = _compute_svd_ftle(embeddings, window=8)
        assert 0.0 < isotropy <= 1.0 + 1e-6, f"Isotropy out of range: {isotropy}"

    def test_degenerate_trajectory_has_low_isotropy(self):
        """Near-1D trajectory → σ_min ≪ σ_max → isotropy near 0."""
        embeddings = make_degenerate_embeddings(n=10)
        _, isotropy, _ = _compute_svd_ftle(embeddings, window=5)
        assert isotropy < 0.1, f"Expected low isotropy for 1D trajectory, got {isotropy}"

    def test_svd_and_logratio_sign_agree_for_clear_cases(self):
        """Both metrics should have the same sign on clearly contracting/expanding trajectories."""
        estimator = LyapunovEstimator(window=8)

        # Contracting
        emb = make_contracting_embeddings(n=20)
        dists = np.linalg.norm(np.diff(emb, axis=0), axis=1).tolist()
        result = estimator.compute(dists, embeddings_matrix=emb)
        assert result.ftle < 0, f"SVD FTLE should be negative: {result.ftle}"
        assert result.step_growth_rate < 0, f"Log-ratio should be negative: {result.step_growth_rate}"

        # Expanding
        emb = make_expanding_embeddings(n=20)
        dists = np.linalg.norm(np.diff(emb, axis=0), axis=1).tolist()
        result = estimator.compute(dists, embeddings_matrix=emb)
        assert result.ftle > 0, f"SVD FTLE should be positive: {result.ftle}"
        assert result.step_growth_rate > 0, f"Log-ratio should be positive: {result.step_growth_rate}"

    def test_fallback_when_no_embeddings(self):
        """Without embeddings_matrix, should fall back to log-ratio (no crash)."""
        estimator = LyapunovEstimator(window=8)
        dists = [0.5, 0.6, 0.5, 0.6, 0.5, 0.6, 0.5, 0.6, 0.5, 0.6]
        result = estimator.compute(dists)  # no embeddings_matrix
        assert isinstance(result.ftle, float)
        assert result.singular_values == []


# ------------------------------------------------------------------
# Tests: PITCHFORK (384-dim k-means)
# ------------------------------------------------------------------

class TestPitchfork384:

    def test_two_clusters_detected(self):
        """Tight two-cluster trajectory → silhouette high → PITCHFORK detected."""
        embeddings = make_two_cluster_embeddings(n=30)
        detector = BifurcationDetector(bimodal_threshold=0.5)
        result = detector._check_pitchfork(embeddings)
        assert result.detected, (
            f"Expected PITCHFORK for two-cluster trajectory. "
            f"Evidence: {result.evidence}"
        )
        assert result.bifurcation_type == BifurcationType.PITCHFORK

    def test_single_cluster_not_detected(self):
        """Slow random walk → roughly one cluster → no PITCHFORK."""
        embeddings = make_single_cluster_embeddings(n=30)
        detector = BifurcationDetector(bimodal_threshold=0.5)
        result = detector._check_pitchfork(embeddings)
        assert not result.detected, (
            f"Expected no PITCHFORK for random walk. Evidence: {result.evidence}"
        )

    def test_cluster_centroids_are_2d(self):
        """Centroids returned for visualization should be (x, y) pairs."""
        embeddings = make_two_cluster_embeddings(n=30)
        detector = BifurcationDetector(bimodal_threshold=0.5)
        result = detector._check_pitchfork(embeddings)
        if result.detected:
            for centroid in result.cluster_centroids:
                assert len(centroid) == 2


# ------------------------------------------------------------------
# Tests: SADDLE_NODE (condition number)
# ------------------------------------------------------------------

class TestSaddleNode:

    def test_degenerate_trajectory_detected(self):
        """Near-1D displacement → high condition number → SADDLE_NODE detected."""
        embeddings = make_degenerate_embeddings(n=10)
        detector = BifurcationDetector(cond_threshold=10.0)
        result = detector._check_saddle_node(embeddings)
        assert result.detected, (
            f"Expected SADDLE_NODE for degenerate trajectory. Evidence: {result.evidence}"
        )
        assert result.bifurcation_type == BifurcationType.SADDLE_NODE

    def test_isotropic_trajectory_not_detected(self):
        """Random walk (isotropic displacements) → low condition number → no SADDLE_NODE."""
        embeddings = make_single_cluster_embeddings(n=30)
        detector = BifurcationDetector(cond_threshold=20.0)
        result = detector._check_saddle_node(embeddings)
        assert not result.detected, (
            f"Expected no SADDLE_NODE for isotropic walk. Evidence: {result.evidence}"
        )

    def test_evidence_contains_condition_number(self):
        """Evidence string should report the actual condition number."""
        embeddings = make_degenerate_embeddings(n=10)
        detector = BifurcationDetector(cond_threshold=5.0)
        result = detector._check_saddle_node(embeddings)
        assert "condition number" in result.evidence.lower() or "cond" in result.evidence.lower()


# ------------------------------------------------------------------
# Tests: HOPF self-monitoring
# ------------------------------------------------------------------

class TestHopfSelfMonitoring:

    def test_hopf_fires_from_ftle_history_alone(self):
        """
        HOPF should fire when FTLE history shows CONVERGING → CYCLING transition,
        even when regime_history is empty.
        """
        W = 5
        # Previous window: clearly converging (λ < -0.05)
        prev_ftle = [-0.15, -0.12, -0.18, -0.10, -0.14]
        # Recent window: neutral/cycling (|λ| < 0.05)
        curr_ftle = [0.01, -0.02, 0.03, -0.01, 0.02]
        ftle_history = prev_ftle + curr_ftle

        detector = BifurcationDetector(hopf_window=W)
        result = detector._check_hopf(regime_history=[], ftle_history=ftle_history)
        assert result.detected, (
            f"Expected HOPF from FTLE history alone (no regime history). "
            f"Evidence: {result.evidence}"
        )

    def test_hopf_does_not_fire_without_transition(self):
        """If always converging, no HOPF."""
        ftle_history = [-0.15] * 12  # always strongly converging
        detector = BifurcationDetector(hopf_window=5)
        result = detector._check_hopf(regime_history=[], ftle_history=ftle_history)
        assert not result.detected


# ------------------------------------------------------------------
# Tests: Persistence round-trip
# ------------------------------------------------------------------

class TestPersistence:

    def test_save_and_load_roundtrip(self, tmp_path, monkeypatch):
        """
        After recording states, save(), then load() on a fresh monitor
        should restore the same number of steps.
        """
        import phase_space as ps

        # Redirect persistence to temp dir
        tmp_file = tmp_path / "session.json"
        monkeypatch.setattr(ps, "PERSIST_PATH", tmp_file)

        # Create a monitor and populate it with synthetic data
        monitor = ps.PhaseSpaceMonitor(capacity=50)
        # Manually insert StateRecords (bypass model loading)
        rng = np.random.default_rng(0)
        for i in range(10):
            rec = ps.StateRecord(
                text=f"step {i}",
                embedding=rng.standard_normal(384).astype(np.float32),
                timestamp=time.time(),
                step_index=i,
            )
            monitor._buffer.append(rec)
            monitor._step_counter = i + 1

        # Save
        monitor.save()
        assert tmp_file.exists(), "Persist file should have been created."

        # Load into a fresh monitor
        monitor2 = ps.PhaseSpaceMonitor(capacity=50)
        restored = monitor2.load()

        assert restored, "load() should return True when data was restored."
        assert monitor2.buffer_size == monitor.buffer_size, (
            f"Buffer size mismatch: {monitor2.buffer_size} != {monitor.buffer_size}"
        )
        assert monitor2._step_counter == monitor._step_counter

    def test_stale_file_not_loaded(self, tmp_path, monkeypatch):
        """Files older than 24h should not be loaded."""
        import phase_space as ps

        tmp_file = tmp_path / "session.json"
        monkeypatch.setattr(ps, "PERSIST_PATH", tmp_file)

        old_data = {
            "version": 1,
            "saved_at": time.time() - 90000,  # 25h ago
            "step_counter": 5,
            "goal_embedding": None,
            "states": [],
        }
        tmp_file.write_text(json.dumps(old_data))

        monitor = ps.PhaseSpaceMonitor()
        restored = monitor.load()
        assert not restored, "Stale file should not be restored."

    def test_corrupt_file_silent_fallback(self, tmp_path, monkeypatch):
        """Corrupt JSON should not raise — silent fallback to empty buffer."""
        import phase_space as ps

        tmp_file = tmp_path / "session.json"
        monkeypatch.setattr(ps, "PERSIST_PATH", tmp_file)
        tmp_file.write_text("this is not valid json {{{")

        monitor = ps.PhaseSpaceMonitor()
        restored = monitor.load()  # should not raise
        assert not restored
        assert monitor.buffer_size == 0


# ------------------------------------------------------------------
# TestInjectPerturbation — Issues 2: missing embeddings_matrix arg
# ------------------------------------------------------------------

class TestInjectPerturbation:
    """
    Verify that the inject_perturbation tool correctly uses the SVD FTLE path.
    Tests use synthetic StateRecord objects injected directly into the monitor
    buffer to avoid loading the SentenceTransformer model.
    """

    def _make_monitor(self, n=12, seed=42):
        """Return a PhaseSpaceMonitor with n synthetic embeddings already in buffer."""
        from phase_space import PhaseSpaceMonitor, StateRecord
        monitor = PhaseSpaceMonitor()
        rng = np.random.default_rng(seed)
        for i in range(n):
            monitor._buffer.append(StateRecord(
                text=f"step {i}",
                embedding=rng.standard_normal(384),
                step_index=i,
            ))
        return monitor

    def test_with_embeddings_produces_singular_values(self):
        """SVD path: compute() with embeddings_matrix returns non-empty singular_values."""
        from lyapunov import LyapunovEstimator
        monitor = self._make_monitor()
        distances = monitor.get_distance_series()
        embeddings = monitor.get_embeddings_matrix()
        lya = LyapunovEstimator(window=8).compute(distances, embeddings_matrix=embeddings)
        assert lya.singular_values != [], "SVD path must produce non-empty singular_values"
        assert lya.isotropy_ratio >= 0.0

    def test_without_embeddings_has_empty_singular_values(self):
        """Fallback path: compute() without embeddings_matrix returns empty singular_values."""
        from lyapunov import LyapunovEstimator
        monitor = self._make_monitor()
        distances = monitor.get_distance_series()
        lya = LyapunovEstimator(window=8).compute(distances)  # no embeddings_matrix
        assert lya.singular_values == [], "Fallback must have empty singular_values"

    def test_plateau_strategy_differs_from_stuck_radical(self):
        """PLATEAU perturbation must use a different strategy than STUCK at high magnitude."""
        from server import _generate_perturbation
        from classifier import Regime
        plateau_result = _generate_perturbation(Regime.PLATEAU, magnitude=0.2)
        stuck_radical  = _generate_perturbation(Regime.STUCK,   magnitude=0.9)
        assert plateau_result["strategy"] != stuck_radical["strategy"], (
            f"PLATEAU strategy '{plateau_result['strategy']}' must differ from "
            f"STUCK radical '{stuck_radical['strategy']}'"
        )
        assert len(plateau_result["text"]) > 20


# ------------------------------------------------------------------
# TestPlateauClassifier — Issue 3: PLATEAU vs STUCK distinction
# ------------------------------------------------------------------

class TestPlateauClassifier:
    """
    Verify that the PLATEAU regime is correctly distinguished from STUCK.
    PLATEAU: is_stuck=True AND distance_trend < -0.01 (agent drifting toward goal)
    STUCK:   is_stuck=True AND distance_trend >= -0.01 (agent genuinely stalled)
    """

    def _make_lya(self):
        from lyapunov import LyapunovResult
        return LyapunovResult(
            ftle=-0.02, step_growth_rate=-0.02, isotropy_ratio=0.8,
            singular_values=[0.3, 0.1, 0.05], ftle_trend=0.0, window_size=8,
            n_valid_steps=8, is_stuck=True, autocorrelation=[0.05, 0.02],
            dominant_lag=1, dominant_autocorr=0.05, stability_label="STUCK",
            raw_increments=[-0.01] * 8, message="test",
        )

    def _make_stats(self, distance_trend: float):
        from phase_space import TrajectoryStats
        return TrajectoryStats(
            n_steps=12, distances=[0.3] * 12, mean_distance=0.3,
            std_distance=0.01, min_distance=0.29, max_distance=0.31,
            distance_trend=distance_trend, goal_distances=[], pca_2d=[],
        )

    def test_plateau_vs_stuck_classification(self):
        """Negative distance_trend + is_stuck → PLATEAU; flat trend + is_stuck → STUCK."""
        from classifier import AttractorClassifier, Regime
        clf = AttractorClassifier()
        plateau = clf.classify(self._make_lya(), self._make_stats(-0.05))
        stuck   = clf.classify(self._make_lya(), self._make_stats(0.0))
        assert plateau.regime == Regime.PLATEAU, \
            f"Expected PLATEAU (is_stuck + negative trend), got {plateau.regime}"
        assert stuck.regime == Regime.STUCK, \
            f"Expected STUCK (is_stuck + flat trend), got {stuck.regime}"


# ------------------------------------------------------------------
# TestEdgeCases — Issue 3 follow-up: degenerate inputs must not crash
# ------------------------------------------------------------------

class TestEdgeCases:
    """
    Very short trajectories, degenerate embeddings, and repeated identical
    states must not raise exceptions or produce NaN / ±inf FTLE values.
    The EPSILON guard in lyapunov.py is the key safety net under test here.
    """

    def test_single_state_no_crash(self):
        """Buffer with one state has no distances; compute() must still return a finite result."""
        from lyapunov import LyapunovEstimator
        from phase_space import PhaseSpaceMonitor, StateRecord
        monitor = PhaseSpaceMonitor()
        monitor._buffer.append(StateRecord(
            text="only", embedding=np.ones(384) / np.sqrt(384), step_index=0
        ))
        lya = LyapunovEstimator(window=8).compute(monitor.get_distance_series())
        assert np.isfinite(lya.ftle), f"Single-state FTLE should be finite, got {lya.ftle}"

    def test_two_state_trajectory_no_crash(self):
        """Two states produce one distance; window > available data must not crash."""
        from lyapunov import LyapunovEstimator
        from phase_space import PhaseSpaceMonitor, StateRecord
        rng = np.random.default_rng(0)
        monitor = PhaseSpaceMonitor()
        for i in range(2):
            monitor._buffer.append(StateRecord(
                text=str(i), embedding=rng.standard_normal(384), step_index=i
            ))
        lya = LyapunovEstimator(window=8).compute(monitor.get_distance_series())
        assert np.isfinite(lya.ftle), f"Two-state FTLE should be finite, got {lya.ftle}"

    def test_repeated_identical_states_finite_ftle(self):
        """
        Identical embeddings → L2 distances all 0.
        In the log-ratio path: log((0 + ε) / (0 + ε)) = log(1) = 0 → FTLE = 0.
        The agent should be classified as is_stuck (mean_distance < STUCK_ABSOLUTE_THRESHOLD).
        """
        from lyapunov import LyapunovEstimator
        from phase_space import PhaseSpaceMonitor, StateRecord
        fixed = np.ones(384) / np.sqrt(384)
        monitor = PhaseSpaceMonitor()
        for i in range(10):
            monitor._buffer.append(StateRecord(
                text="same", embedding=fixed.copy(), step_index=i
            ))
        lya = LyapunovEstimator(window=8).compute(monitor.get_distance_series())
        assert np.isfinite(lya.ftle), f"Repeated-identical FTLE should be finite, got {lya.ftle}"
        assert lya.is_stuck, "All-identical states must trigger is_stuck=True"

    def test_all_zeros_embedding_svd_no_crash(self):
        """
        All-zero embeddings (pathological but possible in synthetic tests).
        SVD of a zero displacement matrix should give σ_max ≈ 0 → log(ε)/W < 0.
        Must not crash or produce NaN.
        """
        from lyapunov import LyapunovEstimator
        from phase_space import PhaseSpaceMonitor, StateRecord
        monitor = PhaseSpaceMonitor()
        for i in range(10):
            monitor._buffer.append(StateRecord(
                text=str(i), embedding=np.zeros(384), step_index=i
            ))
        distances = monitor.get_distance_series()
        embeddings = monitor.get_embeddings_matrix()
        lya = LyapunovEstimator(window=8).compute(distances, embeddings_matrix=embeddings)
        assert np.isfinite(lya.ftle), f"All-zeros SVD FTLE should be finite, got {lya.ftle}"


# ------------------------------------------------------------------
# TestSessionScoping — env-var driven session isolation
# ------------------------------------------------------------------

class TestSessionScoping:
    """
    Verify that ATTRACTORFLOW_SESSION_ID and ATTRACTORFLOW_DISABLE_PERSISTENCE
    are honoured without requiring a server restart.

    _resolve_persist_path() reads os.environ at call time, so monkeypatch.setenv
    works correctly without reloading the module.
    """

    def test_session_id_routes_to_named_file(self, monkeypatch):
        """ATTRACTORFLOW_SESSION_ID=<name> → path contains sessions/<name>.json."""
        monkeypatch.setenv("ATTRACTORFLOW_SESSION_ID", "task-abc-123")
        import phase_space as ps
        path = ps._resolve_persist_path()
        assert "task-abc-123" in str(path), f"Expected 'task-abc-123' in path, got {path}"
        assert "sessions" in str(path), f"Expected 'sessions' subdir in path, got {path}"

    def test_disable_persistence_skips_save(self, monkeypatch, tmp_path):
        """ATTRACTORFLOW_DISABLE_PERSISTENCE=1 → save() writes nothing to disk."""
        monkeypatch.setenv("ATTRACTORFLOW_DISABLE_PERSISTENCE", "1")
        import phase_space as ps
        # Redirect PERSIST_PATH to a temp location so we can assert no file appears
        monkeypatch.setattr(ps, "PERSIST_PATH", tmp_path / "session.json")
        monitor = ps.PhaseSpaceMonitor()
        rng = np.random.default_rng(0)
        monitor._buffer.append(ps.StateRecord(
            text="x", embedding=rng.standard_normal(384), step_index=0
        ))
        monitor.save()
        assert not (tmp_path / "session.json").exists(), \
            "save() must be a no-op when ATTRACTORFLOW_DISABLE_PERSISTENCE=1"
