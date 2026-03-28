# AttractorFlow Plugin — Session Upgrades (2026-03-28)

Two major improvements were made in this session:
1. **Dynamics fidelity** — replaced 1D heuristic proxies with true 384-dim geometry
2. **Zero-setup MCP** — replaced manual venv with `uv run` + PEP 723 inline deps

---

## 1. Dynamics Fidelity Upgrade

### Background

The original implementation collapsed the 384-dim sentence embedding to a scalar distance series immediately, then named the resulting heuristics after rigorous dynamical systems concepts (FTLE, PITCHFORK, SADDLE_NODE) they did not actually implement. Seven gaps were identified and fixed.

---

### 1.1 SVD-based FTLE (`lyapunov.py`)

**Before:** FTLE was the mean of scalar log-ratios of consecutive distances — a 1D approximation with no access to the embedding geometry.

**After:** FTLE is computed from the SVD of the W×384 displacement matrix:

```python
def _compute_svd_ftle(embeddings: np.ndarray, window: int):
    W = min(window, len(embeddings) - 1)
    recent = embeddings[-(W + 1):]       # (W+1) × 384
    deltas = np.diff(recent, axis=0)     # W × 384 displacement matrix M
    _, sigma, _ = np.linalg.svd(deltas, full_matrices=False)
    ftle = float(np.log(sigma[0] + EPSILON) / W)   # log(σ_max) / W
    isotropy = float((sigma[-1] + EPSILON) / (sigma[0] + EPSILON))  # σ_min/σ_max
    return ftle, isotropy, sigma[:3].tolist()
```

**Why:** `log(σ_max(M)) / W` is a legitimate single-trajectory FTLE approximation — σ_max measures the maximum linear stretching direction in the full 384-dim embedding space over the window.

**New fields on `LyapunovResult`:**
- `ftle` — now SVD-based (primary)
- `step_growth_rate` — old log-ratio mean (kept as secondary diagnostic)
- `isotropy_ratio` — σ_min/σ_max; 1.0 = isotropic exploration, ~0 = trajectory collapsed to 1D
- `singular_values` — top-3 σ values for diagnostics

**`compute()` signature:** `compute(distances, embeddings_matrix=None)` — falls back to log-ratio if embeddings not provided (backward compatible).

---

### 1.2 384-dim PITCHFORK detection (`bifurcation.py`)

**Before:** k-means ran on the 2D PCA projection of embeddings — silhouette score computed in 2D lost most cluster separation information.

**After:** k-means runs directly in 384-dim embedding space:

```python
X = embeddings_matrix  # N × 384
kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
labels = kmeans.fit_predict(X)
sil_score = float(silhouette_score(X, labels))   # silhouette in 384-dim
# PCA only used for centroid visualization output:
centroids_2d = PCA(n_components=2).fit(X).transform(kmeans.cluster_centers_)
```

**Why:** Silhouette score in 384-dim is far more discriminative. Two semantically distinct clusters that are inseparable in 2D PCA are clearly separated in the full space.

---

### 1.3 Condition-number SADDLE_NODE (`bifurcation.py`)

**Before:** SADDLE_NODE was detected via variance of FTLE values — not a geometric instability signal.

**After:** Condition number of the W×384 displacement matrix:

```python
recent = embeddings_matrix[-(W + 1):]
deltas = np.diff(recent, axis=0)         # W × 384
_, sigma, _ = np.linalg.svd(deltas, full_matrices=False)
cond = sigma_pos[0] / sigma_pos[-1]      # σ_max / σ_min
# High cond → displacement vectors nearly collinear → trajectory near-1D
```

**Why:** High condition number means all recent steps explore only one semantic direction — the geometric precursor to trajectory stall. This is not a strict saddle-node bifurcation (which requires two fixed points to collide); it is an instability signal that reliably detects trajectory degeneration before the agent fully stalls. The description string was updated accordingly.

---

### 1.4 HOPF self-monitoring (`bifurcation.py`)

**Before:** HOPF detection required `regime_history` to be populated — it only fired if `get_regime()` had been called frequently enough.

**After:** HOPF fires from FTLE history alone:

```python
W = self.hopf_window
prev_ftle = np.array(ftle_history[-(W * 2):-W])
curr_ftle = np.array(ftle_history[-W:])
prev_converging = bool(np.any(prev_ftle < -0.05))
now_cycling = bool(abs(curr_ftle.mean()) < 0.05)
ftle_detected = prev_converging and now_cycling
```

**Why:** In real agent traces, `get_regime()` is often called infrequently (every 5–10 steps). HOPF must detect the CONVERGING → CYCLING transition regardless.

---

### 1.5 Session persistence (`phase_space.py`)

**Before:** All trajectory state was in-memory only — lost on MCP server restart.

**After:** Auto-save on every `record()` call; auto-load on server startup:

```python
PERSIST_PATH = Path.home() / ".attractorflow" / "session.json"
MAX_SESSION_AGE_SECONDS = 86400  # 24h stale guard

def save(self) -> None:
    # Serializes buffer + goal_embedding + step_counter to JSON
    # Best-effort — never raises

def load(self) -> bool:
    # Restores from disk if file exists and is < 24h old
    # Silent fallback on corrupt/stale file
```

The `lifespan` in `server.py` calls `_monitor.load()` on startup.

---

### 1.6 Wang citation removal (`classifier.py`)

**Before:** The OSCILLATING regime rationale cited "Wang et al. 2025" — a paper that was not the actual source of the heuristic.

**After:** Replaced with an accurate description:

```python
"(Empirical: lag-1 anticorrelation characteristic of 2-period limit cycles in LLM generation.)"
```

---

### 1.7 Embeddings wired through `server.py`

All tool handlers now pass `embeddings_matrix` to the Lyapunov estimator and bifurcation detector:

```python
embeddings = _monitor.get_embeddings_matrix()
lya = _lyapunov.compute(distances, embeddings_matrix=embeddings)
result = _bifurcation.analyze(embeddings, _regime_history, _ftle_history)
```

`get_lyapunov` now returns `step_growth_rate`, `isotropy_ratio`, and `singular_values` in its response.

---

### 1.8 Tests (`attractorflow/mcp-server/tests/test_dynamics.py`)

17 unit tests covering all upgraded components. Run with:

```bash
cd attractorflow/mcp-server
python -m pytest tests/test_dynamics.py -v
```

All 17 pass. Test categories:
- `TestSVDFtle` (6 tests) — FTLE sign, isotropy bounds, fallback
- `TestPitchfork384` (3 tests) — two-cluster detected, single-cluster not, centroid shape
- `TestSaddleNode` (3 tests) — degenerate detected, isotropic not, evidence string
- `TestHopfSelfMonitoring` (2 tests) — fires from FTLE only, no false positive
- `TestPersistence` (3 tests) — round-trip, stale rejection, corrupt fallback

---

## 2. Zero-Setup MCP via `uv run`

### Background

The original MCP setup required a manual step — `python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt` — inside the plugin cache directory for every new machine or install location. This is unlike Node-based MCP servers (e.g. `npx -y @modelcontextprotocol/server-xyz`) which auto-install on first use.

### What changed

**`server.py`** — PEP 723 inline script metadata added at the top:

```python
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
```

**`.mcp.json`** — replaced venv-python invocation with `uv run`:

```json
{
  "mcpServers": {
    "attractorflow_mcp": {
      "command": "sh",
      "args": ["-c", "PATH=\"$PATH:$HOME/.local/bin\" uv run \"${CLAUDE_PLUGIN_ROOT:-.}/attractorflow/mcp-server/server.py\""],
      "description": "..."
    }
  }
}
```

**How the shell command works:**
- `PATH="$PATH:$HOME/.local/bin"` — ensures `uv` is found regardless of shell config (uv installs to `~/.local/bin`)
- `${CLAUDE_PLUGIN_ROOT:-.}` — when loaded from the plugin cache, `CLAUDE_PLUGIN_ROOT` is set to the cache directory; when loaded as a project MCP, it falls back to `.` (the project root / CWD)
- `uv run server.py` — reads the `# /// script` block, auto-installs deps into `~/.cache/uv/`, and runs the server

**Runtime behaviour:**
- **First run:** `uv` downloads and installs all dependencies (~30s, done once globally into `~/.cache/uv/`)
- **Subsequent runs:** instant — deps are cached by uv
- **No `venv/` directory in the plugin or project is needed or created**

### Prerequisite for new machines

`uv` must be installed once:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

This is a one-time global install (~5MB binary). After that, any project that uses this plugin or any other `uv run`-based MCP works without further setup.

---

## Applying these changes to a forked/derived project

If you have a project based on this plugin and want to pick up these upgrades:

```bash
# 1. Pull the latest from the attractor-flow upstream
git remote add attractor-flow https://github.com/SharathSPhD/attractor-flow.git
git fetch attractor-flow main

# 2. Cherry-pick or merge the relevant commits
#    Key commits:
#    - Dynamics upgrade: attractorflow/mcp-server/ changes
#    - uv run: .mcp.json + server.py script header
git merge attractor-flow/main  # or cherry-pick specific commits

# 3. Install uv if not already present
curl -LsSf https://astral.sh/uv/install.sh | sh

# 4. Test (no venv needed)
cd attractorflow/mcp-server
PATH="$PATH:$HOME/.local/bin" uv run server.py &
python -m pytest tests/test_dynamics.py -v
```

If you only want the `uv run` simplification without the dynamics changes, the minimum diff is:
1. Add the `# /// script` block to your `server.py`
2. Update `.mcp.json` to use `sh -c` with `uv run` as shown above
3. Install `uv`
