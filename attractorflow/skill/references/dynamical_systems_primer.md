# Dynamical Systems Primer for Agentic AI

Reference for understanding the formal foundations behind AttractorFlow.

## Attractors

A dynamical system dx/dt = f(x) defines a vector field. An **attractor** is a set A
such that trajectories starting near A remain near A and converge to it as t → ∞.

**Fixed-point attractor:** f(x*) = 0. Single converging point.
- Agent analog: agent reaches a stable final answer and stops changing
- Lyapunov function V(x): V > 0 everywhere, dV/dt < 0 guarantees convergence
- Basin of attraction: all starting states that converge to x*

**Limit cycle:** Closed periodic orbit. System oscillates indefinitely.
- Agent analog: healthy dev loop (write → test → refactor → review → write...)
- Stable if nearby trajectories spiral inward to the cycle
- Period = cycle length (e.g., 3-step iteration = period-3 limit cycle)

**Strange attractor:** Bounded, non-repeating, fractal geometry.
- Agent analog: bounded exploration of solution space without premature convergence
- Key property: nearby trajectories diverge (λ > 0) but remain bounded
- Appropriate during design/exploration phases

**2-Period Attractor (Wang et al. 2025):** LLMs specifically converge to
alternating 2-state orbits under paraphrasing/reformulation pressure.
These are stable across models (GPT-4o, Llama3, Qwen2.5), temperatures, and prompts.
AttractorFlow detects this via autocorrelation peak at lag=2.

## Lyapunov Stability

For fixed-point x*: a Lyapunov function V(x) proves stability if:
1. V(x*) = 0, V(x) > 0 for x ≠ x*
2. dV/dt ≤ 0 along trajectories (V decreases or stays same)
3. dV/dt < 0 strictly → asymptotic stability (guaranteed convergence)

**Finite-Time Lyapunov Exponent (FTLE):**
λ(t) = (1/W) × Σ ln(||δ(i+1)|| / ||δ(i)||)

where δ(i) = e_i - e_{i-1} is the step in embedding space and W is window size.

This measures whether nearby trajectories are converging (λ < 0) or
diverging (λ > 0) in the current window.

## Bifurcations

**Saddle-node bifurcation:** As parameter μ crosses μ_c, two fixed points
(one stable, one unstable) collide and disappear. The system loses its attractor.
- Agent analog: a previously viable approach suddenly has no stable solution

**Pitchfork bifurcation:** One fixed point splits into two (symmetry breaking).
- Agent analog: task becomes complex enough that single-basin approach fails;
  two distinct subtask approaches emerge

**Hopf bifurcation:** Stable fixed point loses stability; limit cycle is born.
- Agent analog: direct convergent process becomes iterative (TDD cycles emerge
  when codebase becomes complex enough to require test→fix loops)

## The Helmholtz Decomposition

Any vector field f = -∇V + ∇×A (gradient + curl components).
- Optimization sees only -∇V (gradient descent)
- True dynamics include ∇×A (rotational, cyclic, limit cycle components)
- Limit cycles and strange attractors have NO analog in optimization
- This is why reward optimization cannot model healthy iteration cycles

## Ramsauer et al. Connection (ICLR 2021)

Transformer self-attention is mathematically identical to modern Hopfield
network updates with energy E(ξ, X) = -lse(β, X^T ξ) + ½ξ^T ξ.

Each attention head minimizes energy → forward pass = cascade of energy minimizations.
Three energy landscape regimes by β (inverse temperature):
- Low β (early layers): global averaging (smooth basin)
- Medium β (middle layers): metastable states (multiple shallow basins)
- High β (output): single-pattern retrieval (deep narrow basin)

This means the attractor geometry exists *inside* transformer computation,
not just in the token output sequence. AttractorFlow's embedding-space
analysis is measuring projections of this internal geometry.
