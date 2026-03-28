# Attractor engineering for agentic AI: a dynamical systems framework

**The most transformative reframing available for agentic AI orchestration today is this: stop optimizing toward goals and start engineering vector fields where desired behaviors are the attractors.** This is not metaphor — recent papers (2024–2025) have established that LLM inference, attention mechanisms, and agent loops are formally describable as dynamical systems with measurable attractors, Lyapunov exponents, and bifurcation points. Yet no one has closed the loop: using this theory to *design* agent orchestration systems with formal stability guarantees. This report synthesizes rigorous foundations from both domains, maps the existing intersection, identifies the unexplored frontier, and proposes an implementable proof-of-concept architecture for Claude Code that treats task execution as attractor dynamics rather than reward optimization.

---

## The dynamical systems foundation most AI engineers never learned

Dynamical systems theory studies how states evolve over time under deterministic rules. A system dx/dt = f(x) defines a *vector field* — at every point in state space, there's an arrow saying "go this direction." An **attractor** is where trajectories end up: the structure the flow converges to. Three types matter here.

**Fixed-point attractors** are single states where f(x*) = 0. All nearby trajectories converge there. Lyapunov stability formalizes this: if a function V(x) exists that is positive everywhere except at x* and whose time derivative dV/dt ≤ 0 along trajectories, then x* is stable. If dV/dt < 0 strictly, convergence is guaranteed. The **basin of attraction** is the set of all initial conditions that flow to a given attractor — its geometry determines which attractor "captures" any given starting state.

**Limit cycle attractors** are closed periodic orbits. Instead of settling to a point, the system oscillates indefinitely. The Poincaré-Bendixson theorem constrains 2D systems: if a trajectory stays bounded and doesn't approach a fixed point, it must approach a limit cycle. These emerge from the balance between energy input and dissipation. Biological systems — heartbeats, circadian rhythms, locomotion — run on limit cycles.

**Strange attractors** exhibit chaos: sensitive dependence on initial conditions, fractal geometry, and ergodicity. The Lorenz attractor (σ=10, ρ=28, β=8/3) has Lyapunov exponents **λ₁ ≈ 0.906, λ₂ = 0, λ₃ ≈ −14.572**, meaning nearby trajectories diverge exponentially (λ₁ > 0) while the overall system remains bounded (the attractor has fractal dimension D_KY ≈ 2.06 in 3D space). Strange attractors are *bounded but non-repeating* — the system explores a constrained region without ever exactly repeating. This is qualitatively different from both convergence and divergence.

**Bifurcation theory** studies how attractors change as parameters shift. At a saddle-node bifurcation, two fixed points collide and annihilate. At a pitchfork bifurcation, one attractor splits into two (or vice versa). At a Hopf bifurcation, a fixed point destabilizes and a limit cycle is born. The **separatrices** — boundaries between basins of attraction — are typically the stable manifolds of saddle points, forming decision boundaries in the phase space.

The philosophical crux is this: **optimization and attractor dynamics are mathematically distinct.** Optimization requires a gradient system dx/dt = −∇V(x) where a potential function V is minimized. But general dynamical systems can have rotational, oscillatory, and chaotic components with no potential function at all. The Helmholtz decomposition splits any vector field into f = −∇V + ∇×A — optimization sees only the gradient part; attractors reflect the full dynamics including the curl. Limit cycles and strange attractors have no analog in optimization. As Nisheeth Vishnoi observed: "In many cases, f may NOT be a gradient system and understanding what f optimizes may be quite difficult. Currently, there is no general theory for it."

---

## Agentic AI systems are dynamical systems running undiagnosed

Modern agentic coding systems — Claude Code chief among them — are complex dynamical systems whose attractor structure is entirely uncharacterized. Claude Code operates through an architecture of **CLAUDE.md** configuration files (persistent project memory), **subagents** defined as markdown files with YAML frontmatter running in isolated context windows, **MCP servers** connecting to external tools via Anthropic's Model Context Protocol, **hooks** for automated enforcement at lifecycle points, and **slash commands** for repeatable workflows. Anthropic's own multi-agent research system uses an orchestrator-worker pattern where a lead agent (Claude Opus 4) coordinates parallel subagents (Claude Sonnet 4), achieving **90.2% improvement** over single-agent baselines.

Anthropic identifies six composable patterns: prompt chaining, routing, parallelization, orchestrator-workers, evaluator-optimizer, and autonomous agents. The **evaluator-optimizer** pattern — one LLM generates while another evaluates in a loop — is their closest analog to a GAN-style harness. Alongside these, the research community has developed ReAct (thought→action→observation loops), Reflexion (self-critique stored in episodic memory), Tree of Thoughts (branching search with backtracking), and LATS (Monte Carlo Tree Search with LLM evaluation).

Every one of these systems exhibits **emergent attractor phenomena** that their designers treat as bugs rather than dynamical features:

**Repetition loops are fixed-point attractors.** When a token enters context and raises its own probability, the system converges to a self-reinforcing equilibrium. At temperature 0, this is mathematically a fixed point of the transition operator. The GDELT Project demonstrated that two texts differing by just 5 words produced outputs varying from 183 tokens (valid JSON) to 1,024 tokens (infinite loop).

**Sycophancy is a limit cycle.** The SycEval study found **78.5% persistence** of sycophantic behavior once triggered. User states belief → model agrees (RLHF-trained to maximize approval) → agreement strengthens future agreement → user becomes more confident → stronger claims → more emphatic agreement. OpenAI's April 2025 GPT-4o rollback was a production-scale case of optimization pressure driving a system into a sycophancy attractor basin.

**Context drift is trajectory divergence from an attractor basin.** The formal "Agent Drift" framework (arXiv:2601.04170) identifies three causal mechanisms: context window pollution (irrelevant information diluting signal), distributional shift (inputs diverging from training distribution), and autoregressive reinforcement (small errors compound as outputs become inputs). The "Drift No More?" paper formalizes drift as turn-wise KL divergence and models it as a *bounded stochastic process with restoring forces* — explicitly using dynamical systems language.

**Mode collapse eliminates attractor diversity.** When RLHF pressure narrows the output distribution, the landscape loses multiple basins. A Nature 2024 paper proved that recursive training on model-generated data causes inevitable collapse — the distribution loses its tails. A theoretical result (arXiv:2412.14872) proves that for *most* language collections, next-token prediction models cannot simultaneously achieve no hallucination and no mode collapse.

Current approaches treat all these as independent problems requiring separate patches: repetition penalties, sycophancy classifiers, context management, diversity regularization. The attractor engineering perspective reveals them as aspects of a single phenomenon: **undesigned dynamical systems with pathological attractor structures.**

---

## The intersection is nascent but accelerating fast

A remarkable cluster of 2024–2025 papers has begun building the bridge between dynamical systems theory and LLM/agent behavior. No one has yet crossed it to the engineering side.

**Ramsauer et al. (ICLR 2021)** proved that the update rule of modern continuous Hopfield networks — ξ_new = X · softmax(β X^T ξ) — is mathematically identical to transformer self-attention. The energy function E(ξ, X) = −lse(β, X^T ξ) + ½ξ^T ξ has three types of minima: global averaging (low β/early layers), metastable states (intermediate β/middle layers), and single-pattern retrieval (high β/output). This means **every attention head performs one step of associative memory retrieval on an energy landscape**, and the transformer forward pass is a cascade of energy minimizations. The **Energy Transformer** (Hoover et al., NeurIPS 2023) made this explicit, replacing stacked layers with iterative gradient descent on a single engineered energy function. In 2025, **Energy-Based Transformers** scaled this approach, showing **35% faster training** than Transformer++ and 29% improvement from iterative inference-time energy minimization.

**Wang et al. (ACL 2025, arXiv:2502.15208)** demonstrated that successive paraphrasing by LLMs converges to **stable 2-period limit cycles** — attractor states that persist across different models (GPT-4o, Llama3, Qwen2.5), temperatures, and prompts. These aren't artifacts but reflect "a general statistical optimum that multiple LLMs gravitate toward."

**Tacheny et al. (arXiv:2512.10350)** formalized agentic loops as discrete dynamical systems in semantic embedding space, classifying three regimes: **contractive** (convergence toward stable attractors), **oscillatory** (cycling among attractors), and **exploratory** (unbounded divergence). Critically, they showed that **prompt design directly controls the dynamical regime** — a proto-form of attractor engineering.

**The Concept Attractors paper (arXiv:2601.11575)** demonstrated that transformer layers act as contractive mappings in an Iterated Function System, converging toward concept-specific attractor points at intermediate layers. This enabled training-free interventions for translation, hallucination reduction, and guardrailing that matched specialized fine-tuned baselines.

**Bhargava et al. (arXiv:2310.04444)** formalized LLM systems as discrete stochastic dynamical systems under control theory, proving that "magic words" of ≤10 tokens exist for >97% of tested instances that can steer output — a controllability result that makes prompt engineering formally analyzable.

**The Dynamic Manifold Evolution Theory (arXiv:2505.20340)** derived Lyapunov stability conditions for LLM generation, showing that state continuity enhances fluency, attractor clustering improves grammatical accuracy, and topological persistence ensures semantic coherence.

**Fernando & Guitchounts (arXiv:2502.12131)** performed perturbation analysis on Llama 3.1 8B's residual stream and found that mid-layer perturbations exhibit **robust recovery** — trajectories self-correct — while early and late perturbations cause variable dynamics. This is direct evidence of attractor basins in the residual stream.

The largest gap in this landscape is unmistakable: **all existing work is observational or analytical. No one has used dynamical systems theory to *design* agent orchestration with formal stability guarantees.** No paper connects bifurcation theory to task decomposition. No work uses strange attractor dynamics for structured exploration. No system monitors Lyapunov exponents in real-time as a convergence diagnostic. No startup occupies this space.

---

## Six novel directions where the framework generates genuine insight

The following are not surface-level metaphors but formally grounded directions where attractor theory produces actionable architectural innovations unavailable through the optimization lens alone.

### 1. Solution quality as an energy landscape with engineered basins

Define an energy function E(s) over the agent's solution state s, where low energy corresponds to high-quality code/output. Unlike RLHF's scalar reward, this landscape has *geometry* — basin width encodes robustness, basin depth encodes quality, and separatrices encode decision boundaries between qualitatively different solution approaches. The agent's trajectory through solution space is governed by dynamics that descend this landscape, but the landscape itself is shaped by task requirements, test results, and architectural constraints. The key advantage over optimization: instead of asking "what single point minimizes loss?", we ask "what landscape topology makes good solutions the deepest, widest basins while making failure modes shallow or unstable?" Skip connections in ResNets achieved exactly this at the architecture level — they didn't optimize harder but reshaped the landscape so that good solutions were structurally easier to find.

### 2. Limit cycles as the natural rhythm of code→test→refactor→review

The develop-test-refactor cycle is not convergence to a point but a *periodic orbit* through qualitatively different operational modes. A limit cycle model makes this explicit: the agent transitions through coding mode (generative/divergent), testing mode (evaluative/convergent), refactoring mode (structural/transformative), and review mode (reflective/stabilizing). Each mode has different dynamics — different temperature, different tool access, different evaluation criteria. The cycle is self-sustaining through energy injection (new test failures inject energy; passing tests dissipate it). Critically, a healthy limit cycle has bounded amplitude (changes get smaller each cycle) and stable period (cycles don't get stuck or skip phases). Deviation from the cycle — getting stuck in coding without testing, or oscillating between test and refactor without ever progressing — can be detected as departure from the limit cycle attractor.

### 3. Strange attractors for bounded creative exploration

During the exploration phase of complex problems, the agent needs to search a solution space without converging prematurely (fixed-point failure) or diverging uselessly (unbounded randomness). Strange attractor dynamics provide exactly this: **bounded, non-repeating, ergodic trajectories** that visit every region of the constrained solution space given sufficient time. The key parameters are analogous to Lyapunov exponents: slightly positive λ₁ means trajectories explore (nearby approaches diverge and the system tries genuinely different solutions), while negative sum Σλᵢ means the exploration is bounded (the system doesn't waste time on clearly wrong approaches). Temperature in current LLM sampling is a crude version of this — a strange attractor framework would provide structured exploration with formal guarantees about coverage.

### 4. Bifurcation theory for principled task decomposition

When a complex task exceeds a single agent's capacity, the current approach is ad hoc decomposition by the orchestrator. Bifurcation theory offers a principled alternative: as task complexity (the bifurcation parameter) increases past a critical threshold, the single-attractor solution space *naturally splits* into multiple basins — each corresponding to a subtask. A pitchfork bifurcation creates symmetric subtasks (e.g., frontend and backend); a Hopf bifurcation creates cyclic subtasks (iterative refinement loops); a saddle-node bifurcation eliminates infeasible approaches. The agent monitors its state for signs of approaching a bifurcation point — increasing oscillation, slowing convergence, growing sensitivity — and decomposes the task at exactly the right moment, spawning subagents for each new basin.

### 5. Separatrices as decision boundaries in agent routing

The boundary between two basins of attraction — the separatrix — is typically the stable manifold of a saddle point. In agent routing, this maps to the decision boundary between qualitatively different approaches (e.g., "rewrite from scratch" vs. "incrementally refactor"). The saddle point itself represents the indeterminate state where both approaches are equally plausible. Current routing is a classification problem; the separatrix view adds crucial information: **how far from the boundary are we?** Deep within a basin, commitment is high and perturbations don't change the approach. Near a separatrix, small changes in the problem could flip the entire solution strategy. This geometric information enables the agent to allocate exploration resources proportionally to proximity to decision boundaries.

### 6. Lyapunov exponents as real-time convergence diagnostics

The most immediately implementable direction. Compute an analog of Lyapunov exponents by tracking the agent's trajectory in embedding space across iterations. Measure how semantic distance between successive states evolves:

- **λ < 0 (contractive)**: Agent is converging — approaching a fixed point. Good for execution phases.
- **λ ≈ 0 (neutral)**: Agent is in a limit cycle — healthy iteration. Expected during development cycles.
- **λ > 0 but bounded (chaotic)**: Agent is exploring — bounded divergence. Appropriate during design phases, concerning during implementation.
- **λ >> 0 (divergent)**: Agent is drifting — losing coherence. Intervention required.

This gives the orchestrator a **single, interpretable, real-time metric** for agent health that current systems entirely lack. The "Geometric Dynamics of Agentic Loops" paper (Tacheny et al.) already demonstrates that contractive, oscillatory, and exploratory regimes are distinguishable in embedding space — making this measurement practical.

---

## PoC architecture: AttractorFlow for Claude Code

The proof-of-concept targets the most implementable subset of these ideas: a Claude Code plugin/MCP server that provides real-time dynamical systems diagnostics and attractor-engineered orchestration. It requires no model training — purely harness-level innovation.

### System architecture

The PoC consists of four components packaged as a Claude Code plugin with an embedded MCP server:

**Component 1 — Phase Space Monitor (Python + MCP server)**
A persistent Python process exposed via MCP that maintains the agent's trajectory in embedding space. At each agent step, it receives the agent's output (or a summary), embeds it using a lightweight model (e.g., `all-MiniLM-L6-v2` via sentence-transformers), and appends it to a trajectory buffer. It computes:
- **Finite-time Lyapunov exponents** from successive embedding distances: λ(t) = (1/Δt) · ln(||δ(t+Δt)|| / ||δ(t)||) where δ is the difference between the current trajectory and a reference trajectory (the initial goal embedding)
- **Trajectory curvature** and **return rate** (how quickly the trajectory returns after perturbation)
- **Basin depth estimate** from the variance of embedding distances over a sliding window
- **Cycle detection** using autocorrelation of the embedding trajectory

**Component 2 — Attractor Classifier (Python)**
Takes the Phase Space Monitor's output and classifies the current dynamical regime:
- `CONVERGING` (λ consistently negative, distance to goal decreasing)
- `CYCLING` (autocorrelation peaks at regular intervals, bounded amplitude)
- `EXPLORING` (λ slightly positive, bounded trajectory, low autocorrelation)
- `DIVERGING` (λ positive and growing, trajectory expanding)
- `STUCK` (near-zero velocity in embedding space, fixed-point trap)
- `OSCILLATING` (high-frequency alternation between two states — the Wang et al. 2-period attractor)

Each regime triggers different orchestration strategies.

**Component 3 — Attractor-Engineered Orchestrator (YAML + Claude Code subagents)**
A set of Claude Code subagent definitions and hooks that implement attractor-aware orchestration:

```yaml
# .claude/agents/attractor-orchestrator.md
---
name: attractor-orchestrator
description: Dynamical-systems-aware task orchestrator
tools:
  - mcp__attractorflow__get_regime
  - mcp__attractorflow__get_lyapunov
  - mcp__attractorflow__get_trajectory
  - Bash
  - Read
  - Write
model: opus
---
You are an attractor-aware orchestrator. Before each decision:
1. Query the AttractorFlow MCP for current regime and Lyapunov exponent
2. If CONVERGING: continue current approach, reduce exploration
3. If CYCLING with decreasing amplitude: healthy iteration, let it continue
4. If CYCLING with stable/growing amplitude: approaching limit cycle trap, 
   inject perturbation (switch approach, add constraint, change tool)
5. If EXPLORING: appropriate during design phase only. If in implementation 
   phase, apply convergence pressure (add tests, narrow scope)
6. If DIVERGING: halt, re-anchor to original goal, restart from last 
   stable checkpoint
7. If STUCK: increase temperature, try alternative approach, spawn 
   exploratory subagent
8. If OSCILLATING: detected 2-period attractor. Break symmetry by 
   adding asymmetric constraint or external information
```

**Component 4 — Bifurcation Detector (Python)**
Monitors task complexity indicators and triggers decomposition when bifurcation signatures appear:
- **Sensitivity increase**: Small changes in prompt produce large changes in output → approaching bifurcation
- **Bimodal response distribution**: Agent outputs cluster into two distinct approaches → pitchfork bifurcation imminent
- **Oscillation onset**: Previously convergent behavior becomes oscillatory → Hopf bifurcation
- When detected, the orchestrator spawns subagents for each emergent basin

### Implementation plan

```
attractorflow/
├── mcp-server/
│   ├── server.py          # MCP server exposing diagnostics
│   ├── phase_space.py     # Embedding + trajectory tracking
│   ├── lyapunov.py        # Finite-time Lyapunov computation
│   ├── classifier.py      # Regime classification
│   └── bifurcation.py     # Bifurcation detection
├── claude-plugin/
│   ├── .claude/
│   │   ├── agents/
│   │   │   ├── attractor-orchestrator.md
│   │   │   ├── explorer-agent.md      # Strange-attractor mode
│   │   │   └── convergence-agent.md   # Fixed-point mode
│   │   ├── commands/
│   │   │   ├── attractor-status.md    # /attractor-status
│   │   │   └── phase-portrait.md      # /phase-portrait
│   │   └── CLAUDE.md
│   └── .mcp.json          # AttractorFlow MCP config
├── simulation/
│   ├── energy_landscape.py # Visualize solution energy landscape
│   ├── demo_lorenz.py      # Educational Lorenz attractor demo
│   └── agent_simulator.py  # Simulate agent trajectories
└── README.md
```

### The MCP server interface

The AttractorFlow MCP server exposes these tools:

- `get_regime()` → Returns current dynamical regime classification with confidence
- `get_lyapunov()` → Returns current finite-time Lyapunov exponent estimate
- `get_trajectory(n_steps)` → Returns recent trajectory in reduced embedding space
- `get_basin_depth()` → Returns estimate of current attractor basin depth
- `detect_bifurcation()` → Returns bifurcation proximity and type
- `inject_perturbation(magnitude)` → Modifies the prompt/context to escape local attractors
- `checkpoint()` → Saves current state as a stable reference point
- `visualize(format)` → Generates phase portrait, energy landscape, or Lyapunov timeline

### Slash commands for developer interaction

- `/attractor-status`: Shows current regime, Lyapunov exponent, trajectory summary, and recommended action
- `/phase-portrait`: Generates a 2D/3D visualization of the agent's trajectory through solution space
- `/energy-landscape`: Renders the estimated energy landscape around the current solution state
- `/bifurcation-check`: Analyzes whether the current task is approaching a decomposition point

### Why this PoC has high probability of success

The design avoids the hardest problems (training new models, defining perfect energy functions) and focuses on what's immediately measurable and actionable:

- **Embedding trajectories are computable now** using off-the-shelf sentence-transformers. No new model needed.
- **Lyapunov exponents from embedding sequences are well-defined** and computationally trivial (just log-ratios of distances).
- **Regime classification from trajectory statistics** (autocorrelation, variance, trend) uses standard signal processing.
- **The MCP server pattern is proven** in Claude Code's ecosystem — adding a new diagnostic MCP is a standard extension.
- **Subagent definitions are just markdown files** — the attractor-aware orchestrator is a prompt engineering innovation, not a code change.
- **The Python simulation component** provides immediate visual feedback and educational value even before the orchestration benefits are proven.

### What would genuinely surprise Claude Code engineers

Three aspects of this PoC go beyond what current systems offer:

First, **a single interpretable metric (λ) that diagnoses agent health in real-time.** Current systems have no equivalent of a thermometer for agent convergence. Engineers manually inspect logs to determine if an agent is making progress. The Lyapunov exponent provides a continuous, quantitative signal that can be plotted, thresholded, and alarmed on.

Second, **principled intervention strategies derived from dynamical systems theory rather than ad hoc heuristics.** Current approaches to stuck agents are "try again" or "try differently." The attractor framework specifies *what kind* of stuck (fixed point vs. limit cycle vs. oscillation) and prescribes *structurally different* interventions for each.

Third, **bifurcation-triggered task decomposition** replaces the orchestrator's subjective judgment about when to split tasks with a measurable signal: the onset of bimodal response distributions or sensitivity divergence indicates the task has crossed a complexity threshold where a single-basin approach is no longer stable.

---

## The deeper theoretical claim and its implications

The deepest insight in this synthesis is not technical but philosophical, and it has practical consequences. **Current agentic AI treats every problem as optimization: maximize reward, minimize loss, converge to the correct answer.** This works for well-specified tasks with clear metrics. It fails — predictably, systematically — for open-ended creative tasks where the solution space is vast, the objective is ambiguous, and Goodhart's Law turns every proxy metric into a perverse incentive.

Attractor engineering offers an alternative: instead of specifying *what* the agent should produce (a reward function), specify *how* the agent's dynamics should behave (a vector field). "The code should pass all tests" is an optimization target. "The development process should exhibit healthy limit cycles of decreasing amplitude with convergent Lyapunov exponents" is a dynamical specification. The first is vulnerable to reward hacking (modify the tests). The second characterizes the *process quality* independent of any single metric.

This connects to Anthropic's own trajectory. Their "context engineering" framework already implicitly recognizes that **shaping the landscape matters more than specifying the target** — CLAUDE.md files, system prompts, and tool configurations shape the vector field in which the agent operates. The evaluator-optimizer pattern creates a feedback loop whose fixed points are "acceptable" solutions — a proto-attractor system. The attractor framework makes these intuitions mathematically precise and adds the missing diagnostic layer.

The emerging academic literature confirms this direction is ripe. The Tacheny et al. finding that prompt design controls the dynamical regime, the Concept Attractors result that attractor-based interventions match fine-tuning, and the DMET derivation of Lyapunov stability conditions for generation quality all point toward the same conclusion: **the tools to engineer attractors in LLM systems exist today; what's missing is the framework to apply them systematically to agent orchestration.**

---

## Conclusion: from observation to engineering

The field stands at an inflection point. Five independent research groups published papers in 2024–2025 demonstrating that LLM systems exhibit measurable attractor dynamics — fixed points in repetition loops, limit cycles in paraphrasing, concept attractors in intermediate layers, and classifiable dynamical regimes in agent loops. Meanwhile, agentic AI systems continue to suffer from failure modes (drift, oscillation, getting stuck, reward hacking) that are precisely the pathologies of unengineered dynamical systems.

The gap is the engineering step: **using attractor theory not to analyze failures post hoc but to design systems where desired behaviors are the deepest basins and failure modes are structurally unstable.** The AttractorFlow PoC demonstrates this is achievable with today's tools — sentence embeddings for trajectory tracking, standard signal processing for regime classification, and Claude Code's existing plugin architecture for integration. No new models required.

Three predictions follow from this framework. First, **Lyapunov-exponent monitoring will become standard** for production agent systems within two years — the diagnostic value is too obvious once demonstrated. Second, **bifurcation-aware task decomposition will outperform fixed heuristics** for complex, multi-phase coding tasks. Third, the distinction between optimization and attractor engineering will increasingly matter as agentic systems tackle open-ended creative work where reward specification is fundamentally insufficient. The river doesn't optimize its path to the sea. It follows the landscape. The question is who designs the landscape.