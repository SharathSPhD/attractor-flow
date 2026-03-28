"""
demo_lorenz.py — Lorenz Attractor Demo

Educational script that visualizes the Lorenz strange attractor alongside
a simulated agent trajectory, demonstrating the analogy between chaotic
bounded exploration and productive agent exploration phases.

Usage:
    python demo_lorenz.py

Output: ASCII phase portrait + Lyapunov exponent history
"""

import math
import sys


def lorenz_step(x, y, z, dt=0.01, sigma=10.0, rho=28.0, beta=8/3):
    """One Euler step of the Lorenz system."""
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return x + dx * dt, y + dy * dt, z + dz * dt


def simulate_lorenz(n_steps=2000, dt=0.01):
    """Generate Lorenz trajectory."""
    x, y, z = 0.1, 0.0, 0.0
    trajectory = []
    for _ in range(n_steps):
        x, y, z = lorenz_step(x, y, z, dt)
        trajectory.append((x, y, z))
    return trajectory


def compute_ftle_series(trajectory, window=8):
    """Compute FTLE series from a trajectory."""
    distances = []
    for i in range(1, len(trajectory)):
        dx = trajectory[i][0] - trajectory[i-1][0]
        dy = trajectory[i][1] - trajectory[i-1][1]
        dz = trajectory[i][2] - trajectory[i-1][2]
        d = math.sqrt(dx*dx + dy*dy + dz*dz)
        distances.append(max(d, 1e-10))

    ftle_series = []
    for i in range(window - 1, len(distances) - 1):
        window_vals = distances[i - window + 1: i + 1]
        increments = [
            math.log(window_vals[j+1] / window_vals[j])
            for j in range(len(window_vals) - 1)
        ]
        ftle = sum(increments) / len(increments) if increments else 0.0
        ftle_series.append(ftle)
    return ftle_series


def ascii_phase_portrait(trajectory, width=60, height=20):
    """Render X-Y projection as ASCII art."""
    xs = [p[0] for p in trajectory]
    ys = [p[1] for p in trajectory]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    grid = [[" "] * width for _ in range(height)]

    for x, y, _ in trajectory:
        col = int((x - x_min) / (x_max - x_min + 1e-9) * (width - 1))
        row = height - 1 - int((y - y_min) / (y_max - y_min + 1e-9) * (height - 1))
        col = max(0, min(width - 1, col))
        row = max(0, min(height - 1, row))
        grid[row][col] = "·"

    lines = []
    lines.append("┌" + "─" * width + "┐")
    for row in grid:
        lines.append("│" + "".join(row) + "│")
    lines.append("└" + "─" * width + "┘")
    return "\n".join(lines)


def sparkline(values, width=60, chars="▁▂▃▄▅▆▇█"):
    """Render a list of values as a sparkline."""
    if not values:
        return ""
    values = values[-width:]
    v_min = min(values)
    v_max = max(values)
    span = v_max - v_min + 1e-9
    result = ""
    for v in values:
        idx = int((v - v_min) / span * (len(chars) - 1))
        result += chars[idx]
    return result


def main():
    print("=" * 70)
    print("AttractorFlow — Lorenz Strange Attractor Demo")
    print("Formal λ₁ ≈ +0.906 (known positive Lyapunov exponent)")
    print("=" * 70)
    print()

    print("Simulating 2000-step Lorenz trajectory...")
    traj = simulate_lorenz(n_steps=2000)

    print()
    print("Lorenz Attractor — X-Y Phase Portrait (bounded strange attractor)")
    print("Note: bounded (stays in region) but non-repeating (λ > 0)")
    print()
    print(ascii_phase_portrait(traj, width=60, height=18))
    print()

    ftle_series = compute_ftle_series(traj, window=20)
    mean_ftle = sum(ftle_series) / len(ftle_series) if ftle_series else 0
    print(f"FTLE series (last 60 values):")
    print(f"  {sparkline(ftle_series, width=60)}")
    print(f"  Mean FTLE ≈ {mean_ftle:+.3f} (theory: +0.906 for large n)")
    print()

    print("AttractorFlow Regime Classification:")
    if mean_ftle > 0.25:
        regime = "EXPLORING (strange attractor — bounded divergence)"
        symbol = "🔍"
    elif mean_ftle > 0:
        regime = "EXPLORING (mild expansion)"
        symbol = "🔍"
    elif mean_ftle > -0.05:
        regime = "NEUTRAL / CYCLING"
        symbol = "🔄"
    else:
        regime = "CONVERGING"
        symbol = "⬇️"

    print(f"  {symbol} {regime}")
    print()
    print("Interpretation for agents:")
    print("  A strange attractor in agent space means the agent is exploring")
    print("  a bounded region of solution space without repeating itself.")
    print("  This is appropriate for DESIGN phases. For IMPLEMENTATION,")
    print("  reduce temperature to drive λ negative (toward CONVERGING).")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
