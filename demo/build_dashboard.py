#!/usr/bin/env python3
"""build_dashboard.py — Generate demo/index.html from demo/results.json"""

import json
import os
import math

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results.json")
OUT_PATH = os.path.join(os.path.dirname(__file__), "index.html")

with open(RESULTS_PATH) as f:
    results = json.load(f)


REGIME_COLORS = {
    "CONVERGING":  ("#22c55e", "#dcfce7", "✓"),  # green
    "STUCK":       ("#ef4444", "#fee2e2", "✗"),  # red
    "OSCILLATING": ("#f59e0b", "#fef3c7", "⟳"),  # amber
    "DIVERGING":   ("#f97316", "#ffedd5", "↗"),  # orange
    "EXPLORING":   ("#3b82f6", "#dbeafe", "◉"),  # blue
    "PLATEAU":     ("#a855f7", "#f3e8ff", "≈"),  # purple
    "CYCLING":     ("#06b6d4", "#cffafe", "↻"),  # cyan
    "UNKNOWN":     ("#6b7280", "#f3f4f6", "?"),  # gray
}

ACTION_LABELS = {
    "CONTINUE":             ("", "#6b7280"),
    "REDUCE_TEMPERATURE":   ("Apply Convergence Pressure", "#22c55e"),
    "INJECT_PERTURBATION":  ("Inject Perturbation", "#ef4444"),
    "SPAWN_EXPLORER":       ("Spawn Explorer Agent", "#f97316"),
    "RESTORE_CHECKPOINT":   ("Restore Checkpoint", "#f97316"),
    "BREAK_SYMMETRY":       ("Break Symmetry", "#f59e0b"),
    "DECOMPOSE_TASK":       ("Decompose Task", "#3b82f6"),
    "HALT":                 ("Halt & Escalate", "#7c3aed"),
    "NUDGE":                ("Nudge (gentle push)", "#a855f7"),
}


def sparkline_svg(lambdas, width=160, height=40):
    """Generate an inline SVG sparkline for the λ series."""
    if len(lambdas) < 2:
        return ""
    mn = min(lambdas)
    mx = max(lambdas)
    rng = mx - mn or 0.001
    pts = []
    for i, v in enumerate(lambdas):
        x = 4 + (i / (len(lambdas) - 1)) * (width - 8)
        y = (height - 4) - ((v - mn) / rng) * (height - 8)
        pts.append(f"{x:.1f},{y:.1f}")
    polyline = " ".join(pts)
    # zero line
    zy = (height - 4) - ((0 - mn) / rng) * (height - 8)
    zy = max(4, min(height - 4, zy))
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg">'
        f'<line x1="4" y1="{zy:.1f}" x2="{width-4}" y2="{zy:.1f}" '
        f'stroke="#94a3b8" stroke-width="1" stroke-dasharray="2,2"/>'
        f'<polyline points="{polyline}" fill="none" stroke="#6366f1" stroke-width="2" '
        f'stroke-linejoin="round" stroke-linecap="round"/>'
        f'</svg>'
    )


def phase_portrait_svg(pca_2d, regime, width=200, height=140):
    """2D phase portrait from PCA-projected trajectory."""
    if len(pca_2d) < 2:
        return ""
    xs = [p[0] for p in pca_2d]
    ys = [p[1] for p in pca_2d]
    mnx, mxx = min(xs), max(xs)
    mny, mxy = min(ys), max(ys)
    rx = mxx - mnx or 0.001
    ry = mxy - mny or 0.001
    pad = 12

    def tx(v): return pad + (v - mnx) / rx * (width - 2 * pad)
    def ty(v): return (height - pad) - (v - mny) / ry * (height - 2 * pad)

    color = REGIME_COLORS.get(regime, ("#6366f1", "#e0e7ff", "?"))[0]
    pts = [(tx(p[0]), ty(p[1])) for p in pca_2d]
    lines = " ".join(f"L{p[0]:.1f},{p[1]:.1f}" for p in pts[1:])
    path = f"M{pts[0][0]:.1f},{pts[0][1]:.1f} {lines}"

    dots = ""
    for i, (px, py) in enumerate(pts):
        alpha = 0.3 + 0.7 * (i / len(pts))
        r = 3 if i == len(pts) - 1 else 2
        dots += f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{r}" fill="{color}" opacity="{alpha:.2f}"/>'

    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg" style="background:#0f172a;border-radius:6px">'
        f'<path d="{path}" fill="none" stroke="{color}" stroke-width="1.5" opacity="0.5"/>'
        f'{dots}'
        f'<circle cx="{pts[0][0]:.1f}" cy="{pts[0][1]:.1f}" r="3" fill="#94a3b8"/>'
        f'</svg>'
    )


def render_scenario_card(r):
    color, bg, icon = REGIME_COLORS.get(r["final_regime"], ("#6b7280", "#f3f4f6", "?"))
    action_label, action_color = ACTION_LABELS.get(r["final_action"], ("", "#6b7280"))
    lambdas = [s["lambda"] for s in r["steps"]]
    spark = sparkline_svg(lambdas)
    portrait = phase_portrait_svg(r["pca_2d"], r["final_regime"])

    # Build step table
    step_rows = ""
    for s in r["steps"]:
        rc, rbg, ri = REGIME_COLORS.get(s["regime"], ("#6b7280", "#f3f4f6", "?"))
        lval = f"{s['lambda']:+.4f}"
        step_rows += (
            f'<tr>'
            f'<td style="color:#94a3b8;font-size:11px;padding:2px 6px">{s["step"]}</td>'
            f'<td style="font-size:11px;padding:2px 4px;max-width:280px;overflow:hidden;white-space:nowrap;text-overflow:ellipsis">{s["text"]}</td>'
            f'<td style="font-size:11px;padding:2px 6px;font-family:monospace;color:{"#22c55e" if s["lambda"] < -0.05 else "#ef4444" if s["lambda"] > 0.15 else "#94a3b8"}">{lval}</td>'
            f'<td style="padding:2px 4px"><span style="background:{rbg};color:{rc};font-size:10px;padding:1px 5px;border-radius:4px;font-weight:600">{s["regime"]}</span></td>'
            f'</tr>'
        )

    # Interventions
    intervention_html = ""
    for iv in r.get("interventions", []):
        ic, ibg, ii = REGIME_COLORS.get(iv["regime"], ("#6b7280", "#f3f4f6", "?"))
        al, ac = ACTION_LABELS.get(iv["action"], (iv["action"], "#6b7280"))
        intervention_html += (
            f'<div style="margin-top:8px;padding:10px 12px;background:#1e293b;border-left:3px solid {ic};border-radius:4px">'
            f'<div style="font-size:11px;color:{ic};font-weight:700;margin-bottom:4px">🔔 Step {iv["at_step"]} — {al}</div>'
            f'<div style="font-size:11px;color:#94a3b8;line-height:1.5">{iv["hint"][:200]}{"…" if len(iv["hint"]) > 200 else ""}</div>'
            f'</div>'
        )

    match_badge = (
        '<span style="background:#dcfce7;color:#166534;font-size:10px;padding:2px 7px;border-radius:10px;font-weight:700">✅ PASS</span>'
        if r["match_expected"] else
        f'<span style="background:#fee2e2;color:#991b1b;font-size:10px;padding:2px 7px;border-radius:10px;font-weight:700">⚠ expected {r["expected_final"]}</span>'
    )

    return f"""
<div style="background:#1e293b;border-radius:12px;padding:20px;margin-bottom:24px;border:1px solid #334155">
  <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:16px;flex-wrap:wrap">
    <div style="flex:1;min-width:260px">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">
        <span style="background:{bg};color:{color};font-size:22px;width:36px;height:36px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700">{icon}</span>
        <div>
          <div style="font-size:15px;font-weight:700;color:#f1f5f9">{r["name"]}</div>
          <div style="font-size:11px;color:#64748b;margin-top:1px">{r["tagline"]}</div>
        </div>
      </div>
      <div style="margin:10px 0;font-size:11px;color:#94a3b8">
        <strong style="color:#cbd5e1">Goal:</strong> {r["goal"]}
      </div>
      <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px">
        <span style="background:{bg};color:{color};font-size:12px;padding:3px 10px;border-radius:6px;font-weight:700">{r["final_regime"]}</span>
        <span style="background:#1e3a5f;color:#7dd3fc;font-size:12px;padding:3px 10px;border-radius:6px;font-family:monospace">λ = {r["final_lambda"]:+.4f}</span>
        {f'<span style="background:#292524;color:{action_color};font-size:11px;padding:3px 10px;border-radius:6px">→ {action_label}</span>' if action_label else ''}
        {match_badge}
      </div>
      <div style="font-size:11px;color:#64748b">
        <span>basin: <strong style="color:#94a3b8">{r["basin_label"]}</strong></span>
        <span style="margin-left:12px">trend: <strong style="color:{"#22c55e" if r["distance_trend"] < 0 else "#ef4444"}">{r["distance_trend"]:+.5f}</strong></span>
        <span style="margin-left:12px">mean_d: <strong style="color:#94a3b8">{r["mean_distance"]:.4f}</strong></span>
      </div>
      {intervention_html}
    </div>
    <div style="display:flex;flex-direction:column;gap:10px;align-items:flex-end">
      <div>
        <div style="font-size:10px;color:#475569;margin-bottom:3px;text-align:right">λ over steps</div>
        {spark}
      </div>
      <div>
        <div style="font-size:10px;color:#475569;margin-bottom:3px;text-align:right">phase portrait (PCA 2D)</div>
        {portrait}
      </div>
    </div>
  </div>
  <details style="margin-top:14px">
    <summary style="font-size:11px;color:#475569;cursor:pointer;user-select:none">▶ Step-by-step telemetry ({len(r["steps"])} steps)</summary>
    <div style="margin-top:10px;overflow-x:auto">
      <table style="width:100%;border-collapse:collapse;font-size:11px">
        <thead>
          <tr style="border-bottom:1px solid #334155">
            <th style="text-align:left;color:#475569;padding:3px 6px">#</th>
            <th style="text-align:left;color:#475569;padding:3px 4px">Step summary</th>
            <th style="text-align:left;color:#475569;padding:3px 6px">λ</th>
            <th style="text-align:left;color:#475569;padding:3px 4px">Regime</th>
          </tr>
        </thead>
        <tbody style="color:#cbd5e1">{step_rows}</tbody>
      </table>
    </div>
    <div style="margin-top:10px">
      <div style="font-size:11px;color:#475569;margin-bottom:4px">Final intervention hint from harness:</div>
      <div style="font-size:11px;color:#94a3b8;background:#0f172a;padding:10px;border-radius:6px;line-height:1.6">{r["intervention_hint"]}</div>
    </div>
  </details>
</div>"""


passed = sum(1 for r in results if r["match_expected"])
total = len(results)

cards_html = "\n".join(render_scenario_card(r) for r in results)

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AttractorFlow — Harness Proof-of-Work</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background: #0f172a; color: #e2e8f0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; padding: 24px; }}
    a {{ color: #7dd3fc; }}
    details summary::-webkit-details-marker {{ display: none; }}
  </style>
</head>
<body>
<div style="max-width:900px;margin:0 auto">

  <!-- Header -->
  <div style="margin-bottom:28px">
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:6px">
      <div style="font-size:28px;font-weight:800;color:#f1f5f9">AttractorFlow</div>
      <div style="background:#312e81;color:#a5b4fc;font-size:12px;padding:3px 10px;border-radius:6px;font-weight:600">Harness Proof-of-Work</div>
    </div>
    <div style="font-size:14px;color:#64748b;line-height:1.6">
      Five developer use-cases run through the AttractorFlow MCP harness. Each scenario
      exercises a different dynamical regime. The harness classifies trajectory health in
      real-time using finite-time Lyapunov exponents (FTLE) and prescribes a concrete
      orchestration action.
    </div>
  </div>

  <!-- Summary bar -->
  <div style="background:#1e293b;border-radius:10px;padding:16px 20px;margin-bottom:24px;display:flex;gap:24px;flex-wrap:wrap;border:1px solid #334155">
    <div>
      <div style="font-size:11px;color:#475569;margin-bottom:2px">SCENARIOS PASSED</div>
      <div style="font-size:24px;font-weight:800;color:{"#22c55e" if passed == total else "#f59e0b"}">{passed}/{total}</div>
    </div>
    <div style="border-left:1px solid #334155;padding-left:24px">
      <div style="font-size:11px;color:#475569;margin-bottom:4px">REGIMES DETECTED</div>
      <div style="display:flex;gap:6px;flex-wrap:wrap">
        {"".join(
          f'<span style="background:{REGIME_COLORS[r["final_regime"]][1]};color:{REGIME_COLORS[r["final_regime"]][0]};font-size:11px;padding:2px 8px;border-radius:4px;font-weight:600">{r["final_regime"]}</span>'
          for r in results
        )}
      </div>
    </div>
    <div style="border-left:1px solid #334155;padding-left:24px">
      <div style="font-size:11px;color:#475569;margin-bottom:2px">EMBEDDING MODEL</div>
      <div style="font-size:12px;color:#94a3b8;font-family:monospace">all-MiniLM-L6-v2</div>
    </div>
    <div style="border-left:1px solid #334155;padding-left:24px">
      <div style="font-size:11px;color:#475569;margin-bottom:2px">BUGS FIXED THIS SESSION</div>
      <div style="font-size:12px;color:#94a3b8">
        3 classifier fixes · 1 new PLATEAU regime · drift_diverging signal
      </div>
    </div>
  </div>

  <!-- Fix log -->
  <div style="background:#1e293b;border-radius:10px;padding:16px 20px;margin-bottom:24px;border:1px solid #334155">
    <div style="font-size:13px;font-weight:700;color:#f1f5f9;margin-bottom:10px">🔧 Bugs Found &amp; Fixed During This Demo Session</div>
    <div style="font-size:12px;color:#94a3b8;line-height:1.8">
      <div><strong style="color:#fbbf24">Bug 1 — DIVERGING threshold too high (0.25):</strong> Drift-style topic divergence produces λ ≈ 0.03, well below 0.25. Fixed: added secondary signal <code style="background:#0f172a;padding:1px 4px;border-radius:3px">distance_trend &gt; 0.008 AND mean_distance &gt; 1.0</code>.</div>
      <div style="margin-top:6px"><strong style="color:#fbbf24">Bug 2 — CYCLING masked CONVERGING:</strong> A converging trajectory with cycling autocorrelation pattern was classified as CYCLING. Fixed: CYCLING doesn't fire when <code style="background:#0f172a;padding:1px 4px;border-radius:3px">distance_trend &lt; −0.02</code>.</div>
      <div style="margin-top:6px"><strong style="color:#fbbf24">Bug 3 — PLATEAU/STUCK overlap (design flaw in new PLATEAU regime):</strong> PLATEAU requires low mean_dist but STUCK fires when all(recent_distances &lt; 0.40), creating an unreachable code path. Fixed: PLATEAU now checked BEFORE STUCK using <code style="background:#0f172a;padding:1px 4px;border-radius:3px">is_stuck AND distance_trend &lt; −0.01</code> as the distinguisher.</div>
    </div>
  </div>

  <!-- Scenario cards -->
  {cards_html}

  <!-- Footer -->
  <div style="text-align:center;color:#334155;font-size:11px;margin-top:20px;padding-top:16px;border-top:1px solid #1e293b">
    AttractorFlow — Dynamical systems diagnostics for Claude Code agents
    · Embedding: all-MiniLM-L6-v2 · FTLE window: 4 · Buffer: 100 steps
  </div>
</div>
</body>
</html>"""

with open(OUT_PATH, "w") as f:
    f.write(html)
print(f"Dashboard written → {OUT_PATH}")
