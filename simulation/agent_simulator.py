"""
agent_simulator.py — Simulated Agent Trajectory Generator

Creates synthetic agent trajectories exhibiting known pathological regimes
for testing the AttractorFlow classifier.  Each scenario generates a
sequence of text summaries that, when embedded, reproduce the target regime.

Usage:
    python agent_simulator.py [--scenario all|converging|stuck|oscillating|diverging]

Output: Trajectory analysis for each scenario using the AttractorFlow stack.
"""

import argparse
import sys
import os

# Add server directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../attractorflow/mcp-server"))

from phase_space import PhaseSpaceMonitor
from lyapunov import LyapunovEstimator
from classifier import AttractorClassifier


# ------------------------------------------------------------------
# Scenario definitions — text sequences designed to produce target regimes
# ------------------------------------------------------------------

SCENARIOS = {
    "converging": {
        "description": "Agent refining a quicksort implementation toward completion",
        "goal": "Write a correct, efficient quicksort implementation in Python",
        "steps": [
            # Each step is a refinement of the SAME quicksort code — semantically converging
            "Quicksort implementation: base case returns empty list. Partition uses first pivot.",
            "Quicksort implementation: partition correct. Recursive calls now in place.",
            "Quicksort implementation: all recursive calls correct. Tests passing for basic cases.",
            "Quicksort implementation: edge cases handled. Empty list and single element tested.",
            "Quicksort implementation: duplicate elements handled. All 12 tests passing.",
            "Quicksort implementation: in-place sort finalized. Memory O(log n) confirmed.",
            "Quicksort implementation: complete and tested. All 15 tests passing, O(n log n).",
            "Quicksort implementation: done. Correct, efficient, tested. Implementation complete.",
        ],
    },
    "stuck": {
        "description": "Agent looping on the same failed approach with no variation",
        "goal": "Parse the malformed JSON file and extract the user records",
        "steps": [
            # Nearly identical text — agent is repeating itself without any variation
            "Cannot parse the JSON file. json.loads() raises JSONDecodeError. Blocked.",
            "Cannot parse the JSON file. json.loads() raises JSONDecodeError. Still blocked.",
            "Cannot parse the JSON file. json.loads() raises JSONDecodeError. Still blocked.",
            "Cannot parse the JSON file. json.loads() raises JSONDecodeError. Still blocked.",
            "Cannot parse the JSON file. json.loads() raises JSONDecodeError. Stuck here.",
            "Cannot parse the JSON file. json.loads() raises JSONDecodeError. Still stuck.",
            "Cannot parse the JSON file. json.loads() raises JSONDecodeError. Still stuck.",
        ],
    },
    "oscillating": {
        "description": "Agent alternating between REST and GraphQL with no resolution",
        "goal": "Design the API for the user service",
        "steps": [
            # Stark alternation: REST-focused → GraphQL-focused → repeat
            "Decision: REST API. HTTP endpoints GET /users POST /users DELETE /users/{id}.",
            "Decision: GraphQL API. Single /graphql endpoint with flexible query schema.",
            "Decision: REST API. HTTP verbs, stateless, cacheable, simple CRUD endpoints.",
            "Decision: GraphQL API. Query language, nested resolvers, no overfetching.",
            "Decision: REST API. Standard HTTP, great browser tooling, simple pagination.",
            "Decision: GraphQL API. One trip fetches all data, subscriptions for real-time.",
            "Decision: REST API. Team expertise, proven patterns, easy to debug REST.",
            "Decision: GraphQL API. Schema introspection, type safety, flexible clients.",
        ],
    },
    "diverging": {
        "description": "Agent drifting from a login bug to entirely unrelated ML and infra work",
        "goal": "Fix the login form validation bug",
        "steps": [
            # Each step is in a completely different technical domain — maximum semantic drift
            "Examining login form. The email validation regex has a bug.",
            "Analysing neural network training loss curves. Gradient explosion observed.",
            "Optimising Kubernetes pod autoscaling policies for the inference cluster.",
            "Profiling Apache Spark shuffle partitions in the data warehouse ETL job.",
            "Refactoring the React component tree for the marketing landing page.",
            "Investigating TLS certificate renewal for the CDN edge nodes.",
            "Benchmarking PostgreSQL BRIN index vs B-tree for time-series queries.",
            "Writing Terraform modules for multi-region AWS VPC peering topology.",
        ],
    },
    "cycling": {
        "description": "Agent in healthy TDD cycle writing authentication",
        "goal": "Implement user authentication with tests",
        "steps": [
            "Writing test: test_login_valid_credentials. Expecting 200 and JWT token.",
            "Test fails. Implementing login endpoint. Basic password check.",
            "Login returns 200 but no JWT. Adding JWT generation.",
            "test_login_valid_credentials passes. Writing test_login_invalid_password.",
            "Test fails. Adding error handling for wrong password. Returns 401.",
            "test_login_invalid_password passes. Writing test_login_missing_fields.",
            "Test fails. Adding request validation. Returns 422 for missing fields.",
            "All 3 tests pass. Writing test_login_expired_token.",
            "Test fails. Adding token expiry check.",
            "All 4 tests pass. Code coverage at 94%.",
        ],
    },
}


def analyze_scenario(name: str, scenario: dict) -> None:
    """Run a scenario through the AttractorFlow stack and print analysis."""
    print(f"\n{'='*60}")
    print(f"Scenario: {name.upper()} — {scenario['description']}")
    print(f"Goal: {scenario['goal']}")
    print(f"{'='*60}")

    monitor = PhaseSpaceMonitor()
    estimator = LyapunovEstimator(window=4)
    classifier = AttractorClassifier()

    try:
        monitor.load_model()
    except Exception as e:
        print(f"⚠️  Could not load embedding model: {e}")
        print("    Install dependencies: pip install -r attractorflow/mcp-server/requirements.txt")
        return

    monitor.set_goal(scenario["goal"])

    print("\nStep-by-step trajectory:")
    for i, step_text in enumerate(scenario["steps"]):
        monitor.record(step_text)
        distances = monitor.get_distance_series()
        if len(distances) >= 2:
            lya = estimator.compute(distances)
            print(f"  Step {i+1:2d} | λ={lya.ftle:+.3f} | {lya.stability_label}")
        else:
            print(f"  Step {i+1:2d} | (insufficient data for λ)")

    print()
    distances = monitor.get_distance_series()
    stats = monitor.get_stats()
    lya = estimator.compute(distances)
    result = classifier.classify(lya, stats)

    print(f"Final Classification:")
    print(f"  Regime:     {result.regime.value}")
    print(f"  Confidence: {result.confidence:.0%}")
    print(f"  Action:     {result.action.value}")
    print(f"  λ:          {result.λ:+.4f}")
    print(f"  Rationale:  {result.rationale}")
    print()
    print(f"Intervention hint:")
    # Wrap at 70 chars
    hint = result.intervention_hint
    while hint:
        print(f"  {hint[:70]}")
        hint = hint[70:]


def main():
    parser = argparse.ArgumentParser(description="AttractorFlow Agent Simulator")
    parser.add_argument(
        "--scenario",
        default="all",
        choices=list(SCENARIOS.keys()) + ["all"],
        help="Which scenario to run",
    )
    args = parser.parse_args()

    print("AttractorFlow Agent Trajectory Simulator")
    print("Demonstrates regime classification on synthetic agent traces\n")

    scenarios_to_run = (
        SCENARIOS.items() if args.scenario == "all"
        else [(args.scenario, SCENARIOS[args.scenario])]
    )

    for name, scenario in scenarios_to_run:
        analyze_scenario(name, scenario)

    print(f"\n{'='*60}")
    print("Simulation complete.")
    print("To use AttractorFlow in a real Claude Code project:")
    print("  1. Copy attractorflow/ into your project")
    print("  2. pip install -r attractorflow/mcp-server/requirements.txt")
    print("  3. Add .mcp.json to project root")
    print("  4. Use /attractor-status in any Claude Code session")


if __name__ == "__main__":
    main()
