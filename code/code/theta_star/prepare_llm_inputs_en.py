"""Create English-language LLM inputs from saved Theta* comparison results.

This script does not rerun Theta*, select a route, or generate explanations.
It copies the route selected by Python and the factual metrics from the
initially planned routes. It does not prewrite the reason for the selection.

Place in: USV_LLM_Research/code/theta_star/prepare_llm_inputs_en.py
Run:      python code\\theta_star\\prepare_llm_inputs_en.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

DEFAULT_SOURCE_NAME = "crossing_all130_u5_20261120_075940_800383_comparison.json"

OBJECTIVES = {
    "distance_priority": {
        "label": "Shortest route priority",
        "rule": "Prefer the route with the shorter initial Theta* planned length.",
        "metric": "planned_length_m",
        "minimize": True,
    },
    "separation_priority": {
        "label": "Maximum separation priority",
        "rule": "Prefer the route with the larger minimum vessel-to-vessel separation along the initial planned route.",
        "metric": "planned_minimum_separation_m",
        "minimize": False,
    },
}

# Translate labels from the existing Japanese comparison program.
BYPASS_SIDES = {
    "右舷側": "starboard side",
    "左舷側": "port side",
    "直進付近": "approximately straight",
    "両側": "both sides",
    "starboard side": "starboard side",
    "port side": "port side",
    "approximately straight": "approximately straight",
    "both sides": "both sides",
}


def find_project_root() -> Path:
    """Find the project root when this script is placed under code/theta_star/."""
    here = Path(__file__).resolve()
    if here.parent.name == "theta_star" and here.parent.parent.name == "code":
        return here.parents[2]
    return Path.cwd()


def load_source(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Comparison JSON not found: {path}")
    with path.open(encoding="utf-8") as stream:
        source = json.load(stream)
    for field in ("scenario_id", "routes", "selection_results"):
        if field not in source:
            raise ValueError(f"Missing field in comparison JSON: {field}")
    return source


def finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def route_facts(route: dict, route_id: str) -> dict:
    """Read initial-plan metrics only; tracked-vessel metrics are excluded."""
    if not route.get("found", False):
        raise ValueError(f"Route {route_id} was not generated")
    mapping = {
        "route_length_m": "planned_length_m",
        "estimated_travel_time_s": "planned_travel_time_s",
        "minimum_separation_m": "planned_minimum_separation_m",
    }
    result = {}
    for output_field, source_field in mapping.items():
        result[output_field] = round(finite_number(route.get(source_field), f"Route {route_id}.{source_field}"), 2)
    side = route.get("planned_bypass_side")
    if side not in BYPASS_SIDES:
        raise ValueError(f"Unknown bypass-side label for route {route_id}: {side!r}")
    result["bypass_side"] = BYPASS_SIDES[side]
    return result


def make_llm_input(source: dict, objective_id: str) -> dict:
    """Preserve Python's original selection without generating a reason."""
    definition = OBJECTIVES[objective_id]
    decision = source["selection_results"].get(objective_id)
    if not isinstance(decision, dict):
        raise ValueError(f"Selection result not found for {objective_id}")

    selected = decision.get("selected_route")
    # The original comparison code records successful comparisons in Japanese.
    if decision.get("status") not in ("比較成立", "comparison_valid") or selected not in ("A", "B"):
        raise ValueError(f"No unique selected route for {objective_id}: {decision}")
    if decision.get("metric") != definition["metric"]:
        raise ValueError(f"Unexpected selection metric for {objective_id}")

    routes = source["routes"]
    if not all(isinstance(routes.get(rid), dict) for rid in ("A", "B")):
        raise ValueError("Both route A and route B are required")
    facts = {rid: route_facts(routes[rid], rid) for rid in ("A", "B")}

    # Check the existing decision; do not make a new selection.
    a = finite_number(routes["A"].get(definition["metric"]), "Route A selection metric")
    b = finite_number(routes["B"].get(definition["metric"]), "Route B selection metric")
    if math.isclose(a, b, abs_tol=1e-6, rel_tol=0.0):
        raise ValueError("The selection metric is tied; a unique selection cannot be explained")
    expected = ("A" if a < b else "B") if definition["minimize"] else ("A" if a > b else "B")
    if expected != selected:
        raise ValueError("Python's saved selection conflicts with the saved route metrics")

    return {
        "schema_version": "1.1-en",
        "language": "en",
        "scenario_id": source["scenario_id"],
        "navigation_objective": {
            "id": objective_id,
            "label": definition["label"],
            "rule": definition["rule"],
        },
        "selected_route": selected,
        "routes": facts,
        "data_definitions": {
            "metrics": "Route length, estimated travel time, and minimum separation are calculated for the initial Theta* plan, assuming straight segments and constant own-ship speed.",
            "bypass_side": "Bypass side is relative to the own ship's initial heading. It does not describe the side of the target vessel passed or establish COLREG compliance.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare English LLM inputs from route comparison JSON")
    parser.add_argument("--source", default=DEFAULT_SOURCE_NAME, help="Comparison JSON filename or absolute path")
    parser.add_argument("--project-root", type=Path, default=None, help="Project root (normally detected automatically)")
    args = parser.parse_args()

    root = args.project_root.resolve() if args.project_root else find_project_root()
    source_path = Path(args.source)
    if not source_path.is_absolute():
        source_path = root / "data" / "theta_star" / source_path
    source_path = source_path.resolve()
    source = load_source(source_path)
    scenario_id = source["scenario_id"]
    if not isinstance(scenario_id, str) or not scenario_id or any(char in scenario_id for char in '\\/:*?"<>|'):
        raise ValueError("scenario_id cannot be used as a filename")

    outputs = {objective_id: make_llm_input(source, objective_id) for objective_id in OBJECTIVES}
    out_dir = root / "data" / "llm_inputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {
        objective_id: out_dir / f"{scenario_id}_{objective_id}_llm_input_en.json"
        for objective_id in outputs
    }
    for path in files.values():
        if path.exists():
            raise FileExistsError(f"English input file already exists; it will not be overwritten: {path}")

    print(f"Source comparison JSON: {source_path}")
    for objective_id, payload in outputs.items():
        destination = files[objective_id]
        with destination.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        print(f"{objective_id}: selected route {payload['selected_route']} -> {destination}")
    print("Completed: English factual inputs and Python's selected route were saved. No explanation was included.")


if __name__ == "__main__":
    main()
