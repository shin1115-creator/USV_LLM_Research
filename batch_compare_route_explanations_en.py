"""Batch Qwen route-explanation experiments with unchanged P1/P2 prompts.

Place in: USV_LLM_Research/code/batch_compare_route_explanations_en.py

From the project root (Windows cmd):
    python code\batch_compare_route_explanations_en.py --dry-run
    python code\batch_compare_route_explanations_en.py
    python code\batch_compare_route_explanations_en.py --runs 3
    python code\batch_compare_route_explanations_en.py --report-only

The LLM only explains routes selected by Python. Tied choices are excluded.
Model inference runs locally on the user's PC (no online service).
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import math
import re
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

DEFAULT_MODELS = (
    "qwen2.5-0.5b-instruct-q4_k_m.gguf",
    "qwen2.5-1.5b-instruct-q4_k_m.gguf",
)
OBJECTIVES = ("distance_priority", "separation_priority")
PROMPTS = ("p1", "p2")
METRIC_FIELDS = {
    "distance_priority": "route_length_m",
    "separation_priority": "minimum_separation_m",
}
PROMPT_VERSION_NAMES = {
    "p1": "objective_baseline_en_v1",
    "p2": "objective_structured_en_v2",
}


@dataclass(frozen=True)
class Case:
    scenario: str
    objective: str
    data: dict
    input_path: Path
    source_path: Path | None


def load_module(name: str, path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Required Python script not found: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def as_list(value: str) -> list[str] | None:
    if value.strip().lower() == "all":
        return None
    items = [v.strip() for v in value.split(",") if v.strip()]
    if not items:
        raise ValueError("Expected comma-separated values or 'all'")
    return list(dict.fromkeys(items))


def check_payload(data: dict, objective: str, scenario: str) -> None:
    if data.get("language") != "en" or data.get("scenario_id") != scenario:
        raise ValueError(f"Invalid English input for {scenario}/{objective}")
    nav = data.get("navigation_objective", {})
    if not isinstance(nav, dict) or nav.get("id") != objective:
        raise ValueError(f"Navigation objective mismatch for {scenario}/{objective}")
    selected = data.get("selected_route")
    if selected not in ("A", "B"):
        raise ValueError(f"Missing Python decision for {scenario}/{objective}")
    routes = data.get("routes", {})
    if not isinstance(routes, dict):
        raise ValueError(f"No routes for {scenario}/{objective}")
    for rid in ("A", "B"):
        row = routes.get(rid)
        if not isinstance(row, dict):
            raise ValueError(f"Missing route {rid} for {scenario}/{objective}")
        for field in ("route_length_m", "estimated_travel_time_s", "minimum_separation_m"):
            number = row.get(field)
            if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
                raise ValueError(f"Invalid {rid}.{field} for {scenario}/{objective}")
    key = METRIC_FIELDS[objective]
    a, b = (float(routes[rid][key]) for rid in ("A", "B"))
    if math.isclose(a, b, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"Tied primary metric for {scenario}/{objective}; no unique decision")
    expected = ("A" if a < b else "B") if objective == "distance_priority" else ("A" if a > b else "B")
    if expected != selected:
        raise ValueError(f"Saved Python decision does not match the values for {scenario}/{objective}")


def discover_cases(root: Path, prep, requested: list[str] | None):
    src_dir = root / "data" / "theta_star"
    inp_dir = root / "data" / "llm_inputs"
    files = sorted(src_dir.glob("*_comparison.json"), key=lambda p: (p.stat().st_mtime, p.name))
    chosen_sources: dict[str, Path] = {}
    counts: dict[str, int] = defaultdict(int)
    skips: list[dict] = []
    for file in files:
        try:
            source = prep.load_source(file)
            sid = source["scenario_id"]
            if not isinstance(sid, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", sid):
                raise ValueError("Invalid scenario_id")
            if requested is None or sid in requested:
                chosen_sources[sid] = file
                counts[sid] += 1
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            skips.append({"scenario": file.stem, "objective": "", "reason": f"Invalid source: {exc}", "source": str(file)})

    for sid, count in counts.items():
        if count > 1:
            print(f"NOTICE: {sid} has {count} comparison JSON files. Using latest modified: {chosen_sources[sid].name}")

    cases: dict[tuple[str, str], Case] = {}
    # Validated comparison files are the primary source of truth.
    for sid, path in sorted(chosen_sources.items()):
        source = prep.load_source(path)
        for obj in OBJECTIVES:
            try:
                data = prep.make_llm_input(source, obj)
                check_payload(data, obj, sid)
                input_path = inp_dir / f"{sid}_{obj}_llm_input_en.json"
                if input_path.exists():
                    existing = json.loads(input_path.read_text(encoding="utf-8"))
                    if existing != data:
                        raise RuntimeError(
                            f"Existing LLM input differs from its source comparison JSON: {input_path}\n"
                            "Do not silently overwrite experimental inputs. Review the two JSON files."
                        )
                cases[(sid, obj)] = Case(sid, obj, data, input_path, path)
            except ValueError as exc:
                skips.append({"scenario": sid, "objective": obj, "reason": str(exc), "source": str(path)})

    # Also accept previously prepared English inputs if source comparisons are absent.
    suffixes = {obj: f"_{obj}_llm_input_en.json" for obj in OBJECTIVES}
    for path in sorted(inp_dir.glob("*_llm_input_en.json")):
        obj = next((o for o, suffix in suffixes.items() if path.name.endswith(suffix)), None)
        if obj is None:
            continue
        sid = path.name[: -len(suffixes[obj])]
        if requested is not None and sid not in requested:
            continue
        if sid in chosen_sources or (sid, obj) in cases:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            check_payload(data, obj, sid)
            cases[(sid, obj)] = Case(sid, obj, data, path, None)
        except (ValueError, json.JSONDecodeError) as exc:
            skips.append({"scenario": sid, "objective": obj, "reason": str(exc), "source": str(path)})

    if requested is not None:
        discovered = {case.scenario for case in cases.values()} | set(chosen_sources)
        for sid in requested:
            if sid not in discovered:
                skips.append({"scenario": sid, "objective": "", "reason": "No comparison JSON or prepared input found", "source": ""})
    return sorted(cases.values(), key=lambda c: (c.scenario, c.objective)), skips


def expected_prompt(gen, case: Case, style: str) -> str:
    return gen.build_prompt(case.data, style)


def record_matches(record: dict, case: Case, model_name: str, prompt_style: str, prompt: str, gen) -> bool:
    style = record.get("prompt_style")
    if style is None:
        style = next((k for k, v in PROMPT_VERSION_NAMES.items() if v == record.get("prompt_version")), None)
    if (
        record.get("language") != "en"
        or record.get("scenario_id") != case.scenario
        or record.get("objective_id") != case.objective
        or record.get("model_file") != model_name
        or style != prompt_style
        or record.get("selected_route_by_python") != case.data["selected_route"]
        or record.get("prompt") != prompt
    ):
        return False
    params = record.get("parameters", {})
    setting_names = (
        ("n_ctx", "N_CTX"),
        ("n_gpu_layers", "N_GPU_LAYERS"),
        ("n_threads", "N_THREADS"),
        ("max_tokens", "MAX_TOKENS"),
        ("temperature", "TEMPERATURE"),
    )
    if not isinstance(params, dict) or any(params.get(k) != getattr(gen, attr) for k, attr in setting_names):
        return False
    return params.get("top_p") == 1.0 and isinstance(record.get("generated_explanation"), str)


def read_saved_results(folder: Path) -> list[tuple[Path, dict]]:
    saved = []
    for file in sorted(folder.glob("*.json")):
        try:
            record = json.loads(file.read_text(encoding="utf-8"))
            if isinstance(record, dict):
                saved.append((file, record))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            print(f"NOTICE: Ignoring unreadable result JSON: {file.name}")
    return saved


def matched_results(saved, cases, gen, models, prompts):
    matched = defaultdict(list)
    for case in cases:
        for model in models:
            for style in prompts:
                prompt = expected_prompt(gen, case, style)
                key = (case.scenario, case.objective, model, style)
                for path, record in saved:
                    if record_matches(record, case, model, style, prompt, gen):
                        matched[key].append((path, record))
    return matched


def save_english_input(case: Case):
    if case.input_path.exists():
        return
    case.input_path.parent.mkdir(parents=True, exist_ok=True)
    with case.input_path.open("x", encoding="utf-8") as handle:
        json.dump(case.data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"Prepared input: {case.input_path.name}")


def generate_one(llm, gen, case, style, model_name, root):
    prompt = expected_prompt(gen, case, style)
    started = time.perf_counter()
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You explain a USV route-selection result using only the supplied facts. Write in clear, concise English."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=gen.MAX_TOKENS,
        temperature=gen.TEMPERATURE,
        top_p=1.0,
    )
    elapsed = time.perf_counter() - started
    choice = response["choices"][0]
    explanation = (choice.get("message", {}).get("content") or "").strip()
    stamp = datetime.now()
    out_dir = root / "results" / "llm_explanations"
    out_dir.mkdir(parents=True, exist_ok=True)
    outfile = out_dir / (
        f"{case.scenario}_{case.objective}_{Path(model_name).stem}_en_{style}_"
        f"{stamp.strftime('%Y%m%d_%H%M%S_%f')}.json"
    )
    record = {
        "timestamp": stamp.astimezone().isoformat(),
        "language": "en",
        "scenario_id": case.scenario,
        "objective_id": case.objective,
        "selected_route_by_python": case.data["selected_route"],
        "model_file": model_name,
        "prompt_version": gen.PROMPT_VERSIONS[style],
        "prompt_style": style,
        "parameters": {
            "n_ctx": gen.N_CTX,
            "n_gpu_layers": gen.N_GPU_LAYERS,
            "n_threads": gen.N_THREADS,
            "max_tokens": gen.MAX_TOKENS,
            "temperature": gen.TEMPERATURE,
            "top_p": 1.0,
        },
        "input_file": str(case.input_path.relative_to(root)),
        "comparison_source_file": str(case.source_path.relative_to(root)) if case.source_path else None,
        "prompt": prompt,
        "generated_explanation": explanation,
        "inference_seconds": round(elapsed, 4),
        "finish_reason": choice.get("finish_reason"),
        "token_usage": response.get("usage", {}),
        "evaluation_status": "not_evaluated",
    }
    with outfile.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"  Saved: {outfile.name}; inference={elapsed:.2f}s; finish={choice.get('finish_reason')}")
    return outfile, record


def write_csv(path: Path, headers: list[str], rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    # BOM lets Excel on Windows open UTF-8 text without garbling.
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_reports(root, cases, matched, skips, models, prompts):
    output = root / "results" / "llm_comparison"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    combo_rows = []
    for case in cases:
        metric_name = METRIC_FIELDS[case.objective]
        a, b = (case.data["routes"][rid][metric_name] for rid in ("A", "B"))
        for model in models:
            for style in prompts:
                records = matched[(case.scenario, case.objective, model, style)]
                times = []
                for path, result in records:
                    duration = result.get("inference_seconds")
                    if isinstance(duration, (float, int)) and math.isfinite(duration):
                        times.append(float(duration))
                    text = result.get("generated_explanation", "")
                    # Presence only. This does NOT evaluate whether comparisons are correct.
                    metric_mentions = (f"{a:.2f}" in text and f"{b:.2f}" in text)
                    rows.append({
                        "scenario": case.scenario,
                        "objective": case.objective,
                        "python_selected": case.data["selected_route"],
                        "objective_metric": metric_name,
                        "metric_A": f"{a:.2f}",
                        "metric_B": f"{b:.2f}",
                        "bypass_A": case.data["routes"]["A"].get("bypass_side"),
                        "bypass_B": case.data["routes"]["B"].get("bypass_side"),
                        "model": model,
                        "prompt": style,
                        "inference_seconds": duration,
                        "finish_reason": result.get("finish_reason", ""),
                        "both_metric_numbers_mentioned": "yes" if metric_mentions else "no",
                        "generated_explanation": text,
                        "record_file": str(path.relative_to(root)),
                        "manual_route_0_to_2": "",
                        "manual_metric_0_to_2": "",
                        "manual_reason_0_to_2": "",
                        "manual_other_facts_0_to_2": "",
                        "manual_major_error": "",
                        "manual_notes": "",
                    })
                combo_rows.append({
                    "scenario": case.scenario,
                    "objective": case.objective,
                    "python_selected": case.data["selected_route"],
                    "model": model,
                    "prompt": style,
                    "run_count": len(records),
                    "mean_inference_seconds": round(statistics.mean(times), 4) if times else "",
                    "min_inference_seconds": round(min(times), 4) if times else "",
                    "max_inference_seconds": round(max(times), 4) if times else "",
                    "quality_status": "manual evaluation pending" if records else "not run",
                })

    rows.sort(key=lambda r: (r["scenario"], r["objective"], r["model"], r["prompt"], r["record_file"]))
    # Preserve manual grading when reports are refreshed after additional runs.
    previous_grades = {}
    previous_file = output / "all_explanations_en.csv"
    if previous_file.is_file():
        with previous_file.open(encoding="utf-8-sig", newline="") as handle:
            for old_row in csv.DictReader(handle):
                if old_row.get("record_file"):
                    previous_grades[old_row["record_file"]] = old_row
    grading_fields = (
        "manual_route_0_to_2", "manual_metric_0_to_2", "manual_reason_0_to_2",
        "manual_other_facts_0_to_2", "manual_major_error", "manual_notes",
    )
    for row in rows:
        earlier = previous_grades.get(row["record_file"], {})
        for field in grading_fields:
            row[field] = earlier.get(field, "")
    run_headers = [
        "scenario", "objective", "python_selected", "objective_metric", "metric_A", "metric_B",
        "bypass_A", "bypass_B", "model", "prompt", "inference_seconds", "finish_reason",
        "both_metric_numbers_mentioned", "generated_explanation", "record_file",
        "manual_route_0_to_2", "manual_metric_0_to_2", "manual_reason_0_to_2",
        "manual_other_facts_0_to_2", "manual_major_error", "manual_notes",
    ]
    write_csv(output / "all_explanations_en.csv", run_headers, rows)
    write_csv(output / "condition_summary_en.csv", [
        "scenario", "objective", "python_selected", "model", "prompt", "run_count",
        "mean_inference_seconds", "min_inference_seconds", "max_inference_seconds", "quality_status",
    ], combo_rows)

    grouped = {(r["scenario"], r["objective"], r["model"], r["prompt"]): r for r in combo_rows}
    wide = []
    for case in cases:
        for model in models:
            p1 = grouped.get((case.scenario, case.objective, model, "p1"), {})
            p2 = grouped.get((case.scenario, case.objective, model, "p2"), {})
            wide.append({
                "scenario": case.scenario,
                "objective": case.objective,
                "python_selected": case.data["selected_route"],
                "model": model,
                "p1_runs": p1.get("run_count", ""),
                "p1_mean_seconds": p1.get("mean_inference_seconds", ""),
                "p2_runs": p2.get("run_count", ""),
                "p2_mean_seconds": p2.get("mean_inference_seconds", ""),
                "accuracy_comparison": "not automatically scored",
            })
    write_csv(output / "p1_p2_side_by_side_en.csv", [
        "scenario", "objective", "python_selected", "model", "p1_runs", "p1_mean_seconds",
        "p2_runs", "p2_mean_seconds", "accuracy_comparison",
    ], wide)
    paired = []
    for case in cases:
        for model in models:
            p1_records = sorted(matched[(case.scenario, case.objective, model, "p1")], key=lambda item: item[0].name)
            p2_records = sorted(matched[(case.scenario, case.objective, model, "p2")], key=lambda item: item[0].name)
            for i in range(max(len(p1_records), len(p2_records))):
                p1_item = p1_records[i] if i < len(p1_records) else None
                p2_item = p2_records[i] if i < len(p2_records) else None
                paired.append({
                    "scenario": case.scenario,
                    "objective": case.objective,
                    "python_selected": case.data["selected_route"],
                    "model": model,
                    "run_index": i + 1,
                    "p1_explanation": p1_item[1].get("generated_explanation", "") if p1_item else "",
                    "p1_seconds": p1_item[1].get("inference_seconds", "") if p1_item else "",
                    "p1_file": str(p1_item[0].relative_to(root)) if p1_item else "",
                    "p2_explanation": p2_item[1].get("generated_explanation", "") if p2_item else "",
                    "p2_seconds": p2_item[1].get("inference_seconds", "") if p2_item else "",
                    "p2_file": str(p2_item[0].relative_to(root)) if p2_item else "",
                    "manual_comparison_notes": "",
                })
    previous_pairs = {}
    pair_file = output / "paired_explanations_en.csv"
    if pair_file.is_file():
        with pair_file.open(encoding="utf-8-sig", newline="") as handle:
            for earlier in csv.DictReader(handle):
                key = (earlier.get("scenario"), earlier.get("objective"), earlier.get("model"), earlier.get("run_index"))
                previous_pairs[key] = earlier.get("manual_comparison_notes", "")
    for pair in paired:
        key = (pair["scenario"], pair["objective"], pair["model"], str(pair["run_index"]))
        pair["manual_comparison_notes"] = previous_pairs.get(key, "")
    write_csv(output / "paired_explanations_en.csv", [
        "scenario", "objective", "python_selected", "model", "run_index",
        "p1_explanation", "p1_seconds", "p1_file", "p2_explanation", "p2_seconds", "p2_file",
        "manual_comparison_notes",
    ], paired)
    write_csv(output / "skipped_cases_en.csv", ["scenario", "objective", "reason", "source"], skips)
    print(f"\nReports: {output}")
    for name in ("all_explanations_en.csv", "condition_summary_en.csv", "p1_p2_side_by_side_en.csv", "paired_explanations_en.csv", "skipped_cases_en.csv"):
        print(f"  {name}")
    print(f"Result records in report: {len(rows)}")


def main():
    parser = argparse.ArgumentParser(description="Batch comparison: Qwen 0.5B/1.5B, P1/P2, navigation objectives, all scenarios")
    parser.add_argument("--project-root", type=Path, default=None, help="USV_LLM_Research root (default: script parent/code/..)")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS), help="Comma-separated GGUF filenames under models/")
    parser.add_argument("--prompts", default="p1,p2", help="Comma-separated prompt styles")
    parser.add_argument("--scenarios", default="all", help="Comma-separated scenario IDs, or 'all'")
    parser.add_argument("--runs", type=int, default=1, help="Minimum total runs PER condition, including compatible saved runs")
    flags = parser.add_mutually_exclusive_group()
    flags.add_argument("--dry-run", action="store_true", help="Show intended runs; do not generate files or load models")
    flags.add_argument("--report-only", action="store_true", help="Export CSV from existing results; no inference")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be >= 1")
    try:
        models = as_list(args.models)
        styles = as_list(args.prompts)
        selected = as_list(args.scenarios)
    except ValueError as exc:
        parser.error(str(exc))
    if models is None or styles is None:
        parser.error("--models and --prompts must not be 'all'")
    if any(style not in PROMPTS for style in styles):
        parser.error("Only p1 and p2 are supported")
    root = (args.project_root or Path(__file__).resolve().parent.parent).resolve()
    gen = load_module("route_explanation_batch_source", root / "code" / "generate_route_explanation_en.py")
    prep = load_module("llm_input_batch_source", root / "code" / "theta_star" / "prepare_llm_inputs_en.py")
    if not hasattr(gen, "PROMPT_VERSIONS") or gen.PROMPT_VERSIONS != PROMPT_VERSION_NAMES:
        raise RuntimeError("Please install the included multi-scenario P1/P2 generate_route_explanation_en.py first")
    if not callable(getattr(gen, "build_prompt", None)):
        raise RuntimeError("Missing build_prompt() in generate_route_explanation_en.py")
    cases, skips = discover_cases(root, prep, selected)
    if not cases:
        print("No eligible A/B selection cases found. See data/theta_star and data/llm_inputs.")
        for item in skips:
            print(f"  SKIP {item['scenario']} {item['objective']}: {item['reason']}")
        return 1
    saved = read_saved_results(root / "results" / "llm_explanations")
    matched = matched_results(saved, cases, gen, models, styles)
    pending = []
    for model in models:
        for case in cases:
            for style in styles:
                key = (case.scenario, case.objective, model, style)
                missing = max(0, args.runs - len(matched[key]))
                if missing:
                    pending.append((model, case, style, missing))
    print(f"Eligible scenario/objective cases: {len(cases)}")
    print(f"Available compatible results: {sum(len(x) for x in matched.values())}")
    print(f"New inference runs needed: {sum(x[3] for x in pending)}")
    for item in skips:
        print(f"  SKIP {item['scenario']}/{item['objective']}: {item['reason']}")
    if args.dry_run:
        for model, case, style, missing in pending:
            print(f"  PLAN {case.scenario}/{case.objective}/{Path(model).stem}/{style}: {missing} new runs")
        print("Dry-run complete: no files changed, no models loaded.")
        return 0
    if args.report_only:
        write_reports(root, cases, matched, skips, models, styles)
        return 0
    ggufs = {}
    for model in {item[0] for item in pending}:
        gguf = gen.model_path(root, model)
        if not gguf.is_file():
            raise FileNotFoundError(f"Model file missing: {gguf}")
        ggufs[model] = gguf
    for case in cases:
        save_english_input(case)
    try:
        if pending:
            try:
                from llama_cpp import Llama
            except ImportError as exc:
                raise RuntimeError("llama-cpp-python is unavailable in the active environment") from exc
            for model in models:
                work = [(case, style, amount) for m, case, style, amount in pending if m == model]
                if not work:
                    continue
                print(f"\nLoading model once: {model}")
                llm = Llama(model_path=str(ggufs[model]), n_ctx=gen.N_CTX,
                            n_gpu_layers=gen.N_GPU_LAYERS, n_threads=gen.N_THREADS, verbose=False)
                for case, style, amount in work:
                    for repeat in range(amount):
                        print(f"RUN {case.scenario}/{case.objective}/{model}/{style} ({repeat + 1}/{amount})")
                        file, record = generate_one(llm, gen, case, style, model, root)
                        matched[(case.scenario, case.objective, model, style)].append((file, record))
                del llm
                gc.collect()
    finally:
        # The JSON records survive an interruption; a later --report-only rebuilds CSVs.
        write_reports(root, cases, matched, skips, models, styles)
    print("Batch complete. Explanation quality requires manual review; CSVs do not claim automatic correctness.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
