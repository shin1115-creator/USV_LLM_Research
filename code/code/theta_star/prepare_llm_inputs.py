"""Theta*比較JSONからLLMに渡す入力JSONを作成する。

このスクリプトではTheta*再実行・経路採用の再決定・LLM推論は行わない。
比較JSONの「Pythonが採用した経路」と、初期計画経路の数値を抜き出す。
説明文、計算済みの差分、評価者用の模範説明はLLM入力に入れない。

配置先：USV_LLM_Research/code/theta_star/prepare_llm_inputs.py
実行例：python code\\theta_star\\prepare_llm_inputs.py
別のケース：python code\\theta_star\\prepare_llm_inputs.py --source ファイル名.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# C01：ユーザーが2026年11月20日に保存した正式な比較JSON
DEFAULT_SOURCE_NAME = "crossing_all130_u5_20261120_075940_800383_comparison.json"

OBJECTIVES = {
    "distance_priority": {
        "label": "経路長重視",
        "rule": "Theta*の初期計画経路長が短い候補を優先する",
        "metric": "planned_length_m",
        "minimize": True,
    },
    "separation_priority": {
        "label": "離隔余裕重視",
        "rule": "Theta*の初期計画上の最小離隔距離が大きい候補を優先する",
        "metric": "planned_minimum_separation_m",
        "minimize": False,
    },
}


def find_project_root() -> Path:
    """プロジェクト内に配置して実行することを前提にルートを求める。"""
    here = Path(__file__).resolve()
    if here.parent.name == "theta_star" and here.parent.parent.name == "code":
        return here.parents[2]
    # ファイル単体を別フォルダから試験する場合は --project-root を指定する。
    return Path.cwd()


def load_source(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"比較JSONが見つかりません: {path}\n"
            "まず run_route_comparison.py で生成したファイル名を確認してください。"
        )
    with path.open(encoding="utf-8") as f:
        obj = json.load(f)
    for key in ("scenario_id", "routes", "selection_results"):
        if key not in obj:
            raise ValueError(f"入力JSONに必要な項目がありません: {key}")
    return obj


def route_facts(route: dict, route_id: str) -> dict:
    """今回の説明対象は初期計画経路。追従航跡の値は混ぜない。"""
    if not route.get("found", False):
        raise ValueError(f"経路{route_id}が生成されていないため、2経路比較はできません。")
    fields = {
        "planned_length_m": "経路長",
        "planned_travel_time_s": "推定航行時間",
        "planned_minimum_separation_m": "最小離隔距離",
        "planned_bypass_side": "迂回方向",
    }
    for key, label in fields.items():
        if key not in route:
            raise ValueError(f"経路{route_id}の{label}（{key}）がありません。")
    return {
        "route_length_m": round(float(route["planned_length_m"]), 2),
        "estimated_travel_time_s": round(float(route["planned_travel_time_s"]), 2),
        "minimum_separation_m": round(float(route["planned_minimum_separation_m"]), 2),
        "bypass_side": route["planned_bypass_side"],
    }


def make_llm_input(source: dict, objective_id: str) -> dict:
    """目的別のLLM入力を生成。既存の採用結果は変更しない。"""
    obj = OBJECTIVES[objective_id]
    decisions = source["selection_results"]
    if objective_id not in decisions:
        raise ValueError(f"採用結果がありません: {objective_id}")
    decision = decisions[objective_id]
    selected = decision.get("selected_route")
    if decision.get("status") != "比較成立" or selected not in ("A", "B"):
        raise ValueError(
            f"{objective_id} は採用経路が一意に確定していません: {decision}. "
            "同値ケースの説明は別の実験として扱います。"
        )
    if decision.get("metric") != obj["metric"]:
        raise ValueError(f"{objective_id} の採用指標が想定と異なります: {decision}")

    source_routes = source["routes"]
    if not all(rid in source_routes for rid in ("A", "B")):
        raise ValueError("比較JSONには経路Aと経路Bの両方が必要です。")
    route_info = {rid: route_facts(source_routes[rid], rid) for rid in ("A", "B")}

    # 元の採用結果との単純な整合チェック（小数の丸め前の値で照合）。
    a = float(source_routes["A"][obj["metric"]])
    b = float(source_routes["B"][obj["metric"]])
    if math.isclose(a, b, abs_tol=1e-6, rel_tol=0.0):
        raise ValueError("採用結果が比較成立になっていますが、指標値が同値です。")
    expected = ("A" if a < b else "B") if obj["minimize"] else ("A" if a > b else "B")
    if selected != expected:
        raise ValueError(f"採用結果と数値が矛盾しています: 結果={selected}、数値上={expected}")

    return {
        "schema_version": "1.0",
        "scenario_id": source["scenario_id"],
        "navigation_objective": {
            "id": objective_id,
            "label": obj["label"],
            "rule": obj["rule"],
        },
        "selected_route": selected,
        "routes": route_info,
        "data_definitions": {
            "metrics": "経路長・推定航行時間・最小離隔距離はTheta*初期計画経路の区間直線・等速航行仮定による値",
            "bypass_side": "迂回方向は自船の初期針路に対する右舷側・左舷側であり、他船の通過舷やCOLREGs適否を意味しない",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="目的別のLLM入力JSONを作成")
    parser.add_argument("--source", default=DEFAULT_SOURCE_NAME, help="比較JSONのファイル名またはフルパス")
    parser.add_argument("--project-root", type=Path, default=None, help="研究プロジェクトのルート（通常は不要）")
    args = parser.parse_args()

    root = args.project_root.resolve() if args.project_root else find_project_root()
    source_path = Path(args.source)
    if not source_path.is_absolute():
        source_path = root / "data" / "theta_star" / source_path
    source_path = source_path.resolve()
    source = load_source(source_path)
    scenario_id = source["scenario_id"]
    if not isinstance(scenario_id, str) or not scenario_id or any(ch in scenario_id for ch in '\\/:*?"<>|'):
        raise ValueError("scenario_idがファイル名に使用できません")

    # 先に両入力の検証を終えてから保存する（片方だけ保存することを防止）。
    outputs = {key: make_llm_input(source, key) for key in OBJECTIVES}
    out_dir = root / "data" / "llm_inputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"読み込んだ比較JSON: {source_path}")
    for objective_id, data in outputs.items():
        dest = out_dir / f"{scenario_id}_{objective_id}_llm_input.json"
        if dest.exists():
            raise FileExistsError(
                f"既存のLLM入力JSONがあります: {dest}\n"
                "実験記録を守るため上書きしません。必要なら別フォルダへ退避してください。"
            )
    for objective_id, data in outputs.items():
        dest = out_dir / f"{scenario_id}_{objective_id}_llm_input.json"
        dest.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{objective_id}: 採用経路{data['selected_route']} -> {dest}")
    print("完了：LLM向けの客観的数値と採用結果のみを保存しました。説明文は含みません。")


if __name__ == "__main__":
    main()
