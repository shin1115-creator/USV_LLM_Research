"""Theta*の経路比較結果を、Qwenの説明生成用JSONへ変換する。

配置先 : code/theta_star/build_llm_input.py
使い方 :
    python code\theta_star\build_llm_input.py --scenario headon
    python code\theta_star\build_llm_input.py --scenario overtaking
    python code\theta_star\build_llm_input.py --scenario crossing

このスクリプトは経路の生成・選択を行わない。
行会い・追越しでは、既に算出した「基準Bumper判定」が一意に正解を定める場合のみ
正解経路を確定する。評価用の正解理由はLLM入力と分離する。
"""

import argparse
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data" / "theta_star"


def load_json(filename):
    """所定のデータフォルダからJSONファイルを読み込む。"""
    path = DATA_DIR / filename
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(filename, data):
    """入力ファイルを上書きせず、出力先のJSONに保存する。"""
    path = DATA_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def build_encounter(scenario_type):
    """行会い・追越しの共通した比較結果形式をLLM入力形式に変換する。"""
    setting = {
        "headon": ("theta_headon_001", "head-on"),
        "overtaking": ("theta_overtaking_001", "overtaking"),
    }
    if scenario_type not in setting:
        raise ValueError(f"未対応の遭遇種別です: {scenario_type}")
    scenario_id, encounter_type = setting[scenario_type]
    comparison = load_json(f"{scenario_id}_AB_comparison.json")
    scenario = comparison["scenario"]
    routes = comparison["routes"]

    if comparison["scenario_id"] != scenario_id:
        raise ValueError("シナリオIDが指定された遭遇種別と一致しません。")
    if any(not routes[r].get("found", False) for r in ("A", "B")):
        raise ValueError("経路A/Bが両方とも生成されていません。")

    # 今回の実験ルールでは、基準Bumperへの侵入がないことが必須。
    # A/Bのどちらか一方だけが合格した場合に限り正解を確定する。
    valid_routes = [r for r in ("A", "B") if routes[r]["valid_under_baseline_bumper"] is True]
    if len(valid_routes) != 1:
        raise ValueError(
            "基準Bumper条件だけでは正解経路を一意に定められません。"
            "判断基準を追加してから実験してください。"
        )
    correct_route = valid_routes[0]
    declared_correct = comparison.get("correct_route")
    if declared_correct is not None and declared_correct != correct_route:
        raise ValueError("比較JSONの正解経路と基準Bumper判定が矛盾しています。")

    baseline = scenario["baseline_bumper"]

    def make_route_info(route_id):
        source = routes[route_id]
        generation_bumper = source["generation_bumper"]
        # 縮小倍率は、実際の設定値から計算する（A=1.0、B=0.7を想定）。
        scale = generation_bumper["fore"] / baseline["fore"]
        return {
            "generation_method": "Theta*",
            "bumper_scale": round(scale, 4),
            "waypoints": source["waypoints"],
            "route_length_m": source["path_length_m"],
            "travel_time_sec": source["travel_time_s_straight_segments"],
            "min_separation_m": source["minimum_separation_m"],
            "valid_under_baseline_bumper": source["valid_under_baseline_bumper"],
            "valid_under_reduced_bumper": source["valid_under_reduced_bumper"],
        }

    route_info = {r: make_route_info(r) for r in ("A", "B")}
    incorrect_route = "B" if correct_route == "A" else "A"
    ownship = scenario["ownship"]
    target = scenario["target"]

    selection_criterion = (
        "基準Bumperへの侵入がないことを必須条件とする。"
        "候補経路のうち、この条件を満たす経路を正解とする。"
        "これは本シミュレーション上の判断基準である。"
    )
    explanation = (
        f"経路{correct_route}は基準Bumper条件を満たす一方、"
        f"経路{incorrect_route}は満たさないため、"
        f"実験上の正解経路は{correct_route}である。"
    )
    facts = [
        f"経路{correct_route}は基準Bumper条件を満たす",
        f"経路{incorrect_route}は基準Bumper条件を満たさない",
    ]
    if route_info["A"]["route_length_m"] != route_info["B"]["route_length_m"]:
        shorter = min(("A", "B"), key=lambda r: route_info[r]["route_length_m"])
        facts.append(f"経路{shorter}はもう一方の経路より短い")
    if route_info["A"]["min_separation_m"] != route_info["B"]["min_separation_m"]:
        wider = max(("A", "B"), key=lambda r: route_info[r]["min_separation_m"])
        facts.append(f"経路{wider}の最小離隔距離はもう一方の経路より大きい")

    output = {
        "llm_input": {
            "scenario_id": comparison["scenario_id"],
            "encounter_type": encounter_type,
            "ownship": ownship,
            "target_ships": [target],
            "selection_criterion": selection_criterion,
            "routes": route_info,
            "correct_route": correct_route,
        },
        "evaluation_reference": {
            "correct_route": correct_route,
            "reason": explanation,
            "important_facts": facts,
        },
        "notes": [
            "正解経路はシミュレーション上の基準Bumper判定に従って設定した",
            "評価用の正解理由はLLMへのプロンプトに含めない",
            "最小離隔距離と航行時間は区間直線・等速航行の近似値",
            "基準Bumperの適合は現実の航行安全性やCOLREGsへの適合を保証しない",
            "BCRは現時点では未計算",
        ],
    }
    return output, f"{scenario_id}_llm_input.json"


def build_headon():
    """行会いシナリオ用。"""
    return build_encounter("headon")


def build_overtaking():
    """追越しシナリオ用。"""
    return build_encounter("overtaking")


def build_crossing():
    """従来の横切りシナリオを、元の入力形式のまま作成する。"""
    conditions = load_json("experiment_conditions_AB.json")
    data_a = load_json("theta_001.json")
    data_b = load_json("theta_001_B.json")
    comparison = load_json("theta_001_AB_comparison.json")
    results = comparison["results"]
    correct_route = conditions["correct_route"]
    if correct_route not in ("A", "B"):
        raise ValueError("正解経路がAまたはBに設定されていません。")

    def make_route_info(route_id, route_data):
        result = results[route_id]
        settings = conditions[f"route_{route_id}"]
        return {
            "generation_method": "Theta*",
            "bumper_scale": settings.get("bumper_scale", 1.0),
            "waypoints": route_data["route"],
            "route_length_m": result["route_length_m"],
            "travel_time_sec": result["travel_time_sec"],
            "min_separation_m": result["min_separation_m"],
            "valid_under_baseline_bumper": result["valid_under_A_bumper"],
            "valid_under_reduced_bumper": result["valid_under_B_bumper"],
        }

    output = {
        "llm_input": {
            "scenario_id": conditions["experiment_id"],
            "encounter_type": conditions["encounter_type"],
            "ownship": conditions["ownship"],
            "target_ships": conditions["target_ships"],
            "selection_criterion": conditions["selection_rule"]["description"],
            "routes": {
                "A": make_route_info("A", data_a),
                "B": make_route_info("B", data_b),
            },
            "correct_route": correct_route,
        },
        "evaluation_reference": {
            "correct_route": correct_route,
            "reason": conditions["selection_rule"]["reason"],
            "important_facts": [
                "経路Aは基準Bumperの侵入判定を満たす",
                "経路Bは基準Bumperの侵入判定を満たさない",
                "経路Bは経路Aより短い",
                "経路Aの最小離隔距離は経路Bより大きい",
            ],
        },
        "notes": [
            "正解経路は実験用の基準で設定した",
            "安全性判定は基準Bumperへの侵入有無による",
            "最小離隔距離は一定速力・区間直線航行の近似値",
            "COLREGsへの適合性を保証するものではない",
            "BCRは現時点では未計算",
        ],
    }
    return output, "theta_001_llm_input.json"


def main():
    parser = argparse.ArgumentParser(description="Theta*経路比較結果をLLM入力JSONに変換")
    parser.add_argument(
        "--scenario", choices=("crossing", "headon", "overtaking"), default="crossing",
        help="対象シナリオ。初期値は横切り(crossing)",
    )
    args = parser.parse_args()

    builders = {
        "crossing": build_crossing,
        "headon": build_headon,
        "overtaking": build_overtaking,
    }
    output, output_name = builders[args.scenario]()
    output_path = save_json(output_name, output)
    llm_input = output["llm_input"]
    print("【Qwen入力データ作成結果】")
    print("シナリオ:", llm_input["scenario_id"])
    print("遭遇種別:", llm_input["encounter_type"])
    print("正解経路:", llm_input["correct_route"])
    print("判断基準:", llm_input["selection_criterion"])
    for route_id, route in llm_input["routes"].items():
        print(f"\n経路{route_id}")
        print("  経路長:", route["route_length_m"], "m")
        print("  最小離隔距離:", route["min_separation_m"], "m")
        print("  基準Bumper判定:", route["valid_under_baseline_bumper"])
    print("\n保存完了:", output_path)


if __name__ == "__main__":
    main()
