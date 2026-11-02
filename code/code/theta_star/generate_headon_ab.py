"""行会いシナリオに対してTheta*で経路A/Bを生成し、同じ条件で比較する。

配置先: code/theta_star/generate_headon_ab.py
実行方法: python code\\theta_star\\generate_headon_ab.py
注意: 経路A/Bのいずれも「正解」とは事前に決めない。
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Optional

from coordinate import XY
from planner import Bumper, Planner, Ship

NM = 1852.0
SCENARIO_ID = "theta_headon_001"

# 既存のmain.pyの定義を引き継ぐ。X軸は船首方向、Y軸は右舷方向。
OWN = Ship(XY(0.0, 0.0), course=0.0, speed=5.0)
TARGET = Ship(XY(5 * NM, 0.0), course=180.0, speed=5.0)
GOAL = XY(8 * NM, 0.0)
TARGETS = [TARGET]

GRID_INTERVAL_M = 500.0
FORE_RANGE_M = (-1.0 * NM, 11.0 * NM)
SIDE_RANGE_M = (-10.0 * NM, 10.0 * NM)
DELAY_TIME_S = 10.0

BUMPER_A = Bumper(fore=1.0 * NM, aft=0.5 * NM, stbd=0.8 * NM, port=0.5 * NM)
BUMPER_B = Bumper(**{k: getattr(BUMPER_A, k) * 0.7 for k in ("fore", "aft", "stbd", "port")})


def make_planner(bumper: Bumper) -> Planner:
    return Planner(
        grid_interval=GRID_INTERVAL_M,
        fore_range=list(FORE_RANGE_M),
        side_range=list(SIDE_RANGE_M),
        bumper=bumper,
    )


def path_length(path: list[XY]) -> float:
    return sum(a.dist_from(b) for a, b in zip(path[:-1], path[1:]))


def minimum_separation(path: list[XY], own_speed: float, target: Ship) -> tuple[float, float]:
    """区間直線・等速航行を仮定し、同時刻の船間距離の最小値と時刻を計算。"""
    t_elapsed = 0.0
    best_distance = math.inf
    best_time = 0.0
    v_target = XY(*target.velocity)

    for start, finish in zip(path[:-1], path[1:]):
        segment = finish - start
        seg_len = segment.dist()
        if seg_len == 0:
            continue
        dt = seg_len / own_speed
        v_own = segment * (own_speed / seg_len)
        target_at_start = target.pos + v_target * t_elapsed
        relative_initial = start - target_at_start
        relative_velocity = v_own - v_target
        denom = relative_velocity.dot(relative_velocity)
        tau = 0.0 if denom < 1e-12 else -relative_initial.dot(relative_velocity) / denom
        tau = max(0.0, min(dt, tau))
        distance = (relative_initial + relative_velocity * tau).dist()
        if distance < best_distance:
            best_distance = distance
            best_time = t_elapsed + tau
        t_elapsed += dt
    return best_distance, best_time


def xy_dict(p: XY) -> dict:
    return {"x": round(p.x, 3), "y": round(p.y, 3)}


def bumper_dict(b: Bumper) -> dict:
    return {k: round(getattr(b, k), 3) for k in ("fore", "aft", "stbd", "port")}


def generate_and_evaluate(label: str, bumper: Bumper) -> dict:
    planner = make_planner(bumper)
    path: Optional[list[XY]] = planner.planning(GOAL, OWN, TARGETS, delay_time=DELAY_TIME_S)
    if path is None:
        return {"label": label, "found": False, "reason": "指定された探索範囲内で経路が見つからない"}

    min_distance, t_min = minimum_separation(path, OWN.speed, TARGET)
    length = path_length(path)
    return {
        "label": label,
        "found": True,
        "generation_bumper": bumper_dict(bumper),
        "waypoints": [xy_dict(p) for p in path],
        "path_length_m": round(length, 2),
        "travel_time_s_straight_segments": round(length / OWN.speed, 2),
        "minimum_separation_m": round(min_distance, 2),
        "minimum_separation_time_s": round(t_min, 2),
        "valid_under_baseline_bumper": make_planner(BUMPER_A).check_path(OWN.speed, path, TARGETS),
        "valid_under_reduced_bumper": make_planner(BUMPER_B).check_path(OWN.speed, path, TARGETS),
    }


def main():
    result = {
        "scenario_id": SCENARIO_ID,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "scenario": {
            "ownship": {"position": xy_dict(OWN.pos), "course_deg": OWN.course, "speed_mps": OWN.speed},
            "target": {"position": xy_dict(TARGET.pos), "course_deg": TARGET.course, "speed_mps": TARGET.speed},
            "goal": xy_dict(GOAL),
            "grid_interval_m": GRID_INTERVAL_M,
            "fore_range_m": list(FORE_RANGE_M),
            "side_range_m": list(SIDE_RANGE_M),
            "delay_time_s": DELAY_TIME_S,
            "baseline_bumper": bumper_dict(BUMPER_A),
            "reduced_bumper": bumper_dict(BUMPER_B),
        },
        "routes": {
            "A": generate_and_evaluate("A", BUMPER_A),
            "B": generate_and_evaluate("B", BUMPER_B),
        },
        "correct_route": None,
        "notes": [
            "正解経路は未設定。両経路の判定を確認した後で判断基準を定義する。",
            "最小離隔距離・航行時間は区間直線・等速航行近似であり、船体運動モデルの追従航跡とは異なる。",
            "Bumper適合はシミュレーション上の判定であり、実船の航行安全性を保証しない。",
        ],
    }

    # このスクリプトはcode/theta_starに置く想定。上位2階層がプロジェクトルート。
    project_root = Path(__file__).resolve().parents[2]
    output = project_root / "data" / "theta_star" / f"{SCENARIO_ID}_AB_comparison.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"シナリオ: {SCENARIO_ID}")
    for label, entry in result["routes"].items():
        if not entry["found"]:
            print(f"経路{label}: 生成不可（{entry['reason']}）")
            continue
        print(f"経路{label}: 経由点 {len(entry['waypoints'])}点")
        print(f"  経路長: {entry['path_length_m']:.2f} m")
        print(f"  最小離隔距離: {entry['minimum_separation_m']:.2f} m")
        print(f"  基準Bumper判定: {entry['valid_under_baseline_bumper']}")
        print(f"  縮小Bumper判定: {entry['valid_under_reduced_bumper']}")
        print(f"  経由点: {entry['waypoints']}")
    print("正解経路: 未設定")
    print(f"保存先: {output}")


if __name__ == "__main__":
    main()
