
import json
import math
from pathlib import Path

from coordinate import XY
from planner import Bumper, Planner, Ship

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data" / "theta_star"


def load_json(filename):
    with open(DATA_DIR / filename, "r", encoding="utf-8") as f:
        return json.load(f)


# 既存のファイルを読み込む
conditions = load_json("experiment_conditions_AB.json")
data_a = load_json("theta_001.json")
data_b = load_json("theta_001_B.json")

own = conditions["ownship"]
speed = own["speed_m_s"]

if speed <= 0:
    raise ValueError("自船速力は0より大きくしてください。")

# 比較条件が一致しているか確認
for data in (data_a, data_b):
    if (data["ownship"] != own
            or data["target_ships"] != conditions["target_ships"]):
        raise ValueError("自船・他船の初期条件が一致しません。")

routes = {
    "A": [XY(p["x_m"], p["y_m"]) for p in data_a["route"]],
    "B": [XY(p["x_m"], p["y_m"]) for p in data_b["route"]]
}

targets = [
    Ship(
        XY(s["x_m"], s["y_m"]),
        s["heading_deg"],
        s["speed_m_s"]
    )
    for s in conditions["target_ships"]
]


def create_planner(route_id):
    settings = conditions[f"route_{route_id}"]
    b = settings["bumper_nm"]
    p = settings["planner"]

    return Planner(
        grid_interval=p["grid_interval_m"],
        fore_range=[x * 1852 for x in p["fore_range_nm"]],
        side_range=[x * 1852 for x in p["side_range_nm"]],
        bumper=Bumper(
            fore=b["fore"] * 1852,
            aft=b["aft"] * 1852,
            stbd=b["stbd"] * 1852,
            port=b["port"] * 1852
        )
    )


planner_a = create_planner("A")
planner_b = create_planner("B")


def evaluate_route(route):
    elapsed = 0.0
    total_length = 0.0
    min_distance = float("inf")

    for a, b in zip(route[:-1], route[1:]):
        distance = (b - a).dist()

        if distance == 0:
            continue

        duration = distance / speed
        own_velocity = (b - a) * (1.0 / duration)

        for target in targets:
            target_velocity = XY(*target.velocity)

            relative_position = (
                target.pos
                + target_velocity * elapsed
                - a
            )
            relative_velocity = target_velocity - own_velocity

            velocity_sq = relative_velocity.dot(relative_velocity)

            if velocity_sq > 0:
                t = (
                    -relative_position.dot(relative_velocity)
                    / velocity_sq
                )
                t = max(0.0, min(duration, t))
            else:
                t = 0.0

            closest_distance = (
                relative_position + relative_velocity * t
            ).dist()

            min_distance = min(min_distance, closest_distance)

        total_length += distance
        elapsed += duration

    return {
        "route_length_m": round(total_length, 2),
        "travel_time_sec": round(elapsed, 2),
        "min_separation_m": (
            round(min_distance, 2) if targets else None
        )
    }


results = {}

for route_id, route in routes.items():
    result = evaluate_route(route)

    result["valid_under_A_bumper"] = planner_a.check_path(
        own_speed=speed, path=route, targets=targets
    )

    result["valid_under_B_bumper"] = planner_b.check_path(
        own_speed=speed, path=route, targets=targets
    )

    results[route_id] = result

    print(f"\n【経路{route_id}】")
    print(f"経路長: {result['route_length_m']:.2f} m")
    print(f"航行時間: {result['travel_time_sec']:.2f} 秒")
    print(f"最小離隔距離: {result['min_separation_m']:.2f} m")
    print(f"基準Bumper判定: {result['valid_under_A_bumper']}")
    print(f"縮小Bumper判定: {result['valid_under_B_bumper']}")


# 比較結果を保存
output = {
    "experiment_id": conditions["experiment_id"],
    "comparison_basis": "same_initial_conditions",
    "assumptions": [
        "自船は各経路区間を一定速力で直線航行",
        "他船は一定針路・一定速力",
        "旋回運動は考慮しない",
        "Bumper判定は提供されたplanner.pyを使用"
    ],
    "results": results
}

output_path = DATA_DIR / "theta_001_AB_comparison.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\n比較結果保存完了: {output_path}")
