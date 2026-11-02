
import json
from pathlib import Path

from coordinate import XY
from planner import Bumper, Planner, Ship

# 1. ファイルの場所を設定
NM_TO_M = 1852.0

PROJECT_DIR = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_DIR / "data" / "theta_star"
    / "experiment_conditions_AB.json"
)

OUTPUT_PATH = (
    PROJECT_DIR / "data" / "theta_star"
    / "theta_001_B.json"
)

# 2. 実験条件を読み込む
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    conditions = json.load(f)

settings = conditions["route_B"]

if settings is None:
    raise ValueError("経路Bの設定がありません。")

# 3. 自船の設定
own = conditions["ownship"]

ownship = Ship(
    XY(own["x_m"], own["y_m"]),
    own["heading_deg"],
    own["speed_m_s"]
)

# 4. 他船の設定
targets = [
    Ship(
        XY(s["x_m"], s["y_m"]),
        s["heading_deg"],
        s["speed_m_s"]
    )
    for s in conditions["target_ships"]
]

# 5. 目的地の設定
goal = XY(
    conditions["goal"]["x_m"],
    conditions["goal"]["y_m"]
)

# 6. 経路BのBumper設定
bumper_nm = settings["bumper_nm"]

bumper = Bumper(
    fore=bumper_nm["fore"] * NM_TO_M,
    aft=bumper_nm["aft"] * NM_TO_M,
    stbd=bumper_nm["stbd"] * NM_TO_M,
    port=bumper_nm["port"] * NM_TO_M
)

# 7. Theta*の探索条件
p = settings["planner"]

planner = Planner(
    grid_interval=p["grid_interval_m"],
    fore_range=[
        value * NM_TO_M
        for value in p["fore_range_nm"]
    ],
    side_range=[
        value * NM_TO_M
        for value in p["side_range_nm"]
    ],
    bumper=bumper
)

# 8. 経路Bを探索
route_b = planner.planning(
    goal=goal,
    ownship=ownship,
    targets=targets,
    delay_time=p["delay_time_sec"]
)

if route_b is None:
    raise RuntimeError(
        "経路Bが生成できませんでした。"
    )

# 9. 経路長を計算
route_length = sum(
    (b - a).dist()
    for a, b in zip(route_b[:-1], route_b[1:])
)

# 10. Bumper侵入判定
bumper_valid = planner.check_path(
    own_speed=ownship.speed,
    path=route_b,
    targets=targets
)

# 11. JSONに保存
result = {
    "scenario_id": conditions["experiment_id"],
    "route_id": "B",
    "source": "theta_star",
    "planning_time_sec": 0,
    "ownship": own,
    "target_ships": conditions["target_ships"],
    "goal": conditions["goal"],
    "bumper_nm": bumper_nm,
    "planner": p,
    "route": [
        {"x_m": wp.x, "y_m": wp.y}
        for wp in route_b
    ],
    "route_length_m": round(route_length, 2),
    "bumper_valid_under_B_settings": bumper_valid
}

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

# 12. 結果表示
print("【Theta*による経路B生成結果】")

for i, wp in enumerate(route_b):
    print(
        f"WP{i}: ({wp.x:.2f}, {wp.y:.2f}) m"
    )

print(f"経路長: {route_length:.2f} m")
print(f"Bumper判定: {bumper_valid}")
print(f"保存完了: {OUTPUT_PATH}")
