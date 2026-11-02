
import json
from pathlib import Path

from coordinate import XY
from planner import Planner, Ship, Bumper

# ファイルの場所
project_dir = Path(__file__).resolve().parents[2]
json_path = project_dir / "data" / "theta_star" / "theta_001.json"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 経路座標を取得
route = [
    XY(p["x_m"], p["y_m"])
    for p in data["route"]
]

# 自船情報
own = data["ownship"]

# 他船情報
targets = [
    Ship(
        XY(s["x_m"], s["y_m"]),
        s["heading_deg"],
        s["speed_m_s"]
    )
    for s in data["target_ships"]
]

# main.pyと同じBumper設定
bumper = Bumper(
    fore=1.0 * 1852.0,
    aft=0.5 * 1852.0,
    stbd=0.8 * 1852.0,
    port=0.5 * 1852.0
)

planner = Planner(
    grid_interval=500.0,
    fore_range=[-1.0*1852.0, 11.0*1852.0],
    side_range=[-10.0*1852.0, 10.0*1852.0],
    bumper=bumper
)

# 経路がBumperへの侵入を避けるか確認
is_valid = planner.check_path(
    own_speed=own["speed_m_s"],
    path=route,
    targets=targets
)

print("【経路のBumper侵入判定】")
print("シナリオ:", data["scenario_id"])
print("経路の判定:", is_valid)

if is_valid:
    print("Bumper侵入なし（既存の判定条件）")
else:
    print("Bumper侵入または経路上の障害物が検出されました")
