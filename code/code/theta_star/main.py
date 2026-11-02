#全体のシミュレーション
import math
import json
from pathlib import Path
from matplotlib import pyplot as plt

from ship_model import ShipModel
from coordinate import XY, adjust_180
from planner import Ship, Bumper, Planner

#----- 船設定 -----#
ownship = ShipModel(K=1.0, T=7.0, x=0.0, y=0.0, heading=0.0, u=5.0)
otherships = [
    ShipModel(K=1.0, T=7.0, x=5.0*1852.0, y=5.0*1852.0, heading=270.0, u=5.0)
]

goal = XY(8.0 * 1852, 0.0)

ownship_log = {"x": [], "y": [], "heading": [], "u": [], "r": [], "rud": []}
otherships_log = [{"x": [], "y": [], "heading": [], "u": [], "r": [], "rud": []} for _ in range(len(otherships))]

#----- 避航航路探索設定 -----#
bumper = Bumper(
    fore = 1.0 * 1852.0,
    aft  = 0.5 * 1852.0,
    stbd = 0.8 * 1852.0,
    port = 0.5 * 1852.0
)

planner = Planner(
    grid_interval   = 500.0,
    fore_range      = [-1.0*1852.0, 11.0*1852.0],
    side_range      = [-10.0*1852.0, 10.0*1852.0],
    bumper          = bumper
)

#----- 制御用パラメーター -----#
lookahead = 300.0
wop = 100.0
rud_limit = 15.0
Kp = 1.0
Kd = 12.0

# ログ保存用関数
def append_log(log: dict, ship_model: ShipModel):
    log["x"].append(ship_model.x)
    log["y"].append(ship_model.y)
    log["heading"].append(ship_model.heading)
    log["u"].append(ship_model.u)
    log["r"].append(ship_model.r)
    log["rud"].append(ship_model.rud)

append_log(ownship_log, ownship)
for j, othership in enumerate(otherships):
    append_log(otherships_log[j], othership)

#----- シミュレーションループ -----#
replan_interval = 60

plan            = None
target_wp_idx   = 1
for i in range(3600):
    # 避航航路計画
    if i % replan_interval == 0:
        own = Ship(XY(ownship.x, ownship.y), ownship.heading, ownship.u)
        targets = targets = [Ship(XY(othership.x, othership.y), othership.heading, othership.u) for othership in otherships]
        remain_plan = plan[target_wp_idx - 1:] if plan is not None else None

        # 航路計画
        new_plan = planner.planning(goal, own, targets, plan=remain_plan, delay_time=10.0)

        if new_plan is None:
            if plan is None:
                raise RuntimeError("避航航路は計算できませんでした")
        else:
            plan = new_plan
            target_wp_idx = 1
        
        # 最初の経路探索結果をJSONに保存
        if i == 0 and new_plan is not None:

            project_dir = Path(__file__).resolve().parents[2]
            output_dir = project_dir / "data" / "theta_star"
            output_dir.mkdir(parents=True, exist_ok=True)

            route_data = {
                "scenario_id": "theta_001",
                "source": "theta_star",
                "planning_time_sec": i,
                "ownship": {
                    "x_m": ownship.x,
                    "y_m": ownship.y,
                    "heading_deg": ownship.heading,
                    "speed_m_s": ownship.u
                },
                "target_ships": [
                    {
                        "x_m": ship.x,
                        "y_m": ship.y,
                        "heading_deg": ship.heading,
                        "speed_m_s": ship.u
                    }
                    for ship in otherships
                ],
                "route": [
                    {"x_m": wp.x, "y_m": wp.y}
                    for wp in new_plan
                ]
            }

            output_path = output_dir / "theta_001.json"

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(
                    route_data, f,
                    ensure_ascii=False, indent=2
                )

            print(f"経路データ保存完了: {output_path}")

    # 追従制御
    own_pos = XY(ownship.x, ownship.y)
    course  = plan[target_wp_idx].angle_from(plan[target_wp_idx - 1])
    diff    = own_pos.rotate_from(course, plan[target_wp_idx])
    remain  = -diff.x   # 残り距離
    dev     = diff.y    # 航路偏差

    target_course   = course - math.degrees(math.atan2(dev, lookahead))
    course_d        = adjust_180(target_course - ownship.heading)
    rud             = Kp * course_d - Kd * ownship.r
    if rud > rud_limit:
        rud = rud_limit
    elif rud < -rud_limit:
        rud = -rud_limit
    ownship.rud = rud

    # WP更新
    if remain < wop and target_wp_idx < len(plan) - 1:
        target_wp_idx += 1

    # 船を進める
    ownship.step(1.0)
    append_log(ownship_log, ownship)
    for j, othership in enumerate(otherships):
        othership.step(1.0)
        append_log(otherships_log[j], othership)

    # ゴール到達判定
    if goal.dist_from(XY(ownship.x, ownship.y)) < 100:
        break

fig, ax = plt.subplots()
ax.axis("equal")
ax.grid()
ax.plot([wp.y for wp in plan], [wp.x for wp in plan], color="black")
ax.plot(ownship_log["y"], ownship_log["x"])
for othership_log in otherships_log:
    ax.plot(othership_log["y"], othership_log["x"])

ax.plot(ownship_log["y"][0], ownship_log["x"][0], "s", color="blue", ms=9)
ax.plot(goal.y, goal.x, "*", color="green", ms=15)

plt.show()
