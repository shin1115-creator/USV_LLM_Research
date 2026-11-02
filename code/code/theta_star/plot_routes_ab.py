import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data" / "theta_star"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

def load_json(filename):
    with open(DATA_DIR / filename, "r", encoding="utf-8") as f:
        return json.load(f)

def heading_to_velocity(speed, heading_deg):
    angle = math.radians(heading_deg)
    vx = speed * math.cos(angle)
    vy = speed * math.sin(angle)
    return vx, vy

def analyze_route_min_sep(route, ownship, target):
    """
    route: [{"x_m":..., "y_m":...}, ...]
    ownship, target: dict
    戻り値:
      {
        "route_length_m": ...,
        "travel_time_sec": ...,
        "min_sep_m": ...,
        "min_time_sec": ...,
        "own_point": {"x_m":..., "y_m":...},
        "target_point": {"x_m":..., "y_m":...}
      }
    """
    own_speed = ownship["speed_m_s"]
    tvx, tvy = heading_to_velocity(
        target["speed_m_s"],
        target["heading_deg"]
    )

    total_length = 0.0
    elapsed = 0.0

    min_distance = float("inf")
    min_time = None
    min_own_point = None
    min_target_point = None

    for a, b in zip(route[:-1], route[1:]):
        ax = a["x_m"]
        ay = a["y_m"]
        bx = b["x_m"]
        by = b["y_m"]

        dx = bx - ax
        dy = by - ay
        seg_len = math.hypot(dx, dy)

        if seg_len == 0:
            continue

        duration = seg_len / own_speed

        ovx = dx / duration
        ovy = dy / duration

        # 区間開始時点での相対位置
        rx = target["x_m"] + tvx * elapsed - ax
        ry = target["y_m"] + tvy * elapsed - ay

        rvx = tvx - ovx
        rvy = tvy - ovy
        rv2 = rvx**2 + rvy**2

        if rv2 > 0:
            t_local = -(rx * rvx + ry * rvy) / rv2
            t_local = max(0.0, min(duration, t_local))
        else:
            t_local = 0.0

        own_x = ax + ovx * t_local
        own_y = ay + ovy * t_local

        target_x = target["x_m"] + tvx * (elapsed + t_local)
        target_y = target["y_m"] + tvy * (elapsed + t_local)

        distance = math.hypot(target_x - own_x, target_y - own_y)

        if distance < min_distance:
            min_distance = distance
            min_time = elapsed + t_local
            min_own_point = {"x_m": own_x, "y_m": own_y}
            min_target_point = {"x_m": target_x, "y_m": target_y}

        total_length += seg_len
        elapsed += duration

    return {
        "route_length_m": total_length,
        "travel_time_sec": elapsed,
        "min_sep_m": min_distance,
        "min_time_sec": min_time,
        "own_point": min_own_point,
        "target_point": min_target_point
    }

# データ読み込み
data_a = load_json("theta_001.json")
data_b = load_json("theta_001_B.json")
conditions = load_json("experiment_conditions_AB.json")

ownship = conditions["ownship"]
target = conditions["target_ships"][0]
goal = conditions["goal"]

route_a = data_a["route"]
route_b = data_b["route"]

# 最小離隔解析
result_a = analyze_route_min_sep(route_a, ownship, target)
result_b = analyze_route_min_sep(route_b, ownship, target)

# 他船の予測航跡を描くため、長い方の時間に合わせる
max_time = max(
    result_a["travel_time_sec"],
    result_b["travel_time_sec"]
)

tvx, tvy = heading_to_velocity(
    target["speed_m_s"],
    target["heading_deg"]
)

target_line_x = [
    target["x_m"],
    target["x_m"] + tvx * max_time
]
target_line_y = [
    target["y_m"],
    target["y_m"] + tvy * max_time
]

# 描画
fig, ax = plt.subplots(figsize=(10, 8))
ax.axis("equal")
ax.grid()

# main.py と同じ向きに合わせる
# 横軸: y, 縦軸: x

# 経路A
ax.plot(
    [p["y_m"] for p in route_a],
    [p["x_m"] for p in route_a],
    color="black",
    linewidth=2,
    label="Route A"
)

# 経路B
ax.plot(
    [p["y_m"] for p in route_b],
    [p["x_m"] for p in route_b],
    color="black",
    linestyle="--",
    linewidth=2,
    label="Route B"
)

# 他船の予測航跡
ax.plot(
    target_line_y,
    target_line_x,
    color="red",
    linewidth=1.5,
    label="Target ship track"
)

# 自船スタート
ax.plot(
    ownship["y_m"],
    ownship["x_m"],
    "s",
    color="blue",
    ms=9,
    label="Own ship start"
)

# 他船スタート
ax.plot(
    target["y_m"],
    target["x_m"],
    "o",
    color="red",
    ms=8,
    label="Target ship start"
)

# ゴール
ax.plot(
    goal["y_m"],
    goal["x_m"],
    "*",
    color="green",
    ms=15,
    label="Goal"
)

# Aの最小離隔地点
a_own = result_a["own_point"]
a_target = result_a["target_point"]

ax.plot(
    [a_own["y_m"], a_target["y_m"]],
    [a_own["x_m"], a_target["x_m"]],
    color="blue",
    linewidth=1.5
)

ax.plot(
    a_own["y_m"],
    a_own["x_m"],
    "o",
    color="blue",
    ms=7,
    label="A min-sep point (own)"
)

ax.plot(
    a_target["y_m"],
    a_target["x_m"],
    "x",
    color="blue",
    ms=8,
    label="A min-sep point (target)"
)

ax.text(
    a_own["y_m"],
    a_own["x_m"] + 250,
    f"A min: {result_a['min_sep_m']:.1f} m",
    color="blue",
    fontsize=9
)

# Bの最小離隔地点
b_own = result_b["own_point"]
b_target = result_b["target_point"]

ax.plot(
    [b_own["y_m"], b_target["y_m"]],
    [b_own["x_m"], b_target["x_m"]],
    color="purple",
    linewidth=1.5
)

ax.plot(
    b_own["y_m"],
    b_own["x_m"],
    "o",
    color="purple",
    ms=7,
    label="B min-sep point (own)"
)

ax.plot(
    b_target["y_m"],
    b_target["x_m"],
    "x",
    color="purple",
    ms=8,
    label="B min-sep point (target)"
)

ax.text(
    b_own["y_m"],
    b_own["x_m"] - 350,
    f"B min: {result_b['min_sep_m']:.1f} m",
    color="purple",
    fontsize=9
)

ax.set_title("Route A / Route B comparison")
ax.set_xlabel("Y position [m]")
ax.set_ylabel("X position [m]")
ax.legend(loc="best")

output_path = RESULTS_DIR / "theta_001_routes_AB_main_style.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print("【描画結果】")
print(f"Route A 最小離隔距離: {result_a['min_sep_m']:.2f} m")
print(f"Route B 最小離隔距離: {result_b['min_sep_m']:.2f} m")
print(f"図を保存しました: {output_path}")