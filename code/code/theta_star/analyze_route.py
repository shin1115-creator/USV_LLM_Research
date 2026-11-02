
import json
import math
from pathlib import Path

# 研究フォルダ
PROJECT_DIR = Path(__file__).resolve().parents[2]

INPUT_PATH = PROJECT_DIR / "data" / "theta_star" / "theta_001.json"
OUTPUT_PATH = PROJECT_DIR / "data" / "theta_star" / "theta_001_analysis.json"

# JSONを読み込む
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

own = data["ownship"]
targets = data["target_ships"]
route = data["route"]

if len(targets) != 1:
    raise ValueError("今回は他船1隻のシナリオのみ対応します。")

if len(route) < 2:
    raise ValueError("経路には2点以上の座標が必要です。")

target = targets[0]

# 針路と速力から速度ベクトルを計算
def velocity(speed, heading_deg):
    angle = math.radians(heading_deg)
    return (
        speed * math.cos(angle),
        speed * math.sin(angle)
    )

own_vx, own_vy = velocity(
    own["speed_m_s"], own["heading_deg"]
)

target_vx, target_vy = velocity(
    target["speed_m_s"], target["heading_deg"]
)

# =====================================
# 1. 初期針路を維持した場合のTCPA・DCPA
# =====================================

rx = target["x_m"] - own["x_m"]
ry = target["y_m"] - own["y_m"]

rvx = target_vx - own_vx
rvy = target_vy - own_vy

relative_speed_sq = rvx**2 + rvy**2

if relative_speed_sq > 0:
    tcpa = -(rx * rvx + ry * rvy) / relative_speed_sq
    dcpa = math.hypot(
        rx + rvx * tcpa,
        ry + rvy * tcpa
    )
else:
    tcpa = None
    dcpa = math.hypot(rx, ry)

# =====================================
# 2. 経路の総延長と最小離隔距離
# =====================================

own_speed = own["speed_m_s"]

if own_speed <= 0:
    raise ValueError("自船速力は正の値が必要です。")

total_length = 0.0
elapsed = 0.0

min_distance = float("inf")
min_distance_time = None

for a, b in zip(route[:-1], route[1:]):

    dx = b["x_m"] - a["x_m"]
    dy = b["y_m"] - a["y_m"]

    length = math.hypot(dx, dy)

    if length == 0:
        continue

    duration = length / own_speed

    # この区間を進む自船の速度
    vx = dx / duration
    vy = dy / duration

    # 区間開始時の他船との相対位置
    rx = target["x_m"] + target_vx * elapsed - a["x_m"]
    ry = target["y_m"] + target_vy * elapsed - a["y_m"]

    # 相対速度
    rvx = target_vx - vx
    rvy = target_vy - vy

    v2 = rvx**2 + rvy**2

    if v2 > 0:
        closest_t = -(rx * rvx + ry * rvy) / v2
        closest_t = max(0.0, min(duration, closest_t))
    else:
        closest_t = 0.0

    distance = math.hypot(
        rx + rvx * closest_t,
        ry + rvy * closest_t
    )

    if distance < min_distance:
        min_distance = distance
        min_distance_time = elapsed + closest_t

    total_length += length
    elapsed += duration

# =====================================
# 3. 結果表示
# =====================================

print("【Theta*経路解析結果】")
print(f"シナリオ：{data['scenario_id']}")
print(f"経路長：{total_length:.2f} m")
print(f"航行所要時間：{elapsed:.2f} 秒")

if tcpa is not None:
    print(f"初期針路維持時のTCPA：{tcpa:.2f} 秒")
    print(f"初期針路維持時のDCPA：{dcpa:.2f} m")
else:
    print("初期相対速度が0のためTCPAは未定義")

print(f"経路上の最小離隔距離：{min_distance:.2f} m")
print(f"最小離隔時刻：{min_distance_time:.2f} 秒")

# =====================================
# 4. JSONに保存
# =====================================

analysis = {
    "scenario_id": data["scenario_id"],
    "source": "theta_star_route_analysis",
    "route_length_m": round(total_length, 2),
    "travel_time_sec": round(elapsed, 2),
    "initial_course_tcpa_sec": (
        round(tcpa, 2) if tcpa is not None else None
    ),
    "initial_course_dcpa_m": round(dcpa, 2),
    "route_min_separation_m": round(min_distance, 2),
    "route_min_separation_time_sec": round(min_distance_time, 2),
    "assumptions": [
        "自船は各経路区間を一定速力で直線航行する",
        "他船は一定針路・一定速力で航行する",
        "船舶の旋回運動は考慮しない",
        "TCPA・DCPAは初期針路維持の場合の値"
    ]
}

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(analysis, f, ensure_ascii=False, indent=2)

print(f"\n解析結果保存完了：{OUTPUT_PATH}")
