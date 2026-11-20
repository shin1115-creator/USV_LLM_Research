"""USV Theta* 共通比較プログラム（PCシミュレーション用）。

目的:
    Bumper、船速、遭遇条件を「実験設定」の値だけ変えて繰り返し比較する。
    ①初期Theta*計画、②main.py相当の船体運動・60秒再計画、③図とJSONを保存する。
    経路の採用はPythonの客観的な指標で決定し、説明文は生成しない。
    LLMには別途、採用経路と計算された事実のみ渡す想定。

配置例: USV_LLM_Research/code/theta_star/run_route_comparison.py
必要: 同じフォルダに planner.py, coordinate.py, ship_model.py
実行: python code\\theta_star\\run_route_comparison.py
      python code\\theta_star\\run_route_comparison.py --no-show

注意:
    • Bumperは経路探索条件のみ。生成後のBumper適合判定は実施しない。
    • 初期計画経路の数値と、船体運動モデルの追従航跡の数値は区別する。
    • 右舷／左舷は「自船の初期針路に対する横偏差」で判定する。
      他船のどちら側を通過したか、COLREGs適否、操舵方向を意味しない。
    • MAX_SIM_SECONDSに達しても目的地未到達なら、シミュレーション時間は所要時間ではない。
    • 現実の航行安全性を保証するシミュレーションではない。
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager

from coordinate import XY, adjust_180
from planner import Bumper, Planner, Ship
from ship_model import ShipModel


# ============================================================================
# ▼ 実験設定：基本的にはここだけ変更する
# ============================================================================
NM = 1852.0  # 1海里 [m]
SCENARIO_ID = "crossing_all130_u7"

# 自船・他船の初期位置、針路 [deg]、船速 [m/s]
OWN_INIT = {"x": 0.0, "y": 0.0, "heading": 0.0, "u": 7.0}
TARGET_INIT = {"x": 5.0 * NM, "y": 5.0 * NM, "heading": 270.0, "u": 5.0}
GOAL = {"x": 8.0 * NM, "y": 0.0}

# ベースのBumper寸法 [海里]
BASE_BUMPER_NM = {"fore": 1.0, "aft": 0.5, "stbd": 0.8, "port": 0.5}

# AとBのBumper倍率。ここを書き換えるだけで方向別感度分析ができる。
# 例：船首だけ130% -> Bのforeを1.3
# 例：右舷だけ130% -> Bのstbdを1.3
# 例：左舷だけ130% -> Bのportを1.3
# 例：全方向130% -> Bの4項目をすべて1.3
BUMPER_MULTIPLIERS = {
    "A": {"fore": 1.0, "aft": 1.0, "stbd": 1.0, "port": 1.0},
    "B": {"fore": 1.3, "aft": 1.3, "stbd": 1.3, "port": 1.3},
}

# Theta*と追従制御。通常はmain.pyの設定に合わせて固定する。
GRID_INTERVAL_M = 500.0
FORE_RANGE_NM = (-1.0, 11.0)
SIDE_RANGE_NM = (-10.0, 10.0)
DELAY_TIME_S = 10.0
LOOKAHEAD_M = 300.0
WAYPOINT_SWITCH_M = 100.0
RUD_LIMIT_DEG = 15.0
KP = 1.0
KD = 12.0
REPLAN_INTERVAL_S = 60
MAX_SIM_SECONDS = 6000
GOAL_TOLERANCE_M = 100.0
SIDE_EPSILON_M = 1.0
# ============================================================================


def pxy(p: XY) -> dict:
    return {"x": round(p.x, 3), "y": round(p.y, 3)}


def bumper_from_multipliers(route_id: str) -> Bumper:
    multipliers = BUMPER_MULTIPLIERS[route_id]
    return Bumper(**{
        side: BASE_BUMPER_NM[side] * NM * multipliers[side]
        for side in ("fore", "aft", "stbd", "port")
    })


def bumper_dict(bumper: Bumper) -> dict:
    return {side: getattr(bumper, side) for side in ("fore", "aft", "stbd", "port")}


def planner_with(bumper: Bumper) -> Planner:
    return Planner(
        grid_interval=GRID_INTERVAL_M,
        fore_range=[value * NM for value in FORE_RANGE_NM],
        side_range=[value * NM for value in SIDE_RANGE_NM],
        bumper=bumper,
    )


def input_state() -> tuple[Ship, Ship, XY]:
    own = Ship(XY(OWN_INIT["x"], OWN_INIT["y"]), OWN_INIT["heading"], OWN_INIT["u"])
    target = Ship(XY(TARGET_INIT["x"], TARGET_INIT["y"]), TARGET_INIT["heading"], TARGET_INIT["u"])
    goal = XY(GOAL["x"], GOAL["y"])
    return own, target, goal


def into_initial_frame(p: XY) -> XY:
    return p.rotate_from(OWN_INIT["heading"], XY(OWN_INIT["x"], OWN_INIT["y"]))


def bypass_side(points: list[XY]) -> str:
    """自船の初期針路の右(+Y)、左(-Y)に対する迂回側を返す。"""
    ys = [into_initial_frame(p).y for p in points]
    starboard = any(y > SIDE_EPSILON_M for y in ys)
    port = any(y < -SIDE_EPSILON_M for y in ys)
    if starboard and port:
        return "両側"
    if starboard:
        return "右舷側"
    if port:
        return "左舷側"
    return "直進付近"


def initial_plan(bumper: Bumper) -> list[XY] | None:
    own, target, goal = input_state()
    path = planner_with(bumper).planning(goal, own, [target], delay_time=DELAY_TIME_S)
    if path is None or len(path) < 2:
        return None
    return path


def planned_minimum_separation(path: list[XY]) -> tuple[float, float]:
    """自船が経由点間を一定速力で直進すると仮定した最小船間距離。"""
    _, target, _ = input_state()
    speed = OWN_INIT["u"]
    target_v = XY(*target.velocity)
    elapsed, closest, closest_t = 0.0, math.inf, 0.0
    for a, b in zip(path[:-1], path[1:]):
        d = b - a
        length = d.dist()
        if length <= 0:
            continue
        duration = length / speed
        own_v = d * (speed / length)
        rel = a - (target.pos + target_v * elapsed)
        relative_v = own_v - target_v
        rv2 = relative_v.dot(relative_v)
        tau = 0.0 if rv2 < 1e-12 else -rel.dot(relative_v) / rv2
        tau = max(0.0, min(duration, tau))
        sep = (rel + relative_v * tau).dist()
        if sep < closest:
            closest, closest_t = sep, elapsed + tau
        elapsed += duration
    return closest, closest_t


def simulate_tracking(bumper: Bumper) -> dict:
    """main.py相当：船体運動を1秒更新し、60秒ごとに経路を再計画。"""
    own = ShipModel(K=1.0, T=7.0, **OWN_INIT)
    other = ShipModel(K=1.0, T=7.0, **TARGET_INIT)
    planner = planner_with(bumper)
    goal = XY(GOAL["x"], GOAL["y"])
    own_track, target_track = [XY(own.x, own.y)], [XY(other.x, other.y)]
    plan = None
    waypoint_idx = 1
    replan_count = 0
    reached = False
    for i in range(MAX_SIM_SECONDS):
        if i % REPLAN_INTERVAL_S == 0:
            own_state = Ship(XY(own.x, own.y), own.heading, own.u)
            targets = [Ship(XY(other.x, other.y), other.heading, other.u)]
            remain_plan = plan[waypoint_idx - 1:] if plan is not None else None
            new_plan = planner.planning(
                goal, own_state, targets, plan=remain_plan, delay_time=DELAY_TIME_S
            )
            if new_plan is None:
                if plan is None:
                    return {"found": False, "reason": "開始時の経路再計画に失敗"}
            else:
                plan = new_plan
                waypoint_idx = 1
            replan_count += 1

        own_pos = XY(own.x, own.y)
        course = plan[waypoint_idx].angle_from(plan[waypoint_idx - 1])
        diff = own_pos.rotate_from(course, plan[waypoint_idx])
        remain = -diff.x
        dev = diff.y
        target_course = course - math.degrees(math.atan2(dev, LOOKAHEAD_M))
        course_d = adjust_180(target_course - own.heading)
        rud = KP * course_d - KD * own.r
        own.rud = max(-RUD_LIMIT_DEG, min(RUD_LIMIT_DEG, rud))
        if remain < WAYPOINT_SWITCH_M and waypoint_idx < len(plan) - 1:
            waypoint_idx += 1

        own.step(1.0)
        other.step(1.0)
        own_track.append(XY(own.x, own.y))
        target_track.append(XY(other.x, other.y))
        if goal.dist_from(XY(own.x, own.y)) < GOAL_TOLERANCE_M:
            reached = True
            break

    length = sum(q.dist_from(p) for p, q in zip(own_track[:-1], own_track[1:]))
    closest_idx = min(range(len(own_track)), key=lambda i: own_track[i].dist_from(target_track[i]))
    return {
        "found": True,
        "own_track": own_track,
        "target_track": target_track,
        "length_m": length,
        "travel_time_s": len(own_track) - 1,
        "minimum_separation_m": own_track[closest_idx].dist_from(target_track[closest_idx]),
        "minimum_separation_time_s": closest_idx,
        "bypass_side": bypass_side(own_track),
        "goal_reached": reached,
        "replan_attempts": replan_count,
    }


def evaluate(route_id: str) -> tuple[dict, dict | None]:
    bumper = bumper_from_multipliers(route_id)
    path = initial_plan(bumper)
    if path is None:
        return {"found": False, "bumper_m": bumper_dict(bumper)}, None

    length = sum(q.dist_from(p) for p, q in zip(path[:-1], path[1:]))
    min_sep, min_t = planned_minimum_separation(path)
    sim = simulate_tracking(bumper)
    result = {
        "found": True,
        "bumper_m": bumper_dict(bumper),
        "planned_waypoints": [pxy(p) for p in path],
        "planned_length_m": length,
        "planned_travel_time_s": length / OWN_INIT["u"],
        "planned_minimum_separation_m": min_sep,
        "planned_minimum_separation_time_s": min_t,
        "planned_bypass_side": bypass_side(path),
        "simulation_found": sim["found"],
    }
    if sim["found"]:
        result.update({
            "tracked_length_m": sim["length_m"],
            "tracked_travel_time_s": sim["travel_time_s"],
            "tracked_minimum_separation_m": sim["minimum_separation_m"],
            "tracked_minimum_separation_time_s": sim["minimum_separation_time_s"],
            "tracked_bypass_side": sim["bypass_side"],
            "goal_reached": sim["goal_reached"],
            "replan_attempts": sim["replan_attempts"],
        })
    return result, {"plan": path, "simulation": sim}


def select_objective(routes: dict, metric: str, lower_is_better: bool) -> dict:
    """LLMではなく、事前に定めた数値基準で採用経路を決める。"""
    eligible = [(rid, r[metric]) for rid, r in routes.items() if r.get("found")]
    if not eligible:
        return {"selected_route": None, "status": "経路生成失敗", "metric": metric}
    if len(eligible) == 1:
        return {"selected_route": eligible[0][0], "status": "候補が1つのみ", "metric": metric}
    values = [value for _, value in eligible]
    if math.isclose(max(values), min(values), abs_tol=1e-6, rel_tol=0):
        return {"selected_route": None, "status": "同値：優劣なし", "metric": metric}
    chosen = min(eligible, key=lambda item: item[1]) if lower_is_better else max(eligible, key=lambda item: item[1])
    return {"selected_route": chosen[0], "status": "比較成立", "metric": metric}


def font_setup() -> None:
    available = {f.name for f in font_manager.fontManager.ttflist}
    for font in ("Yu Gothic", "Meiryo", "Noto Sans CJK JP", "Noto Sans Mono CJK JP", "IPAexGothic"):
        if font in available:
            plt.rcParams["font.family"] = font
            break
    plt.rcParams["axes.unicode_minus"] = False


def make_figure(details: dict, image_path: Path, zoom_path: Path, show: bool) -> None:
    """初期計画と追従航跡を左右に分け、拡大図も保存する。"""
    font_setup()
    colors = {"A": "#2166a5", "B": "#d17825"}
    start = into_initial_frame(XY(OWN_INIT["x"], OWN_INIT["y"]))
    goal = into_initial_frame(XY(GOAL["x"], GOAL["y"]))

    def draw(axes, zoom: bool):
        pts = []
        for rid, detail in details.items():
            if detail is None:
                continue
            plan = [into_initial_frame(p) for p in detail["plan"]]
            pts.extend(plan)
            sim = detail["simulation"]
            # 左図：初期計画経路（A・Bとも破線）
            axes[0].plot([p.y / 1000 for p in plan], [p.x / 1000 for p in plan],
                        color=colors[rid], linewidth=2, marker="o", markersize=4,
                        linestyle="--", label=f"{rid} 計画")

            # 右図：初期計画経路を細い破線で表示
            axes[1].plot([p.y / 1000 for p in plan], [p.x / 1000 for p in plan],
                        color=colors[rid], linestyle="--", linewidth=1.2, alpha=0.7)

            # 右図：追従航跡（A・Bとも実線）
            if sim["found"]:
                track = [into_initial_frame(p) for p in sim["own_track"]]
                pts.extend(track)
                axes[1].plot([p.y / 1000 for p in track], [p.x / 1000 for p in track],
                            color=colors[rid], linewidth=2.5,
                            linestyle="-", label=f"{rid} 航跡")
        for ax in axes:
            ax.plot(start.y / 1000, start.x / 1000, "ks", markersize=6, label="開始")
            ax.plot(goal.y / 1000, goal.x / 1000, "g*", markersize=11, label="目的地")
            ax.axvline(0, linewidth=.8, linestyle=":", color="gray")
            ax.set_xlabel("横位置（km）  左舷(−) ← → 右舷(＋)")
            ax.grid(alpha=.25)
            legend = ax.legend(loc="best", fontsize=9)
            if ax is axes[0]:
                for line, text in zip(legend.get_lines(), legend.get_texts()):
                    if text.get_text() in ("A 計画", "B 計画"):
                        line.set_linestyle(":")
        axes[0].set_ylabel("前後位置（km）")
        axes[1].set_ylabel("前後位置（km）")
        axes[1].tick_params(axis="y", labelleft=True)
        axes[0].set_title("Theta*：初期計画経路")
        axes[1].set_title("ShipModel：追従航跡（60秒ごとに再計画）")
        if zoom and pts:
            # 迂回部分が見えるよう、最大横偏差を持つ点を中心に表示。
            p = max(pts, key=lambda pt: abs(pt.y))
            y_vals = [pt.y / 1000 for pt in pts if abs(pt.x - p.x) <= 2500]
            ymin = min(y_vals) if y_vals else p.y / 1000
            ymax = max(y_vals) if y_vals else p.y / 1000
            pad = max(.3, (ymax - ymin) * .3)
            for ax in axes:
                ax.set_xlim(ymin - pad, ymax + pad)
                ax.set_ylim(p.x / 1000 - 2.5, p.x / 1000 + 2.5)
        else:
            extent = max(4.3, max((abs(pt.y) for pt in pts), default=0) / 1000 + .7)
            for ax in axes:
                ax.set_xlim(-extent, extent)
                ax.set_aspect("equal", adjustable="box")

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 6.7), sharex=True, sharey=True)
    draw(axes, zoom=False)
    fig.suptitle(f"{SCENARIO_ID}：初期計画と追従航跡")
    fig.text(.5, .008, "図1　Bumper条件別の初期計画経路と船体運動による追従航跡", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .04, 1, .95))
    image_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(image_path, dpi=155, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    fig2, axes2 = plt.subplots(1, 2, figsize=(13.4, 6.7), sharex=True, sharey=True)
    draw(axes2, zoom=True)
    fig2.suptitle(f"{SCENARIO_ID}：迂回箇所の拡大")
    fig2.text(.5, .008, "図2　迂回箇所の拡大比較", ha="center", fontsize=10)
    fig2.tight_layout(rect=(0, .04, 1, .95))
    fig2.savefig(zoom_path, dpi=155, bbox_inches="tight")
    plt.close(fig2)


def validate_settings() -> None:
    if not SCENARIO_ID or any(ch in SCENARIO_ID for ch in "\\/:*?\"<>|"):
        raise ValueError("SCENARIO_IDはファイル名に使用できる文字で設定してください")
    if OWN_INIT["u"] <= 0 or TARGET_INIT["u"] < 0:
        raise ValueError("自船速力は正、他船速力は0以上で指定してください")
    for rid in ("A", "B"):
        for side in ("fore", "aft", "stbd", "port"):
            if BASE_BUMPER_NM[side] <= 0 or BUMPER_MULTIPLIERS[rid][side] <= 0:
                raise ValueError("Bumper寸法と倍率は正で指定してください")


def output_root() -> Path:
    here = Path(__file__).resolve()
    if here.parent.name == "theta_star" and here.parent.parent.name == "code":
        return here.parents[2]
    return here.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Bumper・船速を設定値だけ変えて行うTheta*共通比較")
    parser.add_argument("--no-show", action="store_true", help="図を画面表示せず保存する")
    args = parser.parse_args()
    validate_settings()
    root = output_root()
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    name = f"{SCENARIO_ID}_{timestamp}"
    results = {}
    details = {}
    for rid in ("A", "B"):
        results[rid], details[rid] = evaluate(rid)

    selection = {
        "distance_priority": select_objective(results, "planned_length_m", True),
        "separation_priority": select_objective(results, "planned_minimum_separation_m", False),
    }
    image = root / "results" / f"{name}_planned_vs_tracked.png"
    zoom = root / "results" / f"{name}_zoom.png"
    if any(details.values()):
        make_figure(details, image, zoom, show=not args.no_show)

    output = {
        "scenario_id": SCENARIO_ID,
        "timestamp": timestamp,
        "settings": {
            "ownship": OWN_INIT,
            "target": TARGET_INIT,
            "goal": GOAL,
            "base_bumper_nm": BASE_BUMPER_NM,
            "bumper_multipliers": BUMPER_MULTIPLIERS,
            "grid_interval_m": GRID_INTERVAL_M,
            "delay_time_s": DELAY_TIME_S,
            "replan_interval_s": REPLAN_INTERVAL_S,
            "simulation_max_seconds": MAX_SIM_SECONDS,
        },
        "routes": results,
        "selection_results": selection,
        "method_notes": [
            "採用判断は初期Theta*計画経路の長さまたは最小離隔距離に基づく。",
            "比較値の同値は差1e-6以内とし、その場合は優劣なしとする。",
            "最小離隔距離は等速直線区間仮定の値である。追従時の最小船間距離とは区別する。",
            "迂回方向は自船初期針路に対する横偏差であり、他船の通過舷やCOLREGs適否ではない。",
            "生成後のBumper適合判定は実施しない。",
            "LLM用の文章による採用理由はPython側では生成しない。",
        ],
    }
    json_path = root / "data" / "theta_star" / f"{name}_comparison.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"シナリオ: {SCENARIO_ID}")
    for rid in ("A", "B"):
        r = results[rid]
        if not r["found"]:
            print(f"経路{rid}: 生成できませんでした")
            continue
        print(f"経路{rid}  Bumper倍率: {BUMPER_MULTIPLIERS[rid]}")
        print(f"  初期計画: 長さ {r['planned_length_m']:.2f} m、推定時間 {r['planned_travel_time_s']:.2f} 秒、最小離隔 {r['planned_minimum_separation_m']:.2f} m、迂回 {r['planned_bypass_side']}")
        if r["simulation_found"]:
            print(f"  追従航跡: 長さ {r['tracked_length_m']:.2f} m、計算時間範囲 {r['tracked_travel_time_s']} 秒、最小船間距離 {r['tracked_minimum_separation_m']:.2f} m、迂回 {r['tracked_bypass_side']}、目的地到達 {r['goal_reached']}")
        else:
            print("  追従航跡: 計算できませんでした")
    print("【目的別の採用経路（初期計画値を使用）】")
    for objective, decision in selection.items():
        print(f"  {objective}: {decision['selected_route'] or '優劣なし'}（{decision['status']}）")
    if any(details.values()):
        print("全体図:", image)
        print("拡大図:", zoom)
    print("比較JSON:", json_path)


if __name__ == "__main__":
    main()
