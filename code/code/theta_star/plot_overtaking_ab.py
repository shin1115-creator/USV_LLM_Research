"""追越しシナリオの経路A・Bを、誤解を招きにくい縮尺で見やすく図示する。

配置先: code/theta_star/plot_overtaking_ab.py
実行方法: python code\\theta_star\\plot_overtaking_ab.py

全体図は進行方向Xに比べて横方向Yだけ約2.9倍強調し、
最小離隔の拡大図は等縮尺で表示する。
この図は区間直線・等速航行を仮定した計画経路比較であり、実航跡ではない。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

SCENARIO_ID = "theta_overtaking_001"
ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data" / "theta_star" / f"{SCENARIO_ID}_AB_comparison.json"
OUTPUT = ROOT / "results" / f"{SCENARIO_ID}_routes_AB.png"

# 値を小さくすると横方向（Y）の強調率が大きくなる。
# 0.35 => Y方向の1mが、X方向の1mの約2.86倍の長さで描画される。
OVERVIEW_ASPECT = 0.35


def configure_japanese_font() -> None:
    """使用できる日本語フォントを優先順位に従って設定する。"""
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Yu Gothic", "Meiryo", "Noto Sans CJK JP", "IPAexGothic", "MS Gothic"):
        if name in available:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def velocity(speed: float, course_deg: float) -> tuple[float, float]:
    rad = math.radians(course_deg)
    return speed * math.cos(rad), speed * math.sin(rad)


def closest_approach(path: list[dict], own_speed: float, target: dict) -> dict:
    """自船が経路上を等速直線移動する仮定で、同時刻の最小離隔を求める。"""
    tx0, ty0 = float(target["position"]["x"]), float(target["position"]["y"])
    tvx, tvy = velocity(float(target["speed_mps"]), float(target["course_deg"]))
    elapsed = 0.0
    best = None
    for a, b in zip(path[:-1], path[1:]):
        x0, y0 = float(a["x"]), float(a["y"])
        dx, dy = float(b["x"]) - x0, float(b["y"]) - y0
        length = math.hypot(dx, dy)
        if length <= 1e-12:
            continue
        dt = length / own_speed
        ovx, ovy = dx / dt, dy / dt
        rx, ry = x0 - (tx0 + tvx * elapsed), y0 - (ty0 + tvy * elapsed)
        rvx, rvy = ovx - tvx, ovy - tvy
        den = rvx**2 + rvy**2
        tau = 0.0 if den < 1e-12 else -(rx * rvx + ry * rvy) / den
        tau = min(max(tau, 0.0), dt)
        t = elapsed + tau
        own = (x0 + ovx * tau, y0 + ovy * tau)
        other = (tx0 + tvx * t, ty0 + tvy * t)
        sep = math.dist(own, other)
        if best is None or sep < best["separation_m"]:
            best = {"separation_m": sep, "time_s": t, "own_pt": own, "target_pt": other}
        elapsed += dt
    if best is None:
        raise ValueError("経路区間がありません。")
    return best


def main() -> None:
    configure_japanese_font()
    if not SOURCE.exists():
        raise FileNotFoundError(f"追越し比較JSONがありません: {SOURCE}")
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    scenario, routes = data["scenario"], data["routes"]
    own = scenario["ownship"]["position"]
    target = scenario["target"]
    other_start = target["position"]
    goal = scenario["goal"]
    own_speed = float(scenario["ownship"]["speed_mps"])
    tvx, tvy = velocity(float(target["speed_mps"]), float(target["course_deg"]))

    # 論文向け: 全体像（横方向だけ穏やかに強調）と等縮尺の拡大図を分ける。
    fig, (ax_all, ax_zoom) = plt.subplots(1, 2, figsize=(14, 8.5),
                                         gridspec_kw={"width_ratios": [1, 1.12]})
    fig.patch.set_facecolor("white")
    for ax in (ax_all, ax_zoom):
        ax.set_facecolor("#fbfcfe")
        ax.grid(True, color="#dfe5ea", linewidth=0.8, alpha=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel("横方向 Y［m］", fontsize=11)
        ax.set_ylabel("進行方向 X［m］", fontsize=11)
        ax.tick_params(labelsize=9, colors="#455468")
        for spine in ax.spines.values():
            spine.set_color("#cbd5e1")

    ax_all.set_title("経路全体図", fontsize=13, fontweight="bold", pad=15)
    ax_zoom.set_title("最接近付近の拡大図", fontsize=13, fontweight="bold", pad=15)
    ax_all.set_xlim(-900, 1850)
    ax_all.set_ylim(-550, 15700)
    ax_all.set_aspect(OVERVIEW_ASPECT, adjustable="box")
    ax_zoom.set_xlim(-350, 1550)
    ax_zoom.set_ylim(7950, 11000)
    ax_zoom.set_aspect("equal", adjustable="box")

    # 進行方向の基準線（Y=0）。Bumper境界とは異なる。
    for ax in (ax_all, ax_zoom):
        ax.axvline(0, linestyle="--", color="#b7c4d4", linewidth=1.1, zorder=1)

    styles = {
        "A": {"color": "#007e87", "ls": "-", "marker": "o", "sep": "#1588c7"},
        "B": {"color": "#d18420", "ls": "--", "marker": "s", "sep": "#a653ae"},
    }
    approaches = {}
    for label in ("A", "B"):
        route = routes[label]
        if not route.get("found"):
            raise ValueError(f"経路{label}が生成されていません。")
        pts = route["waypoints"]
        ap = closest_approach(pts, own_speed, target)
        if abs(ap["separation_m"] - float(route["minimum_separation_m"])) > 0.02:
            raise ValueError(f"経路{label}の計算値が結果JSONと一致しません。")
        approaches[label] = ap
        st = styles[label]
        for ax in (ax_all, ax_zoom):
            ax.plot([p["y"] for p in pts], [p["x"] for p in pts],
                    color=st["color"], linestyle=st["ls"], linewidth=2.7,
                    solid_capstyle="round", zorder=3)
            ox, oy = ap["own_pt"]
            tx, ty = ap["target_pt"]
            ax.plot([oy, ty], [ox, tx], color=st["sep"], linewidth=2.0,
                    zorder=4)
            ax.scatter(oy, ox, marker=st["marker"], s=64,
                       facecolor=st["sep"], edgecolor="white", zorder=5)
            ax.scatter(ty, tx, marker="x", s=64, color=st["sep"], linewidth=2, zorder=5)
        # 経路の折れ曲がり位置を全体図にだけ示す。
        turn = pts[2]
        ax_all.scatter(turn["y"], turn["x"], color=st["color"], s=44,
                       edgecolor="white", linewidth=0.7, zorder=6)
        dxtext = 15 if label == "A" else 15
        dytext = -23 if label == "A" else 25
        ax_all.annotate(f"経路{label}の迂回点\nX={turn['x']:,.0f} m", 
                        xy=(turn["y"], turn["x"]), xytext=(dxtext, dytext),
                        textcoords="offset points", fontsize=9, color=st["color"],
                        ha="left", va="center",
                        arrowprops={"arrowstyle": "-", "color": st["color"], "alpha": 0.55},
                        zorder=8)

    # 他船は一定速・一定針路の予測直線航跡。
    tmax = max(float(routes[l]["travel_time_s_straight_segments"]) for l in ("A", "B"))
    end_other = (float(other_start["x"]) + tvx * tmax,
                 float(other_start["y"]) + tvy * tmax)
    for ax in (ax_all, ax_zoom):
        ax.plot([other_start["y"], end_other[1]], [other_start["x"], end_other[0]],
                color="#bd4c4c", linestyle=":", linewidth=2.2, zorder=2)
        ax.scatter(own["y"], own["x"], marker="s", s=75, color="#2563eb", zorder=7)
        ax.scatter(other_start["y"], other_start["x"], marker="o", s=65,
                   color="#bd4c4c", zorder=7)
        ax.scatter(goal["y"], goal["x"], marker="*", s=175,
                   color="#228b54", zorder=7)

    ax_all.text(0.97, 0.975,
                f"横方向Yだけ約{1 / OVERVIEW_ASPECT:.1f}倍強調\n"
                "※等縮尺ではありません", transform=ax_all.transAxes,
                ha="right", va="top", fontsize=9, color="#475569",
                bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": "#e2e8f0", "alpha": .92})
    ax_zoom.text(0.03, 0.97, "縦横とも等縮尺", transform=ax_zoom.transAxes,
                 ha="left", va="top", fontsize=9, color="#475569",
                 bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "#e2e8f0", "alpha": .95})

    fig.suptitle("追越しシナリオ｜避航経路A・Bと最小離隔", fontsize=17,
                 fontweight="bold", x=0.08, y=0.975, ha="left", color="#172a3a")
    # 折れ線の関係が分かるよう凡例を図の下にまとめる。
    legend = [
        Line2D([0], [0], color=styles["A"]["color"], lw=2.8, label="経路A（基準Bumperに適合）"),
        Line2D([0], [0], color=styles["B"]["color"], ls="--", lw=2.8, label="経路B（基準Bumperに不適合）"),
        Line2D([0], [0], color="#bd4c4c", ls=":", lw=2.2, label="他船の予測航跡"),
        Line2D([0], [0], color=styles["A"]["sep"], lw=2, label=f"Aの最小離隔：{approaches['A']['separation_m']:.2f} m"),
        Line2D([0], [0], color=styles["B"]["sep"], lw=2, label=f"Bの最小離隔：{approaches['B']['separation_m']:.2f} m"),
        Line2D([0], [0], marker="*", linestyle="None", markerfacecolor="#228b54",
               markeredgecolor="#228b54", markersize=11, label="目的地"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, 0.11), frameon=False, columnspacing=2.0)
    fig.text(0.5, 0.056,
             "注：経路は計画経路、他船は等速直進の予測航跡。最小離隔は各区間を等速直線航行すると仮定して算出。",
             fontsize=9, color="#64748b", ha="center")
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.24, top=0.88, wspace=0.23)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=190, facecolor="white")
    if "agg" not in plt.get_backend().lower():
        plt.show()
    plt.close(fig)
    for label, ap in approaches.items():
        print(f"経路{label}：最小離隔距離 {ap['separation_m']:.2f} m、最接近時刻 {ap['time_s']:.2f} 秒")
    print(f"保存先：{OUTPUT}")


if __name__ == "__main__":
    main()
