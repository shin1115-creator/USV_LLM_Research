#避航経路の探索
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import heapq
import math

from coordinate import XY

@dataclass
class Bumper:
    fore: float
    aft : float
    stbd: float
    port: float

    def quadrants(self):
        """四象限の座標
        """
        return [
            (+1, +1, self.fore, self.stbd), # 右舷前方
            (+1, -1, self.fore, self.port), # 左舷前方
            (-1, +1, self.aft, self.stbd),  # 右舷後方
            (-1, -1, self.aft, self.port),  # 左舷後方
        ]

@dataclass
class Ship:
    """船オブジェクト
    """
    pos     : XY
    course  : float
    speed   : float

    @classmethod
    def from_bearing_dist(cls, bearing, dist, course, speed):
        b = math.radians(bearing)
        return cls(XY(dist * math.cos(b), dist * math.sin(b)), course, speed)

    @classmethod
    def from_world_pos(cls, pos: XY, course: float, speed: float, origin_pos: XY, origin_heading: float):
        rp = pos.rotate_from(origin_heading, origin_pos)
        return cls(rp, course - origin_heading, speed)

    @property
    def velocity(self):
        c = math.radians(self.course)
        return self.speed * math.cos(c), self.speed * math.sin(c)

class Planner:
    def __init__(self, grid_interval, fore_range: List[float], side_range: List[float], bumper: Bumper, n_dir: int=8, sampling_step=None):
        self.grid_interval = grid_interval # 格子間隔 [m]
        self.sampling_step = grid_interval / 5 if sampling_step is None else sampling_step
        
        self.bumper     = bumper        # 自船のバンパーエリア

        self.fore_range = fore_range
        self.side_range = side_range
        self.origin  = XY(0.0, 0.0)     # 格子の原点（planning で変針する点に設定する）

        if n_dir == 4:
            self.moves = ((1, 0), (-1, 0), (0, 1), (0, -1))
        else:
            self.moves = ((1, 0), (-1, 0), (0, 1), (0, -1),
                          (1, 1), (1, -1), (-1, 1), (-1, -1))

    def to_node(self, pos: XY):
        """posを一番近い格子点(i, j)に変換
        """
        return (round((pos.x - self.origin.x) / self.grid_interval),
                round((pos.y - self.origin.y) / self.grid_interval))

    def to_pos(self, node):
        """格子点(i, j)をposに変換
        """
        return (self.origin.x + node[0] * self.grid_interval,
                self.origin.y + node[1] * self.grid_interval)

    def dist(self, a, b):
        """格子点から格子点までの距離
        """
        return self.grid_interval * math.hypot(a[0] - b[0], a[1] - b[1])

    def in_range(self, node):
        return self.xi_min <= node[0] <= self.xi_max and self.yi_min <= node[1] <= self.yi_max

    def neighbors(self, node):
        for di, dj in self.moves:
            n_node = (node[0] + di, node[1] + dj)
            if self.in_range(n_node):
                yield n_node

    def planning(self, goal: XY, ownship: Ship, targets: List[Ship]=(), plan: List[XY]|None=None, delay_time: float=0.0) -> List[XY] | None:
        # 従来航路の衝突判定
        if plan is not None and len(plan) >= 2:
            a = plan[0]
            b = plan[1]
            ab = b - a
            ab_2 = ab.dot(ab)
            # 自船位置を航路に投影する
            t = 0.0 if ab_2 == 0.0 else (ownship.pos - a).dot(ab) / ab_2
            t = min(max(t, 0.0), 1.0)
            path = [a + ab * t] + plan[1:]
            if self.check_path(ownship.speed, path, targets):
                return path

        # 航路計画
        targets_rel = [Ship.from_world_pos(target.pos, target.course, target.speed, ownship.pos, ownship.course) for target in targets]
        goal_rel = goal.rotate_from(ownship.course, ownship.pos)
        res = self.planning_relative(ownship.speed, goal_rel, targets_rel, delay_time=delay_time)
        if res is not None:
            plan = [ownship.pos + wp.rotate(-ownship.course) for wp in res]
            return plan
        else:
            return None
    
    def planning_relative(self, own_speed: float, goal_pos: XY, targets=(), is_blocked=None, delay_time=0.0):
        """避航航路計画

        own_speed  : 自船の速力 [m/s]
        goal_pos   : 目的地 [m]（自船を原点、x = 船首方向、y = 右舷方向）
        targets    : 他船のリスト（同じ座標で、計画する時刻の位置）
        is_blocked : 位置 (x, y) [m] を受け取り、通行不可なら True を返す関数
        delay_time : 今の針路のまま直進してから変針するまでの時間 [s]
        return     : 経路 [XY, ...]（同じ座標）。見つからなければ None
        """
        self.targets    = targets
        self.origin     = XY(own_speed * delay_time, 0.0)   # 変針する点（格子の原点）

        self.xi_min = math.ceil((self.fore_range[0] - self.origin.x) / self.grid_interval)    # x座標の最小値(範囲外に行かないように切り上げ)
        self.xi_max = math.floor((self.fore_range[1] - self.origin.x) / self.grid_interval)   # x座標の最大値(範囲外に行かないように切り捨て)
        self.yi_min = math.ceil((self.side_range[0] - self.origin.y) / self.grid_interval)    # y座標の最小値(範囲外に行かないように切り上げ)
        self.yi_max = math.floor((self.side_range[1] - self.origin.y) / self.grid_interval)   # y座標の最大値(範囲外に行かないように切り捨て)

        if is_blocked is None:
            is_blocked = lambda x, y: False
        self.is_blocked = is_blocked

        # delay_time中に危険
        if delay_time > 0.0 and not self.check_in_delay_time(own_speed, delay_time):
            return None

        start_node  = (0, 0)
        goal_node   = self.to_node(goal_pos)
        g           = {start_node: own_speed * delay_time}              # 出発点からの経路長（直進した分を含む）
        self.g      = g
        parent      = {start_node: start_node}                              # 親ノード
        closed_node = set()                                                 # 展開が終わったノード
        open_node   = [(self.dist(start_node, goal_node), 0, start_node)]   # 展開するノード
        counter     = 1                                                     # キューに入れた回数

        if not self.in_range(goal_node):
            return None
        if self.is_blocked(*self.to_pos(goal_node)):
            return None

        # 探索ループ
        while open_node:
            # ノードを取り出す
            _, _, node = heapq.heappop(open_node)
            if node in closed_node:
                continue

            # 探索終了
            if node == goal_node:
                return self.trace(parent, node)

            closed_node.add(node)

            # 隣のノードに進む
            for n_node in self.neighbors(node):
                if n_node in closed_node:
                    continue
                if self.is_blocked(*self.to_pos(n_node)):
                    continue

                p_node = parent[node]
                if self.can_move(own_speed, p_node, n_node): # 元ノードの親から次ノードに直接進める
                    new_parent = p_node
                elif self.can_move(own_speed, node, n_node): # 元ノードから次ノードに進める
                    new_parent = node
                else:
                    continue

                new_g = g[new_parent] + self.dist(new_parent, n_node) # 出発点からのコスト
                if new_g < g.get(n_node, math.inf):
                    g[n_node] = new_g
                    parent[n_node] = new_parent
                    f = new_g + self.dist(n_node, goal_node) # 評価値 (= g + ヒューリスティック)
                    heapq.heappush(open_node, (f, counter, n_node))
                    counter += 1

        return None

    def check_path(self, own_speed: float, path: List[XY], targets=(), is_blocked=None) -> bool:
        """航路をown_speedで進む場合に他船もしくは障害物に衝突しないか判定する
        """
        if is_blocked is None:
            is_blocked = lambda x, y: False

        t = 0.0
        for a, b in zip(path[:-1], path[1:]):
            length = (b - a).dist()
            if length == 0.0:
                continue

            # 障害物
            n = max(1, math.ceil(length / self.sampling_step))
            for k in range(n + 1):
                p = a + (b - a) * (k / n)
                if is_blocked(p.x, p.y):
                    return False

            # 他船
            heading = math.degrees(math.atan2(b.y - a.y, b.x - a.x))
            duration = length / own_speed
            for target in targets:
                if self.calc_intrusion_time(own_speed, target, a, t, heading, duration) is not None:
                    return False
            t += duration
        return True

    def line_of_sight(self, a, b):
        ax, ay = self.to_pos(a)
        bx, by = self.to_pos(b)

        n = max(1, math.ceil(self.dist(a, b) / self.sampling_step))
        for k in range(n + 1):
            t = k / n
            if self.is_blocked(ax + (bx - ax) * t, ay + (by - ay) * t):
                return False

        return True

    def check_in_delay_time(self, own_speed: float, delay_time: float):
        """速力 own_speed [m/s] で delay_time [s] 直進する間に、障害物にも他船にも当たらなければ True
        """
        length = own_speed * delay_time
        n = max(1, math.ceil(length / self.sampling_step))
        for k in range(n + 1):
            if self.is_blocked(length * k / n, 0.0):
                return False
        for target in self.targets:
            if self.calc_intrusion_time(own_speed, target, XY(0.0, 0.0), 0.0, 0.0, delay_time) is not None:
                return False
        return True

    def can_move(self, own_speed: float, a: Tuple[float, float], b: Tuple[float, float]):
        """aからbに移動できるか
        """
        t_start = self.g[a] / own_speed # aに到達する時間
        return self.line_of_sight(a, b) and self.check_collision(own_speed, a, b, t_start)

    def check_collision(self, own_speed: float, a: Tuple[float, float], b: Tuple[float, float], t_start: float):
        """t_start [s]にaを出発してbへ直進する間に船と衝突しないか判定
        """
        ax, ay = self.to_pos(a)
        bx, by = self.to_pos(b)
        length = self.dist(a, b)
        if length == 0.0:
            return True
        heading = math.degrees(math.atan2(by - ay, bx - ax))
        duration = length / own_speed
        for target in self.targets:
            if self.calc_intrusion_time(own_speed, target, XY(ax, ay), t_start, heading, duration) is not None:
                return False
        return True

    def calc_intrusion_time(self, own_speed: float, target: Ship, start: XY, t_start: float, heading: float, duration:float):
        """自船が速力 own_speed [m/s] で進む duration [s]間に他船(target)がバンパーに侵入する時間区間を計算する
        """
        v = XY(*target.velocity)
        # 出発時刻の他船の位置を、出発位置を原点・船首方向を x とした座標で表す
        rel = (target.pos + v * t_start).rotate_from(heading, start)
        # 相対速度 = 他船の速度 − 自船の速度（船体座標では自船は (own_speed, 0)）
        w = v.rotate(heading) - XY(own_speed, 0.0)
        return self.calc_relative_intrusion_time(rel.x, rel.y, w.x, w.y, duration)

    def calc_relative_intrusion_time(self, xi0: float, eta0: float, w_xi: float, w_eta: float, t_max: float) -> Optional[Tuple[float, float]]:
        """他船の相対運動からt_max [s]の間に他船がバンパーに侵入する時間区間を計算する
        船体座標: (ξ: 船首尾方向, η: 正横方向)
        他船の相対位置: (xi0 + w_xi * t, eta0 + w_eta * t)

        xi0, eta0   : 出発時刻の他船の相対位置 [m]
        w_xi, w_eta : 他船の相対速度 [m/s]
        t_max       : 探索時間 [s]
        return      : (入時刻, 出時刻) [s]（0 <= t <= t_max で切り取る) 侵入しなければ None
        """
        lo_all, hi_all = math.inf, -math.inf
        for sx, sy, a, b in self.bumper.quadrants():
            # 1/4楕円の内側: A t^2 + B t + C < 0
            A = (w_xi / a) ** 2 + (w_eta / b) ** 2
            B = 2 * (xi0 * w_xi / a ** 2 + eta0 * w_eta / b ** 2)
            C = (xi0 / a) ** 2 + (eta0 / b) ** 2 - 1
            if A * t_max ** 2 < 1e-12: # 相対速度ゼロ
                if C >= 0:
                    continue
                lo, hi = -math.inf, math.inf
            else:
                disc = B * B - 4 * A * C
                if disc <= 0:
                    continue
                sq = math.sqrt(disc)
                lo, hi = (-B - sq) / (2 * A), (-B + sq) / (2 * A)

            # その象限にいる時間で切り取る
            for iv in (self.calc_half_plane_time(xi0, w_xi, sx), self.calc_half_plane_time(eta0, w_eta, sy)):
                if iv is None:
                    lo, hi = math.inf, -math.inf
                    break
                lo, hi = max(lo, iv[0]), min(hi, iv[1])
            if lo < hi:
                lo_all, hi_all = min(lo_all, lo), max(hi_all, hi)

        lo, hi = max(lo_all, 0.0), min(hi_all, t_max)
        return (lo, hi) if lo < hi else None

    @staticmethod
    def calc_half_plane_time(p0: float, v: float, sign: int) -> Optional[Tuple[float, float]]:
        """座標の1成分 p0 + v * t が、軸の sign の側（sign * (p0 + v * t) >= 0）にいる時間区間を計算する

        p0   : 出発時刻の位置 [m]（ξ または η）
        v    : 速度 [m/s]
        sign : +1 なら 0 以上の側（前方・右舷）、-1 なら 0 以下の側（後方・左舷）
        戻り値 : (入る時刻, 出る時刻) [s]（片側は ±inf）。ずっと反対側にいれば None
        """
        p0 = sign * p0
        v = sign * v
        if abs(v) < 1e-12:
            return (-math.inf, math.inf) if p0 >= 0 else None
        t0 = -p0 / v
        return (t0, math.inf) if v > 0 else (-math.inf, t0)

    def trace(self, parent, node):
        """探索結果をたどる
        """
        nodes = [node]
        while parent[nodes[-1]] != nodes[-1]:
            nodes.append(parent[nodes[-1]])

        path = [XY(*self.to_pos(node)) for node in reversed(nodes)]
        if self.origin != XY(0.0, 0.0):
            path.insert(0, XY(0.0, 0.0))   # 自船の今の位置（ここから変針点まで直進）
        return path

if __name__ == "__main__":
    from matplotlib import pyplot as plt

    def path_length(path: List[XY]) -> float:
        return sum((b - a).dist() for a, b in zip(path[:-1], path[1:]))

    own_speed   = 12.0 * 1852.0 / 3600.0
    bumper      = Bumper(fore=1852.0, aft=740.8, stbd=1111.2, port=740.8)
    sim_time_s  = 3600.0    # シミュレーション時間（予測航跡を描く時間）[s]
    targets = [
        # 名前, 方位[deg], 距離[m], 針路[deg], 速力[m/s]
        Ship.from_bearing_dist(40.0, 9260.0, 270.0, 10.0 * 1852.0 / 3600.0),
        Ship.from_bearing_dist(355.0, 16668.0, 178.0, 9.0 * 1852.0 / 3600.0),
        Ship.from_bearing_dist(300.0, 7408.0, 60.0, 6.0 * 1852.0 / 3600.0),
    ]
    goal = XY(14816.0, 0.0)

    planner = Planner(463.0, [-1852.0, 16668.0], [-7408.0, 7408.0], bumper)
    path = planner.planning(own_speed, goal, targets, delay_time=300.0)

    if path is None:
        print("経路なし")
    else:
        print("length:", path_length(path))
        dist_sum = 0.0
        for k, p in enumerate(path):
            if k > 0:
                dist_sum += (p - path[k - 1]).dist()
            print(f"  ({p.x:8.1f}, {p.y:8.1f})  到着 {dist_sum / own_speed / 60:5.1f} 分")

    fig, ax = plt.subplots()
    nodes = [(i, j) for i in range(planner.xi_min, planner.xi_max + 1) for j in range(planner.yi_min, planner.yi_max + 1)]
    points = [planner.to_pos(node) for node in nodes]
    ax.scatter([point[1] for point in points], [point[0] for point in points], s=3, color="#d0d0d0")

    # 他船: 初期位置（●）と、判定時間内の予測航跡（10 分ごとの目盛付き）
    for target in targets:
        vx, vy = target.velocity
        ts = [600.0 * k for k in range(int(sim_time_s // 600) + 1)]
        xs = [target.pos.x + vx * t for t in ts]
        ys = [target.pos.y + vy * t for t in ts]
        line, = ax.plot(ys, xs, "|--", ms=8)
        ax.plot(target.pos.y, target.pos.x, "o", color=line.get_color(), ms=8)

    if path is not None:
        ax.plot([p.y for p in path], [p.x for p in path], "o-", color="#2a78d6",
                lw=2, ms=4)

    ax.plot(0.0, 0.0, "s", color="#2a78d4", ms=9)
    ax.plot(goal.y, goal.x, "*", color="#0ca30c", ms=15)
    ax.set_aspect("equal")

    plt.show()
