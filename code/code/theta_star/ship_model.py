# 船の運動モデル
import math

class ShipModel:
    def __init__(self, K: float, T: float, x: float, y: float, heading: float, u: float):
        self.K      : float = K
        self.T      : float = T

        self.x      : float = x         # x座標 [m]
        self.y      : float = y         # y座標 [m]
        self.heading: float = heading   # 船首方位 [deg]
        self.u      : float = u         # 速度 [m/s]
        self.r      : float = 0.0       # 旋回角速度 [deg/s]
        self.rud    : float = 0.0       # 舵角 [deg]

    def step(self, dt: float):
        d_x         = self.u * math.cos(math.radians(self.heading))
        d_y         = self.u * math.sin(math.radians(self.heading))
        d_heading   = self.r
        d_r         = (self.K * self.rud - self.r) / self.T

        self.x          += d_x * dt
        self.y          += d_y * dt
        self.heading    += d_heading * dt
        self.r          += d_r * dt
