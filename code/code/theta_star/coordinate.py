#座標計算

from __future__ import annotations
from dataclasses import dataclass
import math

def adjust_180(angle: float) -> float:
    while True:
        if angle > 180:
            angle -= 360
        elif angle < -180:
            angle += 360
        else:
            return angle

@dataclass(frozen=True)
class XY:
    x: float
    y: float

    def __add__(self, other: XY) -> XY:
        return XY(self.x + other.x, self.y + other.y)

    def __sub__(self, other: XY) -> XY:
        return XY(self.x - other.x, self.y - other.y)

    def __mul__(self, k: float) -> XY:
        return XY(self.x * k, self.y * k)
    __rmul__ = __mul__

    def dot(self, other: XY) -> float:
        return self.x * other.x + self.y * other.y

    def dist(self) -> float:
        return math.hypot(self.x, self.y)
    
    def dist_from(self, origin: XY) -> float:
        return math.hypot(self.x - origin.x, self.y - origin.y)

    def rotate(self, angle: float) -> XY:
        a = math.radians(angle)
        cos = math.cos(a)
        sin = math.sin(a)
        return XY(self.x * cos + self.y * sin, -self.x * sin + self.y * cos)

    def unrotate(self, angle: float) -> XY:
        return self.rotate(-angle)

    def rotate_from(self, angle: float, origin: XY) -> XY:
        return (self - origin).rotate(angle)

    def angle_from(self, origin: XY) -> float:
        return math.degrees(math.atan2(self.y - origin.y, self.x - origin.x)) % 360.0
