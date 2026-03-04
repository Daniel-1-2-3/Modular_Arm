import math
from dataclasses import dataclass
from typing import Iterable, Tuple, Optional

@dataclass(frozen=True)
class ArucoMeasurement:
    rvec: Optional[Tuple[float, float, float]]
    tvec: Tuple[float, float, float]

class CalcBearing:
    @staticmethod
    def _as_tvec(tvec: Iterable[float]) -> Tuple[float, float, float]:
        vals = list(tvec)
        if len(vals) != 3:
            raise ValueError("tvec must have 3 elements: (tx, ty, tz)")
        return float(vals[0]), float(vals[1]), float(vals[2])

    @staticmethod # tvec is aruco reading (t_x = horizontal offset, t_y = vetical offset, t_z = fwd dist)
    def get_bearing(tvec: Iterable[float]) -> float:
        tx, _, tz = CalcBearing._as_tvec(tvec)
        return math.atan2(tx, tz)