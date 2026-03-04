import math
from typing import Iterable, Tuple, Optional
import numpy as np
import cv2


class CalcBearing:
    DOT_BACK_FROM_FRONT_M = 0.07
    LATERAL_OFFSET_M = 0.019

    @staticmethod
    def _as3(v: Iterable[float]) -> np.ndarray:
        a = list(v)
        if len(a) != 3:
            raise ValueError("Expected 3 elements")
        return np.array([float(a[0]), float(a[1]), float(a[2])], dtype=np.float64).reshape(3, 1)

    @staticmethod
    def _bearing_from_xz(x: float, z: float) -> float:
        return math.atan2(float(x), float(z))

    @staticmethod
    def get_camera_bearing_from_tvec(tvec_cam_to_tag_m: Iterable[float]) -> float:
        t = CalcBearing._as3(tvec_cam_to_tag_m)
        return CalcBearing._bearing_from_xz(t[0, 0], t[2, 0])

    @staticmethod
    def get_bearing(
        rvec_tag_to_cam: Iterable[float],
        tvec_tag_to_cam_m: Iterable[float],
        camera_is_right: bool = True,
        center_to_camera_cam_m: Optional[Tuple[float, float, float]] = None,
        tag_to_center_tag_m: Optional[Tuple[float, float, float]] = None,
        dot_back_from_front_m: float = None,
        lateral_offset_m: float = None,
    ) -> float:
        """
            Inputs - rvec_tag_to_cam, tvec_tag_to_cam_m: pose of tag in cam frame
            Output is real bearing - center axis to center axis between robots
        """
        if dot_back_from_front_m is None:
            dot_back_from_front_m = CalcBearing.DOT_BACK_FROM_FRONT_M
        if lateral_offset_m is None:
            lateral_offset_m = CalcBearing.LATERAL_OFFSET_M

        camera_side = 1 if camera_is_right else -1
        tag_side = -camera_side

        if center_to_camera_cam_m is None:
            center_to_camera_cam_m = (
                lateral_offset_m * camera_side,
                0.0,
                dot_back_from_front_m,
            )

        if tag_to_center_tag_m is None:
            tag_to_center_tag_m = (
                -lateral_offset_m * tag_side,
                0.0,
                -dot_back_from_front_m,
            )

        rvec = CalcBearing._as3(rvec_tag_to_cam)
        tvec = CalcBearing._as3(tvec_tag_to_cam_m)

        R_tag_to_cam, _ = cv2.Rodrigues(rvec)

        p_tag_to_center_tag = np.array(tag_to_center_tag_m, dtype=np.float64).reshape(3, 1)
        stationary_center_in_cam = (R_tag_to_cam @ p_tag_to_center_tag) + tvec

        p_center_to_camera_cam = np.array(center_to_camera_cam_m, dtype=np.float64).reshape(3, 1)
        robot_center_in_cam = -p_center_to_camera_cam

        stationary_center_from_robot_center_cam = stationary_center_in_cam - robot_center_in_cam

        x = float(stationary_center_from_robot_center_cam[0, 0])
        z = float(stationary_center_from_robot_center_cam[2, 0])

        return CalcBearing._bearing_from_xz(x, z)