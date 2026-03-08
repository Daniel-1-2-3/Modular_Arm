import numpy as np
import math
import cv2
import mujoco
from pathlib import Path
from Arm_Env.arm_env import ArmEnv
import argparse

RING_OUTER_M = 0.04
RING_INNER_FRAC = 0.55
REF_FOCAL = 608.0
REF_W = 640.0
REF_H = 480.0
TOF_H_M = 0.07
CAM_PITCH_DEG = 8.0

RING_L_LOW = 8 # Tune
RING_L_HIGH = 120
BG_L_MAX = 18

CLAHE_CLIP = 4.0
CLAHE_GRID = (8, 8)

MORPH_CLOSE_K = 9
MORPH_OPEN_K = 3

MIN_AREA = 400
MIN_SIDE = 14
MAX_AREA_FRAC = 0.90
BORDER_MARGIN = 2

HOLLOW_DARK_MIN = 0.12
HOLLOW_DARK_MAX = 0.80

USE_MOG2 = True
MOG2_HISTORY = 120
MOG2_THRESHOLD = 30
MOG2_DETECT_SHADOWS = False
MOG2_BLEND = 0.35

STOP_FILL = 0.50

GROUND_CROP_MAX = 0.05
GROUND_CROP_MIN = 0.0

J_MID_LO = 0.0
J_MID_HI = 25.0 / 180.0 * math.pi
J_END_LO = -90.0 / 180.0 * math.pi
J_END_HI = 110.0 / 180.0 * math.pi
MID_LIMIT_THRESH = 0.10

COL_BOX = (0, 255, 0)
COL_X = (0, 220, 255)
COL_TOF = (0, 255, 200)
COL_EST = (0, 165, 255)
COL_ERR_OK = (0, 220, 255)
COL_ERR_BAD = (50, 50, 240)
COL_MISS = (50, 50, 200)


def _preprocess_for_ring(frame_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l_ch = lab[:, :, 0]

    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    l_enh = clahe.apply(l_ch)

    mask = cv2.inRange(l_enh, RING_L_LOW, RING_L_HIGH)

    kc = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_CLOSE_K, MORPH_CLOSE_K))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kc, iterations=2)

    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_OPEN_K, MORPH_OPEN_K))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ko, iterations=1)

    return mask, l_enh


def _hollow_score(l_enh: np.ndarray, cnt) -> float:
    x, y, w, h = cv2.boundingRect(cnt)
    if w < 4 or h < 4:
        return 0.0
    roi = l_enh[y:y + h, x:x + w]
    dark_px = int(np.sum(roi < BG_L_MAX + 10))
    total_px = w * h
    return dark_px / total_px if total_px > 0 else 0.0


def _ring_pixel_size(cnt, bbox, touches_border: bool) -> float:
    x, y, w, h = bbox
    if touches_border:
        return float(max(w, h))
    rect = cv2.minAreaRect(cnt)
    rw, rh = rect[1]
    outer = max(rw, rh) if rw > 1 and rh > 1 else max(w, h)
    return float(outer)


def _perspective_correction(
    psize_px: float,
    cx: float,
    cy: float,
    w_img: int,
    h_img: int,
    fx: float,
    fy: float,
) -> float:
    dx = (cx - w_img / 2.0) / fx
    dy = (cy - h_img / 2.0) / fy
    cos_a = 1.0 / math.sqrt(1.0 + dx * dx + dy * dy)
    return psize_px * cos_a


class ColorDetectControl:
    def __init__(self):
        self.reset()

    def reset(self):
        self._stopped = False
        self._mid_pos: float = 0.0
        self._end_pos: float = 0.0
        self._last_gray_vis: np.ndarray | None = None
        self._last_fill: float = 0.0
        self._mog2 = (
            cv2.createBackgroundSubtractorMOG2(
                history=MOG2_HISTORY,
                varThreshold=MOG2_THRESHOLD,
                detectShadows=MOG2_DETECT_SHADOWS,
            )
            if USE_MOG2
            else None
        )

    @property
    def is_stopped(self) -> bool:
        return self._stopped

    @staticmethod
    def _snap(val: float, deadzone: float = 0.08) -> float:
        if abs(val) < deadzone:
            return 0.0
        sign = 1.0 if val > 0 else -1.0
        return sign * (1.0 if abs(val) >= 0.5 else 0.5)

    @staticmethod
    def _snap_nonzero(val: float, deadzone: float = 0.08) -> float:
        if abs(val) < deadzone:
            return 0.5 if val >= 0 else -0.5
        return (
            1.0
            if val >= 0.5
            else -1.0
            if val <= -0.5
            else 0.5
            if val > 0
            else -0.5
        )

    @staticmethod
    def _center_crop_square(img):
        h, w = img.shape[:2]
        side = min(h, w)
        x0 = (w - side) // 2
        y0 = (h - side) // 2
        return img[y0:y0 + side, x0:x0 + side], x0, y0

    @staticmethod
    def _focal(w_img, h_img):
        fx = REF_FOCAL * (w_img / REF_W)
        fy = REF_FOCAL * (h_img / REF_H)
        return math.sqrt(fx * fy), fx, fy

    @staticmethod
    def _touches_border(cnt, w_img, h_img) -> bool:
        x, y, w, h = cv2.boundingRect(cnt)
        return (
            x <= BORDER_MARGIN
            or y <= BORDER_MARGIN
            or x + w >= w_img - BORDER_MARGIN
            or y + h >= h_img - BORDER_MARGIN
        )

    @staticmethod
    def _poly_score(cnt) -> float:
        peri = cv2.arcLength(cnt, True)
        if peri <= 1e-6:
            return 0.0
        n = len(cv2.approxPolyDP(cnt, 0.04 * peri, True))
        if 3 <= n <= 6:
            return 1.0
        if n == 7:
            return 0.7
        if n <= 10:
            return 0.5
        return 0.2

    def _score_contour(self, cnt, frame_area, w_img, h_img, l_enh=None):
        area = cv2.contourArea(cnt)
        if area < MIN_AREA or area > frame_area * MAX_AREA_FRAC:
            return None

        x, y, w, h = cv2.boundingRect(cnt)
        if w == 0 or h == 0:
            return None

        tb = self._touches_border(cnt, w_img, h_img)
        if not tb and (w < MIN_SIDE or h < MIN_SIDE):
            return None
        if tb and max(w, h) < MIN_SIDE:
            return None

        sq = min(w, h) / max(w, h)
        hull_a = cv2.contourArea(cv2.convexHull(cnt))
        sol = area / hull_a if hull_a > 1e-6 else 0.0

        rect = cv2.minAreaRect(cnt)
        rw, rh = rect[1]
        rsq = (min(rw, rh) / max(rw, rh) if rw > 1e-6 and rh > 1e-6 else 0.0)

        poly = self._poly_score(cnt)

        hollow = 0.0
        if l_enh is not None:
            hollow = _hollow_score(l_enh, cnt)
            if hollow < HOLLOW_DARK_MIN or hollow > HOLLOW_DARK_MAX:
                return None

        if not tb:
            if sq < 0.45 or rsq < 0.45 or sol < 0.45 or poly < 0.3:
                return None
        else:
            if sq < 0.25 or sol < 0.35 or poly < 0.2:
                return None

        psize = _ring_pixel_size(cnt, (x, y, w, h), tb)

        score = 2.0 * max(sq, rsq) + 1.5 * sol + 2.0 * hollow + 0.8 * poly + 0.002 * psize

        return score, x + w / 2.0, y + h / 2.0, psize, (x, y, w, h)

    def _best_from_mask(self, mask: np.ndarray, l_enh: np.ndarray):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        h_img, w_img = mask.shape[:2]
        fa = h_img * w_img
        best_s, best_r = -1.0, None
        for cnt in contours:
            res = self._score_contour(cnt, fa, w_img, h_img, l_enh=l_enh)
            if res is None:
                continue
            s, cx, cy, ps, bb = res
            if s > best_s:
                best_s, best_r = s, (cx, cy, ps, bb)
        return best_r

    def _detect(self, gray_bgr: np.ndarray):
        mask_band, l_enh = _preprocess_for_ring(gray_bgr)
        det = self._best_from_mask(mask_band, l_enh)
        if det is not None:
            return det

        gray = cv2.cvtColor(gray_bgr, cv2.COLOR_BGR2GRAY)
        adapt = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=51,
            C=-5,
        )

        bright_mask = (l_enh > RING_L_HIGH).astype(np.uint8) * 255
        adapt = cv2.bitwise_and(adapt, cv2.bitwise_not(bright_mask))
        return self._best_from_mask(adapt, l_enh)

    def _ground_crop_frac(self) -> float:
        t = min(self._last_fill / STOP_FILL, 1.0)
        return GROUND_CROP_MAX + (GROUND_CROP_MIN - GROUND_CROP_MAX) * t

    def _detect_full(self, frame_bgr_full):
        h_full, w_full = frame_bgr_full.shape[:2]

        crop_rows = int(h_full * self._ground_crop_frac())
        frame_cropped = frame_bgr_full[:h_full - crop_rows, :] if crop_rows > 0 else frame_bgr_full

        mog2_mask = None
        if self._mog2 is not None:
            fg_mask = self._mog2.apply(frame_bgr_full)
            kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            fg_mask = cv2.dilate(fg_mask, kern, iterations=2)
            fg_crop = fg_mask[:h_full - crop_rows, :] if crop_rows > 0 else fg_mask
            side = min(fg_crop.shape[:2])
            x0_fg = (fg_crop.shape[1] - side) // 2
            y0_fg = (fg_crop.shape[0] - side) // 2
            mog2_mask = fg_crop[y0_fg:y0_fg + side, x0_fg:x0_fg + side]

        cropped, x_off, y_off = self._center_crop_square(frame_cropped)

        patch = cropped.copy()
        if mog2_mask is not None and mog2_mask.shape[:2] == patch.shape[:2]:
            fg_float = (mog2_mask > 0).astype(np.float32)
            w_blend = 1.0 - MOG2_BLEND + MOG2_BLEND * fg_float
            patch = np.clip(patch.astype(np.float32) * w_blend[:, :, None], 0, 255).astype(
                np.uint8
            )

        _, l_full = _preprocess_for_ring(frame_bgr_full)
        if crop_rows > 0:
            cv2.line(l_full, (0, h_full - crop_rows), (w_full, h_full - crop_rows), 128, 1)
        self._last_gray_vis = l_full

        det = self._detect(patch)
        if det is None:
            return None

        cx, cy, ps, (bx, by, bw, bh) = det

        side = min(frame_cropped.shape[:2])
        self._last_fill = ps / side if side > 0 else 0.0

        return cx + x_off, cy + y_off, ps, (bx + x_off, by + y_off, bw, bh)

    def _backproject(self, cx, cy, dist, fx, fy, cx_img, cy_img):
        nx = (cx - cx_img) / fx
        ny = (cy - cy_img) / fy
        rl = math.sqrt(nx**2 + ny**2 + 1.0)
        z = dist / rl
        return nx * z, ny * z, z

    def get_distance(self, image_rgb: np.ndarray) -> float:
        h, w = image_rgb.shape[:2]
        fg, fx, fy = self._focal(w, h)
        det = self._detect_full(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        if det is None:
            return float("nan")
        cx, cy, ps, _ = det

        ps_corr = _perspective_correction(ps, cx, cy, w, h, fx, fy)
        d = (RING_OUTER_M * fg) / ps_corr if ps_corr > 0 else float("nan")

        if not math.isfinite(d) or d <= 0:
            return float("nan")
        xc, yc, zc = self._backproject(cx, cy, d, fx, fy, w / 2.0, h / 2.0)
        pr = math.radians(CAM_PITCH_DEG)
        dyt = yc * math.cos(pr) + zc * math.sin(pr) - TOF_H_M
        dzt = -yc * math.sin(pr) + zc * math.cos(pr)
        return float(math.sqrt(xc**2 + dyt**2 + dzt**2))

    def get_angles(self, image_rgb: np.ndarray):
        h, w = image_rgb.shape[:2]
        fg, fx, fy = self._focal(w, h)
        det = self._detect_full(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        if det is None:
            return float("nan"), float("nan"), float("nan")
        cx, cy, ps, _ = det

        ps_corr = _perspective_correction(ps, cx, cy, w, h, fx, fy)
        d = (RING_OUTER_M * fg) / ps_corr if ps_corr > 0 else float("nan")
        if not math.isfinite(d) or d <= 0:
            return float("nan"), float("nan"), float("nan")

        xc, yc, zc = self._backproject(cx, cy, d, fx, fy, w / 2.0, h / 2.0)
        ha = math.degrees(math.atan2(xc, zc))
        va = math.degrees(math.atan2(-yc, zc))
        if zc > 1e-6:
            pr = math.radians(CAM_PITCH_DEG)
            vo = -math.degrees(
                math.atan2(
                    TOF_H_M * math.cos(pr) - zc * math.sin(pr),
                    TOF_H_M * math.sin(pr) + zc * math.cos(pr),
                )
            )
        else:
            vo = float("nan")
        return ha, va, float(vo)

    def _get_fill_fraction(self, image_rgb: np.ndarray) -> float:
        h, w = image_rgb.shape[:2]
        side = min(h, w)
        det = self._detect_full(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        if det is None:
            return 0.0
        return det[2] / side

    def step(self, image_rgb: np.ndarray, tof_reading: float, step_num: int, env=None) -> np.ndarray:
        action = np.zeros(4, dtype=np.float32)

        if env is not None:
            try:
                qpos = env.data.qpos
                self._mid_pos = float(qpos[2])
                self._end_pos = float(qpos[3])
            except Exception:
                pass

        h_img, w_img = image_rgb.shape[:2]
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        det = self._detect_full(bgr)

        vis = bgr.copy()

        if det is None:
            cv2.putText(vis, "NO RING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_MISS, 2)
            cv2.imshow("step", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            if self._last_gray_vis is not None:
                cv2.imshow("gray enhanced", self._last_gray_vis)
            cv2.waitKey(1)
            return action

        cube_cx, cube_cy, psize, (bx, by, bw, bh) = det
        fill = psize / min(h_img, w_img)

        cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), COL_BOX, 2)

        cx, cy = int(cube_cx), int(cube_cy)
        cv2.line(vis, (cx - 8, cy), (cx + 8, cy), COL_X, 1)
        cv2.line(vis, (cx, cy - 8), (cx, cy + 8), COL_X, 1)

        tx, ty = int(w_img / 2), int(h_img * 0.7)
        cv2.line(vis, (tx - 8, ty), (tx + 8, ty), (0, 180, 255), 1)
        cv2.line(vis, (tx, ty - 8), (tx, ty + 8), (0, 180, 255), 1)
        cv2.line(vis, (tx, ty), (cx, cy), (0, 180, 255), 1)

        h_err_frac = (cube_cx - w_img / 2.0) / (w_img / 2.0)
        v_err_frac = (cube_cy - h_img * 0.7) / (h_img / 2.0)
        status = "STOPPED" if self._stopped else "active"
        for i, txt in enumerate(
            [
                f"Fill: {fill:.0%} [{status}]",
                f"H_err: {h_err_frac:+.2f}  V_err: {v_err_frac:+.2f}",
            ]
        ):
            cv2.putText(
                vis,
                txt,
                (10, 25 + i * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 255, 200),
                1,
                cv2.LINE_AA,
            )

        cv2.imshow("step", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        if self._last_gray_vis is not None:
            cv2.imshow("gray enhanced", self._last_gray_vis)
        cv2.waitKey(1)

        if fill >= STOP_FILL and step_num > 50:
            self._stopped = True
            return action

        H_DEADZONE = 0.06
        V_DEADZONE = 0.06

        if abs(h_err_frac) >= H_DEADZONE:
            action[0] = self._snap_nonzero(-h_err_frac)

        mid_range = J_MID_HI - J_MID_LO
        end_range = J_END_HI - J_END_LO
        mid_near_lo = (self._mid_pos - J_MID_LO) < MID_LIMIT_THRESH * mid_range
        end_near_lo = (self._end_pos - J_END_LO) < MID_LIMIT_THRESH * end_range
        mid_near_hi = (J_MID_HI - self._mid_pos) < MID_LIMIT_THRESH * mid_range
        end_near_hi = (J_END_HI - self._end_pos) < MID_LIMIT_THRESH * end_range

        if abs(v_err_frac) >= V_DEADZONE:
            need_to_raise = v_err_frac < 0
            need_to_lower = v_err_frac > 0
            mid_blocked = (need_to_raise and mid_near_lo) or (need_to_lower and mid_near_hi)
            end_blocked = (need_to_raise and end_near_lo) or (need_to_lower and end_near_hi)

            if mid_blocked and not end_blocked:
                action[2] = self._snap_nonzero(v_err_frac)
            elif end_blocked and not mid_blocked:
                action[1] = self._snap_nonzero(v_err_frac)
            else:
                action[1] = self._snap_nonzero(v_err_frac)
                action[2] = self._snap(v_err_frac * 0.4)

        if fill < 0.10:
            action[3] = -1.0
        elif fill < 0.35:
            action[3] = -0.5
        elif fill < STOP_FILL:
            action[3] = -0.25
        else:
            action[3] = 0.0

        return action

    @staticmethod
    def physics_distance_m(env) -> float:
        sid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "tof_site")
        cid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "red_cube")
        if int(sid) < 0 or int(cid) < 0:
            return float("nan")
        return float(np.linalg.norm(env.data.xpos[cid] - env.data.site_xpos[sid]))

    def run_sim(self, env: ArmEnv, pov_scale: int = 1, scene_div: int = 2):
        env._gl_ctx.make_current()

        pov_rgb = env._render_fixedcam(env._mjv_cam_pov)
        pov_bgr = cv2.cvtColor(pov_rgb, cv2.COLOR_RGB2BGR)
        scene_rgb = env._render_fixedcam(env._mjv_cam_scene)
        scene_bgr = cv2.cvtColor(scene_rgb, cv2.COLOR_RGB2BGR)
        env._draw_hud(scene_bgr)

        h_img, w_img = pov_bgr.shape[:2]
        true_dist = self.physics_distance_m(env)
        tof_dist = float(env.last_tof_m)
        det = self._detect_full(pov_bgr)

        pov_vis = cv2.resize(
            pov_bgr,
            (w_img * pov_scale, h_img * pov_scale),
            interpolation=cv2.INTER_NEAREST,
        )
        H, W = pov_vis.shape[:2]

        font = cv2.FONT_HERSHEY_SIMPLEX
        fs, th, gap = 0.42, 1, 17

        side = min(h_img, w_img)
        crop_x = (w_img - side) // 2
        crop_y = (h_img - side) // 2

        cv2.rectangle(
            pov_vis,
            (crop_x * pov_scale, crop_y * pov_scale),
            ((crop_x + side) * pov_scale, (crop_y + side) * pov_scale),
            (100, 100, 100),
            1,
        )

        tx = int(w_img * 0.5 * pov_scale)
        ty = int(h_img * 0.75 * pov_scale)
        cv2.line(pov_vis, (tx - 10, ty), (tx + 10, ty), (0, 180, 255), 1)
        cv2.line(pov_vis, (tx, ty - 10), (tx, ty + 10), (0, 180, 255), 1)

        h_err_str = v_err_str = align_str = "---"
        fill = 0.0
        est_dist = float("nan")

        if det is not None:
            cube_cx, cube_cy, psize, (bx, by, bw, bh) = det
            fill = psize / side

            fg, fx, fy = self._focal(w_img, h_img)
            ps_corr = _perspective_correction(psize, cube_cx, cube_cy, w_img, h_img, fx, fy)
            est_dist = (RING_OUTER_M * fg) / ps_corr if ps_corr > 0 else float("nan")

            cv2.rectangle(
                pov_vis,
                (bx * pov_scale, by * pov_scale),
                ((bx + bw) * pov_scale, (by + bh) * pov_scale),
                COL_BOX,
                2,
            )
            cxp, cyp = int(cube_cx * pov_scale), int(cube_cy * pov_scale)
            arm = max(5, pov_scale)
            cv2.line(pov_vis, (cxp - arm, cyp), (cxp + arm, cyp), COL_X, 1)
            cv2.line(pov_vis, (cxp, cyp - arm), (cxp, cyp + arm), COL_X, 1)
            cv2.line(pov_vis, (tx, ty), (cxp, cyp), (0, 180, 255), 1)

            h_err = (cube_cx - w_img / 2.0) / (w_img / 2.0)
            v_err = (cube_cy - h_img * 0.75) / (h_img / 2.0)
            h_err_str = f"{h_err:+.2f}"
            v_err_str = f"{v_err:+.2f}"
            align = max(
                0.3,
                max(0.0, 1.0 - 2.0 * abs(h_err)) * max(0.0, 1.0 - 2.0 * abs(v_err)),
            )
            align_str = f"{align:.2f}"
        else:
            cv2.putText(pov_vis, "NO RING", (4, H // 2), font, fs, COL_MISS, th, cv2.LINE_AA)

        est_err = (
            est_dist - true_dist
            if (math.isfinite(est_dist) and true_dist > 0)
            else float("nan")
        )
        status = "STOPPED" if self._stopped else "active"

        mid_range = J_MID_HI - J_MID_LO
        mid_headroom = self._mid_pos - J_MID_LO
        mid_pct = 100.0 * mid_headroom / mid_range if mid_range > 1e-9 else 0.0
        mid_saturated = mid_headroom < MID_LIMIT_THRESH * mid_range
        mid_col = (50, 50, 240) if mid_saturated else (180, 255, 180)
        mid_tag = " [END FALLBACK]" if mid_saturated else ""

        rows = [
            (f"True: {true_dist:.4f}m", COL_TOF),
            (
                f"ToF:  {tof_dist:.4f}m" if math.isfinite(tof_dist) else "ToF: ---",
                (200, 200, 0),
            ),
            (
                f"Est:  {est_dist:.4f}m" if math.isfinite(est_dist) else "Est: ---",
                COL_EST,
            ),
            (
                f"Err:  {est_err:+.4f}m" if math.isfinite(est_err) else "Err:  ---",
                COL_ERR_OK if math.isfinite(est_err) and abs(est_err) < 0.03 else COL_ERR_BAD,
            ),
            (f"H_err:{h_err_str}  V_err:{v_err_str}  align:{align_str}", (255, 200, 0)),
            (
                f"Fill:{fill:.0%} [{status}]  crop:{self._ground_crop_frac():.0%}",
                (200, 255, 200),
            ),
            (
                f"mid_pos:{math.degrees(self._mid_pos):.1f}deg  hdroom:{mid_pct:.0f}%{mid_tag}",
                mid_col,
            ),
        ]
        for i, (txt, col) in enumerate(rows):
            cv2.putText(pov_vis, txt, (4, gap + i * gap), font, fs, col, th, cv2.LINE_AA)

        scene_small = cv2.resize(
            scene_bgr,
            (scene_bgr.shape[1] // scene_div, scene_bgr.shape[0] // scene_div),
            interpolation=cv2.INTER_AREA,
        )
        if scene_small.shape[0] != pov_vis.shape[0]:
            nw = int(scene_small.shape[1] * (pov_vis.shape[0] / scene_small.shape[0]))
            scene_small = cv2.resize(
                scene_small,
                (nw, pov_vis.shape[0]),
                interpolation=cv2.INTER_AREA,
            )

        frame = np.concatenate([pov_vis, scene_small], axis=1)
        cv2.imshow("cv2 rendering frame", frame)
        if self._last_gray_vis is not None:
            gray_display = cv2.resize(
                self._last_gray_vis,
                (w_img * pov_scale, h_img * pov_scale),
                interpolation=cv2.INTER_NEAREST,
            )
            cv2.imshow("gray enhanced", gray_display)


def apply_pi_camera_tuning(picam2):
    USE_PI_CAMERA_TUNING = False
    PI_EXPOSURE_US = 10000
    PI_ANALOGUE_GAIN = 1.5
    PI_AWB_ENABLE = False
    PI_COLOUR_GAINS = (1.4, 1.5)
    PI_CONTRAST = 1.4

    if not USE_PI_CAMERA_TUNING:
        return
    controls = {
        "ExposureTime": PI_EXPOSURE_US,
        "AnalogueGain": PI_ANALOGUE_GAIN,
        "AwbEnable": PI_AWB_ENABLE,
        "ColourGains": PI_COLOUR_GAINS,
        "Contrast": PI_CONTRAST,
        "AeEnable": False,
    }
    picam2.set_controls(controls)
    print(f"[PiCam] Manual tuning applied: {controls}")


def run(args):
    env = ArmEnv(
        xml_path=str(Path("Simulation") / "Assets" / "scene.xml"),
        width=args.img_w,
        height=args.img_h,
        display=False,
        discount=0.99,
        episode_horizon=args.episode_horizon,
    )

    ctrl = ColorDetectControl()
    step_count = 0

    def reset_episode():
        nonlocal step_count
        env.reset()
        ctrl.reset()
        step_count = 0

    reset_episode()
    while True:
        ctrl.run_sim(env, pov_scale=args.pov_scale, scene_div=args.scene_div)
        key = cv2.waitKey(args.delay_ms) & 0xFF
        if key in (ord("q"), 27):
            break

        env._gl_ctx.make_current()
        image_rgb = env._render_fixedcam(env._mjv_cam_pov)
        tof_m = float(env.last_tof_m)
        action = ctrl.step(image_rgb, tof_m, step_count, env=env)
        env.step(action)
        step_count += 1

        if ctrl.is_stopped:
            print(
                f"STOPPED (fill >= {ctrl._get_fill_fraction(image_rgb):.0%}) "
                f"at step {step_count}, tof={tof_m:.4f}m"
            )
            while True:
                ctrl.run_sim(env, pov_scale=args.pov_scale, scene_div=args.scene_div)
                k = cv2.waitKey(100) & 0xFF
                if k in (ord("q"), 27):
                    cv2.destroyAllWindows()
                    return

    cv2.destroyAllWindows()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--img_w", type=int, default=640)
    p.add_argument("--img_h", type=int, default=480)
    p.add_argument("--episode_horizon", type=int, default=600)
    p.add_argument(
        "--domain_rand",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument("--pov_scale", type=int, default=1)
    p.add_argument("--scene_div", type=int, default=2)
    p.add_argument("--delay_ms", type=int, default=30)
    run(p.parse_args())