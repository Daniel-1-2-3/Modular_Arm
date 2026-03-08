import math
import numpy as np
import cv2
import mujoco
from pathlib import Path
from Arm_Env.arm_env import ArmEnv
import argparse

CUBE_SIZE_M = 0.04
REF_FOCAL = 608.0
REF_W = 640.0
REF_H = 480.0
TOF_H_M = 0.07
CAM_PITCH_DEG = 8.0

# Grayscale detection for white cube on dark background
# CLAHE boosts contrast; Otsu auto-thresholds; these are validation limits
WHITE_GRAY_MIN = 170        # CHANGED: was 180 — catches dimmer cube faces
WHITE_REGION_MEAN_MIN = 150 # minimum mean gray — raised after bg normalization (reflections land ~136)
CLAHE_CLIP = 6.0            # CHANGED: was 3.0 — stronger local contrast
CLAHE_GRID = (8, 8)         # CLAHE tile grid size
GAMMA = 1.8                 # gamma > 1 crushes darks, preserves brights = more contrast
MORPH_OPEN_K = 7            # kernel size for morphological opening to kill bright spots

MIN_AREA = 600             # CHANGED: was 600 — detect cube at longer range
MIN_SIDE = 15               # CHANGED: was 22 — allow smaller detections
MAX_AREA_FRAC = 1.0         # No max area
BORDER_MARGIN = 2

# Stop when cube fills this fraction of the cropped frame
STOP_FILL = 0.50

# Dynamic bottom crop to exclude bright ground
# At fill=0 crop this fraction off the bottom; at fill>=STOP_FILL crop nothing
GROUND_CROP_MAX = 0.05     # max fraction of height to remove when cube is far
GROUND_CROP_MIN = 0.0       # no crop when cube is close

# Joint limits (radians / metres)
J_MID_LO = 0.0
J_MID_HI = 25.0 / 180.0 * math.pi
J_END_LO = -90.0 / 180.0 * math.pi
J_END_HI = 110.0 / 180.0 * math.pi
# Fraction of range remaining below which mid is considered at limit
MID_LIMIT_THRESH = 0.10

# Display
COL_BOX = (0, 255, 0)
COL_X = (0, 220, 255)
COL_TOF = (0, 255, 200)
COL_EST = (0, 165, 255)
COL_ERR_OK = (0, 220, 255)
COL_ERR_BAD = (50, 50, 240)
COL_MISS = (50, 50, 200)

# Precompute gamma LUT
_GAMMA_LUT = np.array([
    np.clip(pow(i / 255.0, GAMMA) * 255.0, 0, 255)
    for i in range(256)
], dtype=np.uint8)


class ColorDetectControl:
    """
    Control sign conventions for transfer:

        action[0] > 0: base rotates in -radian direction  (e.g. right)
        action[0] < 0: base rotates in +radian direction  (e.g. left)
        action[1] > 0: mid joint moves in -radian direction  (arm down)
        action[1] < 0: mid joint moves in +radian direction  (arm up)
        action[2] > 0: end-effector moves in -radian direction  (tip down)
        action[2] < 0: end-effector moves in +radian direction  (tip up)
        action[3] > 0: rail moves in -metre direction  (backward)
        action[3] < 0: rail moves in +metre direction  (forward / advance)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._stopped = False
        # Tracked joint positions
        self._mid_pos: float = 0.0
        self._end_pos: float = 0.0
        # Last contrast-enhanced grayscale frame for display
        self._last_gray_vis: np.ndarray | None = None
        # Track fill for dynamic ground crop
        self._last_fill: float = 0.0

    @property
    def is_stopped(self) -> bool:
        return self._stopped

    @staticmethod
    def _snap(val: float, deadzone: float = 0.08) -> float:
        # Snap a float to {-1, -0.5, 0, 0.5, 1} with a deadzone
        if abs(val) < deadzone:
            return 0.0
        sign = 1.0 if val > 0 else -1.0
        return sign * (1.0 if abs(val) >= 0.5 else 0.5)

    @staticmethod
    def _snap_nonzero(val: float, deadzone: float = 0.08) -> float:
        # Snap, but not return 0
        if abs(val) < deadzone:
            # Error is small but real — use minimum step in correct direction
            return 0.5 if val >= 0 else -0.5
        return 1.0 if val >= 0.5 else (-1.0 if val <= -0.5 else (0.5 if val > 0 else -0.5))

    def step(self, image_rgb: np.ndarray, tof_reading: float, step_num, env=None) -> np.ndarray:
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

        # Build display frame
        vis = bgr.copy()

        if det is None:
            cv2.putText(vis, "NO CUBE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_MISS, 2)
            cv2.imshow("step", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            if self._last_gray_vis is not None:
                cv2.imshow("gray enhanced", self._last_gray_vis)
            cv2.waitKey(1)
            return action

        cube_cx, cube_cy, psize, (bx, by, bw, bh) = det
        fill = psize / min(h_img, w_img)

        # Draw bounding box
        cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), COL_BOX, 2)

        # Draw cube center crosshair
        cx, cy = int(cube_cx), int(cube_cy)
        cv2.line(vis, (cx - 8, cy), (cx + 8, cy), COL_X, 1)
        cv2.line(vis, (cx, cy - 8), (cx, cy + 8), COL_X, 1)

        # Draw target crosshair
        tx, ty = int(w_img / 2), int(h_img * 0.7)
        cv2.line(vis, (tx - 8, ty), (tx + 8, ty), (0, 180, 255), 1)
        cv2.line(vis, (tx, ty - 8), (tx, ty + 8), (0, 180, 255), 1)

        # Line from target to cube
        cv2.line(vis, (tx, ty), (cx, cy), (0, 180, 255), 1)

        # HUD text
        h_err_frac = (cube_cx - w_img / 2.0) / (w_img / 2.0)
        v_err_frac = (cube_cy - h_img * 0.7) / (h_img / 2.0)
        status = "STOPPED" if self._stopped else "active"
        lines = [
            f"Fill: {fill:.0%} [{status}]",
            f"H_err: {h_err_frac:+.2f}  V_err: {v_err_frac:+.2f}",
        ]
        for i, txt in enumerate(lines):
            cv2.putText(vis, txt, (10, 25 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1, cv2.LINE_AA)

        cv2.imshow("step", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        if self._last_gray_vis is not None:
            cv2.imshow("gray enhanced", self._last_gray_vis)
        cv2.waitKey(1)

        # Stop condition
        if fill >= STOP_FILL and step_num > 50:
            self._stopped = True
            return action

        # Alignment
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
                action[1] = 0.0
                action[2] = self._snap_nonzero(v_err_frac)
            elif end_blocked and not mid_blocked:
                action[1] = self._snap_nonzero(v_err_frac)
                action[2] = 0.0
            else:
                action[1] = self._snap_nonzero(v_err_frac)
                action[2] = self._snap(v_err_frac * 0.4)

        # Advance based on fill fraction
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
    def _center_crop_square(img):
        h, w = img.shape[:2]
        side = min(h, w)
        x0 = (w - side) // 2
        y0 = (h - side) // 2
        return img[y0:y0 + side, x0:x0 + side], x0, y0

    @staticmethod
    def _to_gray_enhanced(frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
        enhanced = clahe.apply(gray)
        
        # Find Otsu threshold to separate background from cube
        otsu_thresh, _ = cv2.threshold(enhanced, 0, 255,
                                    cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # Apply gamma ONLY to pixels below Otsu threshold (background)
        # Cube pixels (above threshold) are left as-is
        result = enhanced.copy()
        bg_mask = enhanced < otsu_thresh
        result[bg_mask] = _GAMMA_LUT[enhanced[bg_mask]]
        
        return result

    @staticmethod
    def _focal(w_img, h_img):
        fx = REF_FOCAL * (w_img / REF_W)
        fy = REF_FOCAL * (h_img / REF_H)
        return math.sqrt(fx * fy), fx, fy

    @staticmethod
    def _pixel_size(cnt, bbox=None, touches_border=False):
        if not touches_border:
            return float(math.sqrt(cv2.contourArea(cnt)))
        if bbox is None:
            x, y, w, h = cv2.boundingRect(cnt)
        else:
            x, y, w, h = bbox
        rect = cv2.minAreaRect(cnt)
        rw, rh = rect[1]
        return float(max([w, h] + ([rw] if rw > 1e-6 else []) + ([rh] if rh > 1e-6 else [])))

    @staticmethod
    def _touches_border(cnt, w_img, h_img):
        x, y, w, h = cv2.boundingRect(cnt)
        return (x <= BORDER_MARGIN or y <= BORDER_MARGIN
                or x + w >= w_img - BORDER_MARGIN or y + h >= h_img - BORDER_MARGIN)

    @staticmethod
    def _poly_score(cnt):
        peri = cv2.arcLength(cnt, True)
        if peri <= 1e-6:
            return 0.0
        n = len(cv2.approxPolyDP(cnt, 0.04 * peri, True))
        if 3 <= n <= 6: return 1.0
        if n == 7: return 0.7
        if n == 8: return 0.5      # CHANGED: was rejected — allow octagons (rounded cube)
        return 0.0

    @staticmethod
    def _is_white_region(gray_enhanced, cnt):
        """Validate contour is bright on the contrast-enhanced grayscale."""
        mask = np.zeros(gray_enhanced.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        if cv2.countNonZero(mask) < 10:
            return False
        roi = gray_enhanced[mask > 0]
        # Mean brightness must be high
        if np.mean(roi) < WHITE_REGION_MEAN_MIN:
            return False
        # CHANGED: was 55 — allow more lighting gradient across cube face
        if np.std(roi) > 70:
            return False
        return True

    def _score_contour(self, cnt, frame_area, w_img, h_img, gray_enhanced=None):
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
        ext = area / (w * h)
        rect = cv2.minAreaRect(cnt)
        rw, rh = rect[1]
        rsq = min(rw, rh) / max(rw, rh) if rw > 1e-6 and rh > 1e-6 else 0.0
        rext = area / (rw * rh) if rw > 1e-6 and rh > 1e-6 else 0.0
        poly = self._poly_score(cnt)

        # CHANGED: relaxed geometry thresholds throughout
        if not tb:
            sol_thresh = 0.70 if area > frame_area * 0.15 else 0.82  # was 0.75 / 0.88
            if sq < 0.55 or rsq < 0.55 or sol < sol_thresh or ext < 0.55 or rext < 0.45 or poly < 0.5:
                return None  # was 0.70/0.70/sol/0.70/0.55/0.7
        else:
            if sq < 0.30 or sol < 0.60 or ext < 0.30 or poly < 0.3:
                return None  # was 0.40/0.70/0.35/0.5

        if gray_enhanced is not None and not self._is_white_region(gray_enhanced, cnt):
            return None

        psize = self._pixel_size(cnt, bbox=(x, y, w, h), touches_border=tb)
        score = 2.0 * max(sq, rsq) + 1.5 * sol + 1.0 * max(ext, rext) + 0.8 * poly + 0.002 * psize
        return score, x + w / 2.0, y + h / 2.0, psize, (x, y, w, h)

    def _best_from_mask(self, gray_enhanced, mask):
        # Large opening kernel removes small bright spots from high contrast
        kern_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                              (MORPH_OPEN_K, MORPH_OPEN_K))
        kern_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern_open, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern_close, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        h_img, w_img = gray_enhanced.shape[:2]
        fa = h_img * w_img
        best_s, best_r = -1.0, None
        for cnt in contours:
            res = self._score_contour(cnt, fa, w_img, h_img, gray_enhanced=gray_enhanced)
            if res is None:
                continue
            s, cx, cy, ps, bb = res
            if s > best_s:
                best_s, best_r = s, (cx, cy, ps, bb)
        return best_r

    def _detect(self, gray_enhanced):
        """Detect white cube on contrast-enhanced grayscale.
        
        Primary: Otsu auto-threshold (best for bimodal dark-bg / white-cube).
        Fallback: fixed bright threshold.
        """
        # Primary: Otsu thresholding — finds optimal split automatically
        _, mask_otsu = cv2.threshold(gray_enhanced, 0, 255,
                                     cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        det = self._best_from_mask(gray_enhanced, mask_otsu)
        if det is not None:
            return det
        # Fallback: fixed high-brightness threshold
        _, mask_fixed = cv2.threshold(gray_enhanced, WHITE_GRAY_MIN, 255,
                                      cv2.THRESH_BINARY)
        det = self._best_from_mask(gray_enhanced, mask_fixed)
        if det is not None:
            return det
        # Last resort: adaptive threshold
        adapt = cv2.adaptiveThreshold(gray_enhanced, 255,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, blockSize=51, C=-15)
        return self._best_from_mask(gray_enhanced, adapt)

    def _ground_crop_frac(self) -> float:
        """Fraction of height to crop from bottom, based on last fill."""
        t = min(self._last_fill / STOP_FILL, 1.0)
        return GROUND_CROP_MAX + (GROUND_CROP_MIN - GROUND_CROP_MAX) * t

    def _detect_full(self, frame_bgr_full):
        h_full, w_full = frame_bgr_full.shape[:2]

        # Dynamic bottom crop — remove bright ground
        crop_frac = self._ground_crop_frac()
        crop_rows = int(h_full * crop_frac)
        if crop_rows > 0:
            frame_cropped = frame_bgr_full[:h_full - crop_rows, :]
        else:
            frame_cropped = frame_bgr_full

        cropped, x_off, y_off = self._center_crop_square(frame_cropped)
        gray_enh = self._to_gray_enhanced(cropped)

        # Store full-frame enhanced grayscale for display (with crop line)
        full_gray = self._to_gray_enhanced(frame_bgr_full)
        # Draw crop line on display frame
        if crop_rows > 0:
            cv2.line(full_gray, (0, h_full - crop_rows),
                     (w_full, h_full - crop_rows), 128, 1)
        self._last_gray_vis = full_gray

        det = self._detect(gray_enh)
        if det is None:
            return None
        cx, cy, ps, (bx, by, bw, bh) = det

        # Update fill tracker
        side = min(frame_cropped.shape[:2])
        self._last_fill = ps / side if side > 0 else 0.0

        return (cx + x_off, cy + y_off, ps, (bx + x_off, by + y_off, bw, bh))

    def _backproject(self, cx, cy, dist, fx, fy, cx_img, cy_img):
        nx = (cx - cx_img) / fx
        ny = (cy - cy_img) / fy
        rl = math.sqrt(nx ** 2 + ny ** 2 + 1.0)
        z = dist / rl
        return nx * z, ny * z, z

    def get_distance(self, image_rgb):
        h, w = image_rgb.shape[:2]
        fg, fx, fy = self._focal(w, h)
        det = self._detect_full(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        if det is None:
            return float("nan")
        cx, cy, ps, _ = det
        d = (CUBE_SIZE_M * fg) / ps if ps > 0 else float("nan")
        if not math.isfinite(d) or d <= 0:
            return float("nan")
        xc, yc, zc = self._backproject(cx, cy, d, fx, fy, w / 2.0, h / 2.0)
        pr = math.radians(CAM_PITCH_DEG)
        dx, dy, dz = xc, yc - TOF_H_M, zc
        dyt = dy * math.cos(pr) + dz * math.sin(pr)
        dzt = -dy * math.sin(pr) + dz * math.cos(pr)
        return float(math.sqrt(dx ** 2 + dyt ** 2 + dzt ** 2))

    def get_angles(self, image_rgb):
        h, w = image_rgb.shape[:2]
        fg, fx, fy = self._focal(w, h)
        det = self._detect_full(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        if det is None:
            return float("nan"), float("nan"), float("nan")
        cx, cy, ps, _ = det
        d = (CUBE_SIZE_M * fg) / ps if ps > 0 else float("nan")
        if not math.isfinite(d) or d <= 0:
            return float("nan"), float("nan"), float("nan")
        xc, yc, zc = self._backproject(cx, cy, d, fx, fy, w / 2.0, h / 2.0)
        ha = math.degrees(math.atan2(xc, zc))
        va = math.degrees(math.atan2(-yc, zc))
        if zc > 1e-6:
            pr = math.radians(CAM_PITCH_DEG)
            vo = -math.degrees(math.atan2(
                TOF_H_M * math.cos(pr) - zc * math.sin(pr),
                TOF_H_M * math.sin(pr) + zc * math.cos(pr)))
        else:
            vo = float("nan")
        return ha, va, float(vo)

    def _get_fill_fraction(self, image_rgb):
        h, w = image_rgb.shape[:2]
        side = min(h, w)
        det = self._detect_full(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        if det is None:
            return 0.0
        return det[2] / side

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

        # Crop
        cv2.rectangle(
            pov_vis,
            (crop_x * pov_scale, crop_y * pov_scale),
            ((crop_x + side) * pov_scale, (crop_y + side) * pov_scale),
            (100, 100, 100),
            1,
        )

        # Crosshair
        tx = int(w_img * 0.5 * pov_scale)
        ty = int(h_img * 0.60 * pov_scale)
        cv2.line(pov_vis, (tx - 10, ty), (tx + 10, ty), (0, 180, 255), 1)
        cv2.line(pov_vis, (tx, ty - 10), (tx, ty + 10), (0, 180, 255), 1)

        h_err_str = "---"
        v_err_str = "---"
        align_str = "---"
        fill = 0.0
        est_dist = float("nan")

        if det is not None:
            cube_cx, cube_cy, psize, (bx, by, bw, bh) = det
            fill = psize / side

            fg, _, _ = self._focal(w_img, h_img)
            est_dist = (CUBE_SIZE_M * fg) / psize if psize > 0 else float("nan")

            # Bbox
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
            v_err = (cube_cy - h_img * 0.60) / (h_img / 2.0)

            h_err_str = f"{h_err:+.2f}"
            v_err_str = f"{v_err:+.2f}"

            align = max(
                0.3,
                max(0.0, 1.0 - 2.0 * abs(h_err)) *
                max(0.0, 1.0 - 2.0 * abs(v_err)),
            )
            align_str = f"{align:.2f}"
        else:
            cv2.putText(pov_vis, "NO CUBE", (4, H // 2), font, fs, COL_MISS, th, cv2.LINE_AA)

        est_err = est_dist - true_dist if (math.isfinite(est_dist) and true_dist > 0) else float("nan")
        status = "STOPPED" if self._stopped else "active"

        mid_range = J_MID_HI - J_MID_LO
        mid_headroom_up = self._mid_pos - J_MID_LO
        mid_pct = 100.0 * mid_headroom_up / mid_range if mid_range > 1e-9 else 0.0
        mid_saturated = mid_headroom_up < MID_LIMIT_THRESH * mid_range
        mid_col = (50, 50, 240) if mid_saturated else (180, 255, 180)
        mid_tag = " [END FALLBACK]" if mid_saturated else ""

        rows = [
            (f"True: {true_dist:.4f}m", COL_TOF),
            (f"ToF:  {tof_dist:.4f}m" if math.isfinite(tof_dist) else "ToF: ---", (200, 200, 0)),
            (f"Est:  {est_dist:.4f}m" if math.isfinite(est_dist) else "Est: ---", COL_EST),
            (f"Err:  {est_err:+.4f}m" if math.isfinite(est_err) else "Err:  ---",
             COL_ERR_OK if math.isfinite(est_err) and abs(est_err) < 0.03 else COL_ERR_BAD),
            (f"H_err: {h_err_str}  V_err: {v_err_str}  align: {align_str}", (255, 200, 0)),
            (f"Fill: {fill:.0%} [{status}]  crop:{self._ground_crop_frac():.0%}", (200, 255, 200)),
            (f"mid_pos: {math.degrees(self._mid_pos):.1f}deg  hdroom:{mid_pct:.0f}%{mid_tag}", mid_col),
        ]

        for i, (txt, col) in enumerate(rows):
            cv2.putText(pov_vis, txt, (4, gap + i * gap), font, fs, col, th, cv2.LINE_AA)

        scene_small = cv2.resize(
            scene_bgr,
            (scene_bgr.shape[1] // scene_div, scene_bgr.shape[0] // scene_div),
            interpolation=cv2.INTER_AREA,
        )

        if scene_small.shape[0] != pov_vis.shape[0]:
            new_w = int(scene_small.shape[1] * (pov_vis.shape[0] / scene_small.shape[0]))
            scene_small = cv2.resize(
                scene_small,
                (new_w, pov_vis.shape[0]),
                interpolation=cv2.INTER_AREA,
            )

        cv2_rendering_frame = np.concatenate([pov_vis, scene_small], axis=1)
        cv2.imshow("cv2 rendering frame", cv2_rendering_frame)
        if self._last_gray_vis is not None:
            gray_display = cv2.resize(
                self._last_gray_vis,
                (w_img * pov_scale, h_img * pov_scale),
                interpolation=cv2.INTER_NEAREST,
            )
            cv2.imshow("gray enhanced", gray_display)


def run(args):
    env = ArmEnv(
        xml_path=str(Path("Simulation") / "Assets" / "scene.xml"),
        width=args.img_w, height=args.img_h,
        display=False, discount=0.99,
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

        # Pass env so ctrl can read joint positions for limit detection
        action = ctrl.step(image_rgb, tof_m, step_count, env=env)
        result = env.step(action)
        step_count += 1

        if ctrl.is_stopped:
            print(f"STOPPED (fill >= {ctrl._get_fill_fraction(image_rgb):.0%}) "
                  f"at step {step_count}, tof={tof_m:.4f}m")
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
    p.add_argument("--domain_rand", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--pov_scale", type=int, default=1)
    p.add_argument("--scene_div", type=int, default=2)
    p.add_argument("--delay_ms", type=int, default=30)
    run(p.parse_args())