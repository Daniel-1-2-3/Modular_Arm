import math
import numpy as np
import cv2
import mujoco
from Arm_Env.arm_env import ArmEnv

# ── Constants ─────────────────────────────────────────────────────────────────

CUBE_SIZE_M = 0.04
REF_FOCAL = 608.0
REF_W = 640.0
REF_H = 480.0
TOF_H_M = 0.07
CAM_PITCH_DEG = 8.0

# Detection
WHITE_MIN = 160
WHITE_SPREAD = 40
MIN_AREA = 600
MIN_SIDE = 22
MAX_AREA_FRAC = 0.25
BORDER_MARGIN = 2

# Stop when cube fills this fraction of the cropped frame
STOP_FILL = 0.40

# Joint limits (radians / metres) — from ArmEnv limits_by_name
J_MID_LO  =  0.0                  # j_b_mid lower limit
J_MID_HI  =  25.0 / 180.0 * math.pi   # j_b_mid upper limit  (~0.436 rad)
J_END_LO  = -90.0 / 180.0 * math.pi   # j_c_end lower limit
J_END_HI  = 110.0 / 180.0 * math.pi   # j_c_end upper limit
# Fraction of range remaining below which mid is considered "at limit"
MID_LIMIT_THRESH = 0.10           # 10% of mid's total range

# Display
COL_BOX = (0, 255, 0)
COL_X = (0, 220, 255)
COL_TOF = (0, 255, 200)
COL_EST = (0, 165, 255)
COL_ERR_OK = (0, 220, 255)
COL_ERR_BAD = (50, 50, 240)
COL_MISS = (50, 50, 200)


class ColorDetectControl:
    """
    Always-incremental vision controller.

    Control sign conventions (derived from sanity_check.py):
      All joints are negated internally by the env — a positive action value
      applies a negative delta in joint space.  Physical directions:

        action[0] > 0  →  base rotates in -radian direction  (e.g. right)
        action[0] < 0  →  base rotates in +radian direction  (e.g. left)

        action[1] > 0  →  mid joint moves in -radian direction  (arm down)
        action[1] < 0  →  mid joint moves in +radian direction  (arm up)

        action[2] > 0  →  end-effector moves in -radian direction  (tip down)
        action[2] < 0  →  end-effector moves in +radian direction  (tip up)

        action[3] > 0  →  rail moves in -metre direction  (backward)
        action[3] < 0  →  rail moves in +metre direction  (forward / advance)

    Usage:
        ctrl = ColorDetectControl()
        ctrl.reset()
        # pass env so joint positions are available for limit detection:
        action = ctrl.step(image_rgb, tof_m, env=env)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Call on episode start."""
        self._stopped = False
        # Tracked joint positions (updated each step when env is provided)
        self._mid_pos: float = 0.0   # j_b_mid current position (rad)
        self._end_pos: float = 0.0   # j_c_end current position (rad)

    @property
    def is_stopped(self) -> bool:
        return self._stopped

    # ══════════════════════════════════════════════════════════════════════════
    # Public: main loop
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _snap(val: float, deadzone: float = 0.08) -> float:
        """
        Snap a float to {-1, -0.5, 0, 0.5, 1} with a deadzone.
          |val| < deadzone          →  0.0   (true noise, do nothing)
          deadzone <= |val| < 0.5   → ±0.5   (small correction)
          |val| >= 0.5              → ±1.0   (large correction)
        """
        if abs(val) < deadzone:
            return 0.0
        sign = 1.0 if val > 0 else -1.0
        return sign * (1.0 if abs(val) >= 0.5 else 0.5)

    @staticmethod
    def _snap_nonzero(val: float, deadzone: float = 0.08) -> float:
        """
        Like _snap but NEVER returns 0 — used when we must always command
        a joint (e.g. the primary alignment axis).
        Collapses the deadzone to ±0.5 so tiny real errors still move.
        """
        if abs(val) < deadzone:
            # Error is small but real — use minimum step in correct direction
            return 0.5 if val >= 0 else -0.5
        return 1.0 if val >= 0.5 else (-1.0 if val <= -0.5 else (0.5 if val > 0 else -0.5))

    def step(self, image_rgb: np.ndarray, tof_reading: float,
             env=None) -> np.ndarray:
        """
        Feed camera frame (RGB uint8) + ToF (metres).
        Optionally pass `env` so joint positions can be read for limit detection.
        Returns (4,) float32 with each element in {-1, -0.5, 0, 0.5, 1}.

        Rules:
          - Per-joint deadzone: joints only move when error is meaningful.
          - BUT: if horizontal OR vertical error exists above a minimum
            threshold, at least the primary joint for that axis is guaranteed
            non-zero — no all-zero stall while misaligned.
          - Drive (action[3]) is always non-zero while fill < STOP_FILL.
          - All axes run simultaneously (no sequential modes).
          - If mid is at its limit for the needed direction, route to end; vice versa.

        Sign conventions (DO NOT CHANGE):
          action[0] < 0  →  base rotates left   |  action[0] > 0  → right
          action[1] < 0  →  mid raises arm up   |  action[1] > 0  → down
          action[2] < 0  →  end tip up          |  action[2] > 0  → down
          action[3] < 0  →  drive forward       |  action[3] > 0  → backward
        """
        action = np.zeros(4, dtype=np.float32)

        # ── Read joint positions from env ──────────────────────────────
        if env is not None:
            try:
                qpos = env.data.qpos
                # Indices: 0=drive, 1=base, 2=mid, 3=end
                self._mid_pos = float(qpos[2])
                self._end_pos = float(qpos[3])
            except Exception:
                pass

        h_img, w_img = image_rgb.shape[:2]
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        det = self._detect_full(bgr)

        # Can't see cube — hold still
        if det is None:
            return action

        cube_cx, cube_cy, psize, _ = det
        fill = psize / min(h_img, w_img)

        # Stop condition — zero action is intentional here
        if fill >= STOP_FILL:
            self._stopped = True
            return action

        # ── Pixel errors ───────────────────────────────────────────────
        h_err_frac = (cube_cx - w_img / 2.0) / (w_img / 2.0)   # +ve → cube right
        v_err_frac = (cube_cy - h_img * 0.7) / (h_img / 2.0)   # +ve → cube below target

        H_DEADZONE = 0.06   # horizontal error below this → base stays still
        V_DEADZONE = 0.06   # vertical error below this → arm stays still

        # ── action[0]: base horizontal centering ──────────────────────
        # cube right (h_err > 0) → action[0] < 0  (keep current negation)
        # Use _snap_nonzero so we never stall horizontally when misaligned
        if abs(h_err_frac) >= H_DEADZONE:
            action[0] = self._snap_nonzero(-h_err_frac)
        # else: truly centred — leave at 0 (deadzone is intentional here)

        # ── action[1]/[2]: vertical alignment with joint-limit fallback ─
        # cube below target (v_err > 0) → need to look down → action > 0
        # cube above target (v_err < 0) → need to raise     → action < 0
        #
        # Joint limit logic (signs as per env convention):
        #   Raising arm  → action[1] < 0  (mid_pos decreases toward J_MID_LO)
        #   Lowering arm → action[1] > 0  (mid_pos increases toward J_MID_HI)
        #   mid_near_lo  → mid can't raise further; route to end
        #   end_near_lo  → end can't raise further; route to mid
        #   (symmetric for lower-limit on the upper side)
        mid_range = J_MID_HI - J_MID_LO
        end_range = J_END_HI - J_END_LO
        mid_near_lo = (self._mid_pos - J_MID_LO) < MID_LIMIT_THRESH * mid_range
        end_near_lo = (self._end_pos - J_END_LO) < MID_LIMIT_THRESH * end_range
        mid_near_hi = (J_MID_HI - self._mid_pos) < MID_LIMIT_THRESH * mid_range
        end_near_hi = (J_END_HI - self._end_pos) < MID_LIMIT_THRESH * end_range

        if abs(v_err_frac) >= V_DEADZONE:
            need_to_raise = v_err_frac < 0
            need_to_lower = v_err_frac > 0

            # Determine which joints are blocked for this direction
            mid_blocked = (need_to_raise and mid_near_lo) or (need_to_lower and mid_near_hi)
            end_blocked = (need_to_raise and end_near_lo) or (need_to_lower and end_near_hi)

            if mid_blocked and not end_blocked:
                # Mid is at its limit — end does all the work (guaranteed non-zero)
                action[1] = 0.0
                action[2] = self._snap_nonzero(v_err_frac)
            elif end_blocked and not mid_blocked:
                # End is at its limit — mid does all the work (guaranteed non-zero)
                action[1] = self._snap_nonzero(v_err_frac)
                action[2] = 0.0
            else:
                # Both available: mid is the primary (guaranteed non-zero),
                # end assists when error is large enough to clear its own deadzone
                action[1] = self._snap_nonzero(v_err_frac)
                action[2] = self._snap(v_err_frac * 0.4)   # may be 0 if error is small
        # else: within vertical deadzone — leave both at 0 (intentional)

        # ── action[3]: drive — always forward, never zero ──────────────
        # action[3] < 0 = forward.  Far → full speed, close → half speed.
        fg, _, _ = self._focal(w_img, h_img)
        distance = (CUBE_SIZE_M * fg) / psize if psize > 0 else float("nan")
        tof_val = float(tof_reading) if (math.isfinite(float(tof_reading))
                                         and float(tof_reading) > 0) else distance

        if math.isfinite(tof_val) and tof_val > 0.06:
            action[3] = -1.0 if tof_val > 0.20 else -0.5
        else:
            action[3] = -0.5   # no range info — creep forward

        return action

    # ══════════════════════════════════════════════════════════════════════════
    # Action clamping
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    # ══════════════════════════════════════════════════════════════════════════
    # Vision: preprocessing
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _center_crop_square(img):
        h, w = img.shape[:2]
        side = min(h, w)
        x0 = (w - side) // 2
        y0 = (h - side) // 2
        return img[y0:y0 + side, x0:x0 + side], x0, y0

    @staticmethod
    def _enhance(frame_bgr):
        smoothed = cv2.bilateralFilter(frame_bgr, d=7, sigmaColor=50, sigmaSpace=50)
        lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_min, l_max = l.min(), l.max()
        if l_max > l_min:
            l = ((l.astype(np.float32) - l_min) / (l_max - l_min) * 255).astype(np.uint8)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # ══════════════════════════════════════════════════════════════════════════
    # Vision: detection internals
    # ══════════════════════════════════════════════════════════════════════════

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
        return 0.0

    @staticmethod
    def _is_white_region(frame_bgr, cnt):
        mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        if cv2.countNonZero(mask) < 10:
            return False
        b, g, r = cv2.mean(frame_bgr, mask=mask)[:3]
        if min(r, g, b) < 130:
            return False
        if max(r, g, b) - min(r, g, b) > 50:
            return False
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if np.std(gray[mask > 0]) > 35:
            return False
        return True

    def _score_contour(self, cnt, frame_area, w_img, h_img, frame_bgr=None):
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

        if not tb:
            if sq < 0.70 or rsq < 0.70 or sol < 0.88 or ext < 0.70 or rext < 0.55 or poly < 0.7:
                return None
        else:
            if sq < 0.40 or sol < 0.70 or ext < 0.35 or poly < 0.5:
                return None

        if frame_bgr is not None and not self._is_white_region(frame_bgr, cnt):
            return None

        psize = self._pixel_size(cnt, bbox=(x, y, w, h), touches_border=tb)
        score = 2.0 * max(sq, rsq) + 1.5 * sol + 1.0 * max(ext, rext) + 0.8 * poly + 0.002 * psize
        return score, x + w / 2.0, y + h / 2.0, psize, (x, y, w, h)

    def _best_from_mask(self, frame_bgr, mask):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        h_img, w_img = frame_bgr.shape[:2]
        fa = h_img * w_img
        best_s, best_r = -1.0, None
        for cnt in contours:
            res = self._score_contour(cnt, fa, w_img, h_img, frame_bgr=frame_bgr)
            if res is None:
                continue
            s, cx, cy, ps, bb = res
            if s > best_s:
                best_s, best_r = s, (cx, cy, ps, bb)
        return best_r

    def _detect(self, frame_bgr):
        b, g, r = cv2.split(frame_bgr)
        bright = (r >= WHITE_MIN) & (g >= WHITE_MIN) & (b >= WHITE_MIN)
        ch_max = np.maximum(np.maximum(r, g), b)
        ch_min = np.minimum(np.minimum(r, g), b)
        spread = (ch_max.astype(np.int16) - ch_min.astype(np.int16)) <= WHITE_SPREAD
        mask1 = (bright & spread).astype(np.uint8) * 255
        det = self._best_from_mask(frame_bgr, mask1)
        if det is not None:
            return det
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, blockSize=51, C=-15)
        return self._best_from_mask(frame_bgr, adapt)

    def _detect_full(self, frame_bgr_full):
        cropped, x_off, y_off = self._center_crop_square(frame_bgr_full)
        enhanced = self._enhance(cropped)
        det = self._detect(enhanced)
        if det is None:
            det = self._detect(cropped)
        if det is None:
            return None
        cx, cy, ps, (bx, by, bw, bh) = det
        return (cx + x_off, cy + y_off, ps, (bx + x_off, by + y_off, bw, bh))

    # ══════════════════════════════════════════════════════════════════════════
    # Vision: geometry
    # ══════════════════════════════════════════════════════════════════════════

    def _backproject(self, cx, cy, dist, fx, fy, cx_img, cy_img):
        nx = (cx - cx_img) / fx
        ny = (cy - cy_img) / fy
        rl = math.sqrt(nx**2 + ny**2 + 1.0)
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
        return float(math.sqrt(dx**2 + dyt**2 + dzt**2))

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

    # ══════════════════════════════════════════════════════════════════════════
    # Utilities
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def physics_distance_m(env) -> float:
        sid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "tof_site")
        cid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "red_cube")
        if int(sid) < 0 or int(cid) < 0:
            return float("nan")
        return float(np.linalg.norm(env.data.xpos[cid] - env.data.site_xpos[sid]))

    # ══════════════════════════════════════════════════════════════════════════
    # Display
    # ══════════════════════════════════════════════════════════════════════════

    def run_sim(self, env: ArmEnv, pov_scale: int = 1, scene_div: int = 2):
        env._gl_ctx.make_current()
        raw_rgb = env._render_fixedcam(env._mjv_cam_pov)
        raw_bgr = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2BGR)
        h_img, w_img = raw_bgr.shape[:2]

        true_dist = self.physics_distance_m(env)
        tof_dist = float(env.last_tof_m)
        det = self._detect_full(raw_bgr)

        s = pov_scale
        vis = cv2.resize(raw_bgr, (w_img * s, h_img * s), interpolation=cv2.INTER_NEAREST)
        H, W = vis.shape[:2]
        font, fs, th, gap = cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1, 17

        # Draw crop region
        side = min(h_img, w_img)
        cx0 = (w_img - side) // 2
        cy0 = (h_img - side) // 2
        cv2.rectangle(vis, (cx0 * s, cy0 * s), ((cx0 + side) * s, (cy0 + side) * s), (100, 100, 100), 1)

        # Draw target crosshair at (50% width, 75% height)
        tx, ty = int(w_img * 0.5 * s), int(h_img * 0.75 * s)
        cv2.line(vis, (tx - 10, ty), (tx + 10, ty), (0, 180, 255), 1)
        cv2.line(vis, (tx, ty - 10), (tx, ty + 10), (0, 180, 255), 1)

        h_err_str = "---"
        v_err_str = "---"
        fill = 0.0
        est_dist = float("nan")

        if det is not None:
            cube_cx, cube_cy, psize, (bx, by, bw, bh) = det
            fill = psize / side
            fg, _, _ = self._focal(w_img, h_img)
            est_dist = (CUBE_SIZE_M * fg) / psize if psize > 0 else float("nan")

            cv2.rectangle(vis, (bx * s, by * s), ((bx + bw) * s, (by + bh) * s), COL_BOX, 2)
            cd = int(cube_cx * s), int(cube_cy * s)
            arm = max(5, s)
            cv2.line(vis, (cd[0] - arm, cd[1]), (cd[0] + arm, cd[1]), COL_X, 1)
            cv2.line(vis, (cd[0], cd[1] - arm), (cd[0], cd[1] + arm), COL_X, 1)

            # Line from target to cube
            cv2.line(vis, (tx, ty), (cd[0], cd[1]), (0, 180, 255), 1)

            h_err_frac = (cube_cx - w_img / 2.0) / (w_img / 2.0)
            v_err_frac = (cube_cy - h_img * 0.75) / (h_img / 2.0)
            h_err_str = f"{h_err_frac:+.2f}"
            v_err_str = f"{v_err_frac:+.2f}"
        else:
            cv2.putText(vis, "NO CUBE", (4, H // 2), font, fs, COL_MISS, th, cv2.LINE_AA)

        ee = est_dist - true_dist if (math.isfinite(est_dist) and true_dist > 0) else float("nan")
        status = "STOPPED" if self._stopped else "active"

        # Mid joint headroom for raising (distance from lower limit)
        mid_range = J_MID_HI - J_MID_LO
        mid_headroom_up = self._mid_pos - J_MID_LO
        mid_saturated = mid_headroom_up < MID_LIMIT_THRESH * mid_range
        mid_pct = mid_headroom_up / mid_range * 100.0
        mid_col = (50, 50, 240) if mid_saturated else (180, 255, 180)
        mid_tag = " [END FALLBACK]" if mid_saturated else ""

        # Recompute alignment factor for display
        if det is not None:
            _hf = (cube_cx - w_img / 2.0) / (w_img / 2.0)
            _vf = (cube_cy - h_img * 0.7) / (h_img / 2.0)
            _af = max(0.3, max(0.0, 1.0 - 2.0 * abs(_hf)) * max(0.0, 1.0 - 2.0 * abs(_vf)))
            align_str = f"{_af:.2f}"
        else:
            align_str = "---"

        rows = [
            (f"True: {true_dist:.4f}m", COL_TOF),
            (f"ToF:  {tof_dist:.4f}m" if math.isfinite(tof_dist) else "ToF: ---", (200, 200, 0)),
            (f"Est:  {est_dist:.4f}m" if math.isfinite(est_dist) else "Est: ---", COL_EST),
            (f"H_err: {h_err_str}  V_err: {v_err_str}  align: {align_str}", (255, 200, 0)),
            (f"Fill: {fill:.0%} [{status}]", (200, 255, 200)),
            (f"mid_pos: {math.degrees(self._mid_pos):.1f}deg  hdroom:{mid_pct:.0f}%{mid_tag}", mid_col),
        ]
        for i, (txt, col) in enumerate(rows):
            cv2.putText(vis, txt, (4, gap + i * gap), font, fs, col, th, cv2.LINE_AA)
        cv2.imshow("pov | detection", vis)

        cropped, _, _ = self._center_crop_square(raw_bgr)
        cv2.imshow("enhanced crop", self._enhance(cropped))

        scene_bgr = cv2.cvtColor(env._render_fixedcam(env._mjv_cam_scene), cv2.COLOR_RGB2BGR)
        env._draw_hud(scene_bgr)
        cv2.imshow("scene", cv2.resize(scene_bgr,
                   (scene_bgr.shape[1] // scene_div, scene_bgr.shape[0] // scene_div),
                   interpolation=cv2.INTER_AREA))