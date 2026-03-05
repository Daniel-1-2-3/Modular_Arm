import os, math, random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import mujoco
from gymnasium.spaces import Dict, Box
from dm_env import StepType

from Arm_Env.randomize_helpers import RandomizeHelpers
import Arm_Env.reward_utils as reward_utils


class ArmEnv:
    def __init__(
        self,
        xml_path: str,
        scene_cam: str = "scene_cam",
        pov_cam: str = "pov_cam",
        width: int = 1000,
        height: int = 1000,
        brace_other_joints: bool = True,
        display: bool = True,
        discount: float = 0.99,
        episode_horizon: int = 300,
    ):
        self.xml_path = xml_path
        self.scene_cam, self.pov_cam = scene_cam, pov_cam
        self.req_width, self.req_height = int(width), int(height)
        self.brace_other_joints = brace_other_joints
        self.display = display

        try:
            self.model = mujoco.MjModel.from_xml_path(self.xml_path)
            self.data = mujoco.MjData(self.model)
            mujoco.mj_forward(self.model, self.data)
            print("Loaded model")
        except ValueError as e:
            print(f"Failed to load XML: {e}")
            raise SystemExit(1)

        self._init_actuators()
        drive_dims = 1 if getattr(self, "drive_joint", None) is not None else 0
        self.action_dim = len(self.controlled_joints) + drive_dims

        self.max_deg_per_step = 2.0 # max joint delta per step, in degs
        self.max_drive_m_per_step = 0.01 # max drive delta per step, in meters

        self._init_renderer()
        if display:
            self._init_display()

        self.last_tof_m = float("nan")
        self.last_align_h_deg = float("nan")
        self.last_align_v_deg = float("nan")
        self.last_opt_v_deg = float("nan")
        self.last_cam_cube_m = float("nan")
        self.last_cam_cube_z = float("nan")
        self.last_cube_in_front = True
        self.last_tof_hit = False
        self._init_cam_cube_m = float("nan")
        self._last_action = None

        self.tof_cam_offset_m = 0.07
        self.cam_pitch_deg = 8.0

        self.discount = float(discount)
        self.episode_horizon = int(episode_horizon)
        self.curr_path_length: int = 0
        self.step_type: StepType = StepType.FIRST
        self.hand_init_pos: np.ndarray | None = None
        self._target_pos: np.ndarray | None = None

        # Pre-compute the ToF site's parent body for ray exclusion and
        # initialise the direction sign as uncalibrated.
        self._tof_site_body_id = self._get_tof_site_body_id()
        self._tof_dir_sign: int | None = None
        self._tof_dir_calibration_count: int = 0

        self.reset()

    def _get_tof_site_body_id(self) -> int:
        """Return the body id that owns tof_site, or -1 if not found."""
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tof_site")
        if int(site_id) < 0:
            return -1
        return int(self.model.site_bodyid[site_id])

    def get_action_space(self) -> Box:
        return Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)

    def get_observation_space(self) -> Dict:
        return Dict({
            "pov": Box(low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8),
            "tof": Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        })

    def evaluate_state(self) -> tuple[float, dict[str, Any]]:
        reward, cam_cube_m, approach, align, hit_bonus = self._compute_reward()
        success = float(bool(self.last_tof_hit) and np.isfinite(self.last_tof_m) and self.last_tof_m <= 0.05)
        info = {
            "success": success,
            "tof_hit": float(self.last_tof_hit),
            "tof_m": float(self.last_tof_m),
            "cam_cube_m": float(cam_cube_m),
            "approach_reward": float(approach),
            "align_reward": float(align),
            "hit_bonus": float(hit_bonus),
            "unscaled_reward": float(reward),
        }
        return float(reward), info

    def reset(self) -> dict[str, Any]:
        self._randomize()
        self.curr_path_length = 0
        self.step_type = StepType.FIRST
        self._last_action = None

        # Reset ToF direction calibration each episode so a bad lock-in
        # from a previous episode doesn't persist.
        self._tof_dir_sign = None
        self._tof_dir_calibration_count = 0

        self.targets_rad = {"j_a_base": 0.0, "j_b_mid": 0.0, "j_c_end": 105 * math.pi / 180}
        if self.drive_joint is not None:
            self.targets_rad[self.drive_joint] = 0.0

        for jn, q in self.targets_rad.items():
            if jn not in self.joint_to_ctrl:
                continue
            self.data.ctrl[self.joint_to_ctrl[jn]] = float(q)

            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            qadr = self.model.jnt_qposadr[jid]
            self.data.qpos[qadr] = float(q)
            self.data.qvel[self.model.jnt_dofadr[jid]] = 0.0

        mujoco.mj_forward(self.model, self.data)

        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.pov_cam)
        self.hand_init_pos = self.data.cam_xpos[cam_id].copy()
        cube_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "red_cube")
        self._target_pos = self.data.xpos[cube_id].copy()

        self._update_sensors()
        # Per-episode normalization for distance shaping
        self._init_cam_cube_m = float(self.last_cam_cube_m) if np.isfinite(self.last_cam_cube_m) else 1.0

        obs = self.get_obs()

        if self.display:
            self._display_obs(obs)

        return {
            "pov_obs": obs["pov"],
            "tof_obs": obs["tof"],
            "step_type": self.step_type,
            "action": None,
            "reward": None,
            "discount": self.discount,
        }

    def get_obs(self) -> dict[str, Any]:
        self._gl_ctx.make_current()

        scene_u8 = self._render_fixedcam(self._mjv_cam_scene)  # (H,W,3) uint8 RGB
        pov_u8 = self._render_fixedcam(self._mjv_cam_pov)      # (H,W,3) uint8 RGB

        tof_norm = 1.0 if (not np.isfinite(self.last_tof_m) or self.last_tof_m <= 0) else float(
            np.clip(self.last_tof_m / 2.0, 0.0, 0.99)
        )

        return {"scene": scene_u8, "pov": pov_u8, "tof": tof_norm}

    def _display_obs(self, obs: dict[str, Any]) -> None:
        scene_bgr = cv2.cvtColor(obs["scene"], cv2.COLOR_RGB2BGR)
        pov_bgr = cv2.cvtColor(obs["pov"], cv2.COLOR_RGB2BGR)
        self._draw_hud(scene_bgr)
        cv2.imshow(self.scene_window_name, scene_bgr)
        cv2.imshow(self.pov_window_name, pov_bgr)

    def step(self, action: npt.NDArray[np.floating]) -> dict[str, Any]:
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.size != self.action_dim:
            raise RuntimeError(f"Action len {a.size} does not match expected {self.action_dim}")

        # Deadzone to reduce jitter
        dead = 0.02
        a[np.abs(a) < dead] = 0.0
        self._last_action = a.copy()

        drive_dims = 1 if self.drive_joint is not None else 0

        arm_a = a[: len(self.controlled_joints)]
        delta_rad = np.deg2rad(self.max_deg_per_step) * arm_a
        for i, jn in enumerate(self.controlled_joints):
            self.targets_rad[jn] += float(delta_rad[i])
            self.targets_rad[jn] = float(np.clip(self.targets_rad[jn], self.lows[i], self.highs[i]))

        if drive_dims == 1:
            drive_a = float(a[-1])
            self.targets_rad[self.drive_joint] += drive_a * float(self.max_drive_m_per_step)
            self.targets_rad[self.drive_joint] = float(
                np.clip(self.targets_rad[self.drive_joint], self.drive_low, self.drive_high)
            )

        for jn in self.controlled_joints:
            self.data.ctrl[self.joint_to_ctrl[jn]] = self.targets_rad[jn]
        if drive_dims == 1:
            self.data.ctrl[self.joint_to_ctrl[self.drive_joint]] = self.targets_rad[self.drive_joint]

        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        self._update_sensors()
        self.curr_path_length += 1
        obs = self.get_obs()

        if self.display:
            self._display_obs(obs)

        reward, info = self.evaluate_state()

        done = self.curr_path_length >= self.episode_horizon
        self.step_type = StepType.LAST if done else StepType.MID

        return {
            "pov_obs": obs["pov"],
            "tof_obs": obs["tof"],
            "step_type": self.step_type,
            "reward": reward,
            "discount": 0.0 if done else self.discount,
        }

    def _update_sensors(self, body_name: str = "red_cube") -> None:
        # Raw ToF sensor reading (rangefinder over tof_site)
        sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "tof_range")
        adr = self.model.sensor_adr[sensor_id]
        dim = self.model.sensor_dim[sensor_id]
        self.last_tof_m = float(self.data.sensordata[adr:adr + dim][0])

        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.pov_cam)
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)

        mujoco.mj_forward(self.model, self.data)

        cam_pos = self.data.cam_xpos[cam_id]
        R = self.data.cam_xmat[cam_id].reshape(3, 3)
        tgt_pos = self.data.xpos[body_id]
        v = tgt_pos - cam_pos

        self.last_cam_cube_m = float(np.linalg.norm(v))

        # MuJoCo camera convention: the camera looks along its local -Z axis (OpenGL-style).
        # cam_xmat columns are the camera local axes expressed in world coordinates.
        # So:
        #   right  = +X
        #   up     = +Y
        #   forward= -Z  (what the camera looks toward)
        right = R[:, 0]
        up = R[:, 1]
        forward = -R[:, 2]
        down = -up
        x = float(np.dot(v, right))
        y = float(np.dot(v, down))
        z = float(np.dot(v, forward))  # SIGNED forward depth

        self.last_cam_cube_z = z
        self.last_cube_in_front = bool(z > 1e-6)

        if self.last_cube_in_front:
            # Angles in degrees, relative to camera frame.
            self.last_align_h_deg = float(math.degrees(math.atan2(x, abs(z))))
            self.last_align_v_deg = float(math.degrees(math.atan2(-y, abs(z))))
            self.last_opt_v_deg = float(self._get_optimal_v_angle_from_z(abs(z), h_m=self.tof_cam_offset_m, pitch_deg=self.cam_pitch_deg))
        else:
            # Cube behind camera: treat as "not visible" for alignment shaping.
            self.last_align_h_deg = float("nan")
            self.last_align_v_deg = float("nan")
            self.last_opt_v_deg = float("nan")

        # Cache ToF hit test (raycast from tof_site in the calibrated sensor direction)
        self.last_tof_hit = bool(self._tof_hitting_target(body_name))

    def _get_optimal_v_angle_from_z(self, z: float, h_m: float, pitch_deg: float) -> float:
        if not np.isfinite(z) or z <= 1e-6:
            return float("nan")
        th = math.radians(pitch_deg)
        num = h_m * math.cos(th) - z * math.sin(th)
        den = h_m * math.sin(th) + z * math.cos(th)
        return float(math.degrees(math.atan2(num, den)))

    def _tof_hitting_target(self, body_name: str = "red_cube") -> bool:
        """Return True iff the ToF *sensor ray* is hitting the target body.

        Key fixes vs. the original implementation:
        1.  **Exclude the sensor's parent body** from the raycast so that the
            arm's own geometry doesn't block the ray.  The built-in MuJoCo
            rangefinder already does this internally; our manual mj_ray call
            must replicate the same exclusion.
        2.  **Reset direction calibration every episode** (done in reset())
            so a bad early lock-in doesn't persist forever.
        3.  **Allow periodic recalibration** — every 50 steps the calibration
            is temporarily opened up to both directions again, guarding
            against a single fluke locking in the wrong sign.
        """
        # If the sensor is reading invalid, treat as no-hit.
        d_sense = float(getattr(self, "last_tof_m", float("nan")))
        if (not np.isfinite(d_sense)) or (d_sense <= 0.0):
            return False

        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tof_site")
        if int(site_id) < 0:
            return False

        cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if int(cube_body_id) < 0:
            return False

        # --- FIX 1: exclude the body that owns tof_site so the arm's own
        # geometry doesn't intercept the ray.  This matches the behaviour of
        # MuJoCo's built-in rangefinder sensor.
        bodyexclude = self._tof_site_body_id

        ray_origin = self.data.site_xpos[site_id].copy()
        R = self.data.site_xmat[site_id].reshape(3, 3)

        geomgroup = np.ones(6, dtype=np.uint8)

        # --- FIX 2 & 3: direction calibration with periodic recalibration.
        # Determine which candidate directions to test this step.
        sign = self._tof_dir_sign
        recalib_interval = 50
        needs_recalib = (
            sign is None
            or (self._tof_dir_calibration_count % recalib_interval == 0)
        )

        if needs_recalib:
            # Test both directions
            cand_dirs = [(+1, R[:, 2]), (-1, -R[:, 2])]
        else:
            # Use the calibrated direction only
            cand_dirs = [(sign, R[:, 2] if sign == +1 else -R[:, 2])]

        self._tof_dir_calibration_count += 1

        best_err = float("inf")
        best_hit_cube = False
        best_sign = sign

        for s, dvec in cand_dirs:
            ray_dir = np.asarray(dvec, dtype=np.float64)
            n = float(np.linalg.norm(ray_dir))
            if n <= 1e-12:
                continue
            ray_dir = (ray_dir / n).astype(np.float64)

            geomid = np.array([-1], dtype=np.int32)
            dist = mujoco.mj_ray(
                self.model, self.data,
                ray_origin, ray_dir,
                geomgroup, 1, bodyexclude,
                geomid,
            )

            if geomid[0] < 0 or (not np.isfinite(dist)) or dist <= 0.0:
                continue

            hit_body_id = int(self.model.geom_bodyid[int(geomid[0])])
            hit_cube = (hit_body_id == int(cube_body_id))
            abs_err = float(abs(float(dist) - d_sense))

            # Track the direction whose distance best matches the sensor.
            if abs_err < best_err:
                best_err = abs_err
                best_hit_cube = hit_cube
                best_sign = s

        # Update calibration if we got a confident match.
        if best_err < 0.03:
            self._tof_dir_sign = int(best_sign)

        return bool(best_hit_cube and best_err < 0.05)

    def _render_fixedcam(self, mjv_cam: mujoco.MjvCamera) -> npt.NDArray[np.uint8]:
            mujoco.mjv_updateScene(
                self.model,
                self.data,
                self._mjv_opt,
                None,
                mjv_cam,
                mujoco.mjtCatBit.mjCAT_ALL,
                self._mjv_scene,
            )
            mujoco.mjr_render(self._mjr_rect, self._mjv_scene, self._mjr_ctx)
            mujoco.mjr_readPixels(self._rgb_u8, None, self._mjr_rect, self._mjr_ctx)
            return np.flipud(self._rgb_u8).copy()

    def _compute_reward(self) -> tuple[float, float, float, float, float]:
        """
        Reward for single wrist camera + ToF:

        - approach: privileged sim-only camera-to-cube distance shaping
        - align: camera-frame angular alignment (only when cube is in front of camera)
        - hit_bonus: small bonus when ToF ray actually hits cube
        - action penalty: reduces wobble

        Returns: (reward, cam_cube_m, approach, align, hit_bonus)
        """
        # Approach shaping (privileged distance, but gated by visibility to avoid "approach from behind" shortcut)
        d = float(self.last_cam_cube_m)
        if not np.isfinite(d):
            d = 10.0

        d0 = float(getattr(self, "_init_cam_cube_m", d))
        if not np.isfinite(d0) or d0 <= 1e-6:
            d0 = max(d, 1.0)

        D_GOAL = 0.06
        approach = reward_utils.tolerance(
            d,
            bounds=(0.0, D_GOAL),
            margin=d0,
            sigmoid="long_tail",
        )

        # Visibility gate: if cube is behind the camera, don't reward "getting closer behind you"
        vis = 1.0 if bool(getattr(self, "last_cube_in_front", True)) else 0.0
        approach = float(approach) * float(vis)

        # Alignment shaping (only when visible)
        if vis > 0.0 and np.isfinite(self.last_align_h_deg) and np.isfinite(self.last_align_v_deg) and np.isfinite(self.last_opt_v_deg):
            v_err_deg = float(self.last_align_v_deg - self.last_opt_v_deg)
            align_err_deg = float(math.sqrt(self.last_align_h_deg**2 + v_err_deg**2))
            align = reward_utils.tolerance(
                align_err_deg,
                bounds=(0.0, 5.0),
                margin=45.0,
                sigmoid="long_tail",
            )
        else:
            align = 0.0

        # ToF confirmation bonus (only forward ray; cached in last_tof_hit)
        hit = 1.0 if bool(getattr(self, "last_tof_hit", False)) else 0.0
        HIT_BONUS = 0.25
        hit_bonus = HIT_BONUS * hit

        reward = 10.0 * (0.75 * approach + 0.25 * float(align)) + float(hit_bonus)
        return (float(reward), d, float(approach), float(align), float(hit_bonus))

    def _init_actuators(self) -> None:
        self.joint_to_ctrl: dict[str, int] = {}
        for act_id in range(self.model.nu):
            if int(self.model.actuator_trntype[act_id]) != int(mujoco.mjtTrn.mjTRN_JOINT):
                continue
            j_id = int(self.model.actuator_trnid[act_id][0])
            j_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j_id)
            if j_name:
                self.joint_to_ctrl[j_name] = act_id

        if not self.joint_to_ctrl:
            raise RuntimeError("No actuators found")

        self.all_actuated_joints = sorted(self.joint_to_ctrl.keys())

        self.drive_joint = "j_drive" if "j_drive" in self.joint_to_ctrl else None
        self.controlled_joints = [j for j in self.all_actuated_joints if j != self.drive_joint]

        self.targets_rad = {}
        for joint in self.all_actuated_joints:
            self.targets_rad[joint] = self._get_joint_qpos_rad(joint)
            self.data.ctrl[self.joint_to_ctrl[joint]] = self.targets_rad[joint]

        limits_by_name = {
            "j_drive": (-0.70, 0.0),
            "j_a_base": (-45 / 180 * np.pi, 45 / 180 * np.pi),
            "j_b_mid": (0, 25 / 180 * np.pi),
            "j_c_end": (-90 / 180 * np.pi, 110 / 180 * np.pi),
        }

        self.manual_limits = np.array([limits_by_name[j] for j in self.controlled_joints], dtype=np.float64)
        self.lows = self.manual_limits[:, 0]
        self.highs = self.manual_limits[:, 1]

        if self.drive_joint is not None:
            self.drive_low, self.drive_high = limits_by_name[self.drive_joint]

    def _init_renderer(self) -> None:
        self._mjv_cam_scene = mujoco.MjvCamera()
        self._mjv_cam_scene.type = mujoco.mjtCamera.mjCAMERA_FIXED

        self._mjv_cam_pov = mujoco.MjvCamera()
        self._mjv_cam_pov.type = mujoco.mjtCamera.mjCAMERA_FIXED

        scene_cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.scene_cam)
        pov_cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.pov_cam)
        if int(scene_cam_id) < 0:
            raise RuntimeError(f"Camera not found: {self.scene_cam}")
        if int(pov_cam_id) < 0:
            raise RuntimeError(f"Camera not found: {self.pov_cam}")

        self._mjv_cam_scene.fixedcamid = int(scene_cam_id)
        self._mjv_cam_pov.fixedcamid = int(pov_cam_id)

        self._mjv_opt = mujoco.MjvOption()
        self._mjv_scene = mujoco.MjvScene(self.model, maxgeom=2000)

        if not hasattr(mujoco, "GLContext") or mujoco.GLContext is None:
            raise RuntimeError("GLContext unavailable. Set MUJOCO_GL before importing mujoco")

        self._gl_ctx = mujoco.GLContext(self.req_width, self.req_height)
        self._gl_ctx.make_current()

        self._mjr_ctx = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_100)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self._mjr_ctx)

        off_w = int(getattr(self._mjr_ctx, "offWidth", 640))
        off_h = int(getattr(self._mjr_ctx, "offHeight", 480))
        self.width = min(self.req_width, off_w)
        self.height = min(self.req_height, off_h)
        self._mjr_rect = mujoco.MjrRect(0, 0, self.width, self.height)
        self._rgb_u8 = np.empty((self.height, self.width, 3), dtype=np.uint8)

    def _init_display(self) -> None:
        self.scene_window_name = f"RGB ({self.scene_cam}) {self.width}x{self.height}"
        self.pov_window_name = f"RGB ({self.pov_cam}) {self.width}x{self.height}"
        cv2.namedWindow(self.scene_window_name, cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(self.pov_window_name, cv2.WINDOW_AUTOSIZE)

    def _get_joint_qpos_rad(self, joint_name: str) -> float:
        jid = self.model.joint(joint_name).id
        qadr = int(self.model.jnt_qposadr[jid])
        return float(self.data.qpos[qadr])

    def _randomize(self, textures_dir: str = os.path.join("Simulation", "Textures")) -> tuple[str, str]:
        tex_dir = Path(textures_dir)
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        files = [p for p in tex_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
        floor_img_path, table_img_path = random.sample(files, 2)

        self._gl_ctx.make_current()
        RandomizeHelpers._randomize_bg_textures("T_floor", str(floor_img_path), self.model, self._mjr_ctx)
        RandomizeHelpers._randomize_bg_textures("T_table", str(table_img_path), self.model, self._mjr_ctx)
        RandomizeHelpers._randomize_skybox_gradient(self.model, self._mjr_ctx)
        RandomizeHelpers._randomize_target_start(self.model, self.data)

        return (str(floor_img_path), str(table_img_path))

    def _draw_hud(self, scene_bgr: npt.NDArray[np.uint8]) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thickness, x0, y0, line_h = 0.55, 1, 10, 22, 20
        color = (0, 0, 0)

        tof_str = f"ToF: {self.last_tof_m:.4f} m" if np.isfinite(self.last_tof_m) else "ToF: nan"
        hit_str = f"ToF_hit: {bool(getattr(self, 'last_tof_hit', False))}"
        d_str = f"d_cam: {self.last_cam_cube_m:.3f} m" if np.isfinite(self.last_cam_cube_m) else "d_cam: nan"

        if bool(getattr(self, "last_cube_in_front", True)) and np.isfinite(self.last_cam_cube_z):
            z_str = f"z_cam: {self.last_cam_cube_z:.3f} (in front)"
        elif np.isfinite(getattr(self, "last_cam_cube_z", float('nan'))):
            z_str = f"z_cam: {self.last_cam_cube_z:.3f} (behind)"
        else:
            z_str = "z_cam: nan"

        h_str = f"h: {self.last_align_h_deg:.2f} deg" if np.isfinite(self.last_align_h_deg) else "h: n/a"
        v_str = f"v: {self.last_align_v_deg:.2f} deg" if np.isfinite(self.last_align_v_deg) else "v: n/a"
        vo_str = f"v*: {self.last_opt_v_deg:.2f} deg" if np.isfinite(self.last_opt_v_deg) else "v*: n/a"

        cv2.putText(scene_bgr, tof_str, (x0, y0), font, scale, color, thickness, cv2.LINE_AA)
        cv2.putText(scene_bgr, hit_str, (x0, y0 + line_h), font, scale, color, thickness, cv2.LINE_AA)
        cv2.putText(scene_bgr, f"{d_str}   {z_str}", (x0, y0 + 2 * line_h), font, scale, color, thickness, cv2.LINE_AA)
        cv2.putText(scene_bgr, f"{h_str}   {v_str}   {vo_str}", (x0, y0 + 3 * line_h), font, scale, color, thickness, cv2.LINE_AA)

        reward, info = self.evaluate_state()
        rew_str = f"R: {reward:.3f}  (app={info['approach_reward']:.2f}  align={info['align_reward']:.2f}  hit={info['hit_bonus']:.2f})"
        cv2.putText(scene_bgr, rew_str, (x0, y0 + 4 * line_h), font, scale, color, thickness, cv2.LINE_AA)