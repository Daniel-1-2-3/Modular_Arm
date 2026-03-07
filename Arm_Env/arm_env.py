import os, math, random
from collections import deque
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
        enable_domain_randomization: bool = True,
        action_delay_steps: int = 2,
        obs_delay_steps: int = 2,
    ):
        self.xml_path = xml_path
        self.scene_cam, self.pov_cam = scene_cam, pov_cam
        self.req_width, self.req_height = int(width), int(height)
        self.brace_other_joints = brace_other_joints
        self.display = display
        self.enable_domain_randomization = bool(enable_domain_randomization)
        self.max_action_delay_steps = int(action_delay_steps)
        self.max_obs_delay_steps = int(obs_delay_steps)

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

        self.max_deg_per_step = 2.0
        self.max_drive_m_per_step = 0.01

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

        self._tof_offset_bias_m = 0.0
        self._tof_ema = float("nan")
        self._tof_ema_alpha = 0.65

        self._camera_profile: dict[str, Any] = {}
        self._frame_profile: dict[str, Any] = {}
        self._obs_delay_steps = 0
        self._obs_fifo: deque[dict[str, Any]] = deque()
        self._last_delivered_obs: dict[str, Any] | None = None
        self._action_delay_steps = 0
        self._action_fifo: deque[np.ndarray] = deque()
        self._pending_action = None
        self._arm_action_scale = None
        self._drive_action_scale = 1.0
        self._joint_response_alpha = 1.0
        self._drive_response_alpha = 1.0
        self._joint_backlash_rad = None
        self._drive_backlash_m = 0.0
        self._joint_cmd_state = {}
        self._joint_realized_targets = {}
        self._joint_zero_offset_rad = {}
        self._motor_noise_std_rad = 0.0
        self._drive_noise_std_m = 0.0

        self.discount = float(discount)
        self.episode_horizon = int(episode_horizon)
        self.curr_path_length: int = 0
        self.step_type: StepType = StepType.FIRST
        self.hand_init_pos: np.ndarray | None = None
        self._target_pos: np.ndarray | None = None

        self._tof_site_body_id = self._get_tof_site_body_id()
        self._tof_dir_sign: int | None = None
        self._tof_dir_calibration_count: int = 0

        # Cache freejoint id for the cube for velocity zeroing.
        self._cube_freejoint_id: int = self._find_body_freejoint("red_cube")

        self._cached_reward: float | None = None
        self._cached_info: dict[str, Any] | None = None

        self.reset()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_tof_site_body_id(self) -> int:
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tof_site")
        if int(site_id) < 0:
            return -1
        return int(self.model.site_bodyid[site_id])

    def _find_body_freejoint(self, body_name: str) -> int:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if int(body_id) < 0:
            return -1
        for j in range(self.model.njnt):
            if int(self.model.jnt_bodyid[j]) == int(body_id) and int(self.model.jnt_type[j]) == 0:
                return j
        return -1

    def _zero_freejoint_vel(self, jnt_id: int) -> None:
        if jnt_id < 0:
            return
        dof_adr = int(self.model.jnt_dofadr[jnt_id])
        self.data.qvel[dof_adr:dof_adr + 6] = 0.0

    def get_action_space(self) -> Box:
        return Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)

    def get_observation_space(self) -> Dict:
        return Dict({
            "pov": Box(low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8),
            "tof": Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        })

    def evaluate_state(self) -> tuple[float, dict[str, Any]]:
        if self._cached_reward is not None and self._cached_info is not None:
            return self._cached_reward, self._cached_info
        reward, cam_cube_m, approach, align, hit_bonus = self._compute_reward()
        success = float(bool(self.last_tof_hit) and np.isfinite(self.last_tof_m) and self.last_tof_m <= 0.05)
        info = {
            "success": success, "tof_hit": float(self.last_tof_hit),
            "tof_m": float(self.last_tof_m), "cam_cube_m": float(cam_cube_m),
            "approach_reward": float(approach), "align_reward": float(align),
            "hit_bonus": float(hit_bonus), "unscaled_reward": float(reward),
        }
        self._cached_reward = float(reward)
        self._cached_info = info
        return float(reward), info

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> dict[str, Any]:
        self._randomize()
        self.curr_path_length = 0
        self.step_type = StepType.FIRST
        self._last_action = None
        self._tof_dir_sign = None
        self._tof_dir_calibration_count = 0

        self._reset_tof_augmentation_state()
        self._reset_camera_augmentation_state()
        self._reset_dynamics_randomization_state()

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

        # FIX: zero the cube freejoint velocities so it doesn't carry
        # momentum from the previous episode.
        self._zero_freejoint_vel(self._cube_freejoint_id)

        mujoco.mj_forward(self.model, self.data)

        self._joint_cmd_state = {jn: float(self.targets_rad[jn]) for jn in self.all_actuated_joints}
        self._joint_realized_targets = {jn: float(self.targets_rad[jn]) for jn in self.all_actuated_joints}

        self._action_fifo = deque()
        for _ in range(self._action_delay_steps):
            self._action_fifo.append(np.zeros(self.action_dim, dtype=np.float32))
        self._pending_action = np.zeros(self.action_dim, dtype=np.float32)

        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.pov_cam)
        self.hand_init_pos = self.data.cam_xpos[cam_id].copy()
        cube_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "red_cube")
        self._target_pos = self.data.xpos[cube_id].copy()

        self._cached_reward = None
        self._cached_info = None
        self._update_sensors()
        self._init_cam_cube_m = float(self.last_cam_cube_m) if np.isfinite(self.last_cam_cube_m) else 1.0

        initial_obs = self._build_obs_now()
        self._obs_fifo = deque()
        self._last_delivered_obs = None
        for _ in range(self._obs_delay_steps):
            self._obs_fifo.append({
                "scene": initial_obs["scene"].copy(),
                "pov": initial_obs["pov"].copy(),
                "tof": initial_obs["tof"].copy(),
            })

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

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _build_obs_now(self) -> dict[str, Any]:
        self._gl_ctx.make_current()
        scene_u8 = self._render_fixedcam(self._mjv_cam_scene)
        pov_u8 = self._render_fixedcam(self._mjv_cam_pov)
        pov_u8 = self._augment_policy_image(pov_u8)
        tof_norm = 1.0 if (not np.isfinite(self.last_tof_m) or self.last_tof_m <= 0) else float(
            np.clip(self.last_tof_m / 2.0, 0.0, 0.99)
        )
        return {"scene": scene_u8, "pov": pov_u8, "tof": np.array([tof_norm], dtype=np.float32)}

    def get_obs(self) -> dict[str, Any]:
        obs_now = self._build_obs_now()
        if self._obs_delay_steps <= 0:
            self._last_delivered_obs = {
                "scene": obs_now["scene"].copy(), "pov": obs_now["pov"].copy(), "tof": obs_now["tof"].copy(),
            }
            return obs_now

        self._obs_fifo.append({
            "scene": obs_now["scene"].copy(), "pov": obs_now["pov"].copy(), "tof": obs_now["tof"].copy(),
        })
        delayed = self._obs_fifo.popleft()

        stale_prob = float(self._frame_profile.get("stale_frame_prob", 0.0))
        if self._last_delivered_obs is not None and np.random.rand() < stale_prob:
            delayed = {
                "scene": self._last_delivered_obs["scene"].copy(),
                "pov": self._last_delivered_obs["pov"].copy(),
                "tof": self._last_delivered_obs["tof"].copy(),
            }

        self._last_delivered_obs = {
            "scene": delayed["scene"].copy(), "pov": delayed["pov"].copy(), "tof": delayed["tof"].copy(),
        }
        return delayed

    def _display_obs(self, obs: dict[str, Any]) -> None:
        scene_bgr = cv2.cvtColor(obs["scene"], cv2.COLOR_RGB2BGR)
        pov_bgr = cv2.cvtColor(obs["pov"], cv2.COLOR_RGB2BGR)
        self._draw_hud(scene_bgr)
        cv2.imshow(self.scene_window_name, scene_bgr)
        cv2.imshow(self.pov_window_name, pov_bgr)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action: npt.NDArray[np.floating]) -> dict[str, Any]:
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.size != self.action_dim:
            raise RuntimeError(f"Action len {a.size} does not match expected {self.action_dim}")
        dead = 0.02
        a[np.abs(a) < dead] = 0.0
        self._last_action = a.copy()

        self._action_fifo.append(a.copy())
        delayed_a = self._action_fifo.popleft().copy()
        self._pending_action = delayed_a.copy()

        drive_dims = 1 if self.drive_joint is not None else 0
        arm_a = delayed_a[: len(self.controlled_joints)] * self._arm_action_scale
        delta_rad = np.deg2rad(self.max_deg_per_step) * arm_a
        for i, jn in enumerate(self.controlled_joints):
            self.targets_rad[jn] += float(delta_rad[i])
            self.targets_rad[jn] = float(np.clip(self.targets_rad[jn], self.lows[i], self.highs[i]))
            target = float(self.targets_rad[jn] + self._joint_zero_offset_rad[jn])
            alpha = float(self._joint_response_alpha)
            prev_cmd = float(self._joint_cmd_state[jn])
            cmd = prev_cmd + alpha * (target - prev_cmd)
            if abs(target - prev_cmd) < float(self._joint_backlash_rad[i]):
                cmd = prev_cmd
            if self._motor_noise_std_rad > 0.0:
                cmd += float(np.random.normal(0.0, self._motor_noise_std_rad))
            cmd = float(np.clip(cmd, self.lows[i], self.highs[i]))
            self._joint_cmd_state[jn] = cmd
            self._joint_realized_targets[jn] = cmd
            self.data.ctrl[self.joint_to_ctrl[jn]] = cmd

        if drive_dims == 1:
            drive_a = float(delayed_a[-1]) * float(self._drive_action_scale)
            self.targets_rad[self.drive_joint] += drive_a * float(self.max_drive_m_per_step)
            self.targets_rad[self.drive_joint] = float(
                np.clip(self.targets_rad[self.drive_joint], self.drive_low, self.drive_high))
            drive_target = float(self.targets_rad[self.drive_joint] + self._joint_zero_offset_rad[self.drive_joint])
            drive_prev = float(self._joint_cmd_state[self.drive_joint])
            drive_cmd = drive_prev + float(self._drive_response_alpha) * (drive_target - drive_prev)
            if abs(drive_target - drive_prev) < float(self._drive_backlash_m):
                drive_cmd = drive_prev
            if self._drive_noise_std_m > 0.0:
                drive_cmd += float(np.random.normal(0.0, self._drive_noise_std_m))
            drive_cmd = float(np.clip(drive_cmd, self.drive_low, self.drive_high))
            self._joint_cmd_state[self.drive_joint] = drive_cmd
            self._joint_realized_targets[self.drive_joint] = drive_cmd
            self.data.ctrl[self.joint_to_ctrl[self.drive_joint]] = drive_cmd

        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        self._cached_reward = None
        self._cached_info = None
        self._update_sensors()
        self.curr_path_length += 1
        obs = self.get_obs()
        if self.display:
            self._display_obs(obs)

        reward, info = self.evaluate_state()
        done = self.curr_path_length >= self.episode_horizon
        self.step_type = StepType.LAST if done else StepType.MID

        return {
            "pov_obs": obs["pov"], "tof_obs": obs["tof"],
            "step_type": self.step_type, "reward": reward,
            "discount": 0.0 if done else self.discount,
        }

    # ------------------------------------------------------------------
    # Sensors
    # ------------------------------------------------------------------

    def _update_sensors(self, body_name: str = "red_cube") -> None:
        self._cached_reward = None
        self._cached_info = None
        sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "tof_range")
        adr = self.model.sensor_adr[sensor_id]
        dim = self.model.sensor_dim[sensor_id]
        raw_tof = float(self.data.sensordata[adr:adr + dim][0])
        self.last_tof_m = self._augment_tof(raw_tof)

        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.pov_cam)
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        mujoco.mj_forward(self.model, self.data)

        cam_pos = self.data.cam_xpos[cam_id]
        R = self.data.cam_xmat[cam_id].reshape(3, 3)
        tgt_pos = self.data.xpos[body_id]
        v = tgt_pos - cam_pos
        self.last_cam_cube_m = float(np.linalg.norm(v))

        right = R[:, 0]; up = R[:, 1]; forward = -R[:, 2]; down = -up
        x = float(np.dot(v, right)); y = float(np.dot(v, down)); z = float(np.dot(v, forward))
        self.last_cam_cube_z = z
        self.last_cube_in_front = bool(z > 1e-6)

        if self.last_cube_in_front:
            self.last_align_h_deg = float(math.degrees(math.atan2(x, abs(z))))
            self.last_align_v_deg = float(math.degrees(math.atan2(-y, abs(z))))
            self.last_opt_v_deg = float(self._get_optimal_v_angle_from_z(abs(z), h_m=self.tof_cam_offset_m, pitch_deg=self.cam_pitch_deg))
        else:
            self.last_align_h_deg = self.last_align_v_deg = self.last_opt_v_deg = float("nan")

        self.last_tof_hit = bool(self._tof_hitting_target(body_name))

    def _get_optimal_v_angle_from_z(self, z, h_m, pitch_deg):
        if not np.isfinite(z) or z <= 1e-6: return float("nan")
        th = math.radians(pitch_deg)
        num = h_m * math.cos(th) - z * math.sin(th)
        den = h_m * math.sin(th) + z * math.cos(th)
        return float(-math.degrees(math.atan2(num, den)))

    def _tof_hitting_target(self, body_name="red_cube"):
        d_sense = float(getattr(self, "last_tof_m", float("nan")))
        if (not np.isfinite(d_sense)) or (d_sense <= 0.0): return False
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tof_site")
        if int(site_id) < 0: return False
        cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if int(cube_body_id) < 0: return False
        bodyexclude = self._tof_site_body_id
        ray_origin = self.data.site_xpos[site_id].copy()
        R = self.data.site_xmat[site_id].reshape(3, 3)
        geomgroup = np.ones(6, dtype=np.uint8)
        sign = self._tof_dir_sign
        needs_recalib = (sign is None or (self._tof_dir_calibration_count % 50 == 0))
        self._tof_dir_calibration_count += 1
        if needs_recalib:
            cand_dirs = [(+1, R[:, 2]), (-1, -R[:, 2])]
        else:
            cand_dirs = [(sign, R[:, 2] if sign == +1 else -R[:, 2])]
        best_err, best_hit_cube, best_sign = float("inf"), False, sign
        for s, dvec in cand_dirs:
            ray_dir = np.asarray(dvec, dtype=np.float64)
            n = float(np.linalg.norm(ray_dir))
            if n <= 1e-12: continue
            ray_dir = (ray_dir / n).astype(np.float64)
            geomid = np.array([-1], dtype=np.int32)
            dist = mujoco.mj_ray(self.model, self.data, ray_origin, ray_dir, geomgroup, 1, bodyexclude, geomid)
            if geomid[0] < 0 or (not np.isfinite(dist)) or dist <= 0.0: continue
            hit_body_id = int(self.model.geom_bodyid[int(geomid[0])])
            hit_cube = (hit_body_id == int(cube_body_id))
            abs_err = float(abs(float(dist) - d_sense))
            if abs_err < best_err:
                best_err, best_hit_cube, best_sign = abs_err, hit_cube, s
        if best_err < 0.03: self._tof_dir_sign = int(best_sign)
        return bool(best_hit_cube and best_err < 0.05)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_fixedcam(self, mjv_cam):
        mujoco.mjv_updateScene(self.model, self.data, self._mjv_opt, None, mjv_cam, mujoco.mjtCatBit.mjCAT_ALL, self._mjv_scene)
        mujoco.mjr_render(self._mjr_rect, self._mjv_scene, self._mjr_ctx)
        mujoco.mjr_readPixels(self._rgb_u8, None, self._mjr_rect, self._mjr_ctx)
        return np.flipud(self._rgb_u8).copy()

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self):
        d = float(self.last_cam_cube_m)
        if not np.isfinite(d): d = 10.0
        d0 = float(getattr(self, "_init_cam_cube_m", d))
        if not np.isfinite(d0) or d0 <= 1e-6: d0 = max(d, 1.0)
        approach = reward_utils.tolerance(d, bounds=(0.0, 0.06), margin=d0, sigmoid="long_tail")
        vis = 1.0 if bool(getattr(self, "last_cube_in_front", True)) else 0.0
        approach = float(approach) * float(vis)
        if vis > 0.0 and np.isfinite(self.last_align_h_deg) and np.isfinite(self.last_align_v_deg) and np.isfinite(self.last_opt_v_deg):
            v_err_deg = float(self.last_align_v_deg - self.last_opt_v_deg)
            align_err_deg = float(math.sqrt(self.last_align_h_deg**2 + v_err_deg**2))
            align = reward_utils.tolerance(align_err_deg, bounds=(0.0, 5.0), margin=45.0, sigmoid="long_tail")
        else:
            align = 0.0
        hit = 1.0 if bool(getattr(self, "last_tof_hit", False)) else 0.0
        tof_close = 0.0
        if hit > 0.0 and np.isfinite(self.last_tof_m) and self.last_tof_m > 0.0:
            tof_close = reward_utils.tolerance(float(self.last_tof_m), bounds=(0.0, 0.03), margin=0.15, sigmoid="long_tail")
        reward = 10.0 * (0.75 * approach + 0.25 * float(align)) + 0.25 * hit + 4.0 * float(tof_close)
        return (float(reward), d, float(approach), float(align), float(hit))

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------

    def _init_actuators(self):
        self.joint_to_ctrl = {}
        for act_id in range(self.model.nu):
            if int(self.model.actuator_trntype[act_id]) != int(mujoco.mjtTrn.mjTRN_JOINT): continue
            j_id = int(self.model.actuator_trnid[act_id][0])
            j_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j_id)
            if j_name: self.joint_to_ctrl[j_name] = act_id
        if not self.joint_to_ctrl: raise RuntimeError("No actuators found")
        self.all_actuated_joints = sorted(self.joint_to_ctrl.keys())
        self.drive_joint = "j_drive" if "j_drive" in self.joint_to_ctrl else None
        self.controlled_joints = [j for j in self.all_actuated_joints if j != self.drive_joint]
        self.targets_rad = {}
        for joint in self.all_actuated_joints:
            self.targets_rad[joint] = self._get_joint_qpos_rad(joint)
            self.data.ctrl[self.joint_to_ctrl[joint]] = self.targets_rad[joint]
        limits_by_name = {
            "j_drive": (-0.70, 0.0), "j_a_base": (-45/180*np.pi, 45/180*np.pi),
            "j_b_mid": (0, 25/180*np.pi), "j_c_end": (-90/180*np.pi, 110/180*np.pi),
        }
        self.manual_limits = np.array([limits_by_name[j] for j in self.controlled_joints], dtype=np.float64)
        self.lows = self.manual_limits[:, 0]; self.highs = self.manual_limits[:, 1]
        if self.drive_joint is not None:
            self.drive_low, self.drive_high = limits_by_name[self.drive_joint]

    def _init_renderer(self):
        self._mjv_cam_scene = mujoco.MjvCamera(); self._mjv_cam_scene.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self._mjv_cam_pov = mujoco.MjvCamera(); self._mjv_cam_pov.type = mujoco.mjtCamera.mjCAMERA_FIXED
        scene_cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.scene_cam)
        pov_cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.pov_cam)
        if int(scene_cam_id) < 0: raise RuntimeError(f"Camera not found: {self.scene_cam}")
        if int(pov_cam_id) < 0: raise RuntimeError(f"Camera not found: {self.pov_cam}")
        self._mjv_cam_scene.fixedcamid = int(scene_cam_id); self._mjv_cam_pov.fixedcamid = int(pov_cam_id)
        self._mjv_opt = mujoco.MjvOption(); self._mjv_scene = mujoco.MjvScene(self.model, maxgeom=2000)
        if not hasattr(mujoco, "GLContext") or mujoco.GLContext is None:
            raise RuntimeError("GLContext unavailable. Set MUJOCO_GL before importing mujoco")
        self._gl_ctx = mujoco.GLContext(self.req_width, self.req_height); self._gl_ctx.make_current()
        self._mjr_ctx = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_100)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self._mjr_ctx)
        off_w = int(getattr(self._mjr_ctx, "offWidth", 640)); off_h = int(getattr(self._mjr_ctx, "offHeight", 480))
        self.width = min(self.req_width, off_w); self.height = min(self.req_height, off_h)
        self._mjr_rect = mujoco.MjrRect(0, 0, self.width, self.height)
        self._rgb_u8 = np.empty((self.height, self.width, 3), dtype=np.uint8)

    def _init_display(self):
        self.scene_window_name = f"RGB ({self.scene_cam}) {self.width}x{self.height}"
        self.pov_window_name = f"RGB ({self.pov_cam}) {self.width}x{self.height}"
        cv2.namedWindow(self.scene_window_name, cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(self.pov_window_name, cv2.WINDOW_AUTOSIZE)

    def _get_joint_qpos_rad(self, joint_name):
        jid = self.model.joint(joint_name).id
        return float(self.data.qpos[int(self.model.jnt_qposadr[jid])])

    # ------------------------------------------------------------------
    # Domain randomization
    # ------------------------------------------------------------------

    def _randomize_cube_rotation(self, body_name="red_cube"):
        jnt_id = self._cube_freejoint_id
        if jnt_id < 0: return
        qadr = int(self.model.jnt_qposadr[jnt_id])
        u1, u2, u3 = np.random.uniform(0,1), np.random.uniform(0,1), np.random.uniform(0,1)
        q = np.array([
            math.sqrt(1-u1)*math.sin(2*math.pi*u2), math.sqrt(1-u1)*math.cos(2*math.pi*u2),
            math.sqrt(u1)*math.sin(2*math.pi*u3), math.sqrt(u1)*math.cos(2*math.pi*u3),
        ], dtype=np.float64)
        self.data.qpos[qadr+3]=q[3]; self.data.qpos[qadr+4]=q[0]; self.data.qpos[qadr+5]=q[1]; self.data.qpos[qadr+6]=q[2]
        self._zero_freejoint_vel(jnt_id)

    def _randomize(self, textures_dir=os.path.join("Simulation","Textures")):
        tex_dir = Path(textures_dir)
        files = [p for p in tex_dir.iterdir() if p.is_file()]
        floor_img_path, table_img_path = random.sample(files, 2)
        self._gl_ctx.make_current()
        RandomizeHelpers._randomize_bg_textures("T_floor", str(floor_img_path), self.model, self._mjr_ctx)
        RandomizeHelpers._randomize_bg_textures("T_table", str(table_img_path), self.model, self._mjr_ctx)
        RandomizeHelpers._randomize_skybox_gradient(self.model, self._mjr_ctx)
        RandomizeHelpers._randomize_target_start(self.model, self.data)
        self._randomize_cube_rotation("red_cube")

        # Scene lighting — moderate diffuse + low ambient so the cube keeps
        # a clear shadow side vs lit side.  High diffuse or ambient washes
        # out all shading on a white object.
        for i in range(self.model.nlight):
            az = np.random.uniform(0.0, 2 * math.pi)
            el = np.random.uniform(math.radians(25), math.radians(70))
            dx = math.cos(el) * math.cos(az)
            dy = math.cos(el) * math.sin(az)
            dz = -math.sin(el)
            self.model.light_dir[i] = [dx, dy, dz]

            self.model.light_pos[i, 0] = np.random.uniform(-0.3, 0.3)
            self.model.light_pos[i, 1] = np.random.uniform(-0.3, 0.3)
            self.model.light_pos[i, 2] = np.random.uniform(0.7, 1.3)

            # Moderate diffuse so the lit face doesn't blow out
            diffuse_scale = np.random.uniform(0.30, 0.60)
            self.model.light_diffuse[i] = [diffuse_scale] * 3

            # Low ambient preserves the dark/shadow side of the cube
            ambient_scale = np.random.uniform(0.04, 0.14)
            self.model.light_ambient[i] = [ambient_scale] * 3
        mujoco.mj_forward(self.model, self.data)
        return (str(floor_img_path), str(table_img_path))

    def _reset_camera_augmentation_state(self):
        # FIX 7: when domain randomization is disabled, use identity / neutral
        # camera augmentation parameters so images pass through unmodified.
        if not self.enable_domain_randomization:
            self._camera_profile = {
                "pose_tx_frac": 0.0,
                "pose_ty_frac": 0.0,
                "pose_angle_deg": 0.0,
                "pose_scale": 1.0,
                "border_val": 0,
                "exposure_ev": 0.0,
                "gamma": 1.0,
                "illum_cx_frac": 0.5,
                "illum_cy_frac": 0.5,
                "illum_sigma_x_frac": 0.5,
                "illum_sigma_y_frac": 0.5,
                "illum_strength": 0.0,
                "blur_k": 1,
                "blur_sigma": 0.0,
                "resample_scale": 1.0,
                "down_interp": int(cv2.INTER_LINEAR),
                "up_interp": int(cv2.INTER_LINEAR),
                "read_std": 0.0,
                "shot_scale": 0.0,
                "noise_mix": 0.0,
                "jpeg_quality": 95,
                "jpeg_prob": 0.0,
                "stale_frame_prob": 0.0,
                "wb_gains": np.array([1.0, 1.0, 1.0], dtype=np.float32),
            }
            self._frame_profile = {
                "exposure_jitter_ev": 0.0,
                "wb_jitter": 0.0,
                "gamma_jitter": 0.0,
                "blur_sigma_jitter": 0.0,
                "read_std_jitter": 0.0,
                "jpeg_quality_jitter": 0,
                "stale_frame_prob": 0.0,
            }
            self._obs_delay_steps = 0
            self._obs_fifo = deque()
            return

        # --- Lighting / exposure / colour kept asymmetric: only darken,
        # never brighten.  Gamma stays >= 1 so midtones compress rather
        # than lift.  WB gains capped at 1.0 per channel.

        profile = {
            # 1) Camera pose / framing drift
            "pose_tx_frac": float(np.random.uniform(-0.030, 0.030)),
            "pose_ty_frac": float(np.random.uniform(-0.030, 0.030)),
            "pose_angle_deg": float(np.random.uniform(-2.5, 2.5)),
            "pose_scale": float(np.random.uniform(0.970, 1.030)),
            "border_val": int(np.random.uniform(0, 30)),

            # 2) Exposure: darken only — never brighten
            "exposure_ev": float(np.random.uniform(-0.50, 0.0)),

            # 3) Gamma >= 1.0 darkens midtones, preserving shadow contrast
            "gamma": float(np.random.uniform(1.00, 1.10)),

            # 4) Illumination vignette: only darkening (strength <= 0)
            "illum_cx_frac": float(np.random.uniform(0.25, 0.75)),
            "illum_cy_frac": float(np.random.uniform(0.25, 0.75)),
            "illum_sigma_x_frac": float(np.random.uniform(0.35, 0.70)),
            "illum_sigma_y_frac": float(np.random.uniform(0.35, 0.70)),
            "illum_strength": float(np.random.uniform(-0.06, 0.0)),

            # 5) Optical blur
            "blur_k": int(np.random.choice([3, 5])),
            "blur_sigma": float(np.random.uniform(0.20, 0.65)),

            # 6) Resolution / resampling degradation
            "resample_scale": float(np.random.uniform(0.60, 0.92)),
            "down_interp": int(np.random.choice([cv2.INTER_AREA, cv2.INTER_LINEAR])),
            "up_interp": int(np.random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC])),

            # 7) Sensor noise
            "read_std": float(np.random.uniform(1.0, 4.0)),
            "shot_scale": float(np.random.uniform(0.003, 0.008)),
            "noise_mix": float(np.random.uniform(0.25, 0.55)),

            # 8) JPEG compression
            "jpeg_quality": int(np.random.uniform(45, 88)),
            "jpeg_prob": float(np.random.uniform(0.20, 0.50)),

            # 9) Stale frame probability
            "stale_frame_prob": float(np.random.uniform(0.00, 0.10)),
        }

        # White-balance gains: each channel in [0.85, 1.0] so they can only
        # *reduce* a channel, never amplify above the renderer output.
        mode = np.random.choice(["neutral", "warm", "cool"], p=[0.35, 0.35, 0.30])
        if mode == "warm":
            gains = np.array([
                np.random.uniform(0.94, 1.00),  # R — almost untouched
                np.random.uniform(0.90, 0.97),  # G — slightly reduced
                np.random.uniform(0.85, 0.93),  # B — most reduced → warm
            ], dtype=np.float32)
        elif mode == "cool":
            gains = np.array([
                np.random.uniform(0.85, 0.93),  # R — most reduced → cool
                np.random.uniform(0.90, 0.97),  # G — slightly reduced
                np.random.uniform(0.94, 1.00),  # B — almost untouched
            ], dtype=np.float32)
        else:
            gains = np.array([
                np.random.uniform(0.90, 1.00),
                np.random.uniform(0.90, 1.00),
                np.random.uniform(0.90, 1.00),
            ], dtype=np.float32)

        profile["wb_gains"] = gains
        self._camera_profile = profile

        # Per-frame jitter: kept smaller than the episode profile so
        # individual frames don't spike outside the intended range.
        self._frame_profile = {
            "exposure_jitter_ev": float(np.random.uniform(0.01, 0.04)),
            "wb_jitter": float(np.random.uniform(0.004, 0.015)),
            "gamma_jitter": float(np.random.uniform(0.005, 0.02)),
            "blur_sigma_jitter": float(np.random.uniform(0.03, 0.12)),
            "read_std_jitter": float(np.random.uniform(0.15, 0.60)),
            "jpeg_quality_jitter": int(np.random.randint(2, 8)),
            "stale_frame_prob": profile["stale_frame_prob"],
        }

        self._obs_delay_steps = int(np.random.randint(0, self.max_obs_delay_steps + 1))
        self._obs_fifo = deque()

    def _reset_dynamics_randomization_state(self):
        if self.enable_domain_randomization:
            self._action_delay_steps = int(np.random.randint(0, self.max_action_delay_steps + 1))
            self._arm_action_scale = np.random.uniform(0.88, 1.12, size=len(self.controlled_joints)).astype(np.float32)
            self._drive_action_scale = float(np.random.uniform(0.88, 1.12))
            self._joint_response_alpha = float(np.random.uniform(0.40, 0.85))
            self._drive_response_alpha = float(np.random.uniform(0.40, 0.85))
            self._joint_backlash_rad = np.deg2rad(
                np.random.uniform(0.0, 0.80, size=len(self.controlled_joints))
            ).astype(np.float32)
            self._drive_backlash_m = float(np.random.uniform(0.0, 0.005))
            self._motor_noise_std_rad = float(np.deg2rad(np.random.uniform(0.0, 0.18)))
            self._drive_noise_std_m = float(np.random.uniform(0.0, 0.002))
            self._joint_zero_offset_rad = {
                jn: float(np.deg2rad(np.random.uniform(-1.5, 1.5)))
                for jn in self.controlled_joints
            }
            if self.drive_joint is not None:
                self._joint_zero_offset_rad[self.drive_joint] = float(np.random.uniform(-0.015, 0.015))
        else:
            self._action_delay_steps = 0
            self._arm_action_scale = np.ones(len(self.controlled_joints), dtype=np.float32)
            self._drive_action_scale = 1.0
            self._joint_response_alpha = 1.0
            self._drive_response_alpha = 1.0
            self._joint_backlash_rad = np.zeros(len(self.controlled_joints), dtype=np.float32)
            self._drive_backlash_m = 0.0
            self._motor_noise_std_rad = 0.0
            self._drive_noise_std_m = 0.0
            self._joint_zero_offset_rad = {jn: 0.0 for jn in self.controlled_joints}
            if self.drive_joint is not None:
                self._joint_zero_offset_rad[self.drive_joint] = 0.0

    def _reset_tof_augmentation_state(self):
        # FIX 7: gate ToF augmentation by domain randomization flag
        if self.enable_domain_randomization:
            self._tof_offset_bias_m = float(np.random.uniform(-0.012, 0.012))
            self._tof_ema_alpha = float(np.random.uniform(0.45, 0.75))
        else:
            self._tof_offset_bias_m = 0.0
            self._tof_ema_alpha = 1.0  # alpha=1 means no smoothing
        self._tof_ema = float("nan")

    def _augment_tof(self, raw_m):
        # FIX 7: skip all augmentation when DR is disabled
        if not self.enable_domain_randomization:
            return raw_m

        if not np.isfinite(raw_m) or raw_m <= 0.0:
            if np.random.rand() < 0.04:
                return float("nan")
            return raw_m

        d = float(raw_m)

        d += self._tof_offset_bias_m

        sigma_prop = 0.015 * d + 0.003
        d += np.random.normal(0.0, sigma_prop)

        d = round(d * 1000.0) / 1000.0

        if d < 0.06:
            p_specular_fail = np.clip((0.06 - d) / 0.06, 0.0, 1.0) * 0.40
            if np.random.rand() < p_specular_fail:
                if np.random.rand() < 0.5:
                    d = d * np.random.uniform(0.4, 0.85)
                else:
                    d = d + np.random.uniform(0.02, 0.10)

        if np.random.rand() < 0.05:
            d += np.random.uniform(0.01, 0.10)

        p_dropout = 0.05 + max(0.0, (d - 1.5) * 0.25)
        if np.random.rand() < p_dropout:
            return float("nan")

        alpha = self._tof_ema_alpha
        if not np.isfinite(self._tof_ema):
            self._tof_ema = d
        else:
            self._tof_ema = alpha * d + (1.0 - alpha) * self._tof_ema

        return float(np.clip(self._tof_ema, 0.0, 2.0))

    def _augment_policy_image(self, img_rgb):
        # FIX 7: skip all image augmentation when DR is disabled
        if not self.enable_domain_randomization:
            return img_rgb

        out = img_rgb.astype(np.float32)
        h, w = out.shape[:2]

        profile = dict(getattr(self, "_camera_profile", {}) or {})
        frame_profile = dict(getattr(self, "_frame_profile", {}) or {})
        if not profile:
            self._reset_camera_augmentation_state()
            profile = dict(self._camera_profile)
            frame_profile = dict(self._frame_profile)

        # 1) Mild camera pose / framing drift
        # Keep this small because the cube is tiny and strong warps can destroy it.
        tx = float(profile["pose_tx_frac"] * w + np.random.normal(0.0, 0.004 * w))
        ty = float(profile["pose_ty_frac"] * h + np.random.normal(0.0, 0.004 * h))
        angle = float(profile["pose_angle_deg"] + np.random.normal(0.0, 0.25))
        scale = float(profile["pose_scale"] + np.random.normal(0.0, 0.005))

        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty

        border_val = int(np.clip(float(profile["border_val"]) + np.random.normal(0.0, 3.0), 0, 50))
        out = cv2.warpAffine(
            out.astype(np.uint8), M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(border_val, border_val, border_val),
        ).astype(np.float32)

        # 2) Exposure / brightness
        # ev_stops clamped to <= 0 so the image can only darken, never blow
        # out whites.  This preserves the cube's shadow-side shading.
        ev_stops = float(profile["exposure_ev"] + np.random.normal(0.0, frame_profile["exposure_jitter_ev"]))
        ev_stops = float(np.clip(ev_stops, -0.60, 0.0))
        out *= (2.0 ** ev_stops)

        # 3) White balance / color temperature
        # Gains are all <= 1.0 from the profile so they can only attenuate,
        # never amplify any channel above the renderer output.
        gains = np.asarray(profile["wb_gains"], dtype=np.float32).copy()
        gains *= (1.0 + np.random.normal(0.0, frame_profile["wb_jitter"], size=3).astype(np.float32))
        gains = np.clip(gains, 0.80, 1.0)
        out *= gains.reshape(1, 1, 3)

        out = np.clip(out, 0, 255)

        # 4) Gamma / tone curve
        # gamma >= 1.0 compresses highlights and darkens midtones, keeping
        # shadow detail visible on the cube.
        gamma = float(profile["gamma"] + np.random.normal(0.0, frame_profile["gamma_jitter"]))
        gamma = float(np.clip(gamma, 1.00, 1.15))
        out_norm = np.clip(out / 255.0, 1e-6, 1.0)
        out = np.clip((out_norm ** gamma) * 255.0, 0, 255)

        # 5) Slight local illumination nonuniformity
        # strength <= 0 means this only *subtracts* light (mild vignette),
        # never adds a bright hotspot that could wash out the cube.
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        cx = float(profile["illum_cx_frac"] * w)
        cy = float(profile["illum_cy_frac"] * h)
        sigma_x = float(profile["illum_sigma_x_frac"] * w)
        sigma_y = float(profile["illum_sigma_y_frac"] * h)

        illum = np.exp(
            -(((xs - cx) ** 2) / (2.0 * sigma_x ** 2) + ((ys - cy) ** 2) / (2.0 * sigma_y ** 2))
        ).astype(np.float32)[..., None]

        strength = float(profile["illum_strength"] + np.random.normal(0.0, 0.008))
        strength = float(min(strength, 0.0))  # hard-cap: never positive
        out = np.clip(out * (1.0 + strength * illum), 0, 255)

        # 6) Optical blur
        k = int(profile["blur_k"])
        sigma = float(profile["blur_sigma"] + np.random.normal(0.0, frame_profile["blur_sigma_jitter"]))
        sigma = float(np.clip(sigma, 0.10, 0.85))
        out = cv2.GaussianBlur(
            out.astype(np.uint8), (k, k),
            sigmaX=sigma, sigmaY=sigma,
        ).astype(np.float32)

        # 7) Camera resolution / resampling degradation
        sc = float(profile["resample_scale"] + np.random.normal(0.0, 0.02))
        sc = float(np.clip(sc, 0.50, 0.96))
        sw = max(8, int(round(w * sc)))
        sh = max(8, int(round(h * sc)))

        small = cv2.resize(out.astype(np.uint8), (sw, sh), interpolation=int(profile["down_interp"]))
        out = cv2.resize(small, (w, h), interpolation=int(profile["up_interp"])).astype(np.float32)

        # 8) Sensor noise
        # Mild read noise plus signal-dependent shot-ish noise.
        read_std = float(np.clip(profile["read_std"], 0.25, 10.0))

        # Single luminance noise map — no RGB dots
        luma_noise = np.random.normal(0.0, read_std, out.shape[:2]).astype(np.float32)
        out = np.clip(out + luma_noise[..., None], 0, 255)

        # Shot noise: std = sqrt(signal) scaled by a small factor
        shot_factor = float(profile["shot_scale"]) * 30.0
        luma = np.mean(np.clip(out, 0.0, None), axis=-1)
        shot_std = np.sqrt(luma) * shot_factor
        shot_noise = (np.random.normal(0.0, 1.0, out.shape[:2]) * shot_std)[..., None]
        out = np.clip(out + shot_noise, 0, 255)

        # 9) Mild JPEG artifacting
        # Good for matching cheap camera / streaming artifacts.
        if np.random.rand() < float(profile["jpeg_prob"]):
            quality = int(np.clip(
                profile["jpeg_quality"] + np.random.randint(
                    -frame_profile["jpeg_quality_jitter"],
                    frame_profile["jpeg_quality_jitter"] + 1),
                30, 95,
            ))
            ok, enc = cv2.imencode(
                ".jpg",
                cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), quality],
            )
            if ok:
                dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
                if dec is not None:
                    out = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB).astype(np.float32)

        return np.clip(out, 0, 255).astype(np.uint8)

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

        # FIX 8: use cached evaluate_state so we don't recompute reward
        reward, info = self.evaluate_state()
        rew_str = f"R: {reward:.3f}  (app={info['approach_reward']:.2f}  align={info['align_reward']:.2f}  hit={info['hit_bonus']:.2f})"
        cv2.putText(scene_bgr, rew_str, (x0, y0 + 4 * line_h), font, scale, color, thickness, cv2.LINE_AA)