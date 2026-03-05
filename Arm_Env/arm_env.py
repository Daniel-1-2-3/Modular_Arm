import os, math, random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import mujoco
from gymnasium.spaces import Dict, Box
from dm_env import StepType

from randomize_helpers import RandomizeHelpers
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

        self.tof_cam_offset_m = 0.07
        self.cam_pitch_deg = 8.0

        self.discount = float(discount)
        self.episode_horizon = int(episode_horizon)
        self.curr_path_length: int = 0
        self.step_type: StepType = StepType.FIRST
        self.hand_init_pos: np.ndarray | None = None
        self._target_pos: np.ndarray | None = None

        self.reset()

    def get_action_space(self) -> Box:
        return Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)

    def get_observation_space(self) -> Dict:
        return Dict({
            "pov": Box(low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8),
            "tof": Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        })

    def evaluate_state(self) -> tuple[float, dict[str, Any]]:
        reward, tof_dist, in_place = self._compute_reward()
        success = float(tof_dist <= 0.05)
        info = {
            "success": success,
            "tof_dist": tof_dist,
            "in_place_reward": in_place,
            "unscaled_reward": reward,
        }
        return reward, info

    def reset(self) -> dict[str, Any]:
        self._randomize()
        self.curr_path_length = 0
        self.step_type = StepType.FIRST

        self.targets_rad = {"j_a_base": 0.0, "j_b_mid": 0.0, "j_c_end": 105 * math.pi / 180}
        if self.drive_joint is not None:
            self.targets_rad[self.drive_joint] = self._get_joint_qpos_rad(self.drive_joint)

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

        self.last_cam_cube_m = float(np.linalg.norm(tgt_pos - cam_pos))

        right, down, forward = R[:, 0], R[:, 1], R[:, 2]
        x, y, z = np.dot(v, right), np.dot(v, down), np.dot(v, forward)
        z = abs(z)

        self.last_align_h_deg = float(math.degrees(math.atan2(x, abs(z))))
        self.last_align_v_deg = float(math.degrees(math.atan2(-y, abs(z))))
        self.last_opt_v_deg = float(self._get_optimal_v_angle_from_z(z, h_m=self.tof_cam_offset_m, pitch_deg=self.cam_pitch_deg))

    def _get_optimal_v_angle_from_z(self, z: float, h_m: float, pitch_deg: float) -> float:
        if not np.isfinite(z) or z <= 1e-6:
            return float("nan")
        th = math.radians(pitch_deg)
        num = h_m * math.cos(th) - z * math.sin(th)
        den = h_m * math.sin(th) + z * math.cos(th)
        return float(math.degrees(math.atan2(num, den)))

    def _tof_hitting_target(self, body_name: str = "red_cube") -> bool:
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.pov_cam)
        cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        arm_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "EndSegment_body")

        ray_origin = self.data.cam_xpos[cam_id].copy()
        R = self.data.cam_xmat[cam_id].reshape(3, 3)
        ray_dir = -R[:, 2]

        geomid = np.array([-1], dtype=np.int32)
        geomgroup = np.ones(6, dtype=np.uint8)

        mujoco.mj_ray(
            self.model, self.data,
            ray_origin, ray_dir,
            geomgroup, 1, arm_body_id,
            geomid,
        )

        if geomid[0] < 0:
            return False

        hit_body_id = self.model.geom_bodyid[geomid[0]]
        return int(hit_body_id) == int(cube_body_id)

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

    def _compute_reward(self) -> tuple[float, float, float]:
        v_err_deg = self.last_align_v_deg - self.last_opt_v_deg
        align_err_deg = math.sqrt(self.last_align_h_deg**2 + v_err_deg**2)
        align = reward_utils.tolerance(
            align_err_deg,
            bounds=(0, 5.0),
            margin=45.0,
            sigmoid="long_tail",
        )

        TARGET_RADIUS = 0.06
        if self._tof_hitting_target():
            in_place_margin = float(np.linalg.norm(self.hand_init_pos - self._target_pos))
            in_place = reward_utils.tolerance(
                self.last_tof_m,
                bounds=(0, TARGET_RADIUS),
                margin=in_place_margin,
                sigmoid="long_tail",
            )
        else:
            in_place = 0.0

        reward = 10 * (0.3 * align + 0.7 * in_place)
        return (reward, self.last_tof_m, in_place)

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
        h_str = f"h: {self.last_align_h_deg:.2f} deg" if np.isfinite(self.last_align_h_deg) else "h: nan"
        v_str = f"v: {self.last_align_v_deg:.2f} deg" if np.isfinite(self.last_align_v_deg) else "v: nan"
        vo_str = f"v*: {self.last_opt_v_deg:.2f} deg" if np.isfinite(self.last_opt_v_deg) else "v*: nan"
        d_str = f"d_cam: {self.last_cam_cube_m:.3f} m" if np.isfinite(self.last_cam_cube_m) else "d_cam: nan"

        cv2.putText(scene_bgr, tof_str, (x0, y0), font, scale, color, thickness, cv2.LINE_AA)
        cv2.putText(scene_bgr, f"{h_str}   {v_str}", (x0, y0 + line_h), font, scale, color, thickness, cv2.LINE_AA)
        cv2.putText(scene_bgr, f"{vo_str}   {d_str}", (x0, y0 + 2 * line_h), font, scale, color, thickness, cv2.LINE_AA)