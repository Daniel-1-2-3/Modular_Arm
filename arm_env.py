import os, time, math, queue

import cv2
import numpy as np
import numpy.typing as npt
import mujoco

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
            print('Loaded model')
        except ValueError as e:
            print(f"Failed to load XML: {e}")
            raise SystemExit(1)

        self._init_actuators()
        self._init_renderer()
        if display:
            self._init_display()

    def step(self, action: npt.NDArray[np.integer]): # Incremental: action +- degree to total ([-1, 0, 1])
        if action.shape[0] != len(self.controlled_joints):
            raise RuntimeError("Action len does not match number of joints")

        delta = action.astype(np.int32) * (np.pi / 180.0)
        for i, jn in enumerate(self.controlled_joints): # Increment then clip
            self.targets_rad[jn] += float(delta[i])
            self.targets_rad[jn] = float(
                np.clip(
                    self.targets_rad[jn],
                    self.lows[i],
                    self.highs[i]
                )
            )

        for jn in self.controlled_joints: # Apply to actuators and step
            self.data.ctrl[self.joint_to_ctrl[jn]] = self.targets_rad[jn]
        for _ in range(20):
            mujoco.mj_step(self.model, self.data)

        obs = self.get_obs()
        if self.display:
            scene_bgr = cv2.cvtColor(obs["scene"], cv2.COLOR_RGB2BGR)
            pov_bgr = cv2.cvtColor(obs["pov"], cv2.COLOR_RGB2BGR)
            cv2.imshow(self.scene_window_name, scene_bgr)
            cv2.imshow(self.pov_window_name, pov_bgr)

    def get_obs(self):
        self._gl_ctx.make_current()
        scene_rgb = self._render_fixedcam(self._mjv_cam_scene)
        pov_rgb = self._render_fixedcam(self._mjv_cam_pov)
        return {"scene": scene_rgb, "pov": pov_rgb}

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
        return np.flipud(self._rgb_u8.copy())

    def _init_actuators(self):
        self.joint_to_ctrl: dict[str, int] = {} # {joint_name: joint_id}
        for act_id in range(self.model.nu):
            if int(self.model.actuator_trntype[act_id]) != int(mujoco.mjtTrn.mjTRN_JOINT):
                continue
            j_id = int(self.model.actuator_trnid[act_id][0])
            j_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j_id)
            if j_name:
                self.joint_to_ctrl[j_name] = act_id

        if not self.joint_to_ctrl:
            raise RuntimeError("No actuators found")
        self.controlled_joints = sorted(self.joint_to_ctrl.keys()) # [] of joint nams in fixed orders

        self.targets_rad = {} # {joint_name: current radian angle of the joint}
        for joint in self.controlled_joints:
            self.targets_rad[joint] = self._get_joint_qpos_rad(joint)
            self.data.ctrl[self.joint_to_ctrl[joint]] = self.targets_rad[joint]

        # Limits for each joint (j_base, j_mid, j_end)
        limits_by_name = {
            "j_a_base": (-45 / 180 * np.pi, 45 / 180 * np.pi),
            "j_b_mid":  (0, 25 / 180 * np.pi),
            "j_c_end":  (-90 / 180 * np.pi, 110 / 180 * np.pi),
        }

        self.manual_limits = np.array([limits_by_name[j] for j in self.controlled_joints], dtype=np.float64)
        self.lows  = self.manual_limits[:, 0]
        self.highs = self.manual_limits[:, 1]

    def _init_renderer(self):
        # Fast offscreen renderer, refrenced MetaWorld
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
        self._mjv_scene = mujoco.MjvScene(self.model, maxgeom=2000) # Max objects rendered in scene

        if not hasattr(mujoco, "GLContext") or mujoco.GLContext is None:
            raise RuntimeError("GLContext unavailable. Set MUJOCO_GL before importing mujoco")

        self._gl_ctx = mujoco.GLContext(self.req_width, self.req_height)
        self._gl_ctx.make_current()

        self._mjr_ctx = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_100)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self._mjr_ctx)

        # Prep for cv2 render window
        off_w = int(getattr(self._mjr_ctx, "offWidth", 640))
        off_h = int(getattr(self._mjr_ctx, "offHeight", 480))
        self.width = min(self.req_width, off_w)
        self.height = min(self.req_height, off_h)
        self._mjr_rect = mujoco.MjrRect(0, 0, self.width, self.height)
        self._rgb_u8 = np.empty((self.height, self.width, 3), dtype=np.uint8)

    def _init_display(self):
        self.scene_window_name = f"RGB ({self.scene_cam}) {self.width}x{self.height}"
        self.pov_window_name = f"RGB ({self.pov_cam}) {self.width}x{self.height}"
        cv2.namedWindow(self.scene_window_name, cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(self.pov_window_name, cv2.WINDOW_AUTOSIZE)

    def _get_joint_qpos_rad(self, joint_name):
        jid = self.model.joint(joint_name).id
        qadr = int(self.model.jnt_qposadr[jid])
        return float(self.data.qpos[qadr])

if __name__ == "__main__":
    xml_path = os.path.join("Simulation", "Assets", "scene.xml")
    arm_env = ArmEnv(xml_path=xml_path)

    # Confirm what joints are actually controllable (actuated) now
    print("controlled_joints =", arm_env.controlled_joints)

    # WASDRF controls the first three joints (base, mid, end) by index
    # a/d: base -,+
    # s/w: mid  -,+
    # f/r: end  -,+
    keymap = {
        ord("a"): (0, -1),
        ord("d"): (0, +1),
        ord("s"): (1, -1),
        ord("w"): (1, +1),
        ord("f"): (2, -1),
        ord("r"): (2, +1),
    }

    while True:
        action = np.zeros(len(arm_env.controlled_joints), dtype=np.int32)

        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break

        if k in keymap:
            idx, delta = keymap[k]
            if idx < action.shape[0]:
                action[idx] = delta

        arm_env.step(action)

    cv2.destroyAllWindows()