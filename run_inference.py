import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "glfw")
os.environ["PYOPENGL_PLATFORM"] = os.environ.get("PYOPENGL_PLATFORM", "glfw")
os.environ.pop("MUJOCO_PLATFORM", None)

import math
import argparse
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from MAE_Model.model import MAEModel
import DrQv2_Architecture.utils as utils
from DrQv2_Architecture.drqv2 import DrQV2Agent, Actor, Critic


class Control:
    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | torch.device | None = None,
        num_frames: int = 3,
        eval_step: int = 500_000,
    ):
        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.num_frames = num_frames
        self.eval_step = eval_step

        self.agent = self._load_agent(checkpoint_path)
        self.agent.train(False)

        self._frame_buf: deque[np.ndarray] = deque(maxlen=num_frames)

    def process(self, cam_input: np.ndarray, tof_input: float | np.ndarray) -> np.ndarray:
        """
        cam_input : np.ndarray
            Raw RGB frame from the POV camera, shape (H, W, 3) uint8.
            H and W must match img_h_size / img_w_size used during training.
        tof_input : float or np.ndarray
            Raw ToF distance in metres (e.g. 0.45).
            Pass float('nan') or a non-positive value if the sensor has no valid reading.

        Returns
        action : np.ndarray, shape (action_dim,), float32 in [-1, 1]
            One value per controlled joint (in the order of controlled_joints (degree)
            followed by the drive joint value (1 cm fwd or backwd)
        """
        frame = self._preprocess_frame(cam_input)
        self._push_frame(frame)
        stacked = self._get_stacked_obs()
        tof = self._preprocess_tof(tof_input)

        obs = {"pov": stacked, "tof": tof}

        with torch.no_grad(), utils.eval_mode(self.agent):
            action = self.agent.act(obs, self.eval_step, eval_mode=True)
        return action.astype(np.float32)

    def reset(self) -> None:
        self._frame_buf.clear()

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected (H, W, 3) RGB frame, got {frame.shape}")
        return np.transpose(frame, (2, 0, 1)).astype(np.uint8)

    def _push_frame(self, frame: np.ndarray) -> None:
        if len(self._frame_buf) == 0:
            for _ in range(self.num_frames):
                self._frame_buf.append(frame)
        else:
            self._frame_buf.append(frame)

    def _get_stacked_obs(self) -> np.ndarray:
        return np.concatenate(list(self._frame_buf), axis=0)

    @staticmethod
    def _preprocess_tof(tof: float | np.ndarray) -> np.ndarray:
        raw = float(np.asarray(tof).flat[0])
        if not math.isfinite(raw) or raw <= 0:
            val = 1.0
        else:
            val = float(np.clip(raw / 2.0, 0.0, 0.99))
        return np.array([val], dtype=np.float32)

    def _load_agent(self, path: str | Path) -> DrQV2Agent:
        ckpt = torch.load(str(path), map_location=self.device)
        cfg = ckpt["cfg"]

        agent = DrQV2Agent(
            action_shape=tuple(cfg["action_shape"]),
            device=self.device,
            mvmae_patch_size=cfg["patch_size"],
            mvmae_encoder_embed_dim=cfg["encoder_embed_dim"],
            mvmae_decoder_embed_dim=cfg["decoder_embed_dim"],
            mvmae_encoder_heads=cfg["encoder_heads"],
            mvmae_decoder_heads=cfg["decoder_heads"],
            in_channels=cfg["in_channels"],
            img_h_size=cfg["img_h_size"],
            img_w_size=cfg["img_w_size"],
            masking_ratio=cfg["masking_ratio"],
            feature_dim=cfg["feature_dim"],
            hidden_dim=cfg["hidden_dim"],
        )

        agent.mvmae.load_state_dict(ckpt["mvmae"])
        agent.trunc.load_state_dict(ckpt["trunc"])
        agent.trunc_target.load_state_dict(ckpt["trunc_target"])
        agent.actor.load_state_dict(ckpt["actor"])
        agent.critic.load_state_dict(ckpt["critic"])
        agent.critic_target.load_state_dict(ckpt["critic_target"])

        print(f"[Control] Loaded checkpoint from '{path}' (step={ckpt.get('step')})")
        return agent


def run_sim(
    checkpoint_path: str,
    xml_path: str = str(Path("Simulation") / "Assets" / "scene.xml"),
    episode_horizon: int = 300,
    display_scale: int = 6,
    img_size: int = 64,
    num_frames: int = 3,
    discount: float = 0.99,
    wait_ms: int = 30,
) -> None:
    from Arm_Env.arm_env import ArmEnv
    from DrQv2_Architecture.env_wrappers import (
        FrameStackWrapper,
        ActionRepeatWrapper,
        ExtendedTimeStepWrapper,
    )

    base_env = ArmEnv(
        xml_path=xml_path,
        width=img_size,
        height=img_size,
        display=False,
        discount=discount,
        episode_horizon=episode_horizon,
    )
    env = FrameStackWrapper(base_env, num_frames=num_frames)
    env = ActionRepeatWrapper(env, num_repeats=2)
    env = ExtendedTimeStepWrapper(env)

    control = Control(checkpoint_path=checkpoint_path, num_frames=num_frames)

    disp_size = img_size * display_scale
    pov_win   = f"POV  ({img_size}px -> {disp_size}px)"
    scene_win = f"Scene ({img_size}px -> {disp_size}px)"
    cv2.namedWindow(pov_win,   cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(scene_win, cv2.WINDOW_AUTOSIZE)

    time_step = env.reset()
    control.reset()

    step = 0
    total_reward = 0.0

    print(f"\n[SimRun] Starting episode  (horizon={episode_horizon}, scale={display_scale}x)")
    print(f"         Checkpoint : {checkpoint_path}")
    print(f"         Press Q to quit early.\n")

    while not time_step.last():
        stacked_chw = time_step.pov
        latest_chw  = stacked_chw[-3:]
        latest_hwc  = np.transpose(latest_chw, (1, 2, 0))
        tof_raw     = float(base_env.last_tof_m)

        action = control.process(latest_hwc, tof_raw)

        base_env._gl_ctx.make_current()
        scene_hwc = base_env._render_fixedcam(base_env._mjv_cam_scene)

        pov_big   = cv2.resize(cv2.cvtColor(latest_hwc, cv2.COLOR_RGB2BGR), (disp_size, disp_size), interpolation=cv2.INTER_NEAREST)
        scene_big = cv2.resize(cv2.cvtColor(scene_hwc,  cv2.COLOR_RGB2BGR), (disp_size, disp_size), interpolation=cv2.INTER_NEAREST)

        font      = cv2.FONT_HERSHEY_SIMPLEX
        hud_scale = 0.5 * display_scale / 4
        lh        = int(20 * display_scale / 4)
        x0, y0   = 8, lh

        tof_str = f"ToF: {base_env.last_tof_m:.4f} m" if math.isfinite(base_env.last_tof_m) else "ToF: nan"
        hit_str = f"Hit: {bool(base_env.last_tof_hit)}"
        h_str   = f"h: {base_env.last_align_h_deg:.1f}" if math.isfinite(base_env.last_align_h_deg) else "h: n/a"
        v_str   = f"v: {base_env.last_align_v_deg:.1f}" if math.isfinite(base_env.last_align_v_deg) else "v: n/a"
        vo_str  = f"v*: {base_env.last_opt_v_deg:.1f}"  if math.isfinite(base_env.last_opt_v_deg)  else "v*: n/a"

        reward_now, info = base_env.evaluate_state()
        rew_str = f"R: {reward_now:.3f}  app={info['approach_reward']:.2f}  align={info['align_reward']:.2f}  hit={info['hit_bonus']:.2f}"

        joint_parts = [f"{jn}: {math.degrees(base_env.targets_rad.get(jn, 0.0)):.1f}" for jn in base_env.controlled_joints]
        if base_env.drive_joint is not None:
            joint_parts.append(f"{base_env.drive_joint}: {base_env.targets_rad.get(base_env.drive_joint, 0.0):.3f}m")
        opt_str = f"opt_v*: {base_env.last_opt_v_deg:.2f}" if math.isfinite(base_env.last_opt_v_deg) else "opt_v*: n/a"
        joint_parts.append(opt_str)

        hud_lines = [tof_str, hit_str, f"{h_str}  {v_str}  {vo_str}", rew_str, "  ".join(joint_parts), f"step: {step}"]
        for i, txt in enumerate(hud_lines):
            cv2.putText(scene_big, txt, (x0, y0 + i * lh), font, hud_scale, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(scene_big, txt, (x0, y0 + i * lh), font, hud_scale, (0, 200, 0), 1, cv2.LINE_AA)

        cv2.imshow(pov_win,   pov_big)
        cv2.imshow(scene_win, scene_big)

        if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
            print("[SimRun] User quit.")
            break

        time_step     = env.step(action)
        total_reward += float(time_step.reward[0]) if time_step.reward is not None else 0.0
        step         += 1

        if step % 50 == 0:
            print(f"  step={step:4d}  total_reward={total_reward:.3f}  tof={base_env.last_tof_m:.4f}  hit={base_env.last_tof_hit}")

    print(f"\n[SimRun] Done.  steps={step}  total_reward={total_reward:.3f}")
    cv2.destroyAllWindows()


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",      type=str, required=True)
    p.add_argument("--xml_path",        type=str, default=str(Path("Simulation") / "Assets" / "scene.xml"))
    p.add_argument("--episode_horizon", type=int, default=300)
    p.add_argument("--display_scale",   type=int, default=6)
    p.add_argument("--img_size",        type=int, default=64)
    p.add_argument("--num_frames",      type=int, default=3)
    p.add_argument("--wait_ms",         type=int, default=30)
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    run_sim(
        checkpoint_path  = args.checkpoint,
        xml_path         = args.xml_path,
        episode_horizon  = args.episode_horizon,
        display_scale    = args.display_scale,
        img_size         = args.img_size,
        num_frames       = args.num_frames,
        wait_ms          = args.wait_ms,
    )