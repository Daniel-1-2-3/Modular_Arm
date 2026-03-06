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
import DrQv2_Architecture.utils as utils
from DrQv2_Architecture.drqv2 import DrQV2Agent

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
        self._frozen = False

    def process(self, cam_input: np.ndarray, tof_m: float) -> np.ndarray:
        if self._frozen:
            return np.zeros(4, dtype=np.float32)
        frame = self._preprocess_frame(cam_input)
        self._push_frame(frame)
        obs = {
            "pov": self._get_stacked_obs(),
            "tof": self._preprocess_tof(tof_m),
        }
        with torch.no_grad(), utils.eval_mode(self.agent):
            action = self.agent.act(obs, self.eval_step, eval_mode=True)
        return action.astype(np.float32)

    def check_and_freeze(self, tof_m: float) -> bool:
        if math.isfinite(tof_m) and 0.0 < tof_m <= 0.06:
            self._frozen = True
            return True
        return self._frozen

    def reset(self) -> None:
        self._frame_buf.clear()
        self._frozen = False

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
    def _preprocess_tof(tof_m: float) -> np.ndarray:
        # Normalise raw metres to [0, 0.99], expected by agent
        val = 1.0 if (not math.isfinite(tof_m) or tof_m <= 0) \
              else float(np.clip(tof_m / 2.0, 0.0, 0.99))
        return np.array([val], dtype=np.float32)

    def _load_agent(self, path: str | Path) -> DrQV2Agent:
        ckpt = torch.load(str(path), map_location=self.device)
        cfg = ckpt["cfg"]
        agent = DrQV2Agent(
            action_shape = tuple(cfg["action_shape"]),
            device = self.device,
            mvmae_patch_size = cfg["patch_size"],
            mvmae_encoder_embed_dim = cfg["encoder_embed_dim"],
            mvmae_decoder_embed_dim = cfg["decoder_embed_dim"],
            mvmae_encoder_heads = cfg["encoder_heads"],
            mvmae_decoder_heads = cfg["decoder_heads"],
            in_channels = cfg["in_channels"],
            img_h_size = cfg["img_h_size"],
            img_w_size = cfg["img_w_size"],
            masking_ratio = cfg["masking_ratio"],
            feature_dim = cfg["feature_dim"],
            hidden_dim = cfg["hidden_dim"],
        )
        agent.mvmae.load_state_dict(ckpt["mvmae"])
        agent.trunc.load_state_dict(ckpt["trunc"])
        agent.trunc_target.load_state_dict(ckpt["trunc_target"])
        agent.actor.load_state_dict(ckpt["actor"])
        agent.critic.load_state_dict(ckpt["critic"])
        agent.critic_target.load_state_dict(ckpt["critic_target"])
        print(f"[Control] Loaded checkpoint from '{path}' (step={ckpt.get('step')})")
        return agent

def _render_hud(
    scene_big: np.ndarray,
    env: object,
    tof_m: float,
    step: int,
    total_reward: float,
    frozen: bool,
    lh: int,
    hud_scale: float,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    reward_now, info = env.evaluate_state()

    tof_str = f"ToF: {tof_m:.4f} m" if math.isfinite(tof_m) else "ToF: nan"
    frozen_str = "  [FROZEN]" if frozen else ""
    h_str = f"h: {env.last_align_h_deg:.1f}"  if math.isfinite(env.last_align_h_deg)  else "h: n/a"
    v_str = f"v: {env.last_align_v_deg:.1f}"  if math.isfinite(env.last_align_v_deg)  else "v: n/a"
    vo_str = f"v*: {env.last_opt_v_deg:.1f}"   if math.isfinite(env.last_opt_v_deg)    else "v*: n/a"
    rew_str = (f"R: {reward_now:.3f}  app={info['approach_reward']:.2f}"
                  f"  align={info['align_reward']:.2f}  hit={info['hit_bonus']:.2f}")
    j_parts = [f"{jn}: {math.degrees(env.targets_rad.get(jn, 0.0)):.1f}"
                  for jn in env.controlled_joints]
    if env.drive_joint:
        j_parts.append(f"{env.drive_joint}: {env.targets_rad.get(env.drive_joint, 0.0):.3f}m")

    hud_lines = [
        tof_str + frozen_str,
        f"Hit: {bool(env.last_tof_hit)}",
        f"{h_str}  {v_str}  {vo_str}",
        rew_str,
        "  ".join(j_parts),
        f"step: {step}  total_R: {total_reward:.3f}",
    ]
    for i, txt in enumerate(hud_lines):
        cv2.putText(scene_big, txt, (8, lh + i * lh), font, hud_scale, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(scene_big, txt, (8, lh + i * lh), font, hud_scale, (0, 200, 0), 1, cv2.LINE_AA)

def run_sim(
    checkpoint_path: str,
    xml_path: str = str(Path("Simulation") / "Assets" / "scene.xml"),
    episode_horizon: int = 300,
    display_scale: int = 6,
    wait_ms: int = 30,
) -> None:
    from Arm_Env.arm_env import ArmEnv

    cfg = torch.load(checkpoint_path, map_location="cpu")["cfg"]
    img_size = cfg["img_h_size"]
    num_frames = cfg.get("num_frames", 3)
    discount = cfg.get("discount", 0.99)
    disp_size = img_size * display_scale
    lh = int(20 * display_scale / 4)
    hud_scale = 0.5 * display_scale / 4

    env = ArmEnv(
        xml_path=xml_path,
        width=img_size, height=img_size,
        display=False,
        discount=discount,
        episode_horizon=episode_horizon,
    )
    
    control = Control(checkpoint_path=checkpoint_path, num_frames=num_frames)
    pov_win = f"POV ({img_size}→{disp_size}px)"
    scene_win = f"Scene ({img_size}→{disp_size}px)"
    cv2.namedWindow(pov_win, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(scene_win, cv2.WINDOW_AUTOSIZE)

    info = env.reset()
    control.reset()
    step, total_reward = 0, 0.0

    cam_hwc = info["pov_obs"]
    tof_m = float(env.last_tof_m)
    action = control.process(cam_hwc, tof_m)

    episode_done = False
    while not episode_done:
        info = env.step(action)
        total_reward += float(info["reward"] or 0.0)
        step += 1
        episode_done = (info["step_type"].value == 2)

        tof_m = float(env.last_tof_m)
        cam_hwc = info["pov_obs"]

        just_frozen = control.check_and_freeze(tof_m)
        action = control.process(cam_hwc, tof_m)

        env._gl_ctx.make_current()
        scene_hwc = env._render_fixedcam(env._mjv_cam_scene)
        pov_big = cv2.resize(cv2.cvtColor(cam_hwc, cv2.COLOR_RGB2BGR), (disp_size, disp_size), interpolation=cv2.INTER_LANCZOS4)
        scene_big = cv2.resize(cv2.cvtColor(scene_hwc, cv2.COLOR_RGB2BGR), (disp_size, disp_size), interpolation=cv2.INTER_LANCZOS4)
        _render_hud(scene_big, env, tof_m, step, total_reward, control._frozen, lh, hud_scale)

        cv2.imshow(pov_win, pov_big)
        cv2.imshow(scene_win, scene_big)
        if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
            print("[SimRun] User quit.")
            break

        if step % 50 == 0:
            print(f"step={step:4d}  total_reward={total_reward:.3f}"
                  f"  tof={tof_m:.4f}  hit={env.last_tof_hit}")

        if just_frozen:
            print(f"[SimRun] Frozen at tof={tof_m:.4f} m, step={step}")
            while True:
                if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
                    break
            break

    print(f"\n[SimRun] steps={step}  total_reward={total_reward:.3f}")
    cv2.destroyAllWindows()

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--xml_path", type=str, default=str(Path("Simulation") / "Assets" / "scene.xml"))
    p.add_argument("--episode_horizon", type=int, default=300)
    p.add_argument("--display_scale", type=int, default=6)
    p.add_argument("--wait_ms", type=int, default=30)
    return p.parse_args()

if __name__ == "__main__":
    args = get_args()
    run_sim(
        args.checkpoint,
        args.xml_path,
        args.episode_horizon,
        args.display_scale,
        args.wait_ms,
    )