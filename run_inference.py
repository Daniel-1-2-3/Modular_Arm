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

    def process(self, cam_input: np.ndarray, tof_input: float | np.ndarray) -> np.ndarray:
        frame = self._preprocess_frame(cam_input)
        self._push_frame(frame)
        obs = {"pov": self._get_stacked_obs(), "tof": self._preprocess_tof(tof_input)}
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
        val = 1.0 if (not math.isfinite(raw) or raw <= 0) else float(np.clip(raw / 2.0, 0.0, 0.99))
        return np.array([val], dtype=np.float32)

    def _load_agent(self, path: str | Path) -> DrQV2Agent:
        ckpt = torch.load(str(path), map_location=self.device)
        cfg  = ckpt["cfg"]
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

def run_sim(
    checkpoint_path: str,
    xml_path: str = str(Path("Simulation") / "Assets" / "scene.xml"),
    episode_horizon: int = 300,
    display_scale: int = 6,
    wait_ms: int = 30,
) -> None:
    from Arm_Env.arm_env import ArmEnv
    from DrQv2_Architecture.env_wrappers import FrameStackWrapper, ActionRepeatWrapper, ExtendedTimeStepWrapper

    cfg = torch.load(checkpoint_path, map_location="cpu")["cfg"]
    img_size = cfg["img_h_size"]
    num_frames = cfg.get("num_frames", 3)
    discount = cfg.get("discount", 0.99)
    disp_size = img_size * display_scale

    base_env = ArmEnv(xml_path=xml_path, width=img_size, height=img_size, display=False, discount=discount, episode_horizon=episode_horizon)
    env = ExtendedTimeStepWrapper(ActionRepeatWrapper(FrameStackWrapper(base_env, num_frames=num_frames), num_repeats=2))
    control = Control(checkpoint_path=checkpoint_path, num_frames=num_frames)

    pov_win, scene_win = f"POV ({img_size} {disp_size}px)", f"Scene ({img_size} {disp_size}px)"
    cv2.namedWindow(pov_win, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(scene_win, cv2.WINDOW_AUTOSIZE)

    time_step = env.reset()
    control.reset()
    step, total_reward = 0, 0.0

    while not time_step.last():
        latest_hwc = np.transpose(time_step.pov[-3:], (1, 2, 0))
        action = control.process(latest_hwc, float(base_env.last_tof_m))

        base_env._gl_ctx.make_current()
        scene_hwc = base_env._render_fixedcam(base_env._mjv_cam_scene)

        pov_big = cv2.resize(cv2.cvtColor(latest_hwc, cv2.COLOR_RGB2BGR), (disp_size, disp_size), interpolation=cv2.INTER_NEAREST)
        scene_big = cv2.resize(cv2.cvtColor(scene_hwc, cv2.COLOR_RGB2BGR), (disp_size, disp_size), interpolation=cv2.INTER_NEAREST)

        font, lh = cv2.FONT_HERSHEY_SIMPLEX, int(20 * display_scale / 4)
        hud_scale = 0.5 * display_scale / 4
        reward_now, info = base_env.evaluate_state()

        tof_str = f"ToF: {base_env.last_tof_m:.4f} m" if math.isfinite(base_env.last_tof_m) else "ToF: nan"
        h_str = f"h: {base_env.last_align_h_deg:.1f}" if math.isfinite(base_env.last_align_h_deg) else "h: n/a"
        v_str = f"v: {base_env.last_align_v_deg:.1f}" if math.isfinite(base_env.last_align_v_deg) else "v: n/a"
        vo_str = f"v*: {base_env.last_opt_v_deg:.1f}" if math.isfinite(base_env.last_opt_v_deg) else "v*: n/a"
        rew_str = f"R: {reward_now:.3f}  app={info['approach_reward']:.2f}  align={info['align_reward']:.2f}  hit={info['hit_bonus']:.2f}"
        j_parts = [f"{jn}: {math.degrees(base_env.targets_rad.get(jn, 0.0)):.1f}" for jn in base_env.controlled_joints]
        if base_env.drive_joint:
            j_parts.append(f"{base_env.drive_joint}: {base_env.targets_rad.get(base_env.drive_joint, 0.0):.3f}m")

        hud_lines = [tof_str, f"Hit: {bool(base_env.last_tof_hit)}", f"{h_str}  {v_str}  {vo_str}",
                     rew_str, "  ".join(j_parts), f"step: {step}"]
        for i, txt in enumerate(hud_lines):
            cv2.putText(scene_big, txt, (8, lh + i * lh), font, hud_scale, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(scene_big, txt, (8, lh + i * lh), font, hud_scale, (0, 200, 0), 1, cv2.LINE_AA)

        cv2.imshow(pov_win, pov_big)
        cv2.imshow(scene_win, scene_big)
        if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
            print("[SimRun] User quit.")
            break

        time_step = env.step(action)
        total_reward += float(time_step.reward[0]) if time_step.reward is not None else 0.0
        step += 1
        if step % 50 == 0:
            print(f"step={step:4d} total_reward={total_reward:.3f} tof={base_env.last_tof_m:.4f}  hit={base_env.last_tof_hit}")

    print(f"\n[SimRun] steps={step} total_reward={total_reward:.3f}")
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
    run_sim(args.checkpoint, args.xml_path, args.episode_horizon, args.display_scale, args.wait_ms)