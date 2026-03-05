import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
# CPU-friendly rendering:
os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "glfw")
os.environ["PYOPENGL_PLATFORM"] = os.environ.get("PYOPENGL_PLATFORM", "glfw")
os.environ.pop("MUJOCO_PLATFORM", None)

from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from Arm_Env.arm_env import ArmEnv
from DrQv2_Architecture.drqv2 import DrQV2Agent


class Control:
    """
    - Env renders at display resolution (big windows)
    - Policy always receives 64x64 stacked POV only
    """

    def __init__(
        self,
        weights_path: str = "Results/Iteration3/agent_weights.pt",
        device: str | None = None,
        num_frames: int = 3,
        policy_hw: int = 64,

        # Fallback defaults (used only if cfg is missing in checkpoint)
        mvmae_patch_size: int = 8,
        mvmae_encoder_embed_dim: int = 256,
        mvmae_decoder_embed_dim: int = 128,
        mvmae_encoder_heads: int = 16,
        mvmae_decoder_heads: int = 16,
        masking_ratio: float = 0.75,
        coef_mvmae: float = 0.005,
        feature_dim: int = 100,
        hidden_dim: int = 1024,
        critic_target_tau: float = 0.001,
        stddev_schedule: str = "linear(1.0,0.1,500000)",
        stddev_clip: float = 0.3,
    ):
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

        self.num_frames = int(num_frames)
        self.policy_hw = int(policy_hw)
        self.in_channels = 3 * self.num_frames
        self._frames = deque(maxlen=self.num_frames)

        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights not found: {weights_path}")

        ckpt = torch.load(weights_path, map_location=self.device)

        if "actor" not in ckpt or not isinstance(ckpt["actor"], dict):
            raise RuntimeError("Checkpoint missing 'actor' state_dict.")

        # Infer action_dim from actor last layer (policy.4.weight)
        action_dim = None
        for k, v in ckpt["actor"].items():
            if k.endswith("policy.4.weight") and isinstance(v, torch.Tensor):
                action_dim = int(v.shape[0])
                break
        if action_dim is None:
            raise RuntimeError("Could not infer action_dim from ckpt['actor'] (expected policy.4.weight).")

        cfg = ckpt.get("cfg", {}) if isinstance(ckpt.get("cfg", {}), dict) else {}
        # Prefer checkpoint cfg when present
        ckpt_patch = int(cfg.get("patch_size", mvmae_patch_size))
        ckpt_feat = int(cfg.get("feature_dim", feature_dim))
        ckpt_hidden = int(cfg.get("hidden_dim", hidden_dim))
        ckpt_mask = float(cfg.get("masking_ratio", masking_ratio))
        ckpt_action_shape = cfg.get("action_shape", (action_dim,))
        if isinstance(ckpt_action_shape, (list, tuple)) and len(ckpt_action_shape) >= 1:
            ckpt_action_dim = int(ckpt_action_shape[0])
        else:
            ckpt_action_dim = action_dim

        if ckpt_action_dim != action_dim:
            raise RuntimeError(f"Action dim mismatch: inferred {action_dim} but cfg says {ckpt_action_dim}")

        # Inference uses your runtime frame stack, so in_channels must match that.
        ckpt_in_ch = int(cfg.get("in_channels", self.in_channels))
        if ckpt_in_ch != self.in_channels:
            raise RuntimeError(
                f"in_channels mismatch: checkpoint cfg has {ckpt_in_ch}, "
                f"but you configured num_frames={self.num_frames} => in_channels={self.in_channels}."
            )

        # Policy image size for the encoder must match training (usually 64).
        ckpt_img_h = int(cfg.get("img_h_size", self.policy_hw))
        ckpt_img_w = int(cfg.get("img_w_size", self.policy_hw))
        if ckpt_img_h != self.policy_hw or ckpt_img_w != self.policy_hw:
            raise RuntimeError(
                f"policy_hw mismatch: checkpoint cfg is ({ckpt_img_h},{ckpt_img_w}) "
                f"but this script uses policy_hw={self.policy_hw}."
            )

        self.agent = DrQV2Agent(
            action_shape=(action_dim,),
            device=self.device,
            lr=1e-4,
            logger=None,

            mvmae_patch_size=ckpt_patch,
            mvmae_encoder_embed_dim=mvmae_encoder_embed_dim,
            mvmae_decoder_embed_dim=mvmae_decoder_embed_dim,
            mvmae_encoder_heads=mvmae_encoder_heads,
            mvmae_decoder_heads=mvmae_decoder_heads,

            in_channels=self.in_channels,
            img_h_size=self.policy_hw,
            img_w_size=self.policy_hw,

            masking_ratio=ckpt_mask,
            coef_mvmae=coef_mvmae,

            feature_dim=ckpt_feat,
            hidden_dim=ckpt_hidden,

            critic_target_tau=critic_target_tau,

            # Hard safety: no exploration randomness in control script
            num_expl_steps=0,

            update_every_steps=2,
            update_mvmae_every_steps=10,
            stddev_schedule=stddev_schedule,
            stddev_clip=stddev_clip,
        )

        missing = []
        for k in ("mvmae", "trunc", "actor", "critic"):
            if k not in ckpt:
                missing.append(k)
        if missing:
            raise RuntimeError(f"Checkpoint missing keys required for inference: {missing}")

        # Load the full inference path
        self.agent.mvmae.load_state_dict(ckpt["mvmae"])
        self.agent.trunc.load_state_dict(ckpt["trunc"])
        self.agent.actor.load_state_dict(ckpt["actor"])
        self.agent.critic.load_state_dict(ckpt["critic"])

        # Targets are optional for inference, but load if present
        if "critic_target" in ckpt:
            self.agent.critic_target.load_state_dict(ckpt["critic_target"])
        else:
            self.agent.critic_target.load_state_dict(self.agent.critic.state_dict())

        if "trunc_target" in ckpt:
            self.agent.trunc_target.load_state_dict(ckpt["trunc_target"])
        else:
            self.agent.trunc_target.load_state_dict(self.agent.trunc.state_dict())

        for p in self.agent.critic_target.parameters():
            p.requires_grad = False
        for p in self.agent.trunc_target.parameters():
            p.requires_grad = False

        self.agent.train(False)

    def reset(self):
        self._frames.clear()

    def _prep_pov_for_policy(self, pov_rgb_u8_hwc: np.ndarray) -> np.ndarray:
        if pov_rgb_u8_hwc.dtype != np.uint8:
            pov_rgb_u8_hwc = np.asarray(pov_rgb_u8_hwc, dtype=np.uint8)

        # Downsample render POV -> 64x64 for policy
        if pov_rgb_u8_hwc.shape[0] != self.policy_hw or pov_rgb_u8_hwc.shape[1] != self.policy_hw:
            pov_small = cv2.resize(pov_rgb_u8_hwc, (self.policy_hw, self.policy_hw), interpolation=cv2.INTER_AREA)
        else:
            pov_small = pov_rgb_u8_hwc

        # HWC -> CHW
        chw = np.transpose(pov_small, (2, 0, 1))  # (3,64,64)

        # Frame stack to (3*num_frames,64,64)
        if len(self._frames) == 0:
            for _ in range(self.num_frames):
                self._frames.append(chw.copy())
        else:
            self._frames.append(chw.copy())

        stacked = np.concatenate(list(self._frames), axis=0).astype(np.uint8)
        return stacked

    def act_from_env_dict(self, ts: dict, step: int) -> np.ndarray:
        pov_u8 = ts["pov_obs"]          # HWC uint8 (render size, big)
        tof_norm = float(ts["tof_obs"]) # already normalized in your ArmEnv.get_obs()

        pov_stack = self._prep_pov_for_policy(pov_u8)

        obs = {
            "pov": pov_stack,  # uint8 (C,H,W)
            "tof": np.array([tof_norm], dtype=np.float32),
        }

        return self.agent.act(obs, step=step, eval_mode=True).astype(np.float32)


def main():
    xml_path = str(Path("Simulation") / "Assets" / "scene.xml")

    display_w = 600
    display_h = 600

    env = ArmEnv(
        xml_path=xml_path,
        width=display_w,
        height=display_h,
        display=True,
        discount=0.99,
        episode_horizon=300,
    )

    ctrl = Control(weights_path="Results/Iteration4/agent_weights.pt", num_frames=3, policy_hw=64)

    ts = env.reset()
    ctrl.reset()

    step = 0
    action_repeat = 2

    while True:
        action = ctrl.act_from_env_dict(ts, step=step)

        for _ in range(action_repeat):
            ts = env.step(action)
            step += 1
            if ts["step_type"].name == "LAST":
                break

        if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
            break

        if ts["step_type"].name == "LAST":
            ts = env.reset()
            ctrl.reset()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()