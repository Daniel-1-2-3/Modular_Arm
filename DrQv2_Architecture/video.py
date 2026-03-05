import json
from pathlib import Path

import cv2
import imageio
import numpy as np
from Arm_Env.arm_env import ArmEnv


class VideoRecorder:
    def __init__(self, root_dir, render_size=256, fps=20, stats_path="DrQv2_Architecture/rgb_stats.json"):
        self.save_dir = None if root_dir is None else (Path(root_dir) / "eval_video")
        if self.save_dir is not None:
            self.save_dir.mkdir(exist_ok=True)

        self.render_size = int(render_size)
        self.fps = int(fps)
        self.frames = []
        self.enabled = False

        stats = json.loads(Path(stats_path).read_text())
        mean = np.array(stats["mean_rgb"], dtype=np.float32)
        std = np.array(stats["std_rgb"], dtype=np.float32)
        if mean.shape != (3,) or std.shape != (3,):
            raise ValueError(f"Expected mean/std shape (3,), got mean={mean.shape}, std={std.shape}")
        self.mean_chw = mean.reshape(3, 1, 1)
        self.std_chw = std.reshape(3, 1, 1)

    def _unwrap_env(self, env):
        cur = env
        for _ in range(32):
            if isinstance(cur, ArmEnv):
                return cur
            nxt = getattr(cur, "env", None)
            if nxt is None:
                nxt = getattr(cur, "_env", None)
            if nxt is None:
                nxt = getattr(cur, "unwrapped", None)
            if nxt is None or nxt is cur:
                return cur
            cur = nxt
        return cur

    def init(self, env, enabled=True):
        self.frames = []
        self.enabled = (self.save_dir is not None and bool(enabled))
        if self.enabled:
            self.record(env)

    def record(self, env):
        if not self.enabled:
            return

        base = self._unwrap_env(env)
        if not hasattr(base, "get_obs"):
            raise RuntimeError("Env must expose get_obs() returning dict with 'scene' and 'pov'")

        obs = base.get_obs()
        scene = self._to_u8_rgb(obs["scene"])
        pov = self._to_u8_rgb(obs["pov"])

        scene = cv2.resize(scene, (self.render_size, self.render_size), interpolation=cv2.INTER_AREA)
        pov = cv2.resize(pov, (self.render_size, self.render_size), interpolation=cv2.INTER_AREA)

        self.frames.append(np.concatenate([pov, scene], axis=1))

    def _to_u8_rgb(self, frame):
        arr = np.asarray(frame)

        if arr.dtype == np.uint8:
            if arr.ndim != 3 or arr.shape[2] != 3:
                raise ValueError(f"Expected uint8 HWC with 3 channels, got shape={arr.shape}, dtype={arr.dtype}")
            return arr

        if arr.dtype not in (np.float32, np.float64):
            arr = arr.astype(np.float32, copy=False)

        if arr.ndim != 3 or arr.shape[0] != 3:
            raise ValueError(f"Expected normalized float CHW (3,H,W), got shape={arr.shape}, dtype={arr.dtype}")

        arr = arr.astype(np.float32, copy=False)
        arr = arr * self.std_chw + self.mean_chw
        arr = np.clip(arr, 0.0, 1.0)

        arr = (arr * 255.0 + 0.5).astype(np.uint8)
        return np.transpose(arr, (1, 2, 0))

    def save(self, file_name):
        if not self.enabled:
            self.frames = []
            return

        stem = Path(file_name).stem
        mp4_path = self.save_dir / f"{stem}.mp4"
        gif_path = self.save_dir / f"{stem}.gif"

        try:
            with imageio.get_writer(str(mp4_path), fps=self.fps) as w:
                for f in self.frames:
                    w.append_data(f)
        except Exception:
            imageio.mimsave(str(gif_path), self.frames, fps=self.fps)

        self.frames = []