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

        stats_path = Path(stats_path)
        with stats_path.open("r") as f:
            stats = json.load(f)

        self.mean = np.array(stats["mean_rgb"], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(stats["std_rgb"], dtype=np.float32).reshape(1, 1, 3)

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

        if hasattr(base, "get_obs"):
            obs = base.get_obs()
            scene = obs["scene"]
            pov = obs["pov"]
        else:
            raise RuntimeError("Env must expose get_obs() returning dict with 'scene' and 'pov'")

        scene = self._to_u8_rgb(scene)
        pov = self._to_u8_rgb(pov)

        scene = cv2.resize(scene, (self.render_size, self.render_size), interpolation=cv2.INTER_AREA)
        pov = cv2.resize(pov, (self.render_size, self.render_size), interpolation=cv2.INTER_AREA)

        frame = np.concatenate([pov, scene], axis=1)
        self.frames.append(frame)

    def _to_u8_rgb(self, frame):
        arr = np.asarray(frame)
        if arr.dtype == np.uint8:
            return arr
        arr = arr.astype(np.float32, copy=False)
        arr = arr * self.std + self.mean
        arr = np.clip(arr, 0.0, 1.0)
        return (arr * 255.0 + 0.5).astype(np.uint8)

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