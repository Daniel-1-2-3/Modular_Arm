from __future__ import annotations

from collections import deque
from typing import Any, NamedTuple, Dict as TypingDict
import numpy as np
from dm_env import StepType
from Arm_Env.arm_env import ArmEnv

def _to_chw_u8(pov_hwc_u8: np.ndarray) -> np.ndarray:
    """(H,W,3) uint8 -> (3,H,W) uint8 (no normalization)."""
    arr = np.asarray(pov_hwc_u8)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        arr = np.transpose(arr, (2, 0, 1))  # CHW
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8, copy=False)
    return arr

class FrameStackWrapper:
    """
    Stacks ONLY the POV image (single view) along channel dimension.

    Input from ArmEnv:
      info["pov_obs"]: (H,W,3) uint8 RGB
      info["tof_obs"]: float scalar (already normalized in ArmEnv)
    Output in info["observation"]:
      dict:
        "pov": (3*num_frames, H, W) float32 in [0,1]
        "tof": (1,) float32
    """
    def __init__(self, env: Any, num_frames: int):
        self._env = env
        self._num_frames = int(num_frames)
        self._frames: deque[np.ndarray] = deque([], maxlen=self._num_frames)

    def _transform(self, info: TypingDict[str, Any]) -> TypingDict[str, Any]:
        pov_hwc = info["pov_obs"]
        tof = info["tof_obs"]

        pov_chw = _to_chw_u8(pov_hwc) # uint8 CHW
        self._frames.append(pov_chw)
        if len(self._frames) != self._num_frames:
            last = self._frames[-1]
            while len(self._frames) < self._num_frames:
                self._frames.append(last)  # no copy

        stacked = np.concatenate(list(self._frames), axis=0)  # uint8 (3F,H,W)
        info["observation"] = {
            "pov": stacked,  # uint8
            "tof": np.array([tof], dtype=np.float32),
        }
        
        return info

    def reset(self):
        self._frames.clear()
        return self._transform(self._env.reset())

    def step(self, action):
        return self._transform(self._env.step(action))

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionRepeatWrapper:
    def __init__(self, env: Any, num_repeats: int):
        self._env = env
        self._num_repeats = int(num_repeats)

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for _ in range(self._num_repeats):
            info = self._env.step(action)
            reward += (info["reward"] or 0.0) * discount
            discount *= info["discount"]
            if info["step_type"] == StepType.LAST:
                break
        info["reward"] = reward
        info["discount"] = discount
        return info

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    pov: Any
    tof: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        return tuple.__getitem__(self, attr)


class ExtendedTimeStepWrapper:
    def __init__(self, env: Any):
        self._env = env

    def reset(self):
        return self._augment(self._env.reset(), action=None)

    def step(self, action):
        info = self._env.step(action)
        return self._augment(info, action=action)

    def _augment(self, info: TypingDict[str, Any], action=None) -> ExtendedTimeStep:
        if action is None:
            # Continuous action interface: zeros in [-1,1]
            action = np.zeros((getattr(self._env, "action_dim", 0) or info.get("action_dim", 0) or 0,), dtype=np.float32)
        obs = info["observation"]
        return ExtendedTimeStep(
            step_type=info["step_type"],
            reward=np.array([info["reward"] if info["reward"] is not None else 0.0], dtype=np.float32),
            discount=np.array([info["discount"]], dtype=np.float32),
            pov=obs["pov"].astype(np.uint8, copy=False),
            tof=obs["tof"].astype(np.float32, copy=False),
            action=np.asarray(action, dtype=np.float32),
        )

    def __getattr__(self, name):
        return getattr(self._env, name)
