import os, json
import numpy as np
import cv2
from Arm_Env.arm_env import ArmEnv

def to_float01(img: np.ndarray) -> np.ndarray:
    x = img
    if x.ndim != 3:
        raise ValueError(f"bad shape {x.shape}")
    if x.shape[0] == 3 and x.shape[-1] != 3:
        x = np.transpose(x, (1, 2, 0))
    if x.shape[-1] != 3:
        raise ValueError(f"bad channels {x.shape}")
    if x.dtype == np.uint8:
        return x.astype(np.float32) / 255.0
    x = x.astype(np.float32, copy=False)
    mn, mx = float(np.nanmin(x)), float(np.nanmax(x))
    if mn >= -1e-3 and mx <= 1.0 + 1e-3:
        return x
    if mn >= -1e-3 and mx <= 255.0 + 1e-3:
        return x / 255.0
    raise ValueError(f"unexpected range min={mn:.4f} max={mx:.4f}")

def combine(n1, mean1, m21, n2, mean2, m22):
    if n1 == 0:
        return n2, mean2, m22
    d = mean2 - mean1
    n = n1 + n2
    mean = mean1 + d * (n2 / n)
    m2 = m21 + m22 + (d * d) * (n1 * n2 / n)
    return n, mean, m2

if __name__ == "__main__":
    env = ArmEnv(xml_path=os.path.join("Simulation", "Assets", "scene.xml"), display=False, episode_horizon=1)

    N = 1000
    n = 0
    mean = np.zeros(3, np.float64)
    m2 = np.zeros(3, np.float64)

    cv2.namedWindow("pov_obs", cv2.WINDOW_NORMAL)

    for i in range(N):
        ts = env.reset()
        if "pov_obs" not in ts:
            raise KeyError(f"missing 'pov_obs'; got {list(ts.keys())}")

        pov = ts["pov_obs"]

        show = pov
        if show.ndim == 3 and show.shape[0] == 3 and show.shape[-1] != 3:
            show = np.transpose(show, (1, 2, 0))
        if show.dtype != np.uint8:
            s = to_float01(show)
            show = np.clip(s * 255.0, 0, 255).astype(np.uint8)

        cv2.imshow("pov_obs", show[:, :, ::-1])
        if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
            break

        img01 = to_float01(pov).reshape(-1, 3).astype(np.float64, copy=False)
        n2 = img01.shape[0]
        mean2 = img01.mean(axis=0)
        m22 = ((img01 - mean2) ** 2).sum(axis=0)
        n, mean, m2 = combine(n, mean, m2, n2, mean2, m22)

        if (i + 1) % 50 == 0:
            print(f"{i+1:>4}/{N} mean={mean} std={np.sqrt(m2/n)}")

    std = np.sqrt(m2 / n) if n else np.zeros(3)
    out = {"n_pixels_total": int(n), "mean_rgb": mean.tolist(), "std_rgb": std.tolist(), "resets_used": int(i + 1)}
    print("\nmean =", mean)
    print("std  =", std)

    with open("MAE_Model/rgb_stats.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    
    with open("DrQv2_Architecture/rgb_stats.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    cv2.destroyAllWindows()