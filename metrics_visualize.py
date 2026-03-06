import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Visualize:
    def __init__(self, metrics_dir: str | Path):
        self.dir = Path(metrics_dir)
        self._load()
        self._plot_eval_mean_reward()
        self._plot_train_reward()
        self._plot_actor_loss()
        self._plot_recon_loss()

    def _load(self):
        def read(name):
            p = self.dir / f"{name}.csv"
            if not p.exists():
                print(f"[skip] {name}.csv not found")
                return pd.DataFrame()
            return pd.read_csv(p)

        self.train_ep = read("train_episodes")
        self.train_loss = read("train_losses")
        self.eval_ep = read("eval_episodes")

    def _save(self, fig, name: str):
        path = self.dir / f"{name}.png"
        fig.savefig(path, dpi=150)

    def _plot_eval_mean_reward(self):
        df = self.eval_ep
        if df.empty or "eval_episode_reward" not in df.columns:
            return

        grouped = df.groupby("frame")["eval_episode_reward"]
        frames = np.array(sorted(grouped.groups.keys()))
        means = grouped.mean().to_numpy()
        stds = grouped.std(ddof=0).fillna(0).to_numpy()

        fig, ax = plt.subplots(figsize=(9, 5))
        fig.suptitle("Eval Mean Reward", fontsize=13)
        ax.errorbar(frames, means, yerr=stds, fmt="o-", capsize=4, linewidth=1.4, markersize=5, label="mean ± std")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Mean Episode Reward")
        ax.legend()
        fig.tight_layout()
        self._save(fig, "eval_mean_reward")

    def _plot_train_reward(self):
        df = self.train_ep
        if df.empty or "episode_reward" not in df.columns:
            return

        frames  = df["frame"].to_numpy()
        rewards = df["episode_reward"].to_numpy()

        fig, ax = plt.subplots(figsize=(9, 5))
        fig.suptitle("Train Episode Reward", fontsize=13)
        ax.plot(frames, rewards, linewidth=0.8, alpha=0.5, color="steelblue", label="episode reward")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Episode Reward")
        ax.legend()
        fig.tight_layout()
        self._save(fig, "train_episode_reward")

    def _plot_actor_loss(self):
        df = self.train_loss
        if df.empty or "actor_loss" not in df.columns:
            return

        mask = df["actor_loss"].notna()
        x = df.loc[mask, "step"].to_numpy() if "step" in df.columns else np.where(mask)[0]
        y = df.loc[mask, "actor_loss"].to_numpy()

        fig, ax = plt.subplots(figsize=(9, 5))
        fig.suptitle("Actor Loss", fontsize=13)
        ax.plot(x, y, linewidth=0.9, alpha=0.85, color="steelblue")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Actor Loss (-Q mean)")
        fig.tight_layout()
        self._save(fig, "actor_loss")

    def _plot_recon_loss(self):
        df = self.train_loss
        if df.empty or "recon_loss" not in df.columns:
            return

        mask = df["recon_loss"].notna()
        x = df.loc[mask, "step"].to_numpy() if "step" in df.columns else np.where(mask)[0]
        y = df.loc[mask, "recon_loss"].to_numpy()

        fig, ax = plt.subplots(figsize=(9, 5))
        fig.suptitle("Recon Loss (MAE)", fontsize=13)
        ax.plot(x, y, linewidth=0.9, alpha=0.85, color="darkorange")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Recon Loss")
        fig.tight_layout()
        self._save(fig, "recon_loss")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python metrics_visualize.py <path/to/Metrics>")
    Visualize(sys.argv[1])