import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

def _numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def plot_eval(df: pd.DataFrame, out_dir: Path, title_prefix: str = "") -> None:
    xcol = _pick_col(df, ["frame", "frames", "global_frame", "step", "steps"])
    if xcol is None:
        raise RuntimeError(f"Could not find x-axis column. Columns: {list(df.columns)}")

    y_ep = _pick_col(df, ["eval_episode_reward"])
    y_mean = _pick_col(df, ["eval_episode_reward_mean"])
    y_len = _pick_col(df, ["eval_episode_length_mean", "eval_episode_length"])

    df = _numeric(df, [xcol, y_ep or "", y_mean or "", y_len or ""])

    eval_rows = df.copy()
    if y_ep is not None:
        eval_rows = eval_rows[eval_rows[y_ep].notna()]
    elif y_mean is not None:
        eval_rows = eval_rows[eval_rows[y_mean].notna()]

    if eval_rows.empty:
        raise RuntimeError("No eval rows found in this CSV (no eval reward columns with numeric values).")

    x = eval_rows[xcol].to_numpy()

    if y_ep is not None:
        y = eval_rows[y_ep].to_numpy()
        plt.figure()
        plt.plot(x, y, marker="o", linestyle="none")
        plt.xlabel(xcol)
        plt.ylabel("eval_episode_reward")
        plt.title(f"{title_prefix}Eval episode reward (scatter)")
        plt.tight_layout()
        plt.savefig(out_dir / "eval_episode_reward_scatter.png", dpi=200)
        plt.close()

        grp = eval_rows[[xcol, y_ep]].dropna().groupby(xcol)[y_ep]
        xs = np.array(list(grp.groups.keys()), dtype=np.float64)
        means = grp.mean().to_numpy()
        stds = grp.std(ddof=0).fillna(0.0).to_numpy()

        order = np.argsort(xs)
        xs, means, stds = xs[order], means[order], stds[order]

        plt.figure()
        plt.errorbar(xs, means, yerr=stds, fmt="o-")
        plt.xlabel(xcol)
        plt.ylabel("eval_episode_reward (mean ± std)")
        plt.title(f"{title_prefix}Eval reward mean ± std")
        plt.tight_layout()
        plt.savefig(out_dir / "eval_reward_mean_std.png", dpi=200)
        plt.close()

    if y_mean is not None:
        rows = df[df[y_mean].notna()].copy()
        rows = rows.sort_values(xcol)
        plt.figure()
        plt.plot(rows[xcol], rows[y_mean], marker="o")
        plt.xlabel(xcol)
        plt.ylabel("eval_episode_reward_mean")
        plt.title(f"{title_prefix}Eval reward mean (logged)")
        plt.tight_layout()
        plt.savefig(out_dir / "eval_reward_mean_logged.png", dpi=200)
        plt.close()

    if y_len is not None:
        rows = df[df[y_len].notna()].copy()
        rows = rows.sort_values(xcol)
        plt.figure()
        plt.plot(rows[xcol], rows[y_len], marker="o")
        plt.xlabel(xcol)
        plt.ylabel(y_len)
        plt.title(f"{title_prefix}Eval episode length")
        plt.tight_layout()
        plt.savefig(out_dir / "eval_episode_length.png", dpi=200)
        plt.close()


def plot_train(df: pd.DataFrame, out_dir: Path, title_prefix: str = "") -> None:
    xcol = _pick_col(df, ["frame", "frames", "global_frame", "step", "steps"])
    if xcol is None:
        raise RuntimeError(f"Could not find x-axis column. Columns: {list(df.columns)}")

    y_train = _pick_col(df, ["episode_reward", "train_episode_reward"])
    y_fps = _pick_col(df, ["fps"])
    y_loss = _pick_col(df, ["actor_loss", "total_loss"])

    cols = [c for c in [xcol, y_train, y_fps, y_loss] if c is not None]
    df = _numeric(df, cols)

    if y_train is not None:
        rows = df[df[y_train].notna()].sort_values(xcol)
        if not rows.empty:
            plt.figure()
            plt.plot(rows[xcol], rows[y_train], marker="o", linestyle="-")
            plt.xlabel(xcol)
            plt.ylabel(y_train)
            plt.title(f"{title_prefix}Train episode reward")
            plt.tight_layout()
            plt.savefig(out_dir / "train_episode_reward.png", dpi=200)
            plt.close()

    if y_fps is not None:
        rows = df[df[y_fps].notna()].sort_values(xcol)
        if not rows.empty:
            plt.figure()
            plt.plot(rows[xcol], rows[y_fps], marker="o", linestyle="-")
            plt.xlabel(xcol)
            plt.ylabel("fps")
            plt.title(f"{title_prefix}FPS")
            plt.tight_layout()
            plt.savefig(out_dir / "fps.png", dpi=200)
            plt.close()

    if y_loss is not None:
        rows = df[df[y_loss].notna()].sort_values(xcol)
        if not rows.empty:
            plt.figure()
            plt.plot(rows[xcol], rows[y_loss], marker="o", linestyle="-")
            plt.xlabel(xcol)
            plt.ylabel(y_loss)
            plt.title(f"{title_prefix}{y_loss}")
            plt.tight_layout()
            plt.savefig(out_dir / f"{y_loss}.png", dpi=200)
            plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=str, help="Path to logger CSV (e.g., Training_Results/log.csv)")
    ap.add_argument("--out", type=str, default=None, help="Output directory for plots (default: alongside CSV)")
    ap.add_argument("--title", type=str, default="", help="Optional title prefix")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    out_dir = Path(args.out) if args.out is not None else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_csv(csv_path)

    plot_eval(df, out_dir, title_prefix=args.title)
    plot_train(df, out_dir, title_prefix=args.title)

    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()