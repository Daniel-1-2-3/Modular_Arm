import sys
from pathlib import Path
import pandas as pd

TRAIN_EPISODE_COLS = ["frame", "episode", "episode_reward"]
EVAL_EPISODE_COLS  = ["frame", "episode", "eval_episode", "eval_episode_reward"]

def split_log(csv_path: Path, out_dir: Path) -> None:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "step" not in df.columns:
        df["step"] = df.index

    def extract(cols, anchor):
        present = [c for c in cols if c in df.columns]
        if anchor not in present:
            return pd.DataFrame(columns=present)
        return df[df[anchor].notna()][present].dropna(how="all").reset_index(drop=True)

    actor_df = extract(["step", "actor_loss"], "actor_loss")
    recon_df = extract(["step", "recon_loss"], "recon_loss")

    if not recon_df.empty:
        recon_df = recon_df[recon_df["recon_loss"] > 0].reset_index(drop=True)

    if not actor_df.empty and not recon_df.empty:
        train_loss = pd.merge(actor_df, recon_df, on="step", how="outer").sort_values("step").reset_index(drop=True)
    elif not actor_df.empty:
        train_loss = actor_df
    else:
        train_loss = recon_df

    tables = {
        "train_episodes": extract(TRAIN_EPISODE_COLS, "episode_reward"),
        "train_losses":   train_loss,
        "eval_episodes":  extract(EVAL_EPISODE_COLS,  "eval_episode_reward"),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        if table.empty:
            print(f"[skip] {name} — no rows found")
            continue
        out_path = out_dir / f"{name}.csv"
        table.to_csv(out_path, index=False)
        print(f"[saved] {out_path}  ({len(table)} rows, {len(table.columns)} cols)")

if __name__ == "__main__": # ie python metrics_reformat.py Results/Iteration4/metrics.csv
    csv_path = Path(sys.argv[1])
    if not csv_path.exists():
        raise SystemExit(f"File not found: {csv_path}")
    out_dir = Path(sys.argv[2]) if len(sys.argv) >= 3 else csv_path.parent / "Metrics"
    split_log(csv_path, out_dir)