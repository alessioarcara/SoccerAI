"""
Run a list of training experiments sequentially and tabulate the validation
metrics of their best epoch (read back from the saved checkpoints).

Each experiment copies `configs/`, overrides `run_name` and any nested keys of
`base.yaml`, trains with `scripts/train.py` (W&B offline unless WANDB_MODE is
set) and appends one JSON line to `<out>/results.jsonl`.

Usage:
    python scripts/run_experiments.py                 # every experiment below
    python scripts/run_experiments.py gcn gcn_seed1   # a subset, by name
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml
from tabulate import tabulate

from soccerai.training.checkpoint import load_checkpoint
from soccerai.training.trainer_config import deep_merge

ROOT = Path(__file__).resolve().parents[1]

# name, model yaml (run_name), overrides of base.yaml, collector frames (0 = no plots)
EXPERIMENTS: list[tuple[str, str, dict[str, Any], int]] = [
    ("gcn", "gcn", {}, 12),
    ("graphsage", "graphsage", {}, 0),
    ("gatv2", "gatv2", {}, 0),
    ("gine", "gine", {}, 0),
    ("gcn2", "gcn2", {}, 0),
    ("graphgps", "graphgps", {}, 0),
    ("diffpool", "diffpool", {}, 0),
    ("gcn_roster_on", "gcn", {"data": {"use_roster_features": True}}, 0),
    (
        "gcn_no_direction_norm",
        "gcn",
        {"data": {"normalize_attack_direction": False}},
        0,
    ),
    ("gcn_no_goal_window", "gcn", {"data": {"goal_window_for_positives": None}}, 0),
    ("gcn_full_chains", "gcn", {"data": {"max_chain_len": None}}, 0),
    ("gcn_no_pos_weight", "gcn", {"trainer": {"pos_weight": None}}, 0),
    ("gcn_seed1", "gcn", {"seed": 1}, 0),
    ("gcn_seed2", "gcn", {"seed": 2}, 0),
]


def run_experiment(
    name: str,
    run_name: str,
    overrides: dict[str, Any],
    collector_frames: int,
    out: Path,
) -> dict[str, Any]:
    cfg_dir = out / f"cfg_{name}"
    shutil.rmtree(cfg_dir, ignore_errors=True)
    shutil.copytree(ROOT / "configs", cfg_dir)
    base = yaml.safe_load((cfg_dir / "base.yaml").read_text())
    base["run_name"] = run_name
    base["collector"]["n_frames"] = collector_frames
    base = deep_merge(base, overrides)
    (cfg_dir / "base.yaml").write_text(yaml.safe_dump(base))

    ckpt_dir = ROOT / "checkpoints" / run_name
    before = set(ckpt_dir.glob("*.pth")) if ckpt_dir.exists() else set()

    t0 = time.time()
    log = out / f"{name}.log"
    env = {"WANDB_MODE": "offline", **os.environ}
    with open(log, "w") as fh:
        proc = subprocess.run(
            [sys.executable, "scripts/train.py", "--config-dir", str(cfg_dir)],
            cwd=ROOT,
            stdout=fh,
            stderr=subprocess.STDOUT,
            env=env,
        )

    row: dict[str, Any] = {
        "name": name,
        "run_name": run_name,
        "exit": proc.returncode,
        "minutes": round((time.time() - t0) / 60, 1),
    }
    epochs = re.findall(
        r"Epoch:\s+\d+%\|[^|]*\|\s*(\d+)/\d+", log.read_text(errors="replace")
    )
    row["epochs"] = int(epochs[-1]) if epochs else None

    new = sorted(set(ckpt_dir.glob("*.pth")) - before, key=lambda p: p.stat().st_mtime)
    if new:
        payload = load_checkpoint(new[-1])
        metrics = payload.get("metrics", {})
        row.update(
            {
                "ckpt": new[-1].name,
                "val_loss": round(payload["best_value"], 4),
                "val_ap": round(metrics.get("val_average_precision", float("nan")), 3),
                "val_auroc": round(metrics.get("val_auroc", float("nan")), 3),
                "val_acc": round(metrics.get("val_accuracy", float("nan")), 3),
                "val_f1": round(metrics.get("val_f1.0_pos_score", float("nan")), 3),
            }
        )
    return row


def main(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    selected = [e for e in EXPERIMENTS if not args.names or e[0] in args.names]

    rows = []
    for name, run_name, overrides, frames in selected:
        row = run_experiment(name, run_name, overrides, frames, out)
        rows.append(row)
        with open(out / "results.jsonl", "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    keys = [
        "name",
        "epochs",
        "minutes",
        "val_loss",
        "val_ap",
        "val_auroc",
        "val_acc",
        "val_f1",
    ]
    print(
        tabulate(
            [[r.get(k) for k in keys] for r in rows], headers=keys, tablefmt="github"
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*", help="subset of experiment names")
    parser.add_argument("--out", default="experiments", help="output directory")
    main(parser.parse_args())
