"""
Run a list of training experiments sequentially and tabulate the validation
metrics of their best epoch (read back from the saved checkpoints).

Each experiment trains with `scripts/train.py` on `configs/base.yaml`, the
model file `configs/models/<model>.yaml` and a file holding its overrides
(W&B offline unless WANDB_MODE is set), then appends one JSON line to
`<out>/results.jsonl`.

Usage:
    python scripts/run_experiments.py                 # every experiment below
    python scripts/run_experiments.py gcn gcn_seed1   # a subset, by name
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from tabulate import tabulate

from soccerai.training.checkpoint import load_checkpoint

ROOT = Path(__file__).resolve().parents[1]

# name, configs/models/<model>.yaml, overrides, collector frames (0 = no plots)
EXPERIMENTS: list[tuple[str, str, dict[str, Any], int]] = [
    ("gcn", "gcn", {}, 12),
    ("graphsage", "graphsage", {}, 0),
    ("gatv2", "gatv2", {}, 0),
    ("gine", "gine", {}, 0),
    ("gcn2", "gcn2", {}, 0),
    ("graphgps", "graphgps", {}, 0),
    ("diffpool", "diffpool", {}, 0),
    ("gcn_roster_on", "gcn", {"data_config": {"use_roster_features": True}}, 0),
    (
        "gcn_no_direction_norm",
        "gcn",
        {"data_config": {"normalize_attack_direction": False}},
        0,
    ),
    (
        "gcn_no_goal_window",
        "gcn",
        {"data_config": {"goal_window_for_positives": None}},
        0,
    ),
    ("gcn_full_chains", "gcn", {"data_config": {"max_chain_len": None}}, 0),
    ("gcn_no_pos_weight", "gcn", {"pos_weight": None}, 0),
    ("gcn_seed1", "gcn", {"seed": 1}, 0),
    ("gcn_seed2", "gcn", {"seed": 2}, 0),
]


def run_experiment(
    name: str,
    model: str,
    overrides: dict[str, Any],
    collector_frames: int,
    out: Path,
) -> dict[str, Any]:
    # the overrides become the last config file of the run (later files win)
    override_path = out / f"{name}.yaml"
    override_path.write_text(
        yaml.safe_dump({**overrides, "collector_frames": collector_frames})
    )
    configs = [
        ROOT / "configs" / "base.yaml",
        ROOT / "configs" / "models" / f"{model}.yaml",
        override_path,
    ]
    run_name = (
        overrides.get("run_name") or yaml.safe_load(configs[1].read_text())["run_name"]
    )

    ckpt_dir = ROOT / "checkpoints" / run_name
    before = set(ckpt_dir.rglob("best_*.pth")) if ckpt_dir.exists() else set()

    t0 = time.time()
    log = out / f"{name}.log"
    env = {"WANDB_MODE": "offline", **os.environ}
    with open(log, "w") as fh:
        proc = subprocess.run(
            [sys.executable, "scripts/train.py", "--configs", *map(str, configs)],
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
    new = sorted(
        set(ckpt_dir.rglob("best_*.pth")) - before, key=lambda p: p.stat().st_mtime
    )
    if new:
        best = new[-1]
        payload = load_checkpoint(best)
        metrics = payload.get("metrics", {})
        last = torch.load(best.parent / "last.pth", map_location="cpu")
        nan = float("nan")
        row.update(
            {
                "run_id": best.parent.name,
                "epochs": int(last["iteration"]),
                "val_loss": round(payload["best_value"], 4),
                "val_ap": round(metrics.get("val/average_precision", nan), 3),
                "val_auroc": round(metrics.get("val/auroc", nan), 3),
                "val_acc": round(metrics.get("val/accuracy", nan), 3),
                "val_f1": round(metrics.get("val/f1.0_pos_score", nan), 3),
            }
        )
    return row


def main(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    selected = [e for e in EXPERIMENTS if not args.names or e[0] in args.names]

    rows = []
    for name, model, overrides, frames in selected:
        row = run_experiment(name, model, overrides, frames, out)
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
