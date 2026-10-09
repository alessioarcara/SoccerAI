"""
Tabular reference models for the shot-prediction task.

The GNNs are compared against two classifiers (logistic regression and
gradient-boosted trees) trained on hand-crafted features of the *last frame*
of each chain, built from the very same processed dataset, split and node
features the GNNs consume. If a GNN cannot beat these numbers on the
validation split, the problem is upstream of the architecture.

Usage: python scripts/baseline.py   (reads configs/base.yaml like train.py)
"""

import argparse
from collections.abc import Sequence

import numpy as np
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal
from xgboost import XGBClassifier

from soccerai.config import DEFAULT_CONFIGS, build_config
from soccerai.data.config import X_GOAL_RIGHT, Y_GOAL
from soccerai.data.temporal_dataset import TemporalChainsDataset

PITCH = np.array([X_GOAL_RIGHT, 2 * Y_GOAL])  # length, width (metres)
PITCH_DIAG = float(np.hypot(*PITCH))


def chain_features(
    chain: DynamicGraphTemporalSignal, feature_names: Sequence[str]
) -> tuple[dict[str, float], float]:
    """Hand-crafted description of the last frame of a chain."""
    idx = {name: i for i, name in enumerate(feature_names)}
    x = chain.features[-1]
    u = chain.u[-1].reshape(-1)
    label = float(chain.targets[-1].reshape(-1)[0])

    possession = x[:, idx["is_possession_team_1"]] == 1
    carrier_mask = x[:, idx["is_ball_carrier_1"]] == 1
    if not carrier_mask.any():
        raise ValueError("The last frame of the chain has no ball carrier")
    carrier = int(np.argmax(carrier_mask))

    xy = x[:, [idx["x"], idx["y"]]] * PITCH  # metres
    c_xy = xy[carrier]
    opponents = xy[~possession]
    teammates = xy[possession & ~carrier_mask]

    # direction and distance from the carrier to the attacked goal
    if "goal_cos" in idx:
        goal_dir = np.array([x[carrier, idx["goal_cos"]], x[carrier, idx["goal_sin"]]])
        goal_dist = float(x[carrier, idx["goal_dist"]] * PITCH_DIAG)
    else:  # attacked goal assumed on the right (normalised frames)
        delta = np.array([PITCH[0], PITCH[1] / 2]) - c_xy
        goal_dist = float(np.linalg.norm(delta) + 1e-6)
        goal_dir = delta / goal_dist
    goal_xy = c_xy + goal_dir * goal_dist

    opp_delta = opponents - c_xy
    opp_dist = np.linalg.norm(opp_delta, axis=1)
    along = opp_delta @ goal_dir
    across = np.abs(opp_delta[:, 0] * goal_dir[1] - opp_delta[:, 1] * goal_dir[0])
    blockers = (along > 0) & (along < goal_dist) & (across < 4.0)

    feats = {
        "goal_dist": goal_dist,
        "goal_cos": float(goal_dir[0]),
        "goal_sin": float(goal_dir[1]),
        "carrier_vx": float(x[carrier, idx["vx"]]),
        "carrier_vy": float(x[carrier, idx["vy"]]),
        "carrier_speed": float(np.hypot(x[carrier, idx["vx"]], x[carrier, idx["vy"]])),
        "nearest_opp": float(opp_dist.min()) if opp_dist.size else PITCH_DIAG,
        "n_opp_3m": float((opp_dist < 3.0).sum()),
        "n_opp_6m": float((opp_dist < 6.0).sum()),
        "n_blockers": float(blockers.sum()),
        "n_opp_near_goal": float(
            (np.linalg.norm(opponents - goal_xy, axis=1) < 20.0).sum()
        ),
        "n_team_near_goal": float(
            (np.linalg.norm(teammates - goal_xy, axis=1) < 20.0).sum()
        ),
        "team_mean_goal_dist": float(
            np.linalg.norm(teammates - goal_xy, axis=1).mean()
        ),
        "chain_len": float(chain.snapshot_count),
    }
    for i, value in enumerate(u):
        feats[f"u{i}"] = float(value)
    return feats, label


def build_table(
    ds: TemporalChainsDataset,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rows, labels = [], []
    for chain in ds.temporal_chains:
        feats, label = chain_features(chain, ds.feature_names)
        rows.append(feats)
        labels.append(label)
    names = list(rows[0].keys())
    X = np.array([[r[n] for n in names] for r in rows], dtype=np.float64)
    return X, np.array(labels), names


def evaluate(name: str, y_true: np.ndarray, scores: np.ndarray) -> list:
    return [
        name,
        f"{average_precision_score(y_true, scores):.3f}",
        f"{roc_auc_score(y_true, scores):.3f}",
        f"{log_loss(y_true, np.clip(scores, 1e-6, 1 - 1e-6)):.3f}",
    ]


def main(args: argparse.Namespace) -> None:
    # same processed data, chains and split as the GNNs
    cfg, _ = build_config(args.configs)
    datasets = {"train": cfg.train_chains, "val": cfg.val_chains}

    X_train, y_train, names = build_table(datasets["train"])
    X_val, y_val, _ = build_table(datasets["val"])
    logger.info(
        "train: {} chains ({:.1%} positive) | val: {} chains ({:.1%} positive) | {} features",
        len(y_train),
        y_train.mean(),
        len(y_val),
        y_val.mean(),
        len(names),
    )

    models = {
        "logistic regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000),
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            # same class balancing as the logistic regression
            scale_pos_weight=float((y_train == 0).sum() / max((y_train == 1).sum(), 1)),
            eval_metric="aucpr",
            random_state=cfg.seed,
            n_jobs=-1,
        ),
    }

    rows = [
        evaluate("constant (prior)", y_val, np.full_like(y_val, y_train.mean())),
        evaluate("-goal distance", y_val, -X_val[:, names.index("goal_dist")]),
    ]
    for name, model in models.items():
        model.fit(X_train, y_train)
        rows.append(
            evaluate(f"{name} (train)", y_train, model.predict_proba(X_train)[:, 1])
        )
        rows.append(evaluate(f"{name} (val)", y_val, model.predict_proba(X_val)[:, 1]))

    print(
        tabulate(rows, headers=["model", "AP", "AUROC", "log-loss"], tablefmt="github")
    )

    if args.importance:
        gbm = models["xgboost"]
        from sklearn.inspection import permutation_importance

        imp = permutation_importance(
            gbm, X_val, y_val, scoring="roc_auc", n_repeats=10, random_state=cfg.seed
        )
        order = np.argsort(-imp.importances_mean)[:12]
        print("\nTop features by permutation importance (val AUROC drop):")
        print(
            tabulate(
                [[names[i], f"{imp.importances_mean[i]:.4f}"] for i in order],
                headers=["feature", "importance"],
                tablefmt="github",
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--importance", action="store_true")
    parser.add_argument(
        "--configs", nargs="+", default=[str(p) for p in DEFAULT_CONFIGS]
    )
    main(parser.parse_args())
