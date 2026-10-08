import argparse
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data.dataloader import DataLoader as TorchDataLoader
from tqdm import tqdm

from soccerai.config import build_config_from_dict
from soccerai.training.checkpoint import find_best_checkpoint, load_checkpoint
from soccerai.training.metrics import BinaryConfusionMatrix, BinaryPrecisionRecallCurve
from soccerai.training.trainer import resolve_device


def evaluate(
    model: nn.Module,
    loader: TorchDataLoader,
    device: str,
    threshold: float,
    fbeta: float,
):
    model.eval()

    cm = BinaryConfusionMatrix(threshold=threshold, fbeta=fbeta, mode="both")
    ap = BinaryPrecisionRecallCurve()

    cm.reset()
    ap.reset()

    with torch.inference_mode():
        for signal in tqdm(loader, desc="signals", leave=False):
            h = c = None
            last_preds: torch.Tensor | None = None

            for snapshot in signal:
                snapshot = snapshot.to(device, non_blocking=True)
                mask = snapshot.masks.bool()

                out, h, c = model(
                    x=snapshot.x,
                    edge_index=snapshot.edge_index,
                    edge_weight=snapshot.edge_attr,
                    edge_attr=snapshot.edge_attr,
                    u=snapshot.u,
                    batch=snapshot.batch,
                    batch_size=snapshot.num_graphs,
                    prev_h=h,
                    prev_c=c,
                )

                if last_preds is None:
                    last_preds = torch.zeros_like(out)

                last_preds[mask] = out[mask]

            assert last_preds is not None

            preds_probs = torch.sigmoid(last_preds)
            true_labels = torch.tensor(signal.targets[0], device=device)

            cm.update(preds_probs, true_labels, snapshot)
            ap.update(preds_probs, true_labels, snapshot)

        cm_results = cm.compute()
        ap_results = ap.compute()

        print("Evaluation results:")
        for name, value in {**cm_results, **ap_results}.items():
            print(f"{name}: {value:.4f}")

        cm_np = cm.cm.cpu().numpy()
        tn, fp = cm_np[0, 0], cm_np[0, 1]
        fn, tp = cm_np[1, 0], cm_np[1, 1]

        print("\n" + "Confusion Matrix".center(35))
        print("\n" + " " * 14 + "Predicted")
        print(" " * 15 + "0     1")
        print(" " * 11 + "┌─────┬─────┐")
        print("  True   0 │ {:>3} │ {:>3} │".format(tn, fp))
        print("  Label    ├─────┼─────┤")
        print("         1 │ {:>3} │ {:>3} │".format(fn, tp))
        print(" " * 11 + "└─────┴─────┘\n")


def main(args):
    device = resolve_device("auto")
    logger.info("Using device: {}", device)

    model_dir = Path("checkpoints") / args.name
    best = find_best_checkpoint(model_dir)
    if best is None:
        logger.warning("No checkpoints were found in {}", model_dir)
        raise SystemExit(1)

    _, ckpt_path = best
    logger.info("Evaluating checkpoint {}", ckpt_path)
    payload = load_checkpoint(ckpt_path)

    # rebuild the run exactly as it was trained, from the YAML it stored
    with tempfile.TemporaryDirectory() as tmp:
        cfg, _ = build_config_from_dict(payload["config"], tmp)

    model = cfg.model
    model.load_state_dict(payload["state_dict"])
    model.to(device)

    evaluate(model, cfg.val_loader, device, args.threshold, args.fbeta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        help="Experiment directory under 'checkpoints/'",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )
    parser.add_argument("--fbeta", type=float, default=1.0)

    args = parser.parse_args()
    main(args)
