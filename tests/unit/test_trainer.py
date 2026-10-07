import os
from pathlib import Path

os.environ.setdefault("WANDB_MODE", "disabled")

import torch.nn as nn  # noqa: E402
from test_temporal_collate import N_FEAT, make_chain  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torch_geometric.nn import global_mean_pool  # noqa: E402

from soccerai.data.temporal_dataset import TemporalChainsDataset  # noqa: E402
from soccerai.training.metrics import (  # noqa: E402
    BinaryConfusionMatrix,
    BinaryPrecisionRecallCurve,
)
from soccerai.training.trainer import TemporalTrainer  # noqa: E402
from soccerai.training.trainer_config import build_config  # noqa: E402

REPO_CONFIGS = Path(__file__).resolve().parents[2] / "configs"


class RecordingModel(nn.Module):
    """Minimal temporal model that records its train/eval mode at every call."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(N_FEAT, 1)
        self.modes = []

    def forward(
        self,
        x,
        edge_index,
        u,
        edge_weight=None,
        edge_attr=None,
        batch=None,
        batch_size=None,
        prev_h=None,
        prev_c=None,
    ):
        self.modes.append(self.training)
        out = global_mean_pool(self.lin(x), batch, size=batch_size)
        return out, prev_h, prev_c


def _make_trainer(n_epochs: int, val_batch_size: int):
    cfg = build_config(REPO_CONFIGS)
    cfg.trainer.n_epochs = n_epochs
    cfg.trainer.eval_rate = 1
    cfg.trainer.bs = 2

    train_chains = [make_chain(3, 1.0, 0), make_chain(1, 0.0, 1)]
    val_chains = [make_chain(2, 1.0, 2), make_chain(4, 0.0, 3), make_chain(1, 1.0, 4)]
    train_loader = DataLoader(
        train_chains, batch_size=2, collate_fn=TemporalChainsDataset.collate
    )
    val_loader = DataLoader(
        val_chains, batch_size=val_batch_size, collate_fn=TemporalChainsDataset.collate
    )
    model = RecordingModel()
    trainer = TemporalTrainer(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device="cpu",
        metrics=[
            BinaryConfusionMatrix(cfg.metrics, -1),
            BinaryPrecisionRecallCurve(-1),
        ],
        callbacks=[],
    )
    return trainer, model


def test_model_is_in_train_mode_during_every_epoch():
    trainer, model = _make_trainer(n_epochs=2, val_batch_size=3)
    trainer.train("unit-test")

    # per epoch: 3 training forwards (T_max=3), then eval on train (3) and val (4)
    assert len(model.modes) == 2 * (3 + 3 + 4)
    assert model.modes[:3] == [True] * 3
    assert model.modes[10:13] == [True] * 3  # second epoch is trained in train mode
    assert not any(model.modes[13:])  # evaluation runs in eval mode
