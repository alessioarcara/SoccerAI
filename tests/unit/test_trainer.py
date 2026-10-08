import os
from typing import Any

os.environ.setdefault("WANDB_MODE", "disabled")

import pytest  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from test_temporal_collate import N_FEAT, make_chain  # noqa: E402
from torch.optim import AdamW  # noqa: E402
from torch.optim.lr_scheduler import OneCycleLR  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torch_geometric.nn import global_mean_pool  # noqa: E402

from soccerai.data.temporal_dataset import TemporalChainsDataset  # noqa: E402
from soccerai.training.metrics import (  # noqa: E402
    BinaryConfusionMatrix,
    BinaryPrecisionRecallCurve,
)
from soccerai.training.trainer import TemporalTrainer  # noqa: E402


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


def _make_trainer(n_epochs: int, val_batch_size: int, model=None):
    train_chains: Any = [make_chain(3, 1.0, 0), make_chain(1, 0.0, 1)]
    val_chains: Any = [
        make_chain(2, 1.0, 2),
        make_chain(4, 0.0, 3),
        make_chain(1, 1.0, 4),
    ]
    train_loader: DataLoader = DataLoader(
        train_chains, batch_size=2, collate_fn=TemporalChainsDataset.collate
    )
    val_loader: DataLoader = DataLoader(
        val_chains, batch_size=val_batch_size, collate_fn=TemporalChainsDataset.collate
    )
    model = model or RecordingModel()
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-3,
        epochs=n_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
    )
    trainer = TemporalTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        n_epochs=n_epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        device="cpu",
        metrics=[
            BinaryConfusionMatrix(ignore_value=-1),
            BinaryPrecisionRecallCurve(-1),
        ],
        callbacks=[],
    )
    return trainer, model


def test_model_is_in_train_mode_during_every_epoch():
    trainer, model = _make_trainer(n_epochs=2, val_batch_size=3)
    trainer.train()

    # per epoch: 3 training forwards (T_max=3), then eval on train (3) and val (4)
    assert len(model.modes) == 2 * (3 + 3 + 4)
    assert model.modes[:3] == [True] * 3
    assert model.modes[10:13] == [True] * 3  # second epoch is trained in train mode
    assert not any(model.modes[13:])  # evaluation runs in eval mode


def test_val_loss_does_not_depend_on_batching():
    torch.manual_seed(0)
    trainer_a, model_a = _make_trainer(n_epochs=1, val_batch_size=3)
    torch.manual_seed(0)
    trainer_b, model_b = _make_trainer(n_epochs=1, val_batch_size=1)
    model_b.load_state_dict(model_a.state_dict())

    trainer_a.eval("val")
    trainer_b.eval("val")
    assert abs(trainer_a.history["val_loss"] - trainer_b.history["val_loss"]) < 1e-5
    assert trainer_a.history["val_auroc"] == trainer_b.history["val_auroc"]


def test_auxiliary_loss_is_added_to_the_training_loss():
    trainer, model = _make_trainer(n_epochs=1, val_batch_size=3)
    batch = next(iter(trainer.train_loader))
    base_loss, _ = trainer._compute_signal_loss_and_last_pred(batch)

    model.aux_loss = torch.tensor(2.0)
    trainer.aux_loss_weight = 0.5
    loss, _ = trainer._compute_signal_loss_and_last_pred(batch)
    assert loss.item() == pytest.approx(base_loss.item() + 0.5 * 2.0, abs=1e-5)


class PaddingAuxModel(RecordingModel):
    """Exposes a large per-graph auxiliary loss only on padded (all-zero) frames."""

    def forward(self, x, edge_index, u, batch=None, batch_size=None, **kwargs):
        out, h, c = super().forward(
            x, edge_index, u, batch=batch, batch_size=batch_size, **kwargs
        )
        mass = global_mean_pool(x.abs().sum(-1, keepdim=True), batch, size=batch_size)
        self.aux_loss = 100.0 * (mass.squeeze(-1) == 0).float()
        return out, h, c


def test_auxiliary_loss_ignores_padded_frames():
    torch.manual_seed(0)
    trainer, model = _make_trainer(n_epochs=1, val_batch_size=3)
    torch.manual_seed(0)
    trainer_pad, model_pad = _make_trainer(
        n_epochs=1, val_batch_size=3, model=PaddingAuxModel()
    )
    model_pad.load_state_dict(model.state_dict())

    batch = next(iter(trainer.train_loader))  # chain lengths 3 and 1
    loss, _ = trainer._compute_signal_loss_and_last_pred(batch)
    loss_pad, _ = trainer_pad._compute_signal_loss_and_last_pred(batch)
    assert loss_pad.item() == pytest.approx(loss.item(), abs=1e-5)


def test_pos_weight_balances_the_loss():
    from soccerai.data.utils import balanced_pos_weight

    assert balanced_pos_weight([1, 0, 0, 0]) == 3.0

    torch.manual_seed(0)
    trainer_a, model_a = _make_trainer(n_epochs=1, val_batch_size=3)
    torch.manual_seed(0)
    trainer_b, model_b = _make_trainer(n_epochs=1, val_batch_size=3)
    model_b.load_state_dict(model_a.state_dict())
    trainer_b.pos_weight = torch.tensor(3.0)
    trainer_b.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=trainer_b.pos_weight)

    batch = next(iter(trainer_a.train_loader))  # chain 0 positive, chain 1 negative
    loss_a, _ = trainer_a._compute_signal_loss_and_last_pred(batch)
    loss_b, _ = trainer_b._compute_signal_loss_and_last_pred(batch)
    assert loss_b.item() > loss_a.item()
