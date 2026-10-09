import math
import os
from typing import Any

os.environ.setdefault("WANDB_MODE", "disabled")

import pytest  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from eztrain import MetricCollection  # noqa: E402
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
        max_iterations=n_epochs,
        eval_freq=1,
        train_loader=train_loader,
        val_loader=val_loader,
        device="cpu",
        metrics=MetricCollection(
            [BinaryConfusionMatrix(ignore_value=-1), BinaryPrecisionRecallCurve(-1)]
        ),
        callbacks=[],
    )
    return trainer, model


def test_model_is_in_train_mode_during_every_epoch():
    trainer, model = _make_trainer(n_epochs=2, val_batch_size=3)
    trainer.fit()

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

    logs_a = trainer_a.evaluate()
    logs_b = trainer_b.evaluate()
    assert abs(logs_a["val/loss"] - logs_b["val/loss"]) < 1e-5
    assert logs_a["val/auroc"] == logs_b["val/auroc"]


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


def test_rank_loss_penalises_drops_on_positive_chains_only():
    logit = lambda p: torch.logit(torch.tensor(p))  # noqa: E731
    # (T_max, B): chain 0 positive rising, chain 1 positive dropping,
    # chain 2 negative dropping, chain 3 positive with a padded last frame
    probs = [[0.2, 0.8, 0.8, 0.5], [0.5, 0.4, 0.4, 0.6], [0.9, 0.4, 0.4, 0.01]]
    logits = torch.stack([logit(row) for row in probs])
    masks = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 0]]).float()
    positive = torch.tensor([True, True, False, True])

    loss = TemporalTrainer._rank_loss(logits, masks, positive)
    # chain 1: drops 0.4 and 0.4 over 3 frames; chain 3: padding ignored
    assert loss.item() == pytest.approx((0.8 / 3) / 4, abs=1e-6)


def test_box_decomposition_combines_the_heads_and_trains_each_on_its_frames():
    trainer = TemporalTrainer.__new__(TemporalTrainer)
    trainer.pos_weight = None
    out = torch.logit(torch.tensor([[0.5, 0.8, 0.1], [0.2, 0.9, 0.3]]))
    y = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    loss, logit = trainer._frame_loss(out, y)

    p_shot = torch.sigmoid(logit)
    assert p_shot.tolist() == pytest.approx(
        [0.5 * 0.8 + 0.5 * 0.1, 0.2 * 0.9 + 0.8 * 0.3]
    )
    bce = lambda p, t: -math.log(p if t else 1 - p)  # noqa: E731
    # frame 0 is in the box: box and shot | box heads; frame 1: box and shot | no box
    assert loss.tolist() == pytest.approx(
        [bce(0.5, 1) + bce(0.8, 1), bce(0.2, 0) + bce(0.3, 0)], rel=1e-5
    )


def test_box_decomposition_removes_the_class_weights_before_combining():
    trainer = TemporalTrainer.__new__(TemporalTrainer)
    trainer.pos_weight = torch.tensor([4.0, 1.0, 9.0])
    # weighted heads predict w times the true odds: true p = 0.2, 0.5, 0.1
    odds = torch.tensor([0.25 * 4, 1.0, (1 / 9) * 9])
    _, logit = trainer._frame_loss(torch.log(odds).unsqueeze(0), torch.zeros(1, 2))
    assert torch.sigmoid(logit).item() == pytest.approx(0.2 * 0.5 + 0.8 * 0.1, rel=1e-5)


def test_lead_multiplier_favours_frames_far_from_the_shot():
    trainer = TemporalTrainer.__new__(TemporalTrainer)
    trainer.lead_weight, trainer.lead_horizon, trainer.device = 2.0, 8.0, "cpu"
    batch = TemporalChainsDataset.collate(
        [make_chain(3, 1.0, 0), make_chain(1, 0.0, 1)]
    )
    # chain 0: shot 5, 3 and 1 s ahead; chain 1: no shot, then padding (NaN)
    m = trainer._lead_multiplier(batch)
    assert m[:, 0].tolist() == pytest.approx([1 + 2 * 5 / 8, 1 + 2 * 3 / 8, 1 + 2 / 8])
    assert m[:, 1].tolist() == [1.0, 1.0, 1.0]
