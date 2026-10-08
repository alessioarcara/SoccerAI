import os

os.environ.setdefault("WANDB_MODE", "disabled")

import pytest  # noqa: E402
import torch  # noqa: E402
from ezconfy.core.exceptions import InstantiationError  # noqa: E402
from eztrain import CheckpointCallback  # noqa: E402
from stubs import MODELS, build_stub_config  # noqa: E402

from soccerai.config import _peek_seed  # noqa: E402
from soccerai.data.temporal_dataset import TemporalChainsDataset  # noqa: E402
from soccerai.training.checkpoint import (  # noqa: E402
    find_best_checkpoint,
    load_checkpoint,
)
from soccerai.training.trainer import TemporalTrainer  # noqa: E402


def test_every_model_config_exists():
    assert MODELS == [
        "diffpool",
        "gatv2",
        "gcn",
        "gcn2",
        "gine",
        "graphgps",
        "graphsage",
    ]


@pytest.mark.parametrize("model", MODELS)
def test_every_model_config_builds_and_trains_one_epoch(tmp_path, model):
    cfg, raw = build_stub_config(model, tmp_path)
    assert cfg.run_name == model
    assert isinstance(cfg.trainer, TemporalTrainer)
    assert isinstance(cfg.train_chains, TemporalChainsDataset)
    # the optimizer owns the parameters of the model the trainer trains
    trainer_params = {id(p) for p in cfg.trainer.model.parameters()}
    optim_params = {id(p) for g in cfg.optim.param_groups for p in g["params"]}
    assert trainer_params == optim_params

    cfg.trainer.config = raw
    cfg.trainer.fit()

    best = find_best_checkpoint(tmp_path / model)
    assert best is not None
    run_id, path = best
    assert run_id == cfg.trainer.run.run_id and run_id.startswith(f"{model}_")
    payload = load_checkpoint(path)
    assert payload["config"]["run_name"] == model
    assert set(payload["metrics"]) >= {"val/loss", "val/average_precision"}
    assert (path.parent / "last.pth").exists()


def test_checkpoint_config_rebuilds_the_same_model(tmp_path):
    cfg, raw = build_stub_config("gcn", tmp_path)
    cfg.trainer.config = raw
    cfg.trainer.fit()
    _, path = find_best_checkpoint(tmp_path / "gcn")
    payload = load_checkpoint(path)

    rebuilt, _ = build_stub_config(
        "gcn", tmp_path, {k: v for k, v in payload["config"].items() if k == "seed"}
    )
    rebuilt.model.load_state_dict(payload["state_dict"])
    for a, b in zip(
        cfg.model.state_dict().values(), rebuilt.model.state_dict().values()
    ):
        assert a.shape == b.shape


def test_widths_are_derived_from_the_modules(tmp_path):
    cfg, _ = build_stub_config("gine", tmp_path)
    # jumping knowledge: 3 layers x 64 channels reach the neck
    assert cfg.backbone.out_dim == 3 * 64
    assert cfg.neck.norm.normalized_shape == (3 * 64 + 32,)
    assert cfg.head.mlp[0].in_channels == cfg.neck.out_dim


def test_carrier_readout_widens_the_neck(tmp_path):
    plain, _ = build_stub_config("gcn", tmp_path)
    carrier, _ = build_stub_config(
        "gcn", tmp_path, {"neck": {"_init_args_": {"carrier_readout": True}}}
    )
    assert carrier.neck.fusion.out_dim == plain.neck.fusion.out_dim + 64


def test_overrides_reach_the_objects(tmp_path):
    cfg, _ = build_stub_config(
        "gcn", tmp_path, {"lr": 5e-4, "max_lr": 2e-3, "n_epochs": 3}
    )
    assert cfg.optim.param_groups[0]["initial_lr"] == pytest.approx(2e-3 / 25)
    assert cfg.scheduler.total_steps == 3 * cfg.steps_per_epoch
    assert cfg.trainer.max_iterations == 3
    saver = next(c for c in cfg.callbacks if isinstance(c, CheckpointCallback))
    assert saver.checkpointer.root == tmp_path


def test_peak_learning_rate_defaults_to_lr(tmp_path):
    cfg, _ = build_stub_config("gcn", tmp_path, {"lr": 3e-3})
    assert cfg.max_lr == 3e-3
    assert cfg.optim.param_groups[0]["max_lr"] == pytest.approx(3e-3)


def test_pos_weight_is_computed_on_the_training_chains(tmp_path):
    cfg, _ = build_stub_config("gcn", tmp_path)
    assert cfg.pos_weight == 1.0  # two positive and two negative chains
    assert float(cfg.trainer.pos_weight) == 1.0


def test_the_seed_is_fixed_before_the_model_is_built(tmp_path):
    a, _ = build_stub_config("gcn", tmp_path)
    b, _ = build_stub_config("gcn", tmp_path)
    c, _ = build_stub_config("gcn", tmp_path, {"seed": 7})
    wa, wb, wc = (
        torch.cat([p.flatten() for p in x.model.parameters()]) for x in (a, b, c)
    )
    assert torch.equal(wa, wb) and not torch.equal(wa, wc)


def test_seed_peeking_follows_the_merge_order(tmp_path):
    first, second = tmp_path / "a.yaml", tmp_path / "b.yaml"
    first.write_text("seed: 1\n")
    second.write_text("lr: 0.1\n")
    assert _peek_seed([first, second], None) == 1
    assert _peek_seed([first, second], {"seed": 3}) == 3


def test_an_unknown_target_fails_loudly(tmp_path):
    with pytest.raises(InstantiationError):
        build_stub_config(
            "gcn",
            tmp_path,
            {"backbone": {"_target_type_": "soccerai.models.backbones:Nope"}},
        )
