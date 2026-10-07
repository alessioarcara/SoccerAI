import torch
from test_models import DatasetStub, load_cfg

from soccerai.models.models import build_model
from soccerai.training.checkpoint import (
    checkpoint_config,
    find_best_checkpoint,
    load_checkpoint,
    save_checkpoint,
)


def test_checkpoint_round_trip_rebuilds_the_model(tmp_path):
    cfg = load_cfg(tmp_path, "gcn")
    model = build_model(cfg, DatasetStub())
    path = tmp_path / "abc123_val_loss_0.5000.pth"
    save_checkpoint(
        path,
        model.state_dict(),
        cfg,
        ["f1", "f2"],
        "val_loss",
        0.5,
        metrics={"val_loss": 0.5, "val_auroc": 0.8},
    )

    payload = load_checkpoint(path)
    assert payload["feature_names"] == ["f1", "f2"]
    assert payload["best_value"] == 0.5
    assert payload["metrics"] == {"val_loss": 0.5, "val_auroc": 0.8}
    rebuilt = build_model(checkpoint_config(payload), DatasetStub())
    rebuilt.load_state_dict(payload["state_dict"])
    for a, b in zip(model.state_dict().values(), rebuilt.state_dict().values()):
        assert torch.equal(a, b)


def test_bare_state_dicts_are_still_loadable(tmp_path):
    path = tmp_path / "old.pth"
    torch.save({"w": torch.zeros(2)}, path)
    payload = load_checkpoint(path)
    assert payload["config"] is None and "w" in payload["state_dict"]


def test_find_best_checkpoint_searches_recursively(tmp_path):
    (tmp_path / "sub").mkdir()
    for name in [
        "aaa_val_loss_0.6006.pth",
        "sub/bbb_val_loss_0.5332.pth",
        "ccc_val_loss_0.5506.pth",
        "notes.txt",
    ]:
        (tmp_path / name).write_bytes(b"")
    run_id, path = find_best_checkpoint(tmp_path)
    assert run_id == "bbb" and path.name == "bbb_val_loss_0.5332.pth"
    assert find_best_checkpoint(tmp_path / "empty") is None
