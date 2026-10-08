import torch
from stubs import build_stub_config

from soccerai.training.checkpoint import (
    CHECKPOINT_FORMAT,
    find_best_checkpoint,
    load_checkpoint,
    save_checkpoint,
)


def test_checkpoint_round_trip_keeps_weights_and_config(tmp_path):
    cfg, raw = build_stub_config("gcn", tmp_path)
    path = tmp_path / "abc123_val_loss_0.5000.pth"
    save_checkpoint(
        path,
        cfg.model.state_dict(),
        raw,
        ["f1", "f2"],
        "val_loss",
        0.5,
        metrics={"val_loss": 0.5, "val_auroc": 0.8},
    )

    payload = load_checkpoint(path)
    assert payload["format"] == CHECKPOINT_FORMAT
    assert payload["config"] == raw
    assert payload["feature_names"] == ["f1", "f2"]
    assert payload["best_value"] == 0.5
    assert payload["metrics"] == {"val_loss": 0.5, "val_auroc": 0.8}
    rebuilt, _ = build_stub_config("gcn", tmp_path, {"seed": 0})
    rebuilt.model.load_state_dict(payload["state_dict"])
    for a, b in zip(
        cfg.model.state_dict().values(), rebuilt.model.state_dict().values()
    ):
        assert torch.equal(a, b)


def test_bare_state_dicts_are_still_loadable(tmp_path):
    path = tmp_path / "old.pth"
    torch.save({"w": torch.zeros(2)}, path)
    payload = load_checkpoint(path)
    assert payload["config"] is None and "w" in payload["state_dict"]


def test_find_best_checkpoint_searches_recursively_and_skips_legacy_files(tmp_path):
    cfg, raw = build_stub_config("gcn", tmp_path)
    state = cfg.model.state_dict()
    (tmp_path / "sub").mkdir()
    for name, value in [("aaa", 0.6006), ("sub/bbb", 0.5332), ("ccc", 0.5506)]:
        path = tmp_path / f"{name}_val_loss_{value:.4f}.pth"
        save_checkpoint(path, state, raw, [], "val_loss", value)
    torch.save({"w": torch.zeros(1)}, tmp_path / "old_val_loss_0.1000.pth")  # bare
    # format 1 stored a pydantic config the current code cannot rebuild
    torch.save(
        {"format": 1, "state_dict": state, "config": {"model": {}}},
        tmp_path / "v1_val_loss_0.2000.pth",
    )
    (tmp_path / "notes.txt").write_bytes(b"")

    run_id, path = find_best_checkpoint(tmp_path)
    assert run_id == "bbb" and path.name == "bbb_val_loss_0.5332.pth"
    run_id, _ = find_best_checkpoint(tmp_path, include_legacy=True)
    assert run_id == "old"
    assert find_best_checkpoint(tmp_path / "empty") is None
