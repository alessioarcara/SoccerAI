import torch
from eztrain import CheckpointCallback
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
    for run_id, value in [("gcn_a", 0.6006), ("gcn_b", 0.5332), ("gcn_c", 0.5506)]:
        (tmp_path / run_id).mkdir()
        path = tmp_path / run_id / f"best_val-loss_{value:.4f}.pth"
        save_checkpoint(path, state, raw, [], "val/loss", value)
    (tmp_path / "gcn_c" / "last.pth").write_bytes(b"")  # not a best file
    torch.save({"w": torch.zeros(1)}, tmp_path / "old_val_loss_0.1000.pth")  # bare
    # format 1 stored a pydantic config the current code cannot rebuild
    torch.save(
        {"format": 1, "state_dict": state, "config": {"model": {}}},
        tmp_path / "v1_val_loss_0.2000.pth",
    )

    run_id, path = find_best_checkpoint(tmp_path)
    assert run_id == "gcn_b" and path.name == "best_val-loss_0.5332.pth"
    _, path = find_best_checkpoint(tmp_path, include_legacy=True)
    assert path.name == "old_val_loss_0.1000.pth"
    assert find_best_checkpoint(tmp_path / "empty") is None


def _fit(tmp_path, overrides=None, stop_after=None):
    cfg, raw = build_stub_config("gcn", tmp_path, overrides)
    trainer = cfg.trainer
    trainer.config = raw
    if stop_after is not None:  # an interrupted run
        trainer.max_iterations = stop_after
    trainer.fit()
    return trainer


def test_only_the_best_checkpoint_is_kept(tmp_path):
    trainer = _fit(tmp_path, {"n_epochs": 3})
    run_dir = tmp_path / "gcn" / trainer.run.run_id
    best = sorted(p.name for p in run_dir.glob("best_*.pth"))
    assert len(best) == 1 and (run_dir / "last.pth").exists()


def test_resume_continues_the_same_run(tmp_path):
    first = _fit(tmp_path, {"n_epochs": 2}, stop_after=1)
    resumed = _fit(tmp_path, {"n_epochs": 2, "resume_from": first.run.run_id})
    assert resumed.run.run_id == first.run.run_id
    assert resumed.start_iteration == 1 and resumed.iteration == 2
    assert resumed.scheduler.last_epoch == resumed.scheduler.total_steps


def test_resume_with_a_new_name_forks_the_weights(tmp_path):
    first = _fit(tmp_path, {"n_epochs": 1})
    cfg, _ = build_stub_config(
        "gcn", tmp_path, {"run_name": "gcn_fork", "resume_from": first.run.run_id}
    )
    fork = cfg.trainer
    saver = next(c for c in fork.callbacks if isinstance(c, CheckpointCallback))
    saver.on_train_start(fork)  # restores before the first epoch
    assert fork.run.run_id != first.run.run_id and fork.start_iteration == 0
    for a, b in zip(
        first.model.state_dict().values(), fork.model.state_dict().values()
    ):
        assert torch.equal(a.cpu(), b.cpu())
