import numpy as np
import pytest
import torch
from test_temporal_collate import make_chain

from soccerai.data.temporal_dataset import TemporalChainsDataset
from soccerai.training.metrics import (
    BinaryConfusionMatrix,
    BinaryPrecisionRecallCurve,
    EarlyWarning,
    chain_level_predictions,
    frames_before_first_shot,
)


def _batch_and_predictions():
    chains = [make_chain(3, 1.0, 0), make_chain(1, 0.0, 1)]
    batch = TemporalChainsDataset.collate(chains)
    preds = torch.tensor([[0.1, 0.9], [0.2, 0.5], [0.8, 0.4]])  # (T_max, B)
    labels = torch.tensor([[1, 0], [1, -1], [1, -1]])
    return batch, preds, labels


def test_chain_level_predictions_take_last_valid_frame():
    batch, preds, labels = _batch_and_predictions()
    p, y = chain_level_predictions(preds, labels, batch.masks)
    np.testing.assert_allclose(p.numpy(), [0.8, 0.9])
    assert y.tolist() == [1, 0]


def test_confusion_matrix_counts_one_entry_per_chain():
    batch, preds, labels = _batch_and_predictions()
    cm = BinaryConfusionMatrix(threshold=0.5, fbeta=1.0, ignore_value=-1)
    cm.update(*chain_level_predictions(preds, labels, batch.masks), batch)
    # chain 0: label 1, last pred 0.8 -> TP ; chain 1: label 0, pred 0.9 -> FP
    assert cm.cm.tolist() == [[0, 1], [0, 1]]
    results = dict(cm.compute())
    assert results["accuracy"] == 0.5


def test_pr_curve_reports_ap_and_auroc():
    batch, preds, labels = _batch_and_predictions()
    m = BinaryPrecisionRecallCurve(ignore_value=-1)
    m.update(*chain_level_predictions(preds, labels, batch.masks), batch)
    results = dict(m.compute())
    assert set(results) == {"average_precision", "auroc"}
    assert results["auroc"] == 0.0  # the negative chain scored higher than the positive


def _early_warning_batch():
    # 2 positive chains (shots 1 s after the last frame, frames 2 s apart)
    # and 2 negative ones
    chains = [
        make_chain(3, 1.0, 0),
        make_chain(2, 1.0, 1),
        make_chain(3, 0.0, 2),
        make_chain(1, 0.0, 3),
    ]
    batch = TemporalChainsDataset.collate(chains)
    preds = torch.tensor(
        [
            [0.1, 0.1, 0.3, 0.2],
            [0.6, 0.1, 0.2, 0.0],
            [0.9, 0.2, 0.1, 0.0],
        ]
    )
    return batch, preds


def test_early_warning_measures_how_early_positives_are_flagged():
    batch, preds = _early_warning_batch()
    m = EarlyWarning(false_alarm_rate=0.5, min_lead=3.0, lead_bins=(0, 2, 4, 6))
    m.update(preds, torch.from_numpy(batch.targets), batch)
    results = m.compute()

    # negative peaks 0.3 and 0.2 -> threshold at their median
    assert results["early_threshold"] == pytest.approx(0.25)
    # chain 0 flagged at its 2nd frame, 3 s before the shot; chain 1 never
    assert results["early_recall"] == 0.5
    assert results["early_recall_3s"] == 0.5
    assert results["early_median_lead_s"] == pytest.approx(3.0)
    # frames 0-2 s before the shot (0.9, 0.1) against the negative frames
    # (0.3, 0.2, 0.1, 0.2): 4 wins, 1 tie out of 8 pairs
    assert results["auroc_lead_0-2s"] == pytest.approx(4.5 / 8)
    # frame targets of make_chain: the chain label on every frame
    assert 0 < results["frame_average_precision"] <= 1
    assert 0 <= results["frame_auroc"] <= 1
    assert set(m.plot()) == {"early_warning_curve"}


def test_frames_before_first_shot_stop_where_the_time_to_shot_jumps():
    inf = np.inf
    assert frames_before_first_shot(np.array([5.0, 3.0, 1.0, inf, inf])) == 3
    assert frames_before_first_shot(np.array([5.0, 1.0, 9.0, 4.0])) == 2
    assert frames_before_first_shot(np.array([2.0, 1.0])) == 2
    assert frames_before_first_shot(np.array([inf, inf])) == 0


def test_early_warning_ignores_alarms_after_the_shot():
    # positive chain flagged only after its shot (time to shot jumps to inf)
    chains = [make_chain(3, 1.0, 0), make_chain(2, 0.0, 1), make_chain(2, 0.0, 2)]
    batch = TemporalChainsDataset.collate(chains)
    batch.time_to_shot[:, 0] = [3.0, 1.0, np.inf]
    preds = torch.tensor([[0.1, 0.2, 0.3], [0.1, 0.1, 0.2], [0.9, -1.0, -1.0]])
    m = EarlyWarning(false_alarm_rate=0.5, min_lead=1.0)
    m.update(preds.clamp(min=0), torch.from_numpy(batch.targets), batch)
    assert m.compute()["early_recall"] == 0.0
