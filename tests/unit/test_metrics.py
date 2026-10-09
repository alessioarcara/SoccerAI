import numpy as np
import torch
from test_temporal_collate import make_chain

from soccerai.data.temporal_dataset import TemporalChainsDataset
from soccerai.training.metrics import (
    BinaryConfusionMatrix,
    BinaryPrecisionRecallCurve,
    chain_level_predictions,
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
