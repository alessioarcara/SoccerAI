import numpy as np
import polars as pl
import pytest
import torch
from factories import make_raw_df
from test_transformers import make_data_cfg, make_dataset_stub

from soccerai.data.converters import (
    BipartiteGraphConverter,
    FullyConnectedGraphConverter,
    create_graph_converter,
)


def _frame_df():
    rng = np.random.default_rng(0)
    xy = rng.uniform(0.1, 0.9, size=(22, 2))
    xy[0] = [0.5, 0.5]  # possession player at the pitch centre
    xy[11] = [0.5 + 5.0 / 105.0, 0.5]  # opponent 5 m away
    xy[12] = [0.5 + 20.0 / 105.0, 0.5]  # opponent 20 m away
    return pl.DataFrame(
        {
            "x": xy[:, 0],
            "y": xy[:, 1],
            "is_possession_team_1": [1.0] * 11 + [0.0] * 11,
        }
    )


def test_bipartite_weights_decay_with_distance_in_metres():
    ei, ew, ea = BipartiteGraphConverter(length_scale=10.0)._create_edges(_frame_df())

    assert ei.shape == (2, 2 * 11 * 11) and ew is ea
    teams = torch.tensor([1] * 11 + [0] * 11)
    assert bool((teams[ei[0]] != teams[ei[1]]).all())  # only player-opponent edges

    w = {(int(s), int(d)): float(x) for s, d, x in zip(ei[0], ei[1], ew)}
    assert w[(0, 11)] == pytest.approx(np.exp(-5.0 / 10.0), rel=1e-6)
    assert w[(0, 12)] == pytest.approx(np.exp(-20.0 / 10.0), rel=1e-6)
    assert w[(0, 11)] == max(w[(0, j)] for j in range(11, 22))
    assert all(0.0 < v <= 1.0 for v in w.values())


def test_fully_connected_has_no_weights():
    ei, ew, ea = FullyConnectedGraphConverter()._create_edges(_frame_df())
    assert ei.shape == (2, 22 * 21) and ew is None and ea is None


def test_factory_passes_the_length_scale():
    conv = create_graph_converter("bipartite", 7.5)
    assert isinstance(conv, BipartiteGraphConverter) and conv.length_scale == 7.5


def test_dataframe_to_graphs_end_to_end():
    raw = make_raw_df(
        [
            dict(game_id=1, chain_id=0, label=1, n_frames=3),
            dict(game_id=1, chain_id=1, label=0, n_frames=2, possession="away"),
        ]
    )
    ds = make_dataset_stub(make_data_cfg())
    df = ds._prepare_dataframe(raw)
    transformed = ds._create_preprocessor(df).fit_transform(df)
    data_list, feature_names = create_graph_converter(
        "bipartite", 10.0
    ).convert_dataframe_to_data_list(transformed)

    assert len(data_list) == 5
    for data in data_list:
        assert data.x.shape == (22, len(feature_names))
        assert data.edge_index.shape == (2, 242) and data.edge_weight.shape == (242,)
        assert data.u.shape[0] == 1 and data.y.shape == (1, 1)
    assert sorted({int(d.chain_id) for d in data_list}) == [0, 1]
    assert "x" in feature_names and "chain_id" not in feature_names
