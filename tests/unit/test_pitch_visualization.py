import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from matplotlib.collections import PathCollection
from torch_geometric.data import Data
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal

from soccerai.data.temporal_dataset import TemporalChainsDataset
from soccerai.training.metrics import ChainCollector
from soccerai.training.utils import plot_chain_frames, plot_pitch_frames_grid

# x and y deliberately are not adjacent: feature ordering must not change plots.
FEATURE_NAMES = ["x", "goal_cos", "is_ball_carrier_1", "y", "is_possession_team_1"]
GRID = {"nrows": 6, "ncols": 4, "figheight": 12}


def frame(t=0):
    features = np.column_stack(
        [
            np.linspace(0.05, 0.90, 22) + 0.01 * t,
            np.ones(22),
            np.arange(22) == 9,
            np.linspace(0.10, 0.90, 22),
            np.arange(22) < 11,
        ]
    ).astype(np.float32)
    return Data(
        x=torch.from_numpy(features),
        jersey_numbers=torch.tensor(list(range(1, 12)) * 2),
    )


def player_points(ax):
    return next(
        c for c in ax.collections if isinstance(c, PathCollection)
    ).get_offsets()


def test_single_pitch_renders_correct_coordinates_with_y_upwards():
    data = frame()
    fig = plot_pitch_frames_grid(
        [(0.91, data)],
        FEATURE_NAMES,
        {"nrows": 1, "ncols": 1, "figheight": 3},
        item_name="Chain",
    )
    try:
        fig.canvas.draw()
        assert len(fig.axes) == 1
        ax = fig.axes[0]
        np.testing.assert_allclose(player_points(ax), data.x[:, [0, 3]].numpy())
        assert not ax.yaxis_inverted()
        assert ax.get_title() == "Chain 1 · p(shot)=0.91"
        assert [t.get_text() for t in fig.legends[0].texts] == [
            "Possession team",
            "Opponents",
            "Ball carrier",
        ]
    finally:
        plt.close(fig)


def test_grid_expands_instead_of_silently_dropping_collected_frames():
    entries = [(0.9, frame(t)) for t in range(5)]
    fig = plot_pitch_frames_grid(
        entries,
        FEATURE_NAMES,
        {"nrows": 1, "ncols": 2, "figheight": 3},
    )
    try:
        fig.canvas.draw()
        visible = [ax for ax in fig.axes if ax.get_visible()]
        assert len(visible) == 5
        for (_, data), ax in zip(entries, visible):
            np.testing.assert_allclose(player_points(ax), data.x[:, [0, 3]].numpy())
    finally:
        plt.close(fig)


def test_grid_uses_only_rows_needed_by_collected_frames():
    fig = plot_pitch_frames_grid([(0.8, frame())] * 5, FEATURE_NAMES, GRID)
    try:
        assert len(fig.axes) == 8  # two rows, rather than the fixed six
        assert sum(ax.get_visible() for ax in fig.axes) == 5
    finally:
        plt.close(fig)


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_attack_arrow_follows_goal_direction(direction):
    data = frame()
    data.x[:, FEATURE_NAMES.index("goal_cos")] = direction
    fig = plot_pitch_frames_grid([(0.8, data)], FEATURE_NAMES, GRID)
    try:
        arrow = next(text for text in fig.axes[0].texts if hasattr(text, "arrow_patch"))
        assert (arrow.xy[0] > arrow.xyann[0]) == (direction > 0)
    finally:
        plt.close(fig)


def test_no_direction_is_invented_when_goal_feature_is_missing():
    data = frame()
    indices = [0, 2, 3, 4]
    data.x = data.x[:, indices]
    features = [FEATURE_NAMES[i] for i in indices]
    fig = plot_pitch_frames_grid([(0.8, data)], features, GRID)
    try:
        assert not any(text.get_text() == "Attack" for text in fig.axes[0].texts)
    finally:
        plt.close(fig)


def visual_chain(length, label):
    return DynamicGraphTemporalSignal(
        edge_indices=[np.array([[0, 1], [1, 0]])] * length,
        edge_weights=[np.ones(2, dtype=np.float32)] * length,
        features=[frame(t).x.numpy() for t in range(length)],
        targets=[np.array([[label]], dtype=np.float32)] * length,
        u=[np.zeros((1, 1), dtype=np.float32)] * length,
        jersey_numbers=[frame().jersey_numbers.numpy()] * length,
        chain_label=[np.array([label], dtype=np.float32)] * length,
        time_to_shot=[np.array([np.inf], dtype=np.float32)] * length,
    )


def test_collector_last_frames_exclude_padding_and_match_confidence_order():
    batch = TemporalChainsDataset.collate(
        [
            visual_chain(3, 1),
            visual_chain(1, 1),
            visual_chain(2, 0),
        ]
    )
    preds = torch.tensor([[0.7, 0.85, 0.9], [0.75, 0.99, 0.88], [0.95, 0.999, 0.998]])
    labels = torch.tensor(batch.targets).squeeze(-1)
    collector = ChainCollector(1, FEATURE_NAMES, n_frames=3)
    collector.update(preds, labels, batch)
    assert [score for score, _ in collector.frames] == pytest.approx([0.95, 0.85])
    np.testing.assert_allclose(collector.frames[0][1].x.numpy(), frame(2).x.numpy())
    np.testing.assert_allclose(collector.frames[1][1].x.numpy(), frame(0).x.numpy())
    media = collector.plot()
    try:
        fig = media["TP_chains_last_frames"].data
        assert [ax.get_title() for ax in fig.axes] == [
            "Chain 1 · p(shot)=0.95",
            "Chain 2 · p(shot)=0.85",
        ]
        for (_, data), ax in zip(collector.frames, fig.axes):
            np.testing.assert_allclose(player_points(ax), data.x[:, [0, 3]].numpy())
        heatmap = media["TP_chains_predictions"].data.axes[0]
        assert [t.get_text() for t in heatmap.get_yticklabels()] == ["1", "2"]
        assert [t.get_text() for t in heatmap.get_xticklabels()] == ["1", "2", "3"]
        assert media["TP_chains_frames"].frames.shape[0] == 3
    finally:
        for item in media.values():
            if isinstance(getattr(item, "data", None), plt.Figure):
                plt.close(item.data)


def test_video_uses_same_coordinate_mapping_as_last_frame_grid():
    images = plot_chain_frames([frame(0), frame(1)], [0.7, 0.9], FEATURE_NAMES)
    assert images.shape[0] == 2 and images.shape[1] == 4
    assert images.shape[2] > 0 and images.shape[3] > 0
    assert not np.array_equal(images[0], images[1])
