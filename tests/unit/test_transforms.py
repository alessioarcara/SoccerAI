import numpy as np
import polars as pl
import torch
from test_temporal_collate import make_chain

from soccerai.data.transformers import ClippedScaler
from soccerai.training.transforms import RandomHorizontalFlip, RandomVerticalFlip

FEATURES = ["x", "y", "vx", "vy", "cos", "sin", "goal_cos", "goal_sin", "dvx", "dvy"]


def test_horizontal_flip_returns_a_consistent_copy_of_the_chain():
    chain = make_chain(3, 1.0, 0)
    # make the chain carry the named features (N_FEAT = 4 -> use the first 4)
    names = FEATURES[:4]
    original = [f.copy() for f in chain.features]

    flip = RandomHorizontalFlip(names, p=1.0)
    flipped = flip(chain)

    assert flipped is not chain
    for t, (f_orig, f_new) in enumerate(zip(original, flipped.features)):
        np.testing.assert_allclose(chain.features[t], f_orig)  # source untouched
        np.testing.assert_allclose(f_new[:, 0], 1.0 - f_orig[:, 0])  # x
        np.testing.assert_allclose(f_new[:, 2], -f_orig[:, 2])  # vx
        np.testing.assert_allclose(f_new[:, [1, 3]], f_orig[:, [1, 3]])  # y, vy
    # the extra signal attributes are carried over
    assert flipped.u is chain.u and flipped.jersey_numbers is chain.jersey_numbers

    twice = flip(flipped)
    for f_orig, f_twice in zip(original, twice.features):
        np.testing.assert_allclose(f_twice, f_orig, atol=1e-6)


def test_vertical_flip_covers_every_y_dependent_feature():
    x = torch.tensor([[0.2, 0.3, 1.0, 2.0, 0.6, 0.8, 0.0, 1.0, 3.0, 4.0]])
    from torch_geometric.data import Data

    data = Data(x=x)
    out = RandomVerticalFlip(FEATURES, p=1.0)(data)
    expected = torch.tensor([[0.2, 0.7, 1.0, -2.0, 0.6, -0.8, 0.0, -1.0, 3.0, -4.0]])
    assert torch.allclose(out.x, expected)
    assert torch.allclose(
        x, torch.tensor([[0.2, 0.3, 1.0, 2.0, 0.6, 0.8, 0.0, 1.0, 3.0, 4.0]])
    )


def test_flip_is_skipped_with_probability_zero():
    chain = make_chain(2, 1.0, 1)
    out = RandomHorizontalFlip(FEATURES[:4], p=0.0)(chain)
    # BaseTransform shallow-copies its input: the arrays must be the same ones
    assert all(a is b for a, b in zip(out.features, chain.features))


def test_clipped_scaler_is_odd_and_bounded():
    df = pl.DataFrame(
        {"vx": [-30.0, -6.0, 0.0, 6.0, 30.0], "vy": [1.0, 2.0, 3.0, 4.0, 5.0]}
    )
    scaler = ClippedScaler(max_abs=12.0).fit(df)
    out = scaler.transform(df)
    np.testing.assert_allclose(out[:, 0], [-1.0, -0.5, 0.0, 0.5, 1.0])
    np.testing.assert_allclose(scaler.transform(-df.to_numpy()), -out)
    assert list(scaler.get_feature_names_out()) == ["vx", "vy"]


def test_clipped_scaler_names_array_columns():
    scaler = ClippedScaler(max_abs=12.0).fit(np.zeros((3, 2)))
    assert list(scaler.get_feature_names_out()) == ["x0", "x1"]
