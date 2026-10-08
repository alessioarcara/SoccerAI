import os

os.environ.setdefault("WANDB_MODE", "disabled")

import pytest  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from test_config import REPO_CONFIGS, _write_configs  # noqa: E402
from test_temporal_collate import N_FEAT, make_chain  # noqa: E402

from soccerai.data.temporal_dataset import TemporalChainsDataset  # noqa: E402
from soccerai.models.models import build_model  # noqa: E402
from soccerai.training.trainer_config import build_config  # noqa: E402

BACKBONES = ["gcn", "gcn2", "graphsage", "gatv2", "gine", "graphgps", "diffpool"]


class DatasetStub:
    num_node_features = N_FEAT
    num_global_features = 3
    # the carrier readout looks the flag up by name: use the first column
    feature_names = ["is_ball_carrier_1"] + [f"f{i}" for i in range(1, N_FEAT)]


def load_cfg(tmp_path, name: str):
    model_cfg = yaml.safe_load((REPO_CONFIGS / f"{name}.yaml").read_text())
    return build_config(_write_configs(tmp_path, name, model_cfg))


def run_chain(model, snapshots, x_override=None):
    h = c = None
    out = None
    for t, snap in enumerate(snapshots):
        x = x_override if (t == 0 and x_override is not None) else snap.x
        out, h, c = model(
            x=x,
            edge_index=snap.edge_index,
            u=snap.u,
            edge_weight=snap.edge_attr,
            edge_attr=snap.edge_attr,
            batch=snap.batch,
            batch_size=snap.num_graphs,
            prev_h=h,
            prev_c=c,
        )
    return out


@pytest.mark.parametrize("name", BACKBONES)
def test_every_backbone_runs_and_backpropagates(tmp_path, name):
    torch.manual_seed(0)
    cfg = load_cfg(tmp_path, name)
    model = build_model(cfg, DatasetStub())
    batch = TemporalChainsDataset.collate(
        [make_chain(3, 1.0, 0), make_chain(2, 0.0, 1)]
    )

    out = run_chain(model, list(batch))
    assert out.shape == (2, 1)
    out.sum().backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)


@pytest.mark.parametrize("mode", ["node", "graph"])
def test_temporal_context_reaches_the_head(tmp_path, mode):
    torch.manual_seed(0)
    cfg = load_cfg(tmp_path, "gcn")
    cfg.model.neck.mode = mode
    if mode == "node":
        cfg.model.head.din = cfg.model.neck.rnn_dout + cfg.model.neck.glob_dout
    model = build_model(cfg, DatasetStub()).eval()

    snapshots = list(TemporalChainsDataset.collate([make_chain(2, 1.0, 0)]))
    reference = run_chain(model, snapshots)
    perturbed = run_chain(model, snapshots, x_override=snapshots[0].x + 1.0)
    # the prediction at the last frame must depend on the first frame
    assert not torch.allclose(reference, perturbed)


def test_gnn_plus_layers_normalise_once(tmp_path):
    from soccerai.models.layers import GNNPlusLayer, Identity

    cfg = load_cfg(tmp_path, "gine")  # plus: True, norm: batch
    model = build_model(cfg, DatasetStub())
    assert all(isinstance(c, GNNPlusLayer) for c in model.backbone.convs)
    assert all(isinstance(n, Identity) for n in model.backbone.norms)


@pytest.mark.parametrize(
    "norm,expected", [("none", "NoneType"), ("layer", "LayerNorm")]
)
def test_graphgps_uses_the_configured_norm(tmp_path, norm, expected):
    cfg = load_cfg(tmp_path, "graphgps")
    cfg.model.backbone.norm = norm
    model = build_model(cfg, DatasetStub())
    assert type(model.backbone.convs[0].norm1).__name__ == expected


def test_diffpool_exposes_auxiliary_losses(tmp_path):
    cfg = load_cfg(tmp_path, "diffpool")
    model = build_model(cfg, DatasetStub())
    batch = TemporalChainsDataset.collate([make_chain(2, 1.0, 0)])
    run_chain(model, list(batch))
    assert model.aux_loss.shape == (1,) and model.aux_loss.requires_grad
    assert float(model.aux_loss.min()) >= 0.0


@pytest.mark.parametrize("name", [b for b in BACKBONES if b != "diffpool"])
def test_carrier_readout_runs_with_every_backbone(tmp_path, name):
    cfg = load_cfg(tmp_path, name)
    cfg.model.neck.carrier_readout = True
    model = build_model(cfg, DatasetStub())
    batch = TemporalChainsDataset.collate(
        [make_chain(3, 1.0, 0), make_chain(2, 0.0, 1)]
    )
    out = run_chain(model, list(batch))
    assert out.shape == (2, 1)
    out.sum().backward()


def test_carrier_readout_takes_the_flagged_node():
    from soccerai.models.necks import GraphGlobalFusion
    from soccerai.training.trainer_config import NeckConfig

    neck_cfg = NeckConfig(
        rnn_type="gru",
        readout="mean",
        glob_dout=2,
        rnn_din=4,
        rnn_dout=4,
        mode="graph",
        raw_features_proj=False,
        proj_dout=2,
    )
    fusion = GraphGlobalFusion(glob_din=1, cfg=neck_cfg, carrier_idx=0)
    x = torch.tensor([[0.0], [1.0], [0.0], [0.0]])  # carrier: node 1, none in graph 1
    z = torch.tensor([[1.0, 1.0], [5.0, 7.0], [2.0, 2.0], [4.0, 4.0]])
    batch = torch.tensor([0, 0, 1, 1])
    out = fusion(z, torch.zeros(2, 1), batch, 2, x)
    assert out[:, 2:4].tolist() == [[5.0, 7.0], [0.0, 0.0]]


def test_node_norm_keeps_the_graph_mean():
    from soccerai.models.backbones import NORMALIZATIONS

    norm = NORMALIZATIONS["node"](4)
    h = torch.randn(22, 4)
    batch = torch.zeros(22, dtype=torch.long)
    shifted = norm(h + 3.0, batch=batch, batch_size=1)
    graph = NORMALIZATIONS["graph"](4)
    # a per-graph norm maps a shifted graph to the same output, a per-node norm
    # reacts to the shift only through each node's own channels
    assert torch.allclose(
        graph(h + torch.tensor([3.0, 0, 0, 0]), batch=batch, batch_size=1),
        graph(h, batch=batch, batch_size=1),
        atol=1e-4,
    )
    assert not torch.allclose(
        norm(h + torch.tensor([3.0, 0, 0, 0]), batch=batch, batch_size=1),
        norm(h, batch=batch, batch_size=1),
        atol=1e-4,
    )
    assert shifted.shape == h.shape
