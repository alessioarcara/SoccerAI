import os

os.environ.setdefault("WANDB_MODE", "disabled")

import pytest  # noqa: E402
import torch  # noqa: E402
from stubs import MODELS, build_stub_config  # noqa: E402
from test_temporal_collate import make_chain  # noqa: E402

from soccerai.data.temporal_dataset import TemporalChainsDataset  # noqa: E402


def build_model(tmp_path, name: str, overrides=None):
    cfg, _ = build_stub_config(name, tmp_path, overrides)
    return cfg.model


def neck_args(**kwargs):
    return {"neck": {"_init_args_": kwargs}}


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


@pytest.mark.parametrize("name", MODELS)
def test_every_backbone_runs_and_backpropagates(tmp_path, name):
    model = build_model(tmp_path, name)
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
    # the head width follows the neck mode on its own
    model = build_model(tmp_path, "gcn", neck_args(mode=mode)).eval()

    snapshots = list(TemporalChainsDataset.collate([make_chain(2, 1.0, 0)]))
    reference = run_chain(model, snapshots)
    perturbed = run_chain(model, snapshots, x_override=snapshots[0].x + 1.0)
    # the prediction at the last frame must depend on the first frame
    assert not torch.allclose(reference, perturbed)


def test_gnn_plus_layers_normalise_once(tmp_path):
    from soccerai.models.layers import GNNPlusLayer, Identity

    model = build_model(tmp_path, "gine")  # plus: true
    assert all(isinstance(c, GNNPlusLayer) for c in model.backbone.convs)
    assert all(isinstance(n, Identity) for n in model.backbone.norms)


@pytest.mark.parametrize(
    "norm,expected", [("none", "NoneType"), ("layer", "LayerNorm")]
)
def test_graphgps_uses_the_configured_norm(tmp_path, norm, expected):
    model = build_model(
        tmp_path, "graphgps", {"backbone": {"_init_args_": {"norm": norm}}}
    )
    assert type(model.backbone.convs[0].norm1).__name__ == expected


def test_diffpool_exposes_auxiliary_losses(tmp_path):
    model = build_model(tmp_path, "diffpool")
    batch = TemporalChainsDataset.collate([make_chain(2, 1.0, 0)])
    run_chain(model, list(batch))
    assert model.aux_loss.shape == (1,) and model.aux_loss.requires_grad
    assert float(model.aux_loss.min()) >= 0.0


@pytest.mark.parametrize("name", [m for m in MODELS if m != "diffpool"])
def test_carrier_readout_runs_with_every_backbone(tmp_path, name):
    model = build_model(tmp_path, name, neck_args(carrier_readout=True))
    batch = TemporalChainsDataset.collate(
        [make_chain(3, 1.0, 0), make_chain(2, 0.0, 1)]
    )
    out = run_chain(model, list(batch))
    assert out.shape == (2, 1)
    out.sum().backward()


def test_carrier_readout_takes_the_flagged_node():
    from soccerai.models.necks import GraphGlobalFusion

    fusion = GraphGlobalFusion(
        node_dout=2, glob_din=1, glob_dout=2, carrier_readout=True, carrier_idx=0
    )
    assert fusion.out_dim == 2 * 2 + 2
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
