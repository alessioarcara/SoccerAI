from typing import Any, Dict, Generic, List, Tuple, TypeVar

import torch
from torch_geometric.seed import seed_everything

T = TypeVar("T")


def fix_random(seed: int):
    seed_everything(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TopKStorage(Generic[T]):
    def __init__(self, k: int) -> None:
        self.k = k
        self._items: List[Tuple[float, T]] = []

    def add(self, entry: Tuple[float, T]) -> None:
        self._items.append(entry)
        self._items.sort(key=lambda x: x[0], reverse=True)
        if len(self._items) > self.k:
            self._items = self._items[: self.k]

    def clear(self) -> None:
        self._items.clear()

    def get_all_entries(self) -> List[Tuple[float, T]]:
        return list(self._items)


def build_dummy_inputs(
    bs: int, feat_dim: int, glob_dim: int, device: torch.device
) -> Dict[str, Any]:
    """
    Creates random tensors to feed `torch_geometric.nn.summary`.
    """
    num_nodes_total = 22 * bs
    num_edges_total = 11 * 22 * bs

    x = torch.rand((num_nodes_total, feat_dim), device=device)
    edge_index = torch.randint(
        0, num_nodes_total, (2, num_edges_total), dtype=torch.long, device=device
    )
    u = torch.rand((bs, glob_dim), device=device)
    batch = torch.tensor([[i] * 22 for i in range(bs)], device=device).view(-1)
    return dict(x=x, edge_index=edge_index, u=u, batch=batch)
