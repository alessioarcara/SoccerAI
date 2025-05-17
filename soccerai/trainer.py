import torch.nn as nn
import wandb
from torch.optim import AdamW
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from tqdm.notebook import tqdm

from soccerai.config import TrainerConfig


class Trainer:
    def __init__(
        self,
        cfg: TrainerConfig,
        model: nn.Module,
        train_loader: DataLoader,
        device: str,
    ):
        self.cfg = cfg
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device

        self.optim = AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)

        self.criterion = nn.BCEWithLogitsLoss()

    def train(self, run_name: str):
        run = wandb.init(project=self.cfg.project_name, name=run_name, config=self.cfg)
        try:
            for epoch in tqdm(range(1, self.cfg.n_epochs + 1), desc="Epoch"):
                self.model.train()
                for batch in tqdm(self.train_loader, desc="Batch", leave=False):
                    loss = self._train_step(batch)

                    run.log({"train/batch_loss": loss})
        finally:
            run.finish()

    def _train_step(self, batch: Batch) -> float:
        self.optim.zero_grad(set_to_none=True)
        batch.to(self.device, non_blocking=True)
        out = self.model(batch)
        loss = self.criterion(out, batch.y.float().unsqueeze(1))
        loss.backward()
        self.optim.step()
        return loss
