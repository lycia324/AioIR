import lightning.pytorch as pl
import torch.nn as nn
import torch.optim as optim

from net import build_network
from utils.schedulers import LinearWarmupCosineAnnealingLR


class BaseIRModel(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.model_opt = opt.get("model", {})
        self.train_opt = opt.get("train", {})

        network_opt = self.model_opt.get("network_g")
        if network_opt is None:
            raise KeyError("model.network_g is required in yml")
        self.net = build_network(network_opt)

        self.loss_fn = self._build_loss(self.train_opt.get("pixel_opt", {"type": "L1Loss"}))

    def forward(self, x):
        return self.net(x)

    def _build_loss(self, loss_opt):
        loss_type = loss_opt.get("type", "L1Loss")
        loss_kwargs = {k: v for k, v in loss_opt.items() if k != "type"}
        if loss_type == "L1Loss":
            return nn.L1Loss(**loss_kwargs)
        if loss_type == "MSELoss":
            return nn.MSELoss(**loss_kwargs)
        raise ValueError(f"Unsupported loss type: {loss_type}")

    def _build_optimizer(self):
        optim_opt = self.train_opt.get("optimizer_g", {"type": "AdamW", "lr": 2e-4})
        optim_type = optim_opt.get("type", "AdamW")
        optim_kwargs = {k: v for k, v in optim_opt.items() if k != "type"}

        if optim_type == "AdamW":
            return optim.AdamW(self.parameters(), **optim_kwargs)
        if optim_type == "Adam":
            return optim.Adam(self.parameters(), **optim_kwargs)
        raise ValueError(f"Unsupported optimizer type: {optim_type}")

    def _build_scheduler(self, optimizer):
        sched_opt = self.train_opt.get("scheduler")
        if sched_opt is None:
            return None

        sched_type = sched_opt.get("type")
        sched_kwargs = {k: v for k, v in sched_opt.items() if k != "type"}

        if sched_type == "LinearWarmupCosineAnnealingLR":
            return LinearWarmupCosineAnnealingLR(optimizer=optimizer, **sched_kwargs)
        raise ValueError(f"Unsupported scheduler type: {sched_type}")

    def configure_optimizers(self):
        optimizer = self._build_optimizer()
        scheduler = self._build_scheduler(optimizer)
        if scheduler is None:
            return optimizer
        return [optimizer], [scheduler]
