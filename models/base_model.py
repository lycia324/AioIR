import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.optim as optim

from net import build_network
from utils.schedulers import LinearWarmupCosineAnnealingLR
from utils.val_utils import compute_psnr_ssim


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

        self.loss_fn = self._build_loss(
            self.train_opt.get("pixel_opt", {"type": "L1Loss"})
        )
        self.val_task_names = self._build_default_val_task_names()
        self._val_metric_state = {}

    def forward(self, x):
        return self.net(x)

    def _build_default_val_task_names(self):
        val_opt = self.opt.get("datasets", {}).get("val", {})
        task_opts = val_opt.get("tasks", [])
        if task_opts:
            return [str(t.get("name", f"task_{i}")) for i, t in enumerate(task_opts)]
        return [str(val_opt.get("name", "val"))]

    def set_val_task_names(self, task_names):
        if task_names:
            self.val_task_names = [str(name) for name in task_names]

    def _get_task_name(self, dataloader_idx):
        if dataloader_idx < len(self.val_task_names):
            return self.val_task_names[dataloader_idx]
        return f"task_{dataloader_idx}"

    def _forward_val_batch(self, batch):
        (_, degrad_patch, _) = batch
        model_out = self.net(degrad_patch)
        if isinstance(model_out, (tuple, list)):
            restored = model_out[0]
        else:
            restored = model_out
        return restored, model_out

    def _compute_val_loss(self, batch, restored, model_out):
        (_, _, clean_patch) = batch
        return self.loss_fn(restored, clean_patch)

    def on_validation_epoch_start(self):
        self._val_metric_state = {}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        (_, _, clean_patch) = batch
        task_name = self._get_task_name(dataloader_idx)
        restored, model_out = self._forward_val_batch(batch)
        val_loss = self._compute_val_loss(batch, restored, model_out)

        psnr, ssim, n = compute_psnr_ssim(restored, clean_patch)
        if task_name not in self._val_metric_state:
            self._val_metric_state[task_name] = {
                "loss_sum": 0.0,
                "psnr_sum": 0.0,
                "ssim_sum": 0.0,
                "count": 0.0,
            }

        state = self._val_metric_state[task_name]
        state["loss_sum"] += float(val_loss.detach().item()) * float(n)
        state["psnr_sum"] += float(psnr) * float(n)
        state["ssim_sum"] += float(ssim) * float(n)
        state["count"] += float(n)

    def on_validation_epoch_end(self):
        if not self._val_metric_state:
            return

        total_loss_sum = 0.0
        total_psnr_sum = 0.0
        total_ssim_sum = 0.0
        total_count = 0.0

        for task_name in self.val_task_names:
            if task_name not in self._val_metric_state:
                continue

            state = self._val_metric_state[task_name]
            stats = torch.tensor(
                [
                    state["loss_sum"],
                    state["psnr_sum"],
                    state["ssim_sum"],
                    state["count"],
                ],
                device=self.device,
                dtype=torch.float64,
            )
            gathered = self.all_gather(stats)
            if gathered.ndim > 1:
                stats = gathered.sum(dim=0)
            else:
                stats = gathered

            loss_sum, psnr_sum, ssim_sum, count = [float(v.item()) for v in stats]
            if count <= 0:
                continue

            task_loss = loss_sum / count
            task_psnr = psnr_sum / count
            task_ssim = ssim_sum / count

            metric_prefix = f"val/{task_name}"
            self.log(
                f"{metric_prefix}_loss",
                task_loss,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )
            self.log(
                f"{metric_prefix}_psnr",
                task_psnr,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )
            self.log(
                f"{metric_prefix}_ssim",
                task_ssim,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )

            total_loss_sum += loss_sum
            total_psnr_sum += psnr_sum
            total_ssim_sum += ssim_sum
            total_count += count

        if total_count > 0:
            self.log(
                "val/avg_loss",
                total_loss_sum / total_count,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
            self.log(
                "val/avg_psnr",
                total_psnr_sum / total_count,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
            self.log(
                "val/avg_ssim",
                total_ssim_sum / total_count,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )

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
