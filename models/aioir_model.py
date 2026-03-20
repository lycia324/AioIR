from models.base_model import BaseIRModel
import torch.nn as nn
import torch.optim as optim
from utils.registry import register_model

"""
Based on PromptIRModel from https://github.com/va1shn9v/PromptIR/blob/main/train.py
"""
@register_model("AioIRModel")
class AioIRModel(BaseIRModel):
    def __init__(self, opt):
        super().__init__(opt)

    def training_step(self, batch, batch_idx):
        (_, degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)
        loss = self.loss_fn(restored, clean_patch)
        self.log("train_loss", loss)
        return loss

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(self.current_epoch)