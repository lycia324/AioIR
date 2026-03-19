from models.base_model import BaseIRModel


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
