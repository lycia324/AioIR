import random

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader

from datasets import build_dataset
from models import build_model
from utils.config import parse_yaml_opt


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_logger(opt):
    logger_opt = opt.get("logger", {})
    use_wandb = logger_opt.get("use_wandb", False)
    if use_wandb:
        return WandbLogger(
            project=logger_opt.get("wandb_project", "AioIR"),
            name=logger_opt.get("name", opt.get("name", "AioIR-Train")),
        )
    return TensorBoardLogger(save_dir=logger_opt.get("tensorboard_dir", "logs/"))


def main():
    opt, opt_path = parse_yaml_opt("AioIR training")
    print(f"Load option file: {opt_path}")

    seed = opt.get("seed", 0)
    setup_seed(seed)

    dataset_opt = opt["datasets"]["train"]
    loader_opt = opt["datasets"]["train_loader"]
    trainset = build_dataset(dataset_opt)
    trainloader = DataLoader(
        trainset,
        batch_size=loader_opt.get("batch_size", 8),
        shuffle=loader_opt.get("shuffle", True),
        drop_last=loader_opt.get("drop_last", True),
        num_workers=loader_opt.get("num_workers", 16),
        pin_memory=loader_opt.get("pin_memory", True),
    )

    model = build_model(opt)

    ckpt_opt = opt.get("path", {})
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_opt.get("ckpt_dir", "train_ckpt"),
        every_n_epochs=ckpt_opt.get("save_every_n_epochs", 1),
        save_top_k=ckpt_opt.get("save_top_k", -1),
    )

    train_opt = opt.get("train", {})
    trainer = pl.Trainer(
        max_epochs=train_opt.get("epochs", 120),
        accelerator=train_opt.get("accelerator", "gpu"),
        devices=train_opt.get("devices", 1),
        strategy=train_opt.get("strategy", "auto"),
        logger=build_logger(opt),
        callbacks=[checkpoint_callback],
    )

    trainer.fit(
        model=model,
        train_dataloaders=trainloader,
        ckpt_path=ckpt_opt.get("resume_ckpt", None),
    )


if __name__ == "__main__":
    main()
