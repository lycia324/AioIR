import random
import json
import os

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader, Subset

from datasets import build_dataset
from models import build_model
from utils.config import parse_yaml_opt, print_opt


def setup_seed(seed):
    pl.seed_everything(seed)


def build_logger(opt):
    logger_opt = opt.get("logger", {})
    use_wandb = logger_opt.get("use_wandb", False)
    if use_wandb:
        return WandbLogger(
            project=logger_opt.get("wandb_project", "AioIR"),
            name=logger_opt.get("name", opt.get("name", "AioIR-Train")),
        )
    return TensorBoardLogger(
        save_dir=logger_opt.get("tensorboard_dir", "logs/"),
        name=logger_opt.get("name", opt.get("name", "lightning_logs")),
    )


def _subset_dataset(dataset, sample_num, rng, precomputed_indices=None):
    if sample_num is None:
        return dataset
    total = len(dataset)
    if sample_num <= 0 or sample_num >= total:
        return dataset

    if precomputed_indices is not None:
        valid_indices = [int(i) for i in precomputed_indices if 0 <= int(i) < total]
        if len(valid_indices) >= sample_num:
            return Subset(dataset, valid_indices[:sample_num])

    indices = list(range(total))
    rng.shuffle(indices)
    return Subset(dataset, indices[:sample_num])


def _load_sample_indices(sample_indices_path):
    if not sample_indices_path or not os.path.exists(sample_indices_path):
        return {}
    with open(sample_indices_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and "task_indices" in payload:
        payload = payload["task_indices"]
    if not isinstance(payload, dict):
        return {}
    return payload


def _save_sample_indices(sample_indices_path, task_indices, sample_seed):
    if not sample_indices_path:
        return
    save_dir = os.path.dirname(sample_indices_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    payload = {
        "sample_seed": sample_seed,
        "task_indices": task_indices,
    }
    with open(sample_indices_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def build_val_dataloaders(opt):
    datasets_opt = opt.get("datasets", {})
    val_opt = datasets_opt.get("val")
    if not val_opt:
        return None, []

    loader_opt = val_opt.get("loader", {})
    default_sample_num = val_opt.get("sample_num", None)
    sample_seed = val_opt.get("sample_seed", opt.get("seed", 0))
    save_sample_indices = val_opt.get("save_sample_indices", False)
    sample_indices_path = val_opt.get("sample_indices_path", None)
    loaded_indices = _load_sample_indices(sample_indices_path)
    generated_indices = {}

    def _build_loader(dataset, sample_num, task_name, task_idx):
        local_seed = sample_seed + task_idx * 1009
        rng = random.Random(local_seed)
        task_key = str(task_name)
        precomputed = loaded_indices.get(task_key)
        dataset = _subset_dataset(
            dataset,
            sample_num=sample_num,
            rng=rng,
            precomputed_indices=precomputed,
        )
        if isinstance(dataset, Subset):
            generated_indices[task_key] = list(dataset.indices)
        else:
            generated_indices[task_key] = list(range(len(dataset)))

        return DataLoader(
            dataset,
            batch_size=loader_opt.get("batch_size", 1),
            shuffle=loader_opt.get("shuffle", False),
            drop_last=loader_opt.get("drop_last", False),
            num_workers=loader_opt.get("num_workers", 4),
            pin_memory=loader_opt.get("pin_memory", True),
        )

    task_opts = val_opt.get("tasks", [])
    if task_opts:
        val_loaders = []
        val_task_names = []
        for task_idx, task_opt in enumerate(task_opts):
            dataset_opt = task_opt.get("dataset_opt")
            if dataset_opt is None:
                raise KeyError("datasets.val.tasks[*].dataset_opt is required")

            task_name = task_opt.get("name", f"task_{task_idx}")

            dataset_kwargs = {}
            if "task" in task_opt:
                dataset_kwargs["task"] = task_opt["task"]
            if "addnoise" in task_opt:
                dataset_kwargs["addnoise"] = task_opt["addnoise"]
            if "sigma" in task_opt:
                dataset_kwargs["sigma"] = task_opt["sigma"]

            dataset = build_dataset(dataset_opt, **dataset_kwargs)
            if hasattr(dataset, "set_sigma") and "sigma" in task_opt:
                dataset.set_sigma(task_opt["sigma"])

            sample_num = task_opt.get("sample_num", default_sample_num)
            val_loaders.append(
                _build_loader(
                    dataset,
                    sample_num=sample_num,
                    task_name=task_name,
                    task_idx=task_idx,
                )
            )
            val_task_names.append(str(task_name))

        if save_sample_indices:
            _save_sample_indices(sample_indices_path, generated_indices, sample_seed)

        return val_loaders, val_task_names

    dataset_opt = val_opt.get("dataset_opt")
    if dataset_opt is None:
        return None, []

    dataset = build_dataset(dataset_opt)
    task_name = val_opt.get("name", "val")
    val_loader = _build_loader(
        dataset,
        sample_num=default_sample_num,
        task_name=task_name,
        task_idx=0,
    )
    if save_sample_indices:
        _save_sample_indices(sample_indices_path, generated_indices, sample_seed)
    return val_loader, [str(task_name)]


def main():
    opt, opt_path = parse_yaml_opt("AioIR training")
    print(f"Load option file: {opt_path}")
    print_opt(opt)

    seed = opt.get("seed", 0)
    setup_seed(seed)

    dataset_opt = opt["datasets"]["train"]
    loader_opt = opt["datasets"]["train_loader"]
    trainset = build_dataset(dataset_opt)
    trainloader = DataLoader(
        trainset,
        batch_size=loader_opt.get("batch_size_per_gpu", 8),
        shuffle=loader_opt.get("shuffle", True),
        drop_last=loader_opt.get("drop_last", True),
        num_workers=loader_opt.get("num_workers", 16),
        pin_memory=loader_opt.get("pin_memory", True),
    )
    val_loaders, val_task_names = build_val_dataloaders(opt)
    model = build_model(opt)
    if hasattr(model, "set_val_task_names"):
        model.set_val_task_names(val_task_names)

    ckpt_opt = opt.get("path", {})
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_opt.get("ckpt_dir", "train_ckpt"),
        every_n_epochs=ckpt_opt.get("save_every_n_epochs", 1),
        save_top_k=ckpt_opt.get("save_top_k", -1),
    )

    train_opt = opt.get("train", {})
    trainer_kwargs = {}
    if val_loaders is not None:
        trainer_kwargs["check_val_every_n_epoch"] = (
            opt.get("datasets", {}).get("val", {}).get("val_every_n_epochs", 1)
        )

    trainer = pl.Trainer(
        max_epochs=train_opt.get("epochs", 120),
        accelerator=train_opt.get("accelerator", "gpu"),
        devices=train_opt.get("devices", 1),
        strategy=train_opt.get("strategy", "auto"),
        logger=build_logger(opt),
        callbacks=[checkpoint_callback],
        **trainer_kwargs,
    )

    fit_kwargs = {
        "model": model,
        "train_dataloaders": trainloader,
        "ckpt_path": ckpt_opt.get("resume_ckpt", None),
    }
    if val_loaders is not None:
        fit_kwargs["val_dataloaders"] = val_loaders

    trainer.fit(
        **fit_kwargs,
    )


if __name__ == "__main__":
    main()
