import importlib
import pkgutil

from utils.registry import DATASET_REGISTRY


def _should_import(module_name):
    if module_name in {"registry"}:
        return False
    if module_name.endswith("_dataset"):
        return True
    return False


def _auto_import_dataset_modules():
    for module_info in pkgutil.iter_modules(__path__):
        module_name = module_info.name
        if _should_import(module_name):
            importlib.import_module(f"{__name__}.{module_name}")


_auto_import_dataset_modules()


def build_dataset(dataset_opt, **kwargs):
    if "type" not in dataset_opt:
        raise KeyError("Dataset option must contain key: type")

    dataset_type = dataset_opt["type"]
    if dataset_type not in DATASET_REGISTRY:
        support = ", ".join(sorted(DATASET_REGISTRY.keys()))
        raise ValueError(f"Unknown dataset type: {dataset_type}. Available: [{support}]")

    init_opt = {k: v for k, v in dataset_opt.items() if k != "type"}
    return DATASET_REGISTRY[dataset_type](**init_opt, **kwargs)


__all__ = ["DATASET_REGISTRY", "build_dataset"]
