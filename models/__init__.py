import importlib
import pkgutil

from utils.registry import MODEL_REGISTRY


def _should_import(module_name):
    if module_name in {"registry"}:
        return False
    if module_name.endswith("_model"):
        return True
    return False


def _auto_import_model_modules():
    for module_info in pkgutil.iter_modules(__path__):
        module_name = module_info.name
        if _should_import(module_name):
            importlib.import_module(f"{__name__}.{module_name}")


_auto_import_model_modules()


def build_model(opt):
    model_opt = opt.get("model", {})
    model_type = model_opt.get("type", "AioIRModel")

    if model_type not in MODEL_REGISTRY:
        support = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown model type: {model_type}. Available: [{support}]")

    return MODEL_REGISTRY[model_type](opt=opt)


__all__ = ["MODEL_REGISTRY", "build_model"]
