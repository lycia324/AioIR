import importlib
import pkgutil

from utils.registry import ARCH_REGISTRY, build_network, register_arch


def _should_import(module_name):
    if module_name in {"registry"}:
        return False
    if module_name.endswith("_arch"):
        return True
    return False

def _auto_import_arch_modules():
    for module_info in pkgutil.iter_modules(__path__):
        module_name = module_info.name
        if _should_import(module_name):
            importlib.import_module(f"{__name__}.{module_name}")


_auto_import_arch_modules()

__all__ = ["ARCH_REGISTRY", "register_arch", "build_network"]
