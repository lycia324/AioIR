from net.OriSSM_aio_arch import OriSSM_better as OriSSM_aio
from net.OriSSM_base import OriSSM_base
from net.OriSSM_super_better import OriSSM_better


ARCH_REGISTRY = {
    "OriSSM_base": OriSSM_base,
    "OriSSM_better": OriSSM_better,
    "OriSSM_aio": OriSSM_aio,
}


def build_network(opt):
    if "type" not in opt:
        raise KeyError("Network option must contain key: type")
    net_type = opt["type"]
    if net_type not in ARCH_REGISTRY:
        raise ValueError(f"Unknown network type: {net_type}")
    kwargs = {k: v for k, v in opt.items() if k != "type"}
    return ARCH_REGISTRY[net_type](**kwargs)
